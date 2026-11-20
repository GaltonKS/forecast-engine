"""HTTP API around the projection engine.

Run locally:  uvicorn app.main:app --reload
Docs:         http://localhost:8000/docs
"""
from __future__ import annotations

import io
import logging
import secrets
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import date, datetime
from decimal import Decimal
from typing import Iterator

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from engine import ENGINE_VERSION, ForecastRequest, ForecastResponse, load_csv, run_forecast, to_monthly
from engine.ingest import IngestError
from engine.schema import LineItem

from .config import Settings
from .models import Account, Actual, Base, Forecast, Tenant

log = logging.getLogger("forecast_api")


# ---------- response models ----------

class TenantIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class TenantOut(BaseModel):
    id: uuid.UUID
    name: str
    created_at: datetime


class UploadSummary(BaseModel):
    rows_received: int
    accounts: int
    new_accounts: int
    months_in_history: int
    history_start: date
    history_end: date
    gaps: list[str]


class MonthlyRow(BaseModel):
    period: date
    revenue: float
    cogs: float
    opex: float
    other: float
    gross_profit: float
    net_income: float


class ForecastRecord(ForecastResponse):
    id: uuid.UUID
    created_at: datetime


class ForecastSummary(BaseModel):
    id: uuid.UUID
    created_at: datetime
    engine_version: str
    horizon_months: int
    last_actual: date
    overrides: dict
    methods: dict[str, str] = {}  # line -> model used, e.g. {"revenue": "ets"}


# ---------- app factory ----------

def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
    engine = create_engine(settings.database_url, pool_pre_ping=True, connect_args=connect_args)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fine for now; switch to Alembic migrations once the schema starts changing.
        Base.metadata.create_all(engine)
        if not settings.api_key:
            log.warning("API_KEY is not set: the API is open. Set it before deploying.")
        yield
        engine.dispose()

    app = FastAPI(title="Financial Projection Engine", version=ENGINE_VERSION, lifespan=lifespan)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # ----- dependencies -----

    def get_db() -> Iterator[Session]:
        with SessionLocal() as session:
            yield session

    def require_api_key(request: Request) -> None:
        if not settings.api_key:
            return
        supplied = request.headers.get("x-api-key") or ""
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            supplied = auth[7:]
        if not secrets.compare_digest(supplied.encode(), settings.api_key.encode()):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    def get_tenant(tenant_id: uuid.UUID, db: Session = Depends(get_db)) -> Tenant:
        tenant = db.get(Tenant, tenant_id)
        if not tenant:
            raise HTTPException(status_code=404, detail="Tenant not found")
        return tenant

    def load_monthly(db: Session, tenant_id: uuid.UUID):
        rows = db.execute(
            select(Actual.period, Actual.amount, Account.name, Account.category)
            .join(Account, Account.id == Actual.account_id)
            .where(Actual.tenant_id == tenant_id)
        ).all()
        if not rows:
            raise HTTPException(status_code=422, detail="No financial data uploaded for this tenant yet")
        items = [
            LineItem(period=p, account=n, category=c, amount=float(a)) for p, a, n, c in rows
        ]
        return to_monthly(items)

    auth = [Depends(require_api_key)]

    # ----- routes -----

    @app.get("/health")
    def health():
        return {"status": "ok", "engine_version": ENGINE_VERSION}

    @app.post("/tenants", response_model=TenantOut, status_code=201, dependencies=auth)
    def create_tenant(body: TenantIn, db: Session = Depends(get_db)):
        tenant = Tenant(name=body.name)
        db.add(tenant)
        db.commit()
        return TenantOut(id=tenant.id, name=tenant.name, created_at=tenant.created_at)

    @app.get("/tenants/{tenant_id}", response_model=TenantOut, dependencies=auth)
    def read_tenant(tenant: Tenant = Depends(get_tenant)):
        return TenantOut(id=tenant.id, name=tenant.name, created_at=tenant.created_at)

    @app.post("/tenants/{tenant_id}/actuals/upload", response_model=UploadSummary, dependencies=auth)
    async def upload_actuals(
        file: UploadFile = File(...),
        tenant: Tenant = Depends(get_tenant),
        db: Session = Depends(get_db),
    ):
        """Upload a CSV (period, account, category, amount). Re-uploading a month
        replaces that account's amount for the month, so corrections are safe."""
        content = await file.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="File too large")
        try:
            items = load_csv(io.BytesIO(content))
        except IngestError as e:
            raise HTTPException(status_code=422, detail=str(e))
        if not items:
            raise HTTPException(status_code=422, detail="CSV has no rows")

        categories: dict[str, str] = {}
        totals: dict[tuple[str, date], float] = defaultdict(float)
        for it in items:
            cat = it.category.value
            if categories.setdefault(it.account, cat) != cat:
                raise HTTPException(
                    status_code=422,
                    detail=f"Account '{it.account}' appears with two categories in this file",
                )
            totals[(it.account, it.period)] += it.amount

        existing = {
            a.name: a
            for a in db.scalars(select(Account).where(Account.tenant_id == tenant.id))
        }
        new_accounts = 0
        for name, cat in categories.items():
            acct = existing.get(name)
            if acct is None:
                acct = Account(tenant_id=tenant.id, name=name, category=cat)
                db.add(acct)
                existing[name] = acct
                new_accounts += 1
            elif acct.category != cat:
                raise HTTPException(
                    status_code=422,
                    detail=f"Account '{name}' is already stored as '{acct.category}', not '{cat}'",
                )
        db.flush()

        for (name, period), amount in totals.items():
            db.merge(
                Actual(
                    tenant_id=tenant.id,
                    account_id=existing[name].id,
                    period=period,
                    amount=Decimal(str(round(amount, 2))),
                )
            )
        db.commit()

        monthly = load_monthly(db, tenant.id)
        return UploadSummary(
            rows_received=len(items),
            accounts=len(categories),
            new_accounts=new_accounts,
            months_in_history=len(monthly),
            history_start=monthly.index.min().date(),
            history_end=monthly.index.max().date(),
            gaps=monthly.attrs.get("gaps", []),
        )

    @app.get("/tenants/{tenant_id}/actuals/monthly", response_model=list[MonthlyRow], dependencies=auth)
    def monthly_actuals(tenant: Tenant = Depends(get_tenant), db: Session = Depends(get_db)):
        """Historical months, ready to chart next to a forecast."""
        m = load_monthly(db, tenant.id)
        return [
            MonthlyRow(
                period=idx.date(),
                revenue=r.revenue,
                cogs=r.cogs,
                opex=r.opex,
                other=r.other,
                gross_profit=round(r.revenue - r.cogs, 2),
                net_income=round(r.revenue - r.cogs - r.opex + r.other, 2),
            )
            for idx, r in m.iterrows()
        ]

    @app.post("/tenants/{tenant_id}/forecasts", response_model=ForecastRecord, status_code=201, dependencies=auth)
    def create_forecast(
        body: ForecastRequest,
        tenant: Tenant = Depends(get_tenant),
        db: Session = Depends(get_db),
    ):
        """Run a forecast. By default each line's model is picked by backtest
        (see `models` in the response). Send overrides to model a scenario, e.g.
        {"horizon_months": 12, "overrides": {"revenue_growth": 0.02}}, or force a
        model with {"method": "ets"}."""
        monthly = load_monthly(db, tenant.id)
        try:
            resp = run_forecast(monthly, body)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        record = Forecast(
            tenant_id=tenant.id,
            engine_version=resp.engine_version,
            horizon_months=body.horizon_months,
            last_actual=resp.last_actual_period,
            overrides=body.overrides.model_dump(exclude_none=True),
            result=resp.model_dump(mode="json"),
        )
        db.add(record)
        db.commit()
        return ForecastRecord(id=record.id, created_at=record.created_at, **resp.model_dump())

    @app.get("/tenants/{tenant_id}/forecasts", response_model=list[ForecastSummary], dependencies=auth)
    def list_forecasts(
        limit: int = 50,
        tenant: Tenant = Depends(get_tenant),
        db: Session = Depends(get_db),
    ):
        rows = db.scalars(
            select(Forecast)
            .where(Forecast.tenant_id == tenant.id)
            .order_by(Forecast.created_at.desc())
            .limit(min(max(limit, 1), 200))
        )
        return [
            ForecastSummary(
                id=f.id,
                created_at=f.created_at,
                engine_version=f.engine_version,
                horizon_months=f.horizon_months,
                last_actual=f.last_actual,
                overrides=f.overrides,
                methods={k: v.get("method", "") for k, v in (f.result.get("models") or {}).items()},
            )
            for f in rows
        ]

    @app.get("/tenants/{tenant_id}/forecasts/{forecast_id}", response_model=ForecastRecord, dependencies=auth)
    def read_forecast(
        forecast_id: uuid.UUID,
        tenant: Tenant = Depends(get_tenant),
        db: Session = Depends(get_db),
    ):
        f = db.get(Forecast, forecast_id)
        if not f or f.tenant_id != tenant.id:
            raise HTTPException(status_code=404, detail="Forecast not found")
        return ForecastRecord(id=f.id, created_at=f.created_at, **f.result)

    return app


app = create_app()
