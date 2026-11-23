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
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Iterator

from typing import Literal

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from engine import ENGINE_VERSION, ForecastRequest, ForecastResponse, load_csv, run_forecast, to_monthly
from engine.ingest import IngestError
from engine.schema import LineItem, ScenarioRequest
from llm.schema import QueryRequest

from .config import Settings
from . import jobs as jobs_mod
from .jobs import NoData, execute_job, expire_stale_jobs, save_forecast
from .jobs import load_monthly as _load_monthly
from .models import Account, Actual, Base, Forecast, Job, Tenant

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
    prompt: str | None = None  # set when the forecast came from /query


class JobSummary(BaseModel):
    id: uuid.UUID
    kind: Literal["forecast", "scenarios", "query"]
    status: Literal["queued", "running", "done", "failed"]
    error: str | None = None
    forecast_id: uuid.UUID | None = None  # forecast jobs: the saved forecast
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class JobOut(JobSummary):
    request: dict
    result: dict | None = None  # ForecastResponse, ScenarioComparison or QueryResult once done


def _job_summary(j: Job) -> dict:
    return dict(
        id=j.id, kind=j.kind, status=j.status, error=j.error, forecast_id=j.forecast_id,
        created_at=j.created_at, started_at=j.started_at, finished_at=j.finished_at,
    )


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
        try:
            return _load_monthly(db, tenant_id)
        except NoData as e:
            raise HTTPException(status_code=422, detail=str(e))

    job_timeout = timedelta(minutes=settings.job_timeout_minutes)
    celery = None
    if settings.broker_url:
        from .worker import make_celery

        celery = make_celery(settings.broker_url)

    def enqueue(db: Session, tenant: Tenant, kind: str, request: dict, background: BackgroundTasks) -> Job:
        load_monthly(db, tenant.id)  # fail fast if there's no data
        job = Job(tenant_id=tenant.id, kind=kind, request=request)
        db.add(job)
        db.commit()
        if celery is None:
            background.add_task(execute_job, SessionLocal, job.id)
            return job
        try:
            from .worker import TASK_NAME

            celery.send_task(TASK_NAME, args=[str(job.id)])
        except Exception as e:
            log.exception("could not enqueue job %s", job.id)
            job.status, job.error = "failed", f"Could not reach the job queue ({e.__class__.__name__})"
            db.commit()
            raise HTTPException(status_code=503, detail="Job queue unavailable, try again shortly")
        return job

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
        record = save_forecast(db, tenant.id, body, resp)
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
                prompt=f.prompt,
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

    # ----- background jobs -----

    @app.post("/tenants/{tenant_id}/forecast-jobs", response_model=JobOut, status_code=202, dependencies=auth)
    def start_forecast_job(
        body: ForecastRequest,
        background: BackgroundTasks,
        tenant: Tenant = Depends(get_tenant),
        db: Session = Depends(get_db),
    ):
        """Queue a forecast (same body as POST /forecasts). Poll
        GET /tenants/{id}/jobs/{job_id} until status is done or failed."""
        job = enqueue(db, tenant, "forecast", body.model_dump(mode="json"), background)
        return JobOut(**_job_summary(job), request=job.request)

    @app.post("/tenants/{tenant_id}/scenario-jobs", response_model=JobOut, status_code=202, dependencies=auth)
    def start_scenario_job(
        body: ScenarioRequest,
        background: BackgroundTasks,
        tenant: Tenant = Depends(get_tenant),
        db: Session = Depends(get_db),
    ):
        """Queue a baseline + scenarios comparison, e.g.
        {"scenarios": [{"name": "Hire 2", "adjustments": [{"line": "opex", "kind": "add",
        "value": 30000, "start": "2026-03-01"}]}]}"""
        job = enqueue(db, tenant, "scenarios", body.model_dump(mode="json"), background)
        return JobOut(**_job_summary(job), request=job.request)

    @app.post("/tenants/{tenant_id}/query", response_model=JobOut, status_code=202, dependencies=auth)
    def start_query(
        body: QueryRequest,
        background: BackgroundTasks,
        tenant: Tenant = Depends(get_tenant),
        db: Session = Depends(get_db),
    ):
        """Ask in plain language, e.g. {"prompt": "What if we hire two engineers in
        March and raise prices 5% from April?"}. A model turns the prompt into
        scenarios, the engine runs them, and the job result has the
        interpretation (check how the prompt was read), any warnings, a summary,
        and the full comparison. {"run": false} only interprets."""
        if not jobs_mod.llm_config().configured:
            raise HTTPException(
                status_code=503,
                detail="LLM not configured on this server: set LLM_BASE_URL and LLM_MODEL",
            )
        job = enqueue(db, tenant, "query", body.model_dump(mode="json"), background)
        return JobOut(**_job_summary(job), request=job.request)

    @app.get("/tenants/{tenant_id}/jobs/{job_id}", response_model=JobOut, dependencies=auth)
    def read_job(job_id: uuid.UUID, tenant: Tenant = Depends(get_tenant), db: Session = Depends(get_db)):
        expire_stale_jobs(db, tenant.id, job_timeout)
        j = db.get(Job, job_id)
        if not j or j.tenant_id != tenant.id:
            raise HTTPException(status_code=404, detail="Job not found")
        return JobOut(**_job_summary(j), request=j.request, result=j.result)

    @app.get("/tenants/{tenant_id}/jobs", response_model=list[JobSummary], dependencies=auth)
    def list_jobs(
        kind: Literal["forecast", "scenarios", "query"] | None = None,
        status: Literal["queued", "running", "done", "failed"] | None = None,
        limit: int = 50,
        tenant: Tenant = Depends(get_tenant),
        db: Session = Depends(get_db),
    ):
        expire_stale_jobs(db, tenant.id, job_timeout)
        q = select(Job).where(Job.tenant_id == tenant.id)
        if kind:
            q = q.where(Job.kind == kind)
        if status:
            q = q.where(Job.status == status)
        rows = db.scalars(q.order_by(Job.created_at.desc()).limit(min(max(limit, 1), 200)))
        return [JobSummary(**_job_summary(j)) for j in rows]

    return app


app = create_app()
