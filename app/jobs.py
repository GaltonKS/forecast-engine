"""Running jobs: shared by the Celery worker and the in-process fallback."""
from __future__ import annotations

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from engine import ForecastRequest, ForecastResponse, run_forecast, to_monthly
from engine.forecast import run_scenarios
from engine.schema import LineItem, ScenarioRequest

from .models import Account, Actual, Forecast, Job, utcnow

log = logging.getLogger("forecast_jobs")


class NoData(ValueError):
    pass


def load_monthly(db: Session, tenant_id: uuid.UUID):
    rows = db.execute(
        select(Actual.period, Actual.amount, Account.name, Account.category)
        .join(Account, Account.id == Actual.account_id)
        .where(Actual.tenant_id == tenant_id)
    ).all()
    if not rows:
        raise NoData("No financial data uploaded for this tenant yet")
    items = [LineItem(period=p, account=n, category=c, amount=float(a)) for p, a, n, c in rows]
    return to_monthly(items)


def save_forecast(db: Session, tenant_id: uuid.UUID, req: ForecastRequest, resp: ForecastResponse) -> Forecast:
    record = Forecast(
        tenant_id=tenant_id,
        engine_version=resp.engine_version,
        horizon_months=req.horizon_months,
        last_actual=resp.last_actual_period,
        overrides=req.overrides.model_dump(exclude_none=True),
        result=resp.model_dump(mode="json"),
    )
    db.add(record)
    db.flush()
    return record


def execute_job(session_factory: sessionmaker, job_id: uuid.UUID) -> None:
    """Run one job to completion. Safe to call twice: finished jobs are skipped,
    and a job left 'running' by a crashed worker is simply run again."""
    with session_factory() as db:
        job = db.get(Job, job_id)
        if job is None or job.status in ("done", "failed"):
            return
        job.status, job.started_at, job.attempts = "running", utcnow(), job.attempts + 1
        db.commit()
        try:
            monthly = load_monthly(db, job.tenant_id)
            if job.kind == "forecast":
                req = ForecastRequest.model_validate(job.request)
                resp = run_forecast(monthly, req)
                job.forecast_id = save_forecast(db, job.tenant_id, req, resp).id
                job.result = resp.model_dump(mode="json")
            elif job.kind == "scenarios":
                req = ScenarioRequest.model_validate(job.request)
                job.result = run_scenarios(monthly, req).model_dump(mode="json")
            else:
                raise ValueError(f"Unknown job kind '{job.kind}'")
            job.status = "done"
        except ValueError as e:  # bad input or not enough data: the caller's to fix
            db.rollback()
            job = db.get(Job, job_id)
            job.status, job.error = "failed", str(e)
        except Exception as e:
            log.exception("job %s failed", job_id)
            db.rollback()
            job = db.get(Job, job_id)
            job.status, job.error = "failed", f"Internal error ({e.__class__.__name__})"
        job.finished_at = utcnow()
        db.commit()
