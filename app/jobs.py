"""Running jobs: shared by the Celery worker and the in-process fallback."""
from __future__ import annotations

import logging
import uuid
from datetime import timedelta

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from engine import ForecastRequest, ForecastResponse, run_forecast, to_monthly
from engine.ingest import IngestError, to_balances
from engine.forecast import run_scenarios
from engine.schema import LineItem, ScenarioComparison, ScenarioRequest
from llm.client import LLMConfig, LLMError
from llm.interpret import interpret
from llm.schema import QueryRequest, QueryResult
from llm.summary import summarize

from .models import Account, Actual, Forecast, Job, utcnow

log = logging.getLogger("forecast_jobs")


class NoData(ValueError):
    pass


def _items(db: Session, tenant_id: uuid.UUID) -> list[LineItem]:
    rows = db.execute(
        select(Actual.period, Actual.amount, Account.name, Account.category)
        .join(Account, Account.id == Actual.account_id)
        .where(Actual.tenant_id == tenant_id)
    ).all()
    return [LineItem(period=p, account=n, category=c, amount=float(a)) for p, a, n, c in rows]


def load_monthly(db: Session, tenant_id: uuid.UUID):
    """The tenant's income statement, one row per month."""
    return load_financials(db, tenant_id)[0]


def load_financials(db: Session, tenant_id: uuid.UUID):
    """(monthly income statement, month-end balances or None)."""
    items = _items(db, tenant_id)
    if not items:
        raise NoData("No financial data uploaded for this tenant yet")
    try:
        monthly = to_monthly(items)
    except IngestError:
        raise NoData("No income statement data uploaded yet (only balances)")
    return monthly, to_balances(items)


def llm_config() -> LLMConfig:
    """Read per call so tests (and a changed env) take effect without a restart."""
    return LLMConfig.from_env()


def save_forecast(
    db: Session, tenant_id: uuid.UUID, req: ForecastRequest, resp: ForecastResponse, prompt: str | None = None
) -> Forecast:
    record = Forecast(
        tenant_id=tenant_id,
        engine_version=resp.engine_version,
        horizon_months=req.horizon_months,
        last_actual=resp.last_actual_period,
        overrides=req.overrides.model_dump(exclude_none=True),
        result=resp.model_dump(mode="json"),
        prompt=prompt,
    )
    db.add(record)
    db.flush()
    return record


def _finish(db: Session, job_id: uuid.UUID, **values) -> bool:
    """Record the outcome only if the job is still ours (status 'running').
    If the API already marked it timed out, the outcome is dropped: a job the
    caller was told failed must not turn into 'done' later."""
    res = db.execute(
        update(Job)
        .where(Job.id == job_id, Job.status == "running")
        .values(finished_at=utcnow(), **values)
    )
    db.commit()
    if res.rowcount == 0:
        log.warning("job %s was marked failed (timed out) before it finished; result dropped", job_id)
        return False
    return True


def execute_job(session_factory: sessionmaker, job_id: uuid.UUID) -> None:
    """Run one job to completion.

    Safe to call twice: finished jobs (done, failed, or timed out by the API)
    are skipped, and a job left 'running' by a crashed worker runs again. Every
    status change is a conditional UPDATE, so the worker and the API's timeout
    can't overwrite each other.
    """
    with session_factory() as db:
        claimed = db.execute(
            update(Job)
            .where(Job.id == job_id, Job.status.in_(("queued", "running")))
            .values(status="running", started_at=utcnow(), attempts=Job.attempts + 1)
        )
        db.commit()
        if claimed.rowcount == 0:
            return  # unknown, finished, or already timed out
        job = db.get(Job, job_id)
        try:
            monthly, balances = load_financials(db, job.tenant_id)
            if job.kind == "forecast":
                req = ForecastRequest.model_validate(job.request)
                resp = run_forecast(monthly, req, balances)
                record = save_forecast(db, job.tenant_id, req, resp)
                result, forecast_id = resp.model_dump(mode="json"), record.id
            elif job.kind == "scenarios":
                req = ScenarioRequest.model_validate(job.request)
                result, forecast_id = run_scenarios(monthly, req, balances).model_dump(mode="json"), None
            elif job.kind == "query":
                result, forecast_id = run_query(db, job.tenant_id, monthly, balances, QueryRequest.model_validate(job.request)), None
            else:
                raise ValueError(f"Unknown job kind '{job.kind}'")
        except LLMError as e:  # endpoint down, or the model's reply unusable
            db.rollback()
            _finish(db, job_id, status="failed", error=str(e))
            return
        except ValueError as e:  # bad input or not enough data: the caller's to fix
            db.rollback()
            _finish(db, job_id, status="failed", error=str(e))
            return
        except Exception as e:
            log.exception("job %s failed", job_id)
            db.rollback()
            _finish(db, job_id, status="failed", error=f"Internal error ({e.__class__.__name__})")
            return
        # The saved forecast row and the job outcome commit together, so a
        # timed-out job leaves no orphan forecast behind.
        res = db.execute(
            update(Job)
            .where(Job.id == job_id, Job.status == "running")
            .values(status="done", result=result, forecast_id=forecast_id, finished_at=utcnow())
        )
        if res.rowcount == 0:
            db.rollback()
            log.warning("job %s was marked failed (timed out) before it finished; result dropped", job_id)
            return
        db.commit()


def run_query(db: Session, tenant_id: uuid.UUID, monthly, balances, req: QueryRequest) -> dict:
    """Prompt -> scenarios (model) -> forecasts (engine) -> summary (code)."""
    interp, warnings, call = interpret(req.prompt, monthly, llm_config(), req.horizon_months, balances)
    result = QueryResult(prompt=req.prompt, interpretation=interp, warnings=warnings, llm=call)
    if not req.run or interp.intent == "unsupported":
        return result.model_dump(mode="json")

    opts = dict(horizon_months=interp.horizon_months, simulations=req.simulations, seed=req.seed)
    if interp.scenarios:
        comparison = run_scenarios(monthly, ScenarioRequest(scenarios=interp.scenarios, **opts), balances)
    else:  # baseline question
        from engine.forecast import ENGINE_VERSION

        comparison = ScenarioComparison(
            engine_version=ENGINE_VERSION, baseline=run_forecast(monthly, ForecastRequest(**opts), balances), scenarios=[]
        )
    # Each scenario is saved as a forecast, with the prompt that produced it.
    ids = []
    for sc_req, sc in zip(interp.scenarios, comparison.scenarios):
        freq = ForecastRequest(overrides=sc_req.overrides, adjustments=sc_req.adjustments, **opts)
        ids.append(save_forecast(db, tenant_id, freq, sc.forecast, prompt=req.prompt).id)

    engine_warnings = {w for sc in comparison.scenarios for w in sc.forecast.warnings} | set(comparison.baseline.warnings)
    result.warnings = warnings + sorted(engine_warnings)
    result.comparison = comparison
    result.summary = summarize(comparison)
    result.forecast_ids = ids
    return result.model_dump(mode="json")


def expire_stale_jobs(db: Session, tenant_id: uuid.UUID, timeout: timedelta) -> int:
    """Mark this tenant's jobs failed if no worker picked them up, or a worker
    never finished them, within `timeout`. Called when jobs are read, so a
    caller polling a job always reaches done or failed."""
    now = utcnow()
    cutoff = now - timeout
    minutes = f"{timeout.total_seconds() / 60:g}"
    queued = db.execute(
        update(Job)
        .where(Job.tenant_id == tenant_id, Job.status == "queued", Job.created_at < cutoff)
        .values(
            status="failed",
            finished_at=now,
            error=f"Timed out: no worker picked this job up within {minutes} minutes. "
            "The worker may be down; try again shortly.",
        )
    )
    running = db.execute(
        update(Job)
        .where(Job.tenant_id == tenant_id, Job.status == "running", Job.started_at < cutoff)
        .values(
            status="failed",
            finished_at=now,
            error=f"Timed out: the worker didn't finish this job within {minutes} minutes "
            "(it may have crashed). Try again.",
        )
    )
    db.commit()
    return queued.rowcount + running.rowcount
