"""Celery worker. Start with:  celery -A app.worker worker --concurrency 1

Reads REDIS_URL (the Valkey connection string) and DATABASE_URL. Results are
written to the jobs table, not to Celery's result backend.
"""
from __future__ import annotations

import uuid

from celery import Celery
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .config import Settings
# Imported here, not inside the task: Celery puts the project root on sys.path
# only while it loads this module, so a lazy import fails in the worker
# processes. Importing now also means a broken import stops the worker at
# startup (visible in the logs) instead of leaving jobs stuck in "queued".
from .jobs import execute_job

TASK_NAME = "forecast.run_job"


def make_celery(broker_url: str | None) -> Celery:
    app = Celery("forecast", broker=broker_url or "memory://")
    app.conf.update(
        task_ignore_result=True,
        task_acks_late=True,  # a job survives a worker crash: it's redelivered
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,  # forecasts are CPU-heavy; take one at a time
        broker_connection_retry_on_startup=True,
        broker_transport_options={"visibility_timeout": 3600},
        broker_connection_timeout=3,
        # If Valkey is down, give up in about a second so the API can answer 503
        # instead of holding the request open.
        task_publish_retry_policy={"max_retries": 2, "interval_start": 0, "interval_step": 0.3},
    )
    return app


settings = Settings()
celery_app = make_celery(settings.broker_url)
_sessions: sessionmaker | None = None


def _session_factory() -> sessionmaker:
    global _sessions
    if _sessions is None:
        engine = create_engine(settings.database_url, pool_pre_ping=True)
        _sessions = sessionmaker(bind=engine, expire_on_commit=False)
    return _sessions


@celery_app.task(name=TASK_NAME)
def run_job(job_id: str) -> None:
    execute_job(_session_factory(), uuid.UUID(job_id))
