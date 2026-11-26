"""Database tables. These models are the source of truth for the schema.

Every row is scoped by tenant_id so one customer's data never reaches
another's forecast. Forecasts are stored with the exact drivers, overrides
and results, so any number can be reproduced and explained later.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    ForeignKey,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

JSONType = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Tenant(Base):
    __tablename__ = "tenants"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Account(Base):
    __tablename__ = "accounts"
    __table_args__ = (UniqueConstraint("tenant_id", "name"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(16))  # revenue | cogs | opex | other
    source: Mapped[str] = mapped_column(String(32), default="csv")


class Actual(Base):
    __tablename__ = "actuals"
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), primary_key=True
    )
    account_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"), primary_key=True
    )
    period: Mapped[date] = mapped_column(Date, primary_key=True)  # first of month
    amount: Mapped[Decimal] = mapped_column(Numeric(18, 2))
    loaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class Forecast(Base):
    __tablename__ = "forecasts"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    engine_version: Mapped[str] = mapped_column(String(32))
    horizon_months: Mapped[int]
    last_actual: Mapped[date] = mapped_column(Date)
    overrides: Mapped[dict] = mapped_column(JSONType, default=dict)
    result: Mapped[dict] = mapped_column(JSONType)  # full ForecastResponse
    prompt: Mapped[str | None] = mapped_column(Text, default=None)  # the prompt, for forecasts made by /query


class Job(Base):
    """A forecast or scenario comparison run in the background.

    queued -> running -> done | failed. The result lives here (and, for a
    forecast, also as a row in forecasts), so polling needs only this table.
    """

    __tablename__ = "jobs"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(16))  # forecast | scenarios | query
    status: Mapped[str] = mapped_column(String(16), default="queued")
    request: Mapped[dict] = mapped_column(JSONType)
    result: Mapped[dict | None] = mapped_column(JSONType, default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    forecast_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, default=None)
    attempts: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)


class ApiKey(Base):
    """An API key, stored as a SHA-256 hash (keys are long random strings, so a
    fast hash is enough). The plaintext is shown once, when it's created.

    service: may act on any tenant and create tenants (e.g. the shared Pilot app).
    tenant:  may act on its one tenant only (e.g. one company's Pilot app).
    The API_KEY environment variable is the admin key: everything, plus
    creating and revoking these keys.
    """

    __tablename__ = "api_keys"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100))
    kind: Mapped[str] = mapped_column(String(16))  # service | tenant
    tenant_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), index=True, default=None
    )
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    prefix: Mapped[str] = mapped_column(String(16))  # first characters, to recognise a key in lists
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
