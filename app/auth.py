"""Who is calling, and which tenants they may touch.

    admin    the API_KEY environment variable: everything, including managing keys
    service  a stored key: any tenant, and creating tenants; not key management
    tenant   a stored key: its own tenant only; other tenants look like they don't exist (404)
    open     no API_KEY configured (local development): everything, with a warning at startup
"""
from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import timedelta

from fastapi import HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import ApiKey, utcnow

KEY_PREFIX = {"service": "fek_svc_", "tenant": "fek_ten_"}
TOUCH_EVERY = timedelta(minutes=1)  # how often last_used_at is written, at most


@dataclass(frozen=True)
class Principal:
    kind: str  # admin | service | tenant | open
    tenant_id: uuid.UUID | None = None
    key_id: uuid.UUID | None = None

    @property
    def can_manage_keys(self) -> bool:
        return self.kind in ("admin", "open")

    @property
    def can_create_tenants(self) -> bool:
        return self.kind in ("admin", "service", "open")

    def may_access(self, tenant_id: uuid.UUID) -> bool:
        return self.kind != "tenant" or self.tenant_id == tenant_id


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def new_key(kind: str) -> str:
    return KEY_PREFIX[kind] + secrets.token_urlsafe(32)


def supplied_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (request.headers.get("x-api-key") or "").strip()


def authenticate(request: Request, db: Session, admin_key: str | None) -> Principal:
    supplied = supplied_key(request)
    if supplied and admin_key and secrets.compare_digest(supplied.encode(), admin_key.encode()):
        return Principal("admin")
    if supplied:
        row = db.scalars(select(ApiKey).where(ApiKey.key_hash == hash_key(supplied))).first()
        if row is not None and row.revoked_at is None:
            now = utcnow()
            last = row.last_used_at
            if last is not None and last.tzinfo is None:  # SQLite returns naive datetimes
                last = last.replace(tzinfo=now.tzinfo)
            if last is None or now - last > TOUCH_EVERY:
                row.last_used_at = now
                db.commit()
            return Principal(row.kind, row.tenant_id, row.id)
    if not admin_key:
        return Principal("open")
    raise HTTPException(status_code=401, detail="Invalid or missing API key")
