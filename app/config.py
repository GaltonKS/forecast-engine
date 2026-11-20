"""Settings, read from environment variables (set them in Render's dashboard)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _normalize_db_url(url: str) -> str:
    # Render gives postgres:// or postgresql://; SQLAlchemy needs the driver named.
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


@dataclass
class Settings:
    database_url: str = field(
        default_factory=lambda: _normalize_db_url(os.getenv("DATABASE_URL", "sqlite:///./local.db"))
    )
    api_key: str | None = field(default_factory=lambda: os.getenv("API_KEY") or None)
    cors_origins: list[str] = field(
        default_factory=lambda: [
            o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()
        ]
    )
    max_upload_bytes: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
    )
