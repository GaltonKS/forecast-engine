import pytest

# Settings the app reads from the environment. Tests set what they need
# explicitly, so a value exported in the shell (say REDIS_URL from running the
# stack locally) can't change which queue, database or model a test talks to.
# TEST_DATABASE_URL is left alone: it's the deliberate switch to Postgres.
APP_ENV = [
    "DATABASE_URL", "API_KEY", "CORS_ORIGINS", "MAX_UPLOAD_MB", "REDIS_URL", "BROKER_URL",
    "JOB_TIMEOUT_MINUTES", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "LLM_TIMEOUT_SECONDS",
    "LLM_GUIDED", "LLM_MAX_TOKENS",
]


@pytest.fixture(autouse=True)
def clean_app_env(monkeypatch):
    for name in APP_ENV:
        monkeypatch.delenv(name, raising=False)
