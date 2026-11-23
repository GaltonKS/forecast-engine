import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings, _normalize_db_url  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models import Base  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402

SAMPLE = ROOT / "sample_data" / "sample_financials.csv"
KEY = "test-key"
H = {"Authorization": f"Bearer {KEY}"}


@pytest.fixture
def client(tmp_path):
    # TEST_DATABASE_URL=postgresql://... runs these against Postgres; tables are
    # dropped first so every test starts clean. Default: a fresh SQLite file.
    url = os.getenv("TEST_DATABASE_URL")
    if url:
        url = _normalize_db_url(url)
        engine = create_engine(url)
        Base.metadata.drop_all(engine)
        engine.dispose()
    else:
        url = f"sqlite:///{tmp_path}/test.db"
    settings = Settings(database_url=url, api_key=KEY, cors_origins=[])
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def tenant(client):
    return client.post("/tenants", json={"name": "Acme"}, headers=H).json()["id"]


def upload(client, tenant, content: bytes, name="data.csv"):
    return client.post(
        f"/tenants/{tenant}/actuals/upload",
        files={"file": (name, content, "text/csv")},
        headers=H,
    )


def test_health_is_public(client):
    assert client.get("/health").json()["status"] == "ok"


def test_api_key_required(client):
    assert client.post("/tenants", json={"name": "x"}).status_code == 401
    assert client.post("/tenants", json={"name": "x"}, headers={"X-API-Key": "nope"}).status_code == 401
    assert client.post("/tenants", json={"name": "x"}, headers={"X-API-Key": KEY}).status_code == 201


def test_full_flow(client, tenant):
    r = upload(client, tenant, SAMPLE.read_bytes())
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["months_in_history"] == 24 and s["accounts"] == 8 and s["gaps"] == []

    hist = client.get(f"/tenants/{tenant}/actuals/monthly", headers=H).json()
    assert len(hist) == 24

    r = client.post(
        f"/tenants/{tenant}/forecasts",
        json={"horizon_months": 6, "overrides": {"revenue_growth": 0.02}},
        headers=H,
    )
    assert r.status_code == 201, r.text
    f = r.json()
    assert len(f["months"]) == 6
    assert f["drivers"]["revenue_growth"]["overridden"] is True
    assert "revenue" in f["formulas"]

    listed = client.get(f"/tenants/{tenant}/forecasts", headers=H).json()
    assert listed[0]["id"] == f["id"] and listed[0]["overrides"] == {"revenue_growth": 0.02}

    again = client.get(f"/tenants/{tenant}/forecasts/{f['id']}", headers=H).json()
    assert again["months"] == f["months"]


def test_reupload_replaces_not_duplicates(client, tenant):
    upload(client, tenant, b"period,account,category,amount\n2024-01-01,Sales,revenue,100\n")
    upload(client, tenant, b"period,account,category,amount\n2024-01-01,Sales,revenue,250\n")
    hist = client.get(f"/tenants/{tenant}/actuals/monthly", headers=H).json()
    assert hist == [
        {"period": "2024-01-01", "revenue": 250.0, "cogs": 0.0, "opex": 0.0, "other": 0.0,
         "gross_profit": 250.0, "net_income": 250.0}
    ]


def test_bad_uploads(client, tenant):
    assert upload(client, tenant, b"nonsense\x00\xff").status_code == 422
    r = upload(client, tenant, b"period,account,category,amount\n2024-01-01,Sales,sales,1\n")
    assert r.status_code == 422 and "row 2" in r.json()["detail"]
    upload(client, tenant, b"period,account,category,amount\n2024-01-01,Sales,revenue,1\n")
    r = upload(client, tenant, b"period,account,category,amount\n2024-02-01,Sales,opex,1\n")
    assert r.status_code == 422 and "already stored" in r.json()["detail"]


def test_forecast_errors(client, tenant):
    r = client.post(f"/tenants/{tenant}/forecasts", json={}, headers=H)
    assert r.status_code == 422 and "No financial data" in r.json()["detail"]
    upload(client, tenant, b"period,account,category,amount\n2024-01-01,Sales,revenue,1\n")
    r = client.post(f"/tenants/{tenant}/forecasts", json={}, headers=H)
    assert r.status_code == 422 and "at least 3 months" in r.json()["detail"]


def test_tenant_isolation(client, tenant):
    upload(client, tenant, SAMPLE.read_bytes())
    fid = client.post(f"/tenants/{tenant}/forecasts", json={}, headers=H).json()["id"]
    other = client.post("/tenants", json={"name": "Other"}, headers=H).json()["id"]
    assert client.get(f"/tenants/{other}/forecasts/{fid}", headers=H).status_code == 404
    assert client.get(f"/tenants/{other}/actuals/monthly", headers=H).status_code == 422
    assert client.get("/tenants/00000000-0000-0000-0000-000000000000", headers=H).status_code == 404


def test_render_db_url_normalized():
    assert _normalize_db_url("postgres://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert _normalize_db_url("postgresql://u:p@h/db") == "postgresql+psycopg://u:p@h/db"


# ---------- step 4: background jobs (in-process fallback, no broker) ----------

def test_forecast_job_runs_and_saves_forecast(client, tenant):
    upload(client, tenant, SAMPLE.read_bytes())
    r = client.post(f"/tenants/{tenant}/forecast-jobs", json={"horizon_months": 6}, headers=H)
    assert r.status_code == 202 and r.json()["status"] == "queued"
    job = client.get(f"/tenants/{tenant}/jobs/{r.json()['id']}", headers=H).json()
    assert job["status"] == "done", job
    assert job["result"]["uncertainty"]["simulations"] == 1000
    saved = client.get(f"/tenants/{tenant}/forecasts/{job['forecast_id']}", headers=H)
    assert saved.status_code == 200
    assert saved.json()["months"] == job["result"]["months"]


def test_scenario_job(client, tenant):
    upload(client, tenant, SAMPLE.read_bytes())
    body = {"scenarios": [{"name": "Hire", "adjustments": [
        {"line": "opex", "kind": "add", "value": 30000, "start": "2026-03-01"}]}]}
    r = client.post(f"/tenants/{tenant}/scenario-jobs", json=body, headers=H)
    job = client.get(f"/tenants/{tenant}/jobs/{r.json()['id']}", headers=H).json()
    assert job["status"] == "done", job
    assert job["result"]["scenarios"][0]["delta_point"]["net_income"] == -300000
    jobs = client.get(f"/tenants/{tenant}/jobs?kind=scenarios", headers=H).json()
    assert [j["id"] for j in jobs] == [job["id"]] and "result" not in jobs[0]


def test_job_without_data_is_rejected_up_front(client, tenant):
    r = client.post(f"/tenants/{tenant}/forecast-jobs", json={}, headers=H)
    assert r.status_code == 422


def test_failed_job_reports_error(client, tenant):
    # 8 months of data, ETS needs 12: the job fails with a readable reason
    rows = "period,account,category,amount\n" + "".join(
        f"2025-{m:02d}-01,Sales,revenue,100\n" for m in range(1, 9))
    upload(client, tenant, rows.encode())
    r = client.post(f"/tenants/{tenant}/forecast-jobs", json={"method": "ets"}, headers=H)
    job = client.get(f"/tenants/{tenant}/jobs/{r.json()['id']}", headers=H).json()
    assert job["status"] == "failed" and "needs 12 months" in job["error"]


def test_jobs_are_tenant_scoped(client, tenant):
    upload(client, tenant, SAMPLE.read_bytes())
    job_id = client.post(f"/tenants/{tenant}/forecast-jobs", json={}, headers=H).json()["id"]
    other = client.post("/tenants", json={"name": "Other"}, headers=H).json()["id"]
    assert client.get(f"/tenants/{other}/jobs/{job_id}", headers=H).status_code == 404
    assert client.get(f"/tenants/{other}/jobs", headers=H).json() == []


def test_queue_down_returns_503_and_marks_job_failed(tmp_path):
    settings = Settings(database_url=f"sqlite:///{tmp_path}/q.db", api_key=KEY, cors_origins=[],
                        broker_url="redis://127.0.0.1:1/0")  # nothing listens on port 1
    with TestClient(create_app(settings)) as c:
        t = c.post("/tenants", json={"name": "Q"}, headers=H).json()["id"]
        upload(c, t, SAMPLE.read_bytes())
        r = c.post(f"/tenants/{t}/forecast-jobs", json={}, headers=H)
        assert r.status_code == 503
        jobs = c.get(f"/tenants/{t}/jobs", headers=H).json()
        assert jobs[0]["status"] == "failed" and "job queue" in jobs[0]["error"]
