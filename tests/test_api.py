import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings, _normalize_db_url  # noqa: E402
from app.main import create_app  # noqa: E402

SAMPLE = ROOT / "sample_data" / "sample_financials.csv"
KEY = "test-key"
H = {"Authorization": f"Bearer {KEY}"}


@pytest.fixture
def client(tmp_path):
    settings = Settings(database_url=f"sqlite:///{tmp_path}/test.db", api_key=KEY, cors_origins=[])
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


def test_put_tenant_is_idempotent(client):
    tid = "6b1f5c2e-8a4d-5e3f-9c1b-2d7e4f6a8b90"
    assert client.put(f"/tenants/{tid}", json={"name": "Acme"}).status_code == 401
    first = client.put(f"/tenants/{tid}", json={"name": "Acme"}, headers=H)
    assert first.status_code == 200 and first.json()["id"] == tid
    again = client.put(f"/tenants/{tid}", json={"name": "Renamed"}, headers=H).json()
    # The existing tenant is returned unchanged (SQLite drops the timezone on read-back,
    # so compare id and name).
    assert (again["id"], again["name"]) == (tid, "Acme")
    assert client.get(f"/tenants/{tid}", headers=H).json()["name"] == "Acme"
    assert client.put("/tenants/not-a-uuid", json={"name": "x"}, headers=H).status_code == 422
