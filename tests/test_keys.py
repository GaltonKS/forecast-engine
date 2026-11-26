"""API keys: admin, service and tenant keys, and tenant isolation."""
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models import ApiKey  # noqa: E402

ADMIN = "admin-key"
SAMPLE = ROOT / "sample_data" / "sample_financials.csv"


def h(key):
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def env(tmp_path):
    url = f"sqlite:///{tmp_path}/keys.db"
    with TestClient(create_app(Settings(database_url=url, api_key=ADMIN, cors_origins=[]))) as c:
        yield c, url


def tenant(c, name="Acme"):
    return c.post("/tenants", json={"name": name}, headers=h(ADMIN)).json()["id"]


def make_key(c, kind, tenant_id=None, name="test"):
    r = c.post("/admin/keys", json={"name": name, "kind": kind, "tenant_id": tenant_id}, headers=h(ADMIN))
    assert r.status_code == 201, r.text
    return r.json()


def upload(c, t, key):
    return c.post(f"/tenants/{t}/actuals/upload", files={"file": ("d.csv", SAMPLE.read_bytes(), "text/csv")}, headers=h(key))


def test_admin_key_still_does_everything(env):
    c, _ = env
    t = tenant(c)
    assert upload(c, t, ADMIN).status_code == 200
    assert c.get("/admin/keys", headers=h(ADMIN)).status_code == 200


def test_tenant_key_reaches_its_tenant_only(env):
    c, _ = env
    acme, bravo = tenant(c, "Acme"), tenant(c, "Bravo")
    key = make_key(c, "tenant", acme)["key"]
    assert key.startswith("fek_ten_")
    assert upload(c, acme, key).status_code == 200
    assert c.get(f"/tenants/{acme}", headers=h(key)).json()["name"] == "Acme"
    f = c.post(f"/tenants/{acme}/forecasts", json={"simulations": 0}, headers=h(key))
    assert f.status_code == 201
    # every route under another tenant looks like the tenant doesn't exist
    upload(c, bravo, ADMIN)
    bf = c.post(f"/tenants/{bravo}/forecasts", json={"simulations": 0}, headers=h(ADMIN)).json()["id"]
    for method, path, kw in [
        ("GET", f"/tenants/{bravo}", {}),
        ("GET", f"/tenants/{bravo}/actuals/monthly", {}),
        ("GET", f"/tenants/{bravo}/actuals/balances", {}),
        ("GET", f"/tenants/{bravo}/forecasts", {}),
        ("GET", f"/tenants/{bravo}/forecasts/{bf}", {}),
        ("POST", f"/tenants/{bravo}/forecasts", {"json": {}}),
        ("POST", f"/tenants/{bravo}/forecast-jobs", {"json": {}}),
        ("POST", f"/tenants/{bravo}/scenario-jobs", {"json": {"scenarios": [{"name": "x"}]}}),
        ("POST", f"/tenants/{bravo}/query", {"json": {"prompt": "x"}}),
        ("GET", f"/tenants/{bravo}/jobs", {}),
        ("GET", f"/tenants/{bravo}/jobs/{uuid.uuid4()}", {}),
    ]:
        r = c.request(method, path, headers=h(key), **kw)
        assert r.status_code == 404 and r.json()["detail"] == "Tenant not found", (method, path, r.status_code)
    r = c.post(f"/tenants/{bravo}/actuals/upload", files={"file": ("d.csv", b"x", "text/csv")}, headers=h(key))
    assert r.status_code == 404
    # and its own tenant's forecast id under another tenant's path doesn't help either
    acme_f = f.json()["id"]
    assert c.get(f"/tenants/{bravo}/forecasts/{acme_f}", headers=h(key)).status_code == 404


def test_tenant_key_cannot_create_tenants_or_manage_keys(env):
    c, _ = env
    key = make_key(c, "tenant", tenant(c))["key"]
    assert c.post("/tenants", json={"name": "Sneaky"}, headers=h(key)).status_code == 403
    assert c.get("/admin/keys", headers=h(key)).status_code == 403
    assert c.post("/admin/keys", json={"name": "x", "kind": "service"}, headers=h(key)).status_code == 403


def test_service_key_reaches_all_tenants_but_not_key_management(env):
    c, _ = env
    key = make_key(c, "service")["key"]
    assert key.startswith("fek_svc_")
    t = c.post("/tenants", json={"name": "Made by service"}, headers=h(key))
    assert t.status_code == 201
    other = tenant(c, "Other")
    assert c.get(f"/tenants/{other}", headers=h(key)).status_code == 200
    assert upload(c, t.json()["id"], key).status_code == 200
    assert c.get("/admin/keys", headers=h(key)).status_code == 403
    assert c.delete(f"/admin/keys/{uuid.uuid4()}", headers=h(key)).status_code == 403


def test_revoked_key_stops_working(env):
    c, _ = env
    t = tenant(c)
    k = make_key(c, "tenant", t)
    assert c.get(f"/tenants/{t}", headers=h(k["key"])).status_code == 200
    r = c.delete(f"/admin/keys/{k['id']}", headers=h(ADMIN))
    assert r.status_code == 200 and r.json()["revoked_at"]
    assert c.get(f"/tenants/{t}", headers=h(k["key"])).status_code == 401
    assert c.delete(f"/admin/keys/{k['id']}", headers=h(ADMIN)).status_code == 200  # idempotent
    assert c.get("/admin/keys", headers=h(ADMIN)).json() == []
    assert len(c.get("/admin/keys?include_revoked=true", headers=h(ADMIN)).json()) == 1


def test_keys_are_shown_once_and_stored_hashed(env):
    c, url = env
    t = tenant(c)
    k = make_key(c, "tenant", t, name="Pilot: Acme")
    listed = c.get(f"/admin/keys?tenant_id={t}", headers=h(ADMIN)).json()
    assert len(listed) == 1 and "key" not in listed[0] and listed[0]["prefix"] == k["key"][:12]
    with Session(create_engine(url)) as db:
        row = db.scalars(select(ApiKey)).one()
        assert row.key_hash != k["key"] and k["key"] not in (row.key_hash, row.prefix + "x")
        assert len(row.key_hash) == 64


def test_last_used_is_recorded(env):
    c, _ = env
    t = tenant(c)
    k = make_key(c, "tenant", t)
    assert c.get("/admin/keys", headers=h(ADMIN)).json()[0]["last_used_at"] is None
    c.get(f"/tenants/{t}", headers=h(k["key"]))
    assert c.get("/admin/keys", headers=h(ADMIN)).json()[0]["last_used_at"] is not None


@pytest.mark.parametrize("body, msg", [
    ({"name": "x", "kind": "tenant"}, "existing tenant"),
    ({"name": "x", "kind": "tenant", "tenant_id": str(uuid.uuid4())}, "existing tenant"),
    ({"name": "x", "kind": "service", "tenant_id": "TENANT"}, "leave tenant_id empty"),
])
def test_key_creation_validation(env, body, msg):
    c, _ = env
    if body.get("tenant_id") == "TENANT":
        body["tenant_id"] = tenant(c)
    r = c.post("/admin/keys", json=body, headers=h(ADMIN))
    assert r.status_code == 422 and msg in r.json()["detail"]


def test_wrong_or_missing_key_is_401(env):
    c, _ = env
    t = tenant(c)
    assert c.get(f"/tenants/{t}").status_code == 401
    assert c.get(f"/tenants/{t}", headers=h("fek_ten_madeup")).status_code == 401
    assert c.get(f"/tenants/{t}", headers={"X-API-Key": ADMIN}).status_code == 200  # header form still works


def test_open_mode_without_admin_key_still_scopes_tenant_keys(tmp_path):
    s = Settings(database_url=f"sqlite:///{tmp_path}/open.db", api_key=None, cors_origins=[])
    with TestClient(create_app(s)) as c:
        a = c.post("/tenants", json={"name": "A"}).json()["id"]
        b = c.post("/tenants", json={"name": "B"}).json()["id"]
        key = c.post("/admin/keys", json={"name": "x", "kind": "tenant", "tenant_id": a}).json()["key"]
        assert c.get(f"/tenants/{b}").status_code == 200  # open: no key, everything
        assert c.get(f"/tenants/{b}", headers=h(key)).status_code == 404  # a tenant key is still a tenant key
