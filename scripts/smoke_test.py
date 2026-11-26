"""End-to-end check of a deployed pipeline: API, Postgres, Valkey, worker, model.

    python scripts/smoke_test.py --url https://forecast-api.onrender.com --key "$API_KEY"

Runs in order and stops at the first failure, saying which part of the
pipeline is the likely cause:

  1. health            API is up (Render web service)
  2. auth              API key is enforced
  3. tenant + upload   Postgres reads and writes; sample income statement + balances
     tenant keys       a tenant key reaches its tenant only (needs the admin key; SKIP otherwise)
  4. forecast          engine runs in the API (three statements balance)
  5. forecast job      Valkey + worker pick up and finish a job
  6. scenario job      scenarios and paired deltas through the worker
  7. query             model endpoint reachable and returns usable scenarios
                       (SKIP if the server has no LLM configured)

Creates tenants named "smoke-test <timestamp>" each run (two when it checks
tenant keys). There is no delete endpoint yet, so test tenants stay in the
database. The test tenant key it makes is revoked before it finishes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = [ROOT / "sample_data" / "sample_financials.csv", ROOT / "sample_data" / "sample_balances.csv"]


class Failed(Exception):
    def __init__(self, msg: str, hint: str = ""):
        super().__init__(msg)
        self.hint = hint


class Smoke:
    def __init__(self, url: str, key: str | None, job_timeout: float, query_timeout: float):
        self.c = httpx.Client(base_url=url.rstrip("/"), timeout=60)
        self.h = {"X-API-Key": key} if key else {}
        self.key = key
        self.job_timeout, self.query_timeout = job_timeout, query_timeout
        self.tenant = None

    def call(self, method, path, expect=(200,), **kw):
        try:
            r = self.c.request(method, path, headers=self.h, **kw)
        except httpx.HTTPError as e:
            raise Failed(f"{method} {path}: {e.__class__.__name__}: {e}", "Is the URL right and the service running?")
        if r.status_code not in expect:
            hint = "The API key is wrong: use the API_KEY from the forecast-api service's Environment tab." \
                if r.status_code == 401 else ""
            raise Failed(f"{method} {path} -> {r.status_code}: {r.text[:300]}", hint)
        return r.json() if r.content else None

    def poll(self, job_id: str, timeout: float) -> dict:
        start = time.monotonic()
        status = None
        while time.monotonic() - start < timeout:
            job = self.call("GET", f"/tenants/{self.tenant}/jobs/{job_id}")
            status = job["status"]
            if status in ("done", "failed"):
                return job
            time.sleep(1.5)
        hint = ("Still 'queued': no worker picked it up. Check the forecast-worker service is running "
                "and its REDIS_URL matches the API's." if status == "queued" else
                "Still 'running': the worker started but didn't finish. Check forecast-worker logs.")
        raise Failed(f"job {job_id} still '{status}' after {timeout:.0f}s", hint)

    # ----- checks -----

    def health(self):
        h = self.call("GET", "/health")
        return f"engine {h['engine_version']}"

    def auth(self):
        if not self.key:
            return "SKIP: no --key given (if the server has no API_KEY set, it's open to anyone)"
        r = self.c.get("/tenants/00000000-0000-0000-0000-000000000000")
        if r.status_code != 401:
            raise Failed(f"request without a key got {r.status_code}, expected 401",
                         "API_KEY isn't set on the web service, so the API is open.")
        return "requests without the key are rejected"

    def upload(self):
        t = self.call("POST", "/tenants", expect=(201,), json={"name": f"smoke-test {time.strftime('%Y-%m-%d %H:%M:%S')}"})
        self.tenant = t["id"]
        for f in SAMPLES:
            s = self.call("POST", f"/tenants/{self.tenant}/actuals/upload", files={"file": (f.name, f.read_bytes(), "text/csv")})
        if not s["balance_sheet_ready"]:
            raise Failed(f"balance sheet not ready after upload: {s}")
        return f"tenant {self.tenant}, {s['months_in_history']} months, balances to {s['balances_end']}"

    def tenant_keys(self):
        r = self.c.post("/admin/keys", headers=self.h, json={"name": "smoke test", "kind": "tenant", "tenant_id": self.tenant})
        if r.status_code == 403:
            return "SKIP: --key isn't the admin key (API_KEY), so it can't create keys"
        if r.status_code != 201:
            raise Failed(f"POST /admin/keys -> {r.status_code}: {r.text[:300]}")
        k = r.json()
        kh = {"Authorization": f"Bearer {k['key']}"}
        try:
            other = self.call("POST", "/tenants", expect=(201,), json={"name": f"smoke-test other {time.strftime('%H:%M:%S')}"})["id"]
            own = self.c.get(f"/tenants/{self.tenant}", headers=kh).status_code
            theirs = self.c.get(f"/tenants/{other}", headers=kh).status_code
            create = self.c.post("/tenants", headers=kh, json={"name": "should fail"}).status_code
            if (own, theirs, create) != (200, 404, 403):
                raise Failed(f"tenant key: own tenant {own} (want 200), other tenant {theirs} (want 404), "
                             f"create tenant {create} (want 403)", "Tenant isolation is broken: don't put client data in.")
        finally:
            self.c.delete(f"/admin/keys/{k['id']}", headers=self.h)
        if self.c.get(f"/tenants/{self.tenant}", headers=kh).status_code != 401:
            raise Failed("a revoked key still works")
        return "own tenant 200, other tenant 404, can't create tenants, revoked key rejected"

    def forecast(self):
        f = self.call("POST", f"/tenants/{self.tenant}/forecasts", expect=(201,),
                      json={"horizon_months": 12, "simulations": 200})
        worst = max(abs(r["check"]) for r in f["balance_sheet"])
        if worst > 0.05:
            raise Failed(f"balance sheet off by {worst}")
        models = {k: v["method"] for k, v in f["models"].items()}
        return f"models {models}, ending cash {f['cash']['ending_cash']:,.0f}, balance check {worst}"

    def forecast_job(self):
        job = self.call("POST", f"/tenants/{self.tenant}/forecast-jobs", expect=(202,), json={"simulations": 200})
        job = self.poll(job["id"], self.job_timeout)
        if job["status"] != "done":
            raise Failed(f"forecast job failed: {job['error']}", "Check forecast-worker logs.")
        took = _seconds(job)
        return f"done by the worker in {took}"

    def scenario_job(self):
        body = {"simulations": 200, "scenarios": [{"name": "Raise 500k", "adjustments": [
            {"line": "equity", "kind": "add", "value": 500000, "start": "2026-06-01", "end": "2026-06-01"}]}]}
        job = self.call("POST", f"/tenants/{self.tenant}/scenario-jobs", expect=(202,), json=body)
        job = self.poll(job["id"], self.job_timeout)
        if job["status"] != "done":
            raise Failed(f"scenario job failed: {job['error']}")
        delta = job["result"]["scenarios"][0]["delta_point"]["ending_cash"]
        if abs(delta - 500000) > 1:
            raise Failed(f"equity raise moved ending cash by {delta}, expected 500000")
        return f"ending cash +{delta:,.0f} as expected, {_seconds(job)}"

    def query(self, prompt: str):
        r = self.c.post(f"/tenants/{self.tenant}/query", headers=self.h, json={"prompt": prompt, "simulations": 200})
        if r.status_code == 503 and "not configured" in r.text:
            return "SKIP: no model configured (set LLM_BASE_URL and LLM_MODEL on the API and worker)"
        if r.status_code != 202:
            raise Failed(f"POST /query -> {r.status_code}: {r.text[:300]}")
        job = self.poll(r.json()["id"], self.query_timeout)
        if job["status"] != "done":
            raise Failed(f"query failed: {job['error']}",
                         "Model endpoint errors: check LLM_BASE_URL ends in /v1, LLM_MODEL matches the served "
                         "name, and LLM_API_KEY. A cold GPU can take minutes; retry with --query-timeout.")
        res = job["result"]
        i = res["interpretation"]
        out = [f"read as '{i['intent']}': {i['explanation']}  ({res['llm']['latency_seconds']:.1f}s, "
               f"{res['llm']['attempts']} attempt(s), model {res['llm']['model']})"]
        out += [f"      {line}" for line in res.get("summary", [])]
        out += [f"      warning: {w}" for w in res.get("warnings", [])]
        return "\n".join(out)


def _seconds(job: dict) -> str:
    from datetime import datetime

    if job.get("started_at") and job.get("finished_at"):
        s = (datetime.fromisoformat(job["finished_at"]) - datetime.fromisoformat(job["started_at"])).total_seconds()
        return f"{s:.1f}s of work"
    return "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="API base URL, e.g. https://forecast-api.onrender.com")
    ap.add_argument("--key", help="API_KEY of the web service")
    ap.add_argument("--job-timeout", type=float, default=120, help="seconds to wait for forecast/scenario jobs")
    ap.add_argument("--query-timeout", type=float, default=420, help="seconds to wait for /query (GPU cold start)")
    ap.add_argument("--prompt", default="What if we raise 500k from investors in June?")
    ap.add_argument("--skip-query", action="store_true")
    args = ap.parse_args()

    s = Smoke(args.url, args.key, args.job_timeout, args.query_timeout)
    checks = [
        ("health", s.health), ("auth", s.auth), ("tenant + upload", s.upload), ("tenant keys", s.tenant_keys),
        ("forecast", s.forecast),
        ("forecast job", s.forecast_job), ("scenario job", s.scenario_job),
    ]
    if not args.skip_query:
        checks.append(("query", lambda: s.query(args.prompt)))
    for name, fn in checks:
        start = time.monotonic()
        try:
            detail = fn()
        except Failed as e:
            print(f"FAIL  {name:16s} {e}")
            if e.hint:
                print(f"      hint: {e.hint}")
            return 1
        mark = "SKIP" if str(detail).startswith("SKIP") else "PASS"
        detail = str(detail).removeprefix("SKIP: ")
        print(f"{mark}  {name:16s} {time.monotonic() - start:5.1f}s  {detail}")
    print("\nPipeline OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
