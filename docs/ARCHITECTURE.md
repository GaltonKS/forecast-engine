# Architecture

**The LLM reads the prompt; the engine does the math.**

![Financial Projection Engine architecture](architecture.webp)

## Components (target)

| Component | Where | Role |
|---|---|---|
| Existing frontend | outside | Sends prompts, shows results |
| FastAPI (`/query`, `/scenario`, `/forecast`) | Render | Public API |
| Key Value (Valkey) | Render | Celery job queue (`noeviction`, so queued jobs are never dropped) |
| Celery worker | Render | Runs forecast jobs and calls the model |
| Postgres | Render | Financial data, saved forecasts and their assumptions |
| Forecast engine | Render (in the worker) | Driver model producing 3 statements; Prophet / ARIMA / LightGBM; Monte Carlo P10/P50/P90; backtest on recent months |
| Open model (vLLM) | GPU on Modal / RunPod | Turns a prompt into scenario JSON using guided JSON output |
| Data sources | outside | QuickBooks, Xero, CSV and bank feeds, ingested into Postgres |

## Build order

1. Data schema + simple driver forecast — **done**
2. FastAPI wired to frontend — **done** (API built; frontend wiring pending)
3. Statistical / ML forecasting — **done** (seasonality, ETS, ARIMA, Prophet, backtest selection; LightGBM deferred, see below)
4. Scenarios + Monte Carlo — **done** (dated adjustments, scenario comparison, calibrated P10/P50/P90, Valkey + Celery queue)
5. LLM prompt layer (open model) — **code done**; needs a deployed model endpoint and an eval run (see below)

## Where the code stands vs. the plan (after step 5)

**In place:** FastAPI, Postgres (tenants, actuals, saved forecasts with assumptions),
CSV ingest, API-key auth, per-tenant query scoping, Render blueprint. Forecast
engine: seasonality detection, driver / seasonal naive / ETS / ARIMA / Prophet
models per line, rolling-origin backtest with model selection, and the latest
backtest window returned for charting. Monte Carlo P10/P50/P90 ranges checked
against the backtest, dated adjustments, scenario comparison on paired
simulations. Valkey + Celery job queue with a worker service, and job
endpoints to queue and poll. `/query`: the worker sends the prompt and a short
company context to an OpenAI-compatible endpoint (vLLM), gets scenario JSON
back under a JSON schema, checks it, runs it, and returns the interpretation,
warnings, a code-written summary and the full comparison.

**Gaps to close:**

- **Endpoint names.** The diagram has `/query`, `/scenario` and `/forecast`.
  The code has `/tenants/{id}/query`, `/scenario-jobs`, `/forecast-jobs` and
  the synchronous `/forecasts`: tenant-scoped versions of the same three.
- **Job queue: in place.** Forecasts and scenario comparisons can be queued
  (`/forecast-jobs`, `/scenario-jobs`) and polled (`/jobs/{id}`). The worker
  takes one job at a time (`--concurrency 1`); add workers or raise
  concurrency if jobs start waiting. The synchronous `POST /forecasts` stays
  for quick runs. A job interrupted by a hard crash is redelivered after
  Valkey's visibility timeout (1 hour); after a normal deploy it finishes first.
  A job that no worker picks up, or that a worker never finishes, is marked
  failed after `JOB_TIMEOUT_MINUTES` (default 10), so polling always ends.
- **LightGBM not included.** Gradient boosting needs far more points than one
  company's 24-60 monthly values. It would lose to ETS/driver in the backtest
  and add a dependency. Worth revisiting with daily or weekly data, or pooled
  training across tenants.
- **One statement, not three.** The engine produces an income statement only
  (revenue → operating income → net income). The balance sheet and cash flow need
  extra drivers (payment timing, capex, debt) and likely more categories than the
  current four (`revenue`, `cogs`, `opex`, `other`).
- **CSV only.** QuickBooks, Xero and bank feeds should each be a connector that
  outputs the same `LineItem` format, so the engine and API don't change.
- **Model not deployed or chosen yet.** Everything around the model is built
  and tested with a stand-in. Still to do: deploy vLLM on Modal or RunPod, set
  `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY`, run `scripts/eval_llm.py`, and
  tune `llm/prompt.py` until the eval passes. A 7-8B instruct model with
  constrained decoding is a reasonable first try; the eval decides.
- **One prompt, no conversation.** Each `/query` stands alone. Follow-ups
  ("now make it 3 engineers") would need the previous interpretation passed
  back in.
- **Uncertainty is per line, not joint.** Revenue and opex are simulated
  independently, which understates risk if costs fall when revenue does (or
  overstates it if they move together). Fine until there's data to estimate it.
- **No migrations.** Tables are created with `create_all`, which adds new
  tables (step 4's `jobs`) but never changes existing ones. The next change to
  an existing column needs Alembic.
