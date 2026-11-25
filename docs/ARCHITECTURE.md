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
| Forecast engine | Render (in the worker) | Driver model producing 3 statements (income, balance sheet, cash flow); Prophet / ARIMA / LightGBM; Monte Carlo P10/P50/P90; backtest on recent months |
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
- **Three statements: in place.** Balance sheet and cash flow are projected
  from the income statement with working-capital days, capex and depreciation
  fitted from uploaded balances; cash runway comes with Monte Carlo ranges.
  Income tax (with losses carried forward, paid quarterly), interest from the
  debt balance, and depreciation on new capex are included. Not modelled yet:
  interest on cash, pre-forecast tax losses, per-loan rates, and correlation
  between the uncertainty in revenue and costs.
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

## Data retention

| What | How long | Why |
|---|---|---|
| Raw uploads, before name-stripping | Deleted on successful parse; 7 days if parse fails | Shortest possible window on client PII |
| Client roster used to scrub free text | Destroyed at end of the parse run | Already the design |
| De-identified snapshot + declared inputs | Engagement + 30 days | Product needs it; 30 days to request an export |
| Prompts and figures sent to the model | Not retained by us | Nothing accumulates |
| Access / audit log | 24 months | Outlives Xero's twice-yearly audit cycle |

**Where the code stands against this (not yet enforced):**

- **Raw uploads:** the API never writes the uploaded file anywhere. It parses
  the CSV in memory and stores only the line items (period, account name,
  category, amount), whether or not parsing succeeds. There is no
  name-stripping step yet, so account names are stored as uploaded.
- **Client roster / free-text scrubbing:** not built yet.
- **Snapshot and inputs:** actuals, saved forecasts and jobs are kept with no
  end date. Meeting "engagement + 30 days" needs an engagement end date per
  tenant, a scheduled purge, and an export endpoint.
- **Prompts sent to the model:** this conflicts with the code today. `/query`
  stores the prompt in `jobs.request` and `forecasts.prompt`, and the model's
  interpretation in `jobs.result`. Meeting "not retained" means dropping or
  redacting those, and confirming that the model host (vLLM on Modal or
  RunPod) keeps no request logs.
- **Access / audit log:** none exists yet. API calls aren't recorded per
  user or tenant beyond the platform's own request logs.
