# Financial Projection Engine

A financial forecasting engine with an HTTP API. You upload a business's
monthly financials (income statement, plus month-end balances for the balance
sheet), it fits growth, margin and working-capital drivers, backtests several
forecasting models on the most recent months, and returns projections with
every assumption, formula and accuracy score included: income statement,
balance sheet, cash flow and cash runway, with P10/P50/P90 ranges from a Monte
Carlo simulation. Scenarios (driver overrides and dated
changes like a hire or a price rise) can be compared side by side. The math is
deterministic. Plain-language questions go through `/query`: an open model
turns the prompt into scenario JSON, and the engine does the math.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the target architecture,
build order, and what's still missing.

```
engine/            the math (step 1)
  schema.py        Pydantic contracts: line items, drivers, overrides, forecast response
  ingest.py        CSV -> validated line items -> one row per month
  forecast.py      fit drivers, pick a model per line, apply overrides, project forward
  stats.py         seasonality detection + models: driver, seasonal naive, ETS, ARIMA, Prophet (step 3)
  backtest.py      rolling-origin backtest and model selection (step 3)
  simulate.py      Monte Carlo ranges, calibration, dated adjustments (step 4)
  statements.py    balance sheet, cash flow and runway from the income statement forecast
llm/               the prompt layer (step 5): prompt -> checked scenario JSON
  schema.py        what the model must return (sent as a JSON schema) + API models
  prompt.py        instructions and the company context sent with each prompt
  client.py        OpenAI-compatible client for vLLM (Modal / RunPod)
  interpret.py     parse, check, retry once, flag suspicious values
  summary.py       plain-language results, written by code from engine numbers
  evals.json       prompts with the scenario each should produce
app/               the API (step 2)
  main.py          FastAPI routes
  models.py        database tables (source of truth for the schema)
  jobs.py          runs a queued forecast/scenario job (step 4)
  worker.py        Celery worker that pulls jobs from Valkey (step 4)
  config.py        settings from environment variables
render.yaml        Render blueprint: web service, worker, Valkey, Postgres
scripts/           sample data, example run, eval_llm.py, fake_llm_server.py
tests/             pytest suite (engine, models, scenarios, LLM layer, API)
```

## Run locally

```bash
pip install -r requirements.txt
python scripts/make_sample_data.py
uvicorn app.main:app --reload          # uses SQLite (local.db) by default
# open http://localhost:8000/docs for interactive API docs
pytest -q
```

Without `REDIS_URL`, jobs run inside the API process right after the response
is sent, so no queue is needed locally. To run the real queue:

```bash
redis-server --maxmemory-policy noeviction &       # or a Valkey container
export REDIS_URL=redis://localhost:6379/0
uvicorn app.main:app --reload &
celery -A app.worker worker --concurrency 1 --loglevel INFO
```

To run the API tests against Postgres instead of SQLite (tables in that
database are dropped and recreated):

```bash
TEST_DATABASE_URL=postgresql://user@localhost/forecast_test pytest -q tests/test_api.py
```

## Deploy to Render

1. Push this repo to GitHub.
2. In Render: **New -> Blueprint**, then pick the repo. `render.yaml` creates the
   `forecast-api` web service, the `forecast-worker` background worker, the
   `forecast-queue` Valkey instance and the `forecast-db` Postgres database.
3. When prompted, set `CORS_ORIGINS` to your frontend's URL (for example
   `https://app.example.com`). `API_KEY` is generated for you. Copy it from the
   service's Environment tab.
4. Tables are created automatically on first start (including the new `jobs`
   table on an existing database).

| env var        | purpose                                                    |
|----------------|------------------------------------------------------------|
| `DATABASE_URL` | Postgres connection (set by Render); SQLite if unset       |
| `API_KEY`      | required on every call except `/health`; API is open if unset |
| `CORS_ORIGINS` | comma-separated origins allowed to call from a browser     |
| `REDIS_URL`    | Valkey connection for the job queue (set by Render); jobs run in-process if unset |
| `LLM_BASE_URL` | model endpoint for `/query`, ending in `/v1`; `/query` returns 503 if unset |
| `LLM_MODEL`    | model name the endpoint serves                             |
| `LLM_API_KEY`  | sent as a Bearer token, if the endpoint needs one          |
| `LLM_GUIDED`   | `json_schema` (default), `guided_json` (older vLLM) or `none` |
| `LLM_TIMEOUT_SECONDS` | wait for the model (default 180, for GPU cold starts) |
| `JOB_TIMEOUT_MINUTES` | a job still queued or running after this long is marked failed (default 10) |
| `MAX_UPLOAD_MB`| upload size limit (default 10)                             |

## API

Send the key as `Authorization: Bearer <API_KEY>` or `X-API-Key: <API_KEY>`.

| method | path                                         | what it does                          |
|--------|----------------------------------------------|---------------------------------------|
| GET    | `/health`                                    | liveness check (no key needed)        |
| POST   | `/tenants`                                   | create a customer `{"name": "Acme"}`  |
| GET    | `/tenants/{id}`                              | read a customer                       |
| POST   | `/tenants/{id}/actuals/upload`               | upload CSV (multipart field `file`)   |
| GET    | `/tenants/{id}/actuals/monthly`              | history by month, ready to chart      |
| POST   | `/tenants/{id}/forecasts`                    | run and save a forecast (waits for it) |
| GET    | `/tenants/{id}/forecasts`                    | list saved forecasts                  |
| GET    | `/tenants/{id}/forecasts/{forecast_id}`      | read one saved forecast               |
| POST   | `/tenants/{id}/forecast-jobs`                | queue a forecast, returns a job (202) |
| POST   | `/tenants/{id}/scenario-jobs`                | queue baseline + scenarios comparison |
| POST   | `/tenants/{id}/query`                        | ask in plain language (queues a job)  |
| GET    | `/tenants/{id}/jobs/{job_id}`                | poll a job; `result` once `done`      |
| GET    | `/tenants/{id}/jobs`                         | list jobs (`?kind=`, `?status=`)      |

Use the job endpoints from the frontend: queue, then poll every second or two
until `status` is `done` or `failed` (`error` says why). A finished forecast job
is also saved as a forecast (`forecast_id`).

Every job reaches `done` or `failed`. If no worker picks a job up, or a worker
dies mid-job, the job is marked `failed` with a "Timed out" error once it has
been waiting or running for `JOB_TIMEOUT_MINUTES` (checked whenever jobs are
read). A timed-out job stays failed: if a worker reaches it later, it skips it,
and a result that arrives after the timeout is discarded rather than saved.

Forecast request (all fields optional):

```json
{ "horizon_months": 12,
  "overrides": { "revenue_growth": 0.02, "opex_growth": 0.0 },
  "adjustments": [
    { "line": "opex", "kind": "add", "value": 30000, "start": "2026-03-01", "label": "Hire 2 engineers" },
    { "line": "revenue", "kind": "pct", "value": 0.05, "start": "2026-04-01", "end": null, "label": "Price rise" } ],
  "method": "auto",
  "seasonal": null,
  "simulations": 1000,
  "seed": 0 }
```

`adjustments` are dated changes on top of the forecast. `add` is currency per
month (negative to subtract); `pct` scales the line (0.05 = +5%). `line` is
`revenue`, `cogs`, `opex` or `other`; `end` is inclusive, `null` = to the
horizon. A `pct` change on revenue carries through to COGS at the same margin.

Scenario comparison request (same options, plus up to 10 named scenarios):

```json
{ "horizon_months": 12,
  "scenarios": [
    { "name": "Hire 2", "adjustments": [{ "line": "opex", "kind": "add", "value": 30000, "start": "2026-03-01" }] },
    { "name": "Growth halves", "overrides": { "revenue_growth": 0.016 } } ] }
```

The result has the `baseline` and each scenario's full forecast, `delta_point`
(difference in horizon totals) and `delta` (P10/P50/P90 of the difference in
total revenue and net income).

`method` is `auto` (default), `driver`, `seasonal_naive`, `ets`, `arima` or
`prophet`. `seasonal` forces month-of-year seasonality on or off; `null` detects it.

The response has the monthly projections, each driver's value and how it was
derived (`method`, `overridden`), the `formulas`, `models` (which model each
line used, why, its seasonality, every model's backtest score, and the chosen
model's predictions vs. actuals for the latest window) and `warnings` (short
history, missing months), and `uncertainty`: P10/P50/P90 per month for revenue,
gross profit, opex and net income, horizon totals, the probability of a loss
month, the fitted noise, and how well the ranges held up in the backtest.
Your frontend can show "the math" straight from it. See
`sample_data/example_response.json`.

Re-uploading a month replaces that account's amount, so corrections are safe.

### Example

```bash
API=https://forecast-api.onrender.com; KEY=...
T=$(curl -s -X POST $API/tenants -H "X-API-Key: $KEY" \
      -H 'content-type: application/json' -d '{"name":"Demo"}' | jq -r .id)
curl -X POST $API/tenants/$T/actuals/upload -H "X-API-Key: $KEY" \
     -F file=@sample_data/sample_financials.csv
curl -X POST $API/tenants/$T/forecasts -H "X-API-Key: $KEY" \
     -H 'content-type: application/json' -d '{"overrides":{"revenue_growth":0.02}}'

# queued scenario comparison, then poll
J=$(curl -s -X POST $API/tenants/$T/scenario-jobs -H "X-API-Key: $KEY" \
      -H 'content-type: application/json' \
      -d '{"scenarios":[{"name":"Hire 2","adjustments":[{"line":"opex","kind":"add","value":30000,"start":"2026-03-01"}]}]}' \
      | jq -r .id)
curl -s $API/tenants/$T/jobs/$J -H "X-API-Key: $KEY" | jq '.status, .result.scenarios[0].delta'
```

## CSV format

| period     | account              | category | amount   |
|------------|----------------------|----------|----------|
| 2024-01-01 | Subscription revenue | revenue  | 80000.00 |
| 2024-01-01 | Hosting              | cogs     | 14000.00 |

| 2025-12-01 | Bank accounts        | cash        | 720708.19 |
| 2025-12-01 | Accounts receivable  | receivables | 204078.40 |

Map each account in your chart of accounts to one category, once per customer.

| category | statement | amount |
|---|---|---|
| `revenue`, `cogs`, `opex`, `other` | income statement | the month's total (`other` is signed: interest income +, expense -) |
| `depreciation` | income statement | the month's depreciation; counted in opex and added back as non-cash |
| `cash`, `receivables`, `inventory`, `fixed_assets` (net), `other_assets` | balance sheet: assets | month-end balance |
| `payables`, `debt`, `other_liability` | balance sheet: liabilities | month-end balance |
| `equity` | balance sheet | month-end balance (checked; see below) |

Income statement and balance rows can be in one file or separate uploads, in
either order (see `sample_data/sample_balances.csv`). The balance sheet,
cash flow and runway need balances for the **last month of income statement
data**; earlier months improve the fitted drivers. If assets - liabilities
doesn't equal the equity you upload, equity is set to assets - liabilities and
the forecast warns, since it usually means an account is missing.

## Balance sheet and cash flow

Built from the income statement forecast, starting at the last actual
month's balances:

| line | how it's projected |
|---|---|
| receivables | revenue x `dso_days` / 30.4 |
| inventory | cogs x `dio_days` / 30.4 |
| payables | (cogs + opex - depreciation) x `dpo_days` / 30.4 |
| fixed assets | previous + capex - depreciation |
| debt | previous + borrowed - repaid (from `debt` adjustments) |
| equity | previous + net income + raised (from `equity` adjustments) |
| other assets / liabilities | held at the opening balance |
| cash | previous + operating + investing + financing cash flow |

Cash flow is the indirect method: net income + depreciation - increases in
receivables and inventory + increases in payables, minus capex, plus
financing. Cash is the one line no driver sets, so the balance sheet balances
every month (`check` is always 0).

| driver | default fit |
|---|---|
| `dso_days` | receivables / monthly revenue x 30.4, last 3 months with balances |
| `dio_days` | inventory / monthly cogs x 30.4 |
| `dpo_days` | payables / monthly cash costs x 30.4 |
| `depreciation_monthly` | average of the last 3 months of `depreciation` rows |
| `capex_monthly` | change in fixed assets + depreciation, last 3 consecutive months; equal to depreciation if there's no history |

All five can be overridden in a scenario. `capex`, `debt` and `equity` are
also adjustment lines (kind `add`), e.g. a one-off equipment purchase, a loan
drawn in March and repaid monthly, or a funding round in June.

The response adds `opening_balance`, `balance_sheet`, `cash_flow` and `cash`
(opening, ending and lowest cash, and the first month cash goes negative). With
simulations, `uncertainty` adds cash P10/P50/P90 per month, `prob_cash_out`
(share of paths where cash goes negative) and `runway` (months until it does:
`p10` is the pessimistic end; null means beyond the horizon). Scenario deltas
include ending cash.

**Not modelled yet:** income tax; interest calculated from the debt balance
(interest stays in `other` at its historical level, so a new loan's interest
must be added as an `other` adjustment); depreciation on new capex; payment
terms that differ by customer or season.

## Driver fitting

| driver         | default fit                                                                     |
|----------------|---------------------------------------------------------------------------------|
| revenue_base   | last 3 months (seasonally adjusted), each rolled forward to the last month at revenue_growth, averaged |
| revenue_growth | log-linear trend over the last 12 months (seasonally adjusted)                  |
| gross_margin   | (revenue - cogs) / revenue, last 6 months                                       |
| opex_base      | same as revenue_base, for opex                                                  |
| opex_growth    | same as revenue_growth, for opex                                                |
| other_net      | average of the last 6 months                                                    |

## How each line is forecast

Revenue and opex each get their own model. COGS follows revenue through
`gross_margin`; other income stays flat at `other_net`.

1. **Seasonality.** With 24+ months, the engine tests for a month-of-year
   pattern (ANOVA on detrended log values, p < 0.01, swing of at least 3%). If
   found, the driver model multiplies by a seasonal index per calendar month.
2. **Backtest.** Every model is fitted on history up to an earlier month and
   asked to predict the next 6 months (3 when history is under 24 months),
   which we already know. This repeats from up to 3 starting points. The score is
   WAPE: total absolute error / total actual.
3. **Selection.** The lowest WAPE wins, but a more complex model has to beat a
   simpler one by more than 10% (relative) + 0.2 points. Order, simplest first:
   driver, seasonal naive, ETS, ARIMA, Prophet.
4. **Scenarios.** Overriding a line's `_base` or `_growth` switches that line to
   the driver formula, so "2% growth" means exactly 2%. Backtesting is skipped
   for that line.

| history       | what happens                                                        |
|---------------|---------------------------------------------------------------------|
| < 3 months    | rejected                                                            |
| 3-14 months   | driver model only, no backtest                                      |
| 15-23 months  | backtest on 3-month windows; no seasonality                         |
| 24+ months    | backtest on 6-month windows; seasonality detected                   |

## Asking in plain language (`/query`)

```json
{ "prompt": "What if we hire two engineers in March at 120k a year and raise prices 5% from April?",
  "horizon_months": null, "simulations": 1000, "run": true }
```

This queues a job. When it's done, `result` has:

- `interpretation`: how the prompt was read: `intent` (`scenarios`, `baseline`
  or `unsupported`), a one-line `explanation`, the `assumptions` the model had
  to make (e.g. a salary it guessed), and the exact `scenarios` it built. Show
  this to the user so they can check and correct it.
- `warnings`: values worth a second look (e.g. 30%/month growth, which is
  usually an annual rate typed as monthly) plus the engine's own warnings.
- `summary`: plain-language results such as "Hire 2: net income -200k vs
  baseline". Written by code from the engine's numbers, never by the model.
- `comparison`: the full baseline + scenarios result (same as `/scenario-jobs`).
- `forecast_ids`: each scenario is saved as a forecast with the prompt attached.

`"run": false` only interprets the prompt, so a UI can show "here's what I
understood" before running anything.

**How the model is used.** The model sees the instructions, a short company
context (last actual month, first forecast month, current drivers, recent
averages) and the prompt. It must answer in the `LLMOutput` JSON shape, enforced
by constrained decoding on vLLM. Code then checks every value; an unusable reply
gets one retry with the error. The model never sees other tenants' data, and
its output can only become scenario inputs, so a prompt can't make it do
anything else. If the endpoint is down or slow, the job fails with a clear
error ("Model endpoint returned 503", "timed out after 180s").

The company context (summary figures, not raw rows) is sent to the model
endpoint, so host it somewhere you're comfortable sending financial data.

### Choosing and checking a model

`llm/evals.json` holds 18 prompts with the scenario each should produce (on the
sample data). Score any endpoint:

```bash
LLM_BASE_URL=https://.../v1 LLM_MODEL=<model> python scripts/eval_llm.py
```

To build the frontend before a GPU endpoint exists, run the stand-in model. It
answers the eval prompts with their ideal replies:

```bash
uvicorn scripts.fake_llm_server:app --port 8800
export LLM_BASE_URL=http://localhost:8800/v1 LLM_MODEL=fake
```

## Ranges (Monte Carlo)

Each of the 1,000 simulated paths (default) multiplies a line's point forecast
by noise fitted to its last 12 months, on a log scale so it acts as a percentage:

- **level error**: uncertainty in where the business is right now (drawn once per path)
- **growth error**: uncertainty in the growth rate, which compounds over the horizon
- **monthly noise**: month-to-month wobble, AR(1) so a bad month can carry into the next

Gross margin varies around its fitted value (a per-path shift plus monthly
noise), and other income varies month to month. Revenue and opex are simulated
independently. P50 equals the point forecast.

**Calibration.** The engine replays the backtest: for each past origin it
builds the 80% range from the data available then and checks how many actual
months fell inside. If fewer than 80% did, ranges are widened to match. They are
never narrowed. The result is in `uncertainty.calibration`.

**Scenarios.** A scenario's own assumptions carry no uncertainty: overriding
`revenue_growth` removes the growth error for revenue, and a fixed `add`
adjustment is exactly that amount. Baseline and scenarios use the same random
draws (same seed), so `delta` shows the effect of the assumptions, not
simulation noise. A fixed $30k/month hire shows as exactly -$30k/month.

## Security notes

- The single `API_KEY` is meant for server-to-server calls. If your frontend
  calls this API straight from the browser, the key is visible to users. Put
  your frontend's backend in between, or replace this with your app's user auth
  (for example verifying its JWTs) before real customer data goes in.
- Every query is scoped by tenant. A forecast from one tenant returns 404 under another.
