# Financial Projection Engine

A financial forecasting engine with an HTTP API. You upload a business's
monthly financials, it fits growth and margin drivers, backtests several
forecasting models on the most recent months, and returns projections with
every assumption, formula and accuracy score included. The math is
deterministic. A future LLM layer will only turn prompts into driver overrides.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the target architecture,
build order, and what's still missing.

```
engine/            the math (step 1)
  schema.py        Pydantic contracts: line items, drivers, overrides, forecast response
  ingest.py        CSV -> validated line items -> one row per month
  forecast.py      fit drivers, pick a model per line, apply overrides, project forward
  stats.py         seasonality detection + models: driver, seasonal naive, ETS, ARIMA, Prophet (step 3)
  backtest.py      rolling-origin backtest and model selection (step 3)
app/               the API (step 2)
  main.py          FastAPI routes
  models.py        database tables (source of truth for the schema)
  config.py        settings from environment variables
render.yaml        Render blueprint: web service + Postgres
scripts/           sample data generator + example run
tests/             pytest suite (engine, models, API)
```

## Run locally

```bash
pip install -r requirements.txt
python scripts/make_sample_data.py
uvicorn app.main:app --reload          # uses SQLite (local.db) by default
# open http://localhost:8000/docs for interactive API docs
pytest -q
```

## Deploy to Render

1. Push this repo to GitHub.
2. In Render: **New -> Blueprint**, then pick the repo. `render.yaml` creates the
   `forecast-api` web service and the `forecast-db` Postgres database.
3. When prompted, set `CORS_ORIGINS` to your frontend's URL (for example
   `https://app.example.com`). `API_KEY` is generated for you. Copy it from the
   service's Environment tab.
4. Tables are created automatically on first start.

| env var        | purpose                                                    |
|----------------|------------------------------------------------------------|
| `DATABASE_URL` | Postgres connection (set by Render); SQLite if unset       |
| `API_KEY`      | required on every call except `/health`; API is open if unset |
| `CORS_ORIGINS` | comma-separated origins allowed to call from a browser     |
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
| POST   | `/tenants/{id}/forecasts`                    | run and save a forecast               |
| GET    | `/tenants/{id}/forecasts`                    | list saved forecasts                  |
| GET    | `/tenants/{id}/forecasts/{forecast_id}`      | read one saved forecast               |

Forecast request (all fields optional):

```json
{ "horizon_months": 12,
  "overrides": { "revenue_growth": 0.02, "opex_growth": 0.0 },
  "method": "auto",
  "seasonal": null }
```

`method` is `auto` (default), `driver`, `seasonal_naive`, `ets`, `arima` or
`prophet`. `seasonal` forces month-of-year seasonality on or off; `null` detects it.

The response has the monthly projections, each driver's value and how it was
derived (`method`, `overridden`), the `formulas`, `models` (which model each
line used, why, its seasonality, every model's backtest score, and the chosen
model's predictions vs. actuals for the latest window) and `warnings` (short
history, missing months). Your frontend can show "the math" straight from it. See
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
```

## CSV format

| period     | account              | category | amount   |
|------------|----------------------|----------|----------|
| 2024-01-01 | Subscription revenue | revenue  | 80000.00 |
| 2024-01-01 | Hosting              | cogs     | 14000.00 |

`category` is one of `revenue`, `cogs`, `opex`, `other`.

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

## Security notes

- The single `API_KEY` is meant for server-to-server calls. If your frontend
  calls this API straight from the browser, the key is visible to users. Put
  your frontend's backend in between, or replace this with your app's user auth
  (for example verifying its JWTs) before real customer data goes in.
- Every query is scoped by tenant. A forecast from one tenant returns 404 under another.
