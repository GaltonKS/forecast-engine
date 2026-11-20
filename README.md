# Financial Projection Engine: Step 1

Data schema plus a driver-based forecast. No API or LLM yet.

```
engine/schema.py    Pydantic contracts (line items, drivers, overrides, forecast response)
engine/ingest.py    CSV -> validated line items -> one row per month
engine/forecast.py  fit drivers from history, apply overrides, project forward
db/schema.sql       Postgres tables (tenant-scoped, forecasts saved with their assumptions)
scripts/            sample data generator + example run
tests/              pytest suite
```

## Run it

```bash
pip install -r requirements.txt
python scripts/make_sample_data.py
python scripts/run_example.py
pytest -q
```

## Input CSV format

| period     | account              | category | amount   |
|------------|----------------------|----------|----------|
| 2024-01-01 | Subscription revenue | revenue  | 80000.00 |
| 2024-01-01 | Hosting              | cogs     | 14000.00 |

`category` is one of `revenue`, `cogs`, `opex`, `other`. You map your real
chart of accounts to these four once per customer.

## How the forecast works

Drivers are fitted from history. You can override any of them per scenario.

| driver         | default fit                                   |
|----------------|-----------------------------------------------|
| revenue_base   | average of the last 3 months                  |
| revenue_growth | log-linear trend over the last 12 months      |
| gross_margin   | (revenue - cogs) / revenue, last 6 months     |
| opex_base      | average of the last 3 months                  |
| opex_growth    | log-linear trend over the last 12 months      |
| other_net      | average of the last 6 months                  |

Every response includes the driver values, how each was derived, the formulas,
and warnings (short history, missing months). Your frontend can show "the math"
directly from it. See `sample_data/example_response.json`.

`DriverOverrides` is the contract the LLM layer will fill in later. A prompt
becomes overrides, and the engine does the math.
# forecast-engine
