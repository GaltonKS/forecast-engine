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

## Design proposal: one shared Pilot app (option A)

**Status: draft for discussion. Nothing here is built.**

### Goal

One engine (this repo) is the background for every client, and one Pilot app
serves every client's screens. Each company keeps its own look and its own
address. Onboarding a company is publishing its brain, not deploying an app.

Pilot (`GaltonKS/Pilot`) already splits the parts that never change (engine,
screens) from the parts that do (a `brain.js` and `data.js` per company). This
design keeps that split and changes where the per-company parts live: from
folders baked into one Fly image per company, to records looked up at request
time.

### Today vs proposed

| | Today (one Fly app per company) | Proposed (one shared app) |
|---|---|---|
| Screens | Copied into every company's image | One app, one deploy |
| Brain, logo, sign-in branding | Folder in the image | Stored per company, loaded by hostname |
| Financial data | `data.js` in the image, CSV uploads on the volume | Engine tenant: actuals, balances, forecasts |
| Forecast math | `app/engine.js` in the browser | This engine, called by the Pilot server; the browser draws the results |
| Assistant | Browser builds the prompt; server forwards it to Claude | Engine `/query` (model choice open, see below) |
| Sign-in | Client and owner passwords per app | Credentials per company, checked against the hostname |
| Saved ideas, chat, usage log | Files on each app's volume | One database, scoped by company |
| Onboarding | `tools/deploy-fly.sh` | `tools/publish.js` (new); live in seconds |
| Offboarding | `fly apps destroy` | `tools/offboard.js` (new): export, then delete the company's records |

### Request flow

```
browser: https://acme.<yoursite>/            (or a custom domain)
  -> Pilot app
       1. Host header -> company (exact match in a hostnames table; unknown host -> plain 404 naming no one)
       2. Session cookie must be for that company (company id inside the signed cookie)
       3. Brain + theme for that company (cached; a publish clears the cache)
       4. Page, branded for that company
  browser API calls -> Pilot app -> engine, with the Pilot app's service key and that company's tenant id
```

The company is **always** taken from the hostname, never from anything the
browser sends (no `?tenant=`, no body field). Editing a request can't reach
another company.

### Where the per-company records live

**Recommendation: a separate Pilot database, next to the engine's.** Brains,
hostnames, sign-in credentials, saved ideas and the usage log are about
presentation. The engine keeps financial data and math, so other frontends can
use it later without inheriting Pilot's screens. Both can live on the same
Render Postgres instance as two databases. Each Pilot company record points at
one engine tenant id.

New Pilot tables: `companies` (brain JSON and version, theme, sign-in
branding, logo, engine tenant id), `hostnames` (hostname, company, subdomain
or custom, verified), `credentials`, `ui_state` (saved ideas, chat, corrected
numbers, per role), `usage_events`.

Engine changes: a service key for the Pilot app that must name a tenant on
every call (the current single `API_KEY` gives access to all tenants and
should be retired), and per-tenant keys for anything else that calls it.

### Sign-in

- **Phase 1 (same model as today):** a client password and an owner password
  per company, stored hashed. The session cookie is host-only (no `Domain`
  attribute), so a session on `acme` is never sent to `bravo`, and it carries
  the company id, so it's rejected if it is. Rate limits and the daily
  assistant cap become per company.
- **Phase 2:** real user accounts per company. GaltonKS staff sign in once on
  a separate admin host and can open any company's Pilot desk; every staff
  view of a company is written to the audit log (see Data retention).

### Publishing a company (the cofounder's flow)

Steps 1 to 3 are today's flow, unchanged:

1. `node tools/new-pilot.js pilots/acme examples/home-services`
2. Write the brain and prepare `data.js`; `node tools/check.js pilots/acme`
3. Preview locally: `PILOT_DIR=pilots/acme npm start` (folder mode stays for previews)

Then, instead of deploying:

4. `node tools/publish.js pilots/acme --host acme` runs the check, then:
   - stores the brain, logo and sign-in branding as a new version of the company;
   - converts `data.js` and loads it into the company's engine tenant (created on first publish);
   - claims `acme.<yoursite>`;
   - on first publish, sets the two passwords and prints them once (as `deploy-fly.sh` does).

   The company is live straight away. Re-running `publish` updates the brain or
   refreshes the snapshot. `--env staging` publishes to a staging copy first.
5. Each publish keeps the previous brain version, so a bad publish is undone
   by republishing the last good one.

Data mapping from Pilot's `data.js` to the engine (first version):

| `data.js` | engine category |
|---|---|
| `pnl.lines` with group `income` | `revenue` |
| `pnl.lines` with group `cogs` | `cogs` |
| `pnl.lines` with group `expense` | `opex` |
| `balance.cash` / `ar` / `ap` (as of the 1st, used for the month before) | `cash` / `receivables` / `payables` |

### Engine gaps to close first

Pilot's screens model things this engine doesn't yet. Option 2 means the
engine must cover them before those screens can switch:

| Pilot lever | Engine today |
|---|---|
| Sales by unit (technician, chair, location) and by offering, weekly | Monthly totals per line only |
| Prices and their effect on volume | Revenue `pct` adjustments, no price/volume split |
| Units in or out, unit pay or terms | No unit concept |
| Volume shocks per offering | Revenue adjustments for the whole line |
| Weekly cash range | Monthly cash and runway |
| Monthly bills, cost rates, one-time cash | Covered: opex adjustments, gross margin, `other` / capex / equity adjustments |
| Taxes (US rules table, installments) | Flat rate with loss carry-forward, paid quarterly |

**Transition option:** keep Pilot's browser engine for the levers the engine
lacks, and switch screen by screen as the engine catches up. The engine's
three statements, runway and model selection can appear in Pilot first, as
new panels.

### Hosting

- **Pilot app on Render, in the same region as the engine.** The engine can
  then be a private service with no public URL, reachable only from the Pilot
  app. Keeping Pilot on Fly is possible, but then the engine stays public and
  every call crosses providers.
- At least two instances, so a deploy or a crash doesn't take every company
  down, and a one-step rollback.
- DNS: a wildcard `*.<yoursite>` record and certificate for subdomains, and a
  certificate per custom domain (`forecast.clientco.com`). Confirm the plan you
  pick supports wildcard domains before committing to subdomains.

### Isolation checklist (the cost of sharing one app)

- Company comes from the hostname only; every engine call carries that company's tenant id.
- A test for every route that tries to read or write another company's data with a valid session.
- Host-only cookies, with the company id inside the signed session.
- Per-company rate limits and assistant caps.
- Logo uploads: images only, size-capped, served with `nosniff` (as today).
- Staff access across companies is logged.
- The usage log records events, not prompts (see Data retention).

### Migration

1. Engine: service key plus tenant scoping, and the `data.js` loader.
2. Pilot: a shared mode in `server.js` (company from hostname, records from
   the database), alongside today's folder mode for previews and existing pilots.
3. Staging: publish one sample company and compare it side by side with its Fly version.
4. Move clients one at a time. Export each Fly app's usage log, then destroy it.

### Open questions

1. Pilot records in their own database (recommended) or inside the engine's?
2. Subdomains of one domain, custom domains per client, or both?
3. Sign-in: keep two passwords per company at first, or go straight to user accounts?
4. Assistant: Claude, as Pilot uses today, or an open model as the engine's `/query` was built for?
5. Engine gaps: build units, offerings, prices and weekly cash into the engine
   before switching, or switch screen by screen with Pilot's browser engine
   covering the rest?
6. Who on the GaltonKS side can open a client's Pilot desk, and what gets logged when they do?
