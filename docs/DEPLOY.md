# Deploying the full pipeline

Three things go live, in this order. Each ends with a check, so a problem
shows up at the step that caused it.

1. Render: API, worker, Valkey queue, Postgres
2. Modal: the GPU model for `/query` (optional to start; everything else works without it)
3. Your frontend, calling the API from its backend

You'll need accounts on Render and Modal, and this repo on GitHub.

## 1. Render

1. In Render: **New → Blueprint**, pick this repo. If the Blueprint already
   exists, open it and **Sync** so it picks up the current `render.yaml`.
   It creates or updates:
   - `forecast-api`: web service (starter)
   - `forecast-worker`: background worker (default 0.5 CPU / 512 MB). **Paid; no free plan.**
   - `forecast-queue`: Valkey, `noeviction`, private network only. **Paid**, because the free plan doesn't persist queued jobs.
   - `forecast-db`: Postgres

   Everything is created in **Ohio**, next to the Pilot apps on Fly (Chicago).
   Render can't move a resource between regions later. If you already created
   any of these in another region, delete them and let the Blueprint create
   them again (nothing is lost if no real data is in them yet).
2. When prompted:
   - `CORS_ORIGINS`: your frontend's URL(s), comma-separated.
   - `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`: **leave empty for now** (step 2).
3. Wait for both services to deploy, then check the logs:
   - `forecast-api`: `Application startup complete`
   - `forecast-worker`: `celery@... ready`. If it stops at startup with an
     import error, the logs show it; jobs won't run until it's fixed.
4. Copy `API_KEY` from `forecast-api` → **Environment**.

**Check**: from a machine with this repo and `pip install -r requirements.txt`:

```bash
python scripts/smoke_test.py --url https://forecast-api.onrender.com --key "$API_KEY" --skip-query
```

Expect every line to say `PASS` and `Pipeline OK.` The script creates one
tenant called `smoke-test <time>` per run (there's no delete endpoint yet).

## 2. The model on Modal

1. Install and log in:

   ```bash
   pip install modal && modal setup
   ```

2. Create the key that protects the endpoint (any long random string; keep it):

   ```bash
   modal secret create forecast-llm VLLM_API_KEY=<long random string>
   ```

3. Deploy:

   ```bash
   modal deploy deploy/modal_vllm.py
   ```

   It prints the endpoint URL. Default model `Qwen/Qwen2.5-7B-Instruct` on
   one L4 GPU; change with `LLM_HF_MODEL=... LLM_GPU=... modal deploy ...`.

4. Check the endpoint directly. The **first request starts a GPU and
   downloads the weights (several minutes)**; until it's up you'll get 503s.
   Later cold starts are faster because the weights are cached.

   ```bash
   curl "$MODAL_URL/v1/chat/completions" -H "Authorization: Bearer $VLLM_API_KEY" \
        -H 'content-type: application/json' \
        -d '{"model":"forecast-llm","messages":[{"role":"user","content":"Reply with OK"}]}'
   ```

5. Score it on the eval set before connecting it:

   ```bash
   LLM_BASE_URL="$MODAL_URL/v1" LLM_MODEL=forecast-llm LLM_API_KEY="$VLLM_API_KEY" python scripts/eval_llm.py
   ```

   25 cases: hires, price changes, monthly vs annual growth, loans, tax,
   equipment, runway, off-topic and injection prompts. Failures print what the
   model produced. Tune `llm/prompt.py` or try a larger model until it passes.

6. In Render, set on **both** `forecast-api` and `forecast-worker`:

   | variable | value |
   |---|---|
   | `LLM_BASE_URL` | the Modal URL + `/v1` |
   | `LLM_MODEL` | `forecast-llm` |
   | `LLM_API_KEY` | the `VLLM_API_KEY` value |

   Both services redeploy. Leave `LLM_GUIDED` unset (`json_schema` is right for
   current vLLM; `guided_json` was removed in vLLM 0.12).

**Check**: the full smoke test, including `/query`:

```bash
python scripts/smoke_test.py --url https://forecast-api.onrender.com --key "$API_KEY"
```

The `query` line prints how the prompt was read and the summary. If the GPU
was idle it waits for the cold start (up to `LLM_TIMEOUT_SECONDS`, default
180s, then the job fails with "still unavailable"; raise it if first starts
take longer).

## 3. The frontend

- **Call the API from your frontend's backend, never from the browser.** The
  API key gives full access to every tenant.
- Flow: `POST /tenants` once per customer, `POST /tenants/{id}/actuals/upload`
  for their CSV(s), then either `POST /tenants/{id}/forecasts` (waits) or the
  job endpoints (`/forecast-jobs`, `/scenario-jobs`, `/query`) and poll
  `GET /tenants/{id}/jobs/{job_id}` every 1-2 s until `done` or `failed`.
- `/query` results include `interpretation`; show it so users can check how
  their question was read.
- Interactive docs for every endpoint: `https://forecast-api.onrender.com/docs`.

## Costs while testing

- Render: web service, worker and Valkey are each billed monthly while they exist.
- Modal: the GPU bills only while a container runs: from a request until 10
  idle minutes later (`scaledown_window`). No requests, no GPU cost.

## When something fails

| smoke test says | likely cause |
|---|---|
| `health` can't connect | wrong URL, or `forecast-api` still deploying |
| `auth`: request without a key wasn't rejected | `API_KEY` not set on `forecast-api` (the API is open) |
| 401 on `tenant + upload` | wrong `--key` |
| `forecast job` still `queued` | `forecast-worker` not running, or its `REDIS_URL` differs from the API's |
| `forecast job` still `running` | worker crashed mid-job; check its logs |
| `query` SKIP | `LLM_BASE_URL` / `LLM_MODEL` not set on `forecast-api` |
| `query` fails "still unavailable" | GPU cold start longer than `LLM_TIMEOUT_SECONDS`, or the Modal app isn't deployed |
| `query` fails "rejected the request (401)" | `LLM_API_KEY` doesn't match `VLLM_API_KEY` |
| `query` fails "rejected the request (404)" | `LLM_BASE_URL` missing `/v1`, or `LLM_MODEL` isn't `forecast-llm` |

## Before real customer data

See **Data retention** in `docs/ARCHITECTURE.md`: `/query` currently stores
prompts, and there's no name-stripping, purge or audit log yet.
