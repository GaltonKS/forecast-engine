"""Score the LLM layer against llm/evals.json, and a stand-in model for tests."""
from __future__ import annotations

import json
from pathlib import Path

import httpx

from .client import LLMError
from .schema import Interpretation

EVALS = Path(__file__).with_name("evals.json")
SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "sample_financials.csv"
SAMPLE_BALANCES = SAMPLE.with_name("sample_balances.csv")


def sample_data():
    """(monthly, balances) for the sample company the eval cases are written for."""
    from engine import load_csv, to_monthly
    from engine.ingest import to_balances

    items = load_csv(SAMPLE) + load_csv(SAMPLE_BALANCES)
    return to_monthly(items), to_balances(items)


def load_cases() -> list[dict]:
    return json.loads(EVALS.read_text())["cases"]


def check(case: dict, interp: Interpretation | None, warnings: list[str], error: str | None = None) -> list[str]:
    """Return the list of failures; empty means the case passed."""
    exp = case["expect"]
    if interp is None:
        return [f"no usable interpretation: {error}"]
    fails = []
    if interp.intent != exp["intent"]:
        fails.append(f"intent {interp.intent!r}, expected {exp['intent']!r}")
    if "horizon" in exp and interp.horizon_months != exp["horizon"]:
        fails.append(f"horizon {interp.horizon_months}, expected {exp['horizon']}")
    if "warning_contains" in exp and not any(exp["warning_contains"] in w for w in warnings):
        fails.append(f"expected a warning containing {exp['warning_contains']!r}")
    want = exp.get("scenarios", [])
    if len(interp.scenarios) != len(want):
        fails.append(f"{len(interp.scenarios)} scenarios, expected {len(want)}")
        return fails
    for i, (got, w) in enumerate(zip(interp.scenarios, want), 1):
        ov = got.overrides.model_dump(exclude_none=True)
        for key, (value, tol) in w["overrides"].items():
            if key not in ov:
                fails.append(f"scenario {i}: missing override {key}")
            elif abs(ov[key] - value) > tol:
                fails.append(f"scenario {i}: {key}={ov[key]:.4f}, expected {value}±{tol}")
        for key in set(ov) - set(w["overrides"]):
            fails.append(f"scenario {i}: unexpected override {key}={ov[key]}")
        unmatched = list(got.adjustments)
        for wa in w["adjustments"]:
            hit = next(
                (a for a in unmatched
                 if a.line == wa["line"] and a.kind == wa["kind"]
                 and a.start.strftime("%Y-%m") == wa["start"]
                 and (a.end.strftime("%Y-%m") if a.end else None) == wa["end"]
                 and abs(a.value - wa["value"]) <= abs(wa["value"]) * wa["rel_tol"] + 1e-9),
                None,
            )
            if hit is None:
                fails.append(
                    f"scenario {i}: no adjustment {wa['line']} {wa['kind']} {wa['value']} "
                    f"from {wa['start']}" + (f" to {wa['end']}" if wa["end"] else "")
                    + "; got " + ", ".join(
                        f"{a.line} {a.kind} {a.value:g} {a.start:%Y-%m}" + (f"..{a.end:%Y-%m}" if a.end else "")
                        for a in got.adjustments) or "none"
                )
            else:
                unmatched.remove(hit)
        for a in unmatched:
            fails.append(f"scenario {i}: unexpected adjustment {a.line} {a.kind} {a.value:g} from {a.start:%Y-%m}")
    return fails


def fake_transport(replies: dict[str, str] | None = None) -> httpx.MockTransport:
    """An OpenAI-compatible chat endpoint that answers each eval prompt with
    its ideal reply (or a given override reply), for tests and self-checks."""
    by_prompt = {c["prompt"]: json.dumps(c["ideal"]) for c in load_cases()}
    by_prompt.update(replies or {})

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = next(m["content"] for m in body["messages"] if m["role"] == "user")
        content = by_prompt.get(prompt)
        if content is None:
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]})
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1500, "completion_tokens": 120},
        })

    return httpx.MockTransport(handler)


def run_case(case: dict, monthly, cfg, balances=None) -> dict:
    from .interpret import interpret

    try:
        interp, warnings, call = interpret(case["prompt"], monthly, cfg, balances=balances)
        error = None
    except LLMError as e:
        interp, warnings, call, error = None, [], None, str(e)
    fails = check(case, interp, warnings, error)
    return {
        "id": case["id"],
        "passed": not fails,
        "failures": fails,
        "latency_seconds": call.latency_seconds if call else None,
        "attempts": call.attempts if call else None,
        "interpretation": interp.model_dump(mode="json") if interp else None,
        "warnings": warnings,
    }
