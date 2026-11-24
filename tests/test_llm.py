"""Step 5: the LLM prompt layer, tested against a stand-in model.

What a real model will answer can't be tested here; scripts/eval_llm.py does
that against an endpoint. These tests cover everything around the model:
the request we send, parsing, checks, retries, failures, the job and the API.
"""
import json
import re
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import load_csv, to_monthly  # noqa: E402
from llm.client import LLMBadOutput, LLMConfig, LLMNotConfigured, LLMUnavailable, chat_json  # noqa: E402
from llm.evals import SAMPLE, check, fake_transport, load_cases, run_case, sample_data  # noqa: E402
from llm.interpret import interpret  # noqa: E402
from llm.prompt import build_context, messages, render_context  # noqa: E402
from llm.schema import Interpretation, LLMOutput  # noqa: E402
from llm.summary import fmt  # noqa: E402

MONTHLY = to_monthly(load_csv(SAMPLE))
MONTHLY_B, BALANCES = sample_data()
CASES = {c["id"]: c for c in load_cases()}
EMPTY_OV = dict(revenue_base=None, revenue_growth=None, gross_margin=None, opex_base=None, opex_growth=None, other_net=None,
                dso_days=None, dio_days=None, dpo_days=None, capex_monthly=None, depreciation_monthly=None,
                useful_life_months=None, interest_rate=None, tax_rate=None)


@pytest.fixture(autouse=True)
def fake_clock(monkeypatch):
    """Retries wait on a fake clock, so cold-start tests run instantly."""
    import llm.client as client

    clock = {"t": 0.0, "slept": []}
    monkeypatch.setattr(client, "_now", lambda: clock["t"])

    def sleep(s):
        clock["slept"].append(s)
        clock["t"] += s

    monkeypatch.setattr(client, "_sleep", sleep)
    return clock


def reply(content, finish="stop", status=200):
    return httpx.Response(status, json={
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    })


def scripted(*responses):
    """A model that answers with the given responses in order, recording requests."""
    seen = []
    it = iter(responses)

    def handler(request):
        seen.append({"url": str(request.url), "headers": dict(request.headers), "body": json.loads(request.content)})
        r = next(it)
        return r(request) if callable(r) else r

    return httpx.MockTransport(handler), seen


def cfg(transport, **kw):
    return LLMConfig(base_url="http://llm.test/v1", model="test-model", transport=transport, **kw)


def output(**kw):
    base = dict(intent="scenarios", explanation="x", assumptions=[], horizon_months=None, scenarios=[], unsupported_reason=None)
    base.update(kw)
    return json.dumps(base)


def scenario(name="S", adjustments=(), **overrides):
    return dict(name=name, overrides={**EMPTY_OV, **overrides}, adjustments=list(adjustments))


def adj(line="opex", kind="add", value=1000, start="2026-03", end=None, label="a", life=None):
    return dict(line=line, kind=kind, value=value, start=start, end=end, label=label, useful_life_months=life)


# ---------- the request we send ----------

def test_request_is_openai_compatible_with_json_schema():
    t, seen = scripted(reply(json.dumps(CASES["price-rise"]["ideal"])))
    interpret(CASES["price-rise"]["prompt"], MONTHLY, cfg(t, api_key="secret"))
    req = seen[0]
    assert req["url"] == "http://llm.test/v1/chat/completions"
    assert req["headers"]["authorization"] == "Bearer secret"
    body = req["body"]
    assert body["model"] == "test-model" and body["temperature"] == 0
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][1]["content"] == CASES["price-rise"]["prompt"]
    rf = body["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] == LLMOutput.model_json_schema()
    assert "guided_json" not in body


@pytest.mark.parametrize("mode", ["guided_json", "none"])
def test_other_structured_output_modes(mode):
    t, seen = scripted(reply("```json\n" + json.dumps(CASES["margin"]["ideal"]) + "\n```"))
    interp, _, _ = interpret(CASES["margin"]["prompt"], MONTHLY, cfg(t, guided=mode))
    assert interp.scenarios[0].overrides.gross_margin == 0.75  # fenced reply still parsed
    body = seen[0]["body"]
    assert "response_format" not in body
    assert ("guided_json" in body) == (mode == "guided_json")


def test_generation_schema_requires_every_field():
    """Strict JSON-schema decoding needs all properties required and no extras."""
    schema = LLMOutput.model_json_schema()

    def walk(s):
        if s.get("type") == "object" and "properties" in s:
            assert set(s["required"]) == set(s["properties"]), s.get("title")
            assert s.get("additionalProperties") is False, s.get("title")
        for v in s.get("$defs", {}).values():
            walk(v)

    walk(schema)


def test_context_tells_the_model_dates_and_current_drivers():
    ctx = build_context(MONTHLY)
    text = render_context(ctx)
    assert "First forecast month: 2026-01 (January 2026)" in text
    assert "revenue_growth = 0.0328 per month" in text
    system = messages(ctx, "hi")[0]["content"]
    assert "MONTHLY" in system and "COMPANY CONTEXT" in system


# ---------- parsing and checks ----------

def test_retry_after_unusable_reply_feeds_back_the_error():
    good = json.dumps(CASES["growth-monthly"]["ideal"])
    t, seen = scripted(reply("not json at all"), reply(good))
    interp, _, call = interpret("anything", MONTHLY, cfg(t))
    assert call.attempts == 2 and interp.scenarios[0].overrides.revenue_growth == 0.02
    retry_msgs = seen[1]["body"]["messages"]
    assert retry_msgs[-2] == {"role": "assistant", "content": "not json at all"}
    assert "can't be used" in retry_msgs[-1]["content"]


def test_gives_up_after_two_bad_replies():
    t, _ = scripted(reply("{}"), reply("{}"))
    with pytest.raises(LLMBadOutput, match="after 2 attempts"):
        interpret("anything", MONTHLY, cfg(t))


@pytest.mark.parametrize("bad, msg", [
    (output(scenarios=[]), "no scenarios"),
    (output(intent="baseline", scenarios=[scenario()]), "use intent 'scenarios'"),
    (output(intent="unsupported"), "unsupported_reason is empty"),
    (output(scenarios=[scenario(adjustments=[adj(start="2026-05", end="2026-03")])]), "before it starts"),
    (output(scenarios=[scenario(gross_margin=1.5)]), "gross_margin"),
    (output(scenarios=[scenario(adjustments=[adj(start="2026-13")])]), "start"),
])
def test_unusable_replies_are_retried_with_the_reason(bad, msg):
    good = json.dumps(CASES["margin"]["ideal"])
    t, seen = scripted(reply(bad), reply(good))
    interpret("anything", MONTHLY, cfg(t))
    assert msg in seen[1]["body"]["messages"][-1]["content"]


def test_warnings_for_suspicious_values():
    out = output(scenarios=[scenario(
        "Odd",
        adjustments=[adj(value=900_000, label="Huge"), adj(start="2025-10", label="Early"),
                     adj(line="revenue", kind="pct", value=1.5, label="Double")],
        opex_growth=0.2,
    )])
    t, _ = scripted(reply(out))
    interp, warnings, _ = interpret("anything", MONTHLY, cfg(t))
    text = " | ".join(warnings)
    assert "unusually high" in text and "Huge" in text and "Double" in text
    assert "Early' starts 2025-10, before the forecast; it applies from 2026-01" in text
    early = next(a for a in interp.scenarios[0].adjustments if a.label == "Early")
    assert str(early.start) == "2026-01-01"


def test_duplicate_scenario_names_are_made_unique():
    t, _ = scripted(reply(output(scenarios=[scenario("Plan"), scenario("plan")])))
    interp, _, _ = interpret("anything", MONTHLY, cfg(t))
    assert [s.name for s in interp.scenarios] == ["Plan", "plan (2)"]


def test_request_horizon_beats_the_models():
    t, _ = scripted(reply(json.dumps(CASES["baseline-24"]["ideal"])), reply(json.dumps(CASES["baseline-24"]["ideal"])))
    assert interpret("x", MONTHLY, cfg(t))[0].horizon_months == 24
    assert interpret("x", MONTHLY, cfg(t), horizon_override=6)[0].horizon_months == 6


# ---------- endpoint failures ----------

def test_not_configured():
    with pytest.raises(LLMNotConfigured):
        chat_json(LLMConfig(), [], {}, "x")


@pytest.mark.parametrize("response, error, msg", [
    (httpx.Response(503, text="loading"), LLMUnavailable, "returned 503; still unavailable after 30s"),
    (httpx.Response(429, text="slow down"), LLMUnavailable, "returned 429; still unavailable after 30s"),
    (httpx.Response(401, text="bad key"), LLMUnavailable, "rejected the request (401)"),
    (httpx.Response(200, text="<html>"), LLMBadOutput, "unexpected format"),
    (reply("{", finish="length"), LLMBadOutput, "cut off"),
])
def test_endpoint_errors_have_clear_messages(response, error, msg):
    t = httpx.MockTransport(lambda request: response)  # same answer every time
    with pytest.raises(error, match=re.escape(msg)):
        interpret("anything", MONTHLY, cfg(t, timeout_seconds=30))


def test_cold_start_is_waited_out(fake_clock):
    good = reply(json.dumps(CASES["margin"]["ideal"]))
    t, seen = scripted(httpx.Response(503), httpx.Response(503), httpx.Response(502), good)
    interp, _, call = interpret("anything", MONTHLY, cfg(t, timeout_seconds=300))
    assert interp.scenarios[0].overrides.gross_margin == 0.75
    assert len(seen) == 4 and fake_clock["slept"] == [2.0, 4.0, 8.0]  # backoff
    assert call.latency_seconds == pytest.approx(14.0)


def test_backoff_stops_at_the_deadline(fake_clock):
    t = httpx.MockTransport(lambda request: httpx.Response(503))
    with pytest.raises(LLMUnavailable, match="still unavailable after 60s"):
        interpret("anything", MONTHLY, cfg(t, timeout_seconds=60))
    assert sum(fake_clock["slept"]) == pytest.approx(60)
    assert max(fake_clock["slept"]) == 15.0  # capped


def test_timeout_and_unreachable():
    def boom(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LLMUnavailable, match="timed out after 5s"):
        interpret("x", MONTHLY, cfg(httpx.MockTransport(boom), timeout_seconds=5))

    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LLMUnavailable, match=r"unreachable \(ConnectError\); still unavailable after 20s"):
        interpret("x", MONTHLY, cfg(httpx.MockTransport(refused), timeout_seconds=20))


# ---------- eval harness ----------

def test_ideal_replies_pass_every_eval_case():
    c = cfg(fake_transport())
    failed = {cid: r["failures"] for cid, case in CASES.items()
              if not (r := run_case(case, MONTHLY_B, c, BALANCES))["passed"]}
    assert failed == {}


def test_eval_catches_wrong_answers():
    case = CASES["growth-annual"]
    wrong = dict(case["ideal"])
    wrong["scenarios"] = [scenario("x", revenue_growth=0.24)]  # annual rate typed as monthly
    r = run_case(case, MONTHLY, cfg(fake_transport({case["prompt"]: json.dumps(wrong)})))
    assert not r["passed"] and "revenue_growth=0.2400" in r["failures"][0]

    case = CASES["quarter-cut"]
    wrong = dict(case["ideal"])
    wrong["scenarios"] = [scenario("x", adjustments=[adj(value=-20000, start="2026-07", end=None)])]  # never ends
    r = run_case(case, MONTHLY, cfg(fake_transport({case["prompt"]: json.dumps(wrong)})))
    assert not r["passed"] and "no adjustment opex add -20000 from 2026-07 to 2026-09" in r["failures"][0]


def test_eval_flags_unexpected_extras():
    interp = Interpretation(intent="scenarios", explanation="", assumptions=[], horizon_months=12, scenarios=[
        {"name": "x", "overrides": {"revenue_growth": 0.02, "opex_growth": 0.0}}])
    fails = check(CASES["growth-monthly"], interp, [])
    assert fails == ["scenario 1: unexpected override opex_growth=0.0"]


# ---------- summary ----------

def test_money_formatting():
    assert fmt(1_057_306) == "1.06M" and fmt(-300_000, signed=True) == "-300k"
    assert fmt(94_320, signed=True) == "+94k" and fmt(0, signed=True) == "0"


def test_useful_life_kept_for_capex_only():
    out = output(scenarios=[scenario(adjustments=[
        adj(line="capex", value=90000, start="2026-03", end="2026-03", label="Machine", life=36),
        adj(line="opex", value=1000, label="Rent", life=24),  # nonsense on opex: dropped
    ])])
    t, _ = scripted(reply(out))
    interp, _, _ = interpret("x", MONTHLY_B, cfg(t), balances=BALANCES)
    lives = {a.label: a.useful_life_months for a in interp.scenarios[0].adjustments}
    assert lives == {"Machine": 36, "Rent": None}


def test_percentage_typed_interest_rate_is_flagged():
    t, _ = scripted(reply(output(scenarios=[scenario("Loan", interest_rate=0.9)])))
    _, warnings, _ = interpret("x", MONTHLY_B, cfg(t), balances=BALANCES)
    assert any("interest_rate of 90%" in w for w in warnings)


def test_context_shows_tax_and_interest_rates():
    text = render_context(build_context(MONTHLY_B, BALANCES))
    assert "tax_rate       = 0.0000 on profit" in text and "interest_rate = 0.0000 per year on debt" in text
