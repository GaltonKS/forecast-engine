"""Prompt -> validated scenarios. The model proposes; this module checks.

Checks fall in two groups:
- errors (the reply can't be used): bad JSON, wrong shape, impossible values
  such as end before start. The model gets one retry with the error message.
- warnings (usable but worth a look): growth that looks like an annual rate
  typed as monthly, a change far larger than the line it changes, a start
  month before the forecast. These are returned so the user can see them.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
from pydantic import ValidationError

from engine.schema import Adjustment, DriverOverrides, Scenario

from .client import LLMBadOutput, LLMConfig, chat_json, dumps
from .prompt import DEFAULT_HORIZON, Context, build_context, messages
from .schema import Interpretation, LLMCall, LLMOutput

MAX_ATTEMPTS = 2
HIGH_MONTHLY_GROWTH = 0.10  # 10%/month is 214%/year: likely an annual rate typed as monthly
LARGE_CHANGE_RATIO = 3.0  # an "add" bigger than 3x the line's recent monthly average
LARGE_PCT = 1.0  # +100% or more


def _month(s: str) -> date:
    y, m = s.split("-")
    return date(int(y), int(m), 1)


def _short(e: Exception) -> str:
    if isinstance(e, ValidationError):
        return "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()[:5]
        )
    return str(e)


def convert(out: LLMOutput, ctx: Context, horizon_override: int | None) -> tuple[Interpretation, list[str]]:
    """Turn the model's reply into engine scenarios. Raises ValueError /
    ValidationError for replies that can't be used."""
    warnings: list[str] = []
    if out.intent == "scenarios" and not out.scenarios:
        raise ValueError("intent is 'scenarios' but no scenarios were given")
    if out.intent != "scenarios" and out.scenarios:
        raise ValueError(f"intent is '{out.intent}' but scenarios were given; use intent 'scenarios'")
    if out.intent == "unsupported" and not out.unsupported_reason:
        raise ValueError("intent is 'unsupported' but unsupported_reason is empty")

    scenarios, seen = [], set()
    for sc in out.scenarios:
        name = sc.name.strip() or "Scenario"
        base, i = name, 2
        while name.lower() in seen:  # names identify scenarios in the result
            name, i = f"{base} ({i})", i + 1
        seen.add(name.lower())

        overrides = DriverOverrides(**sc.overrides.model_dump())  # validates margin range etc.
        for key in ("revenue_growth", "opex_growth"):
            g = getattr(overrides, key)
            if g is not None and abs(g) >= HIGH_MONTHLY_GROWTH:
                warnings.append(
                    f"{name}: {key} of {g:.1%} per month ({(1 + g) ** 12 - 1:.0%} per year) is unusually "
                    "high. Check it wasn't meant as an annual rate."
                )
        for key in ("dso_days", "dio_days", "dpo_days"):
            v = getattr(overrides, key)
            if v is not None and v > 180:
                warnings.append(f"{name}: {key} of {v:.0f} days is unusually long. Check the number.")
        if overrides.gross_margin is not None and overrides.gross_margin < 0:
            warnings.append(f"{name}: gross_margin of {overrides.gross_margin:.0%} means selling below cost.")

        adjustments = []
        for a in sc.adjustments:
            start = _month(a.start)
            end = _month(a.end) if a.end else None
            if end and end < start:
                raise ValueError(f"adjustment '{a.label}' ends ({a.end}) before it starts ({a.start})")
            if start < ctx.forecast_start:
                if end and end < ctx.forecast_start:
                    warnings.append(f"{name}: '{a.label}' is entirely before the forecast starts and has no effect.")
                else:
                    warnings.append(
                        f"{name}: '{a.label}' starts {a.start}, before the forecast; it applies from "
                        f"{ctx.forecast_start:%Y-%m}."
                    )
                start = max(start, ctx.forecast_start)
            if a.kind == "pct" and a.value >= LARGE_PCT:
                warnings.append(f"{name}: '{a.label}' changes {a.line} by {a.value:+.0%}. Check the size.")
            recent = abs(ctx.recent.get(a.line, 0.0))
            if a.kind == "add" and recent > 0 and abs(a.value) > LARGE_CHANGE_RATIO * recent:
                warnings.append(
                    f"{name}: '{a.label}' adds {a.value:,.0f} per month to {a.line}, over "
                    f"{LARGE_CHANGE_RATIO:g}x its recent monthly average ({recent:,.0f}). Check the size."
                )
            if a.line in ("capex", "debt", "equity") and not ctx.opening:
                warnings.append(f"{name}: '{a.label}' is a cash item, but there's no balance sheet; it has no effect.")
            adjustments.append(
                Adjustment(line=a.line, kind=a.kind, value=a.value, start=start, end=end, label=a.label)
            )
        scenarios.append(Scenario(name=name, overrides=overrides, adjustments=adjustments))

    interp = Interpretation(
        intent=out.intent,
        explanation=out.explanation,
        assumptions=out.assumptions,
        horizon_months=horizon_override or out.horizon_months or DEFAULT_HORIZON,
        scenarios=scenarios,
        unsupported_reason=out.unsupported_reason if out.intent == "unsupported" else None,
    )
    return interp, warnings


def interpret(
    prompt: str,
    monthly: pd.DataFrame,
    cfg: LLMConfig,
    horizon_override: int | None = None,
    balances: pd.DataFrame | None = None,
) -> tuple[Interpretation, list[str], LLMCall]:
    ctx = build_context(monthly, balances)
    msgs = messages(ctx, prompt)
    schema = LLMOutput.model_json_schema()
    latency, ptok, ctok, last_error = 0.0, 0, 0, None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        res = chat_json(cfg, msgs, schema, "scenario_interpretation")
        latency += res.latency_seconds
        ptok += res.prompt_tokens or 0
        ctok += res.completion_tokens or 0
        try:
            out = LLMOutput.model_validate_json(res.content)
            interp, warnings = convert(out, ctx, horizon_override)
        except (ValidationError, ValueError) as e:
            last_error = _short(e)
            msgs = msgs + [
                {"role": "assistant", "content": res.content},
                {"role": "user", "content": f"That reply can't be used: {last_error}. "
                 "Reply again with the corrected JSON object only."},
            ]
            continue
        call = LLMCall(
            model=cfg.model, attempts=attempt, latency_seconds=round(latency, 3),
            prompt_tokens=ptok or None, completion_tokens=ctok or None,
        )
        return interp, warnings, call
    raise LLMBadOutput(f"The model's reply couldn't be used after {MAX_ATTEMPTS} attempts: {last_error}")


__all__ = ["interpret", "convert", "dumps"]
