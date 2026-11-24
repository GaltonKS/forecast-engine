"""Plain-language summary of a comparison, written by code from the engine's
numbers. The model never writes a number the user reads."""
from __future__ import annotations

from engine.schema import Band, ScenarioComparison


def fmt(x: float, signed: bool = False) -> str:
    sign = ("+" if x > 0 else "-" if x < 0 else "") if signed else ("-" if x < 0 else "")
    a = abs(x)
    if a >= 1e6:
        s = f"{a / 1e6:.2f}M"
    elif a >= 1e3:
        s = f"{a / 1e3:.0f}k"
    else:
        s = f"{a:.0f}"
    return sign + s


def _band(b: Band, signed: bool = False) -> str:
    # A fixed change (e.g. a salary) has no spread of its own; don't print "x to x".
    if fmt(b.p10, signed) == fmt(b.p90, signed):
        return fmt(b.p50, signed)
    return f"{fmt(b.p50, signed)} (likely range {fmt(b.p10, signed)} to {fmt(b.p90, signed)})"


def _rev(delta: float) -> str:
    return "revenue unchanged" if fmt(delta) == "0" else f"revenue {fmt(delta, signed=True)}"


def summarize(c: ScenarioComparison) -> list[str]:
    base = c.baseline
    h = len(base.months)
    lines = []
    u = base.uncertainty
    if u:
        lines.append(
            f"Baseline, next {h} months: revenue {_band(u.totals['revenue'])}, "
            f"net income {_band(u.totals['net_income'])}."
        )
        if u.prob_loss_month >= 0.05:
            lines.append(f"Baseline: {u.prob_loss_month:.0%} chance of at least one loss-making month.")
    else:
        rev = sum(m.revenue for m in base.months)
        ni = sum(m.net_income for m in base.months)
        lines.append(f"Baseline, next {h} months: revenue {fmt(rev)}, net income {fmt(ni)}.")

    lines += _cash_lines("Baseline", base)

    for sc in c.scenarios:
        if sc.delta:
            lines.append(
                f"{sc.name}: net income {_band(sc.delta.net_income, signed=True)} vs baseline; "
                f"{_rev(sc.delta.revenue.p50)}."
            )
        else:
            lines.append(
                f"{sc.name}: net income {fmt(sc.delta_point['net_income'], signed=True)} vs baseline; "
                f"{_rev(sc.delta_point['revenue'])}."
            )
        if sc.delta and sc.delta.ending_cash:
            lines.append(f"{sc.name}: ending cash {_band(sc.delta.ending_cash, signed=True)} vs baseline.")
        elif "ending_cash" in sc.delta_point:
            lines.append(f"{sc.name}: ending cash {fmt(sc.delta_point['ending_cash'], signed=True)} vs baseline.")
        lines += [line for line in _cash_lines(sc.name, sc.forecast) if "runs out" in line or "chance" in line]
        su = sc.forecast.uncertainty
        if su and su.prob_loss_month >= 0.05 and (not u or su.prob_loss_month > u.prob_loss_month + 0.05):
            lines.append(f"{sc.name}: {su.prob_loss_month:.0%} chance of at least one loss-making month.")
    return lines


def _cash_lines(name: str, f) -> list[str]:
    if not f.cash:
        return []
    c, u = f.cash, f.uncertainty
    out = []
    if c.cash_out_period:
        out.append(f"{name}: cash runs out in {c.cash_out_period:%Y-%m} ({c.runway_months} months).")
    else:
        out.append(
            f"{name}: cash goes from {fmt(c.opening_cash)} to {fmt(c.ending_cash)}; "
            f"lowest {fmt(c.lowest_cash)} in {c.lowest_cash_period:%Y-%m}."
        )
    if u and u.prob_cash_out is not None and u.prob_cash_out >= 0.05:
        r = u.runway
        span = ""
        if r and r.p10 is not None:
            p90 = f"{r.p90} months" if r.p90 is not None else "beyond the forecast"
            span = f" (runway {r.p10} months in a bad case, {p90} in a good one)"
        out.append(f"{name}: {u.prob_cash_out:.0%} chance cash runs out within {len(f.months)} months{span}.")
    return out
