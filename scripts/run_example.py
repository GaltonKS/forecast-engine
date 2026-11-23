"""Run a baseline, a scenario and a scenario comparison on the sample data."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import DriverOverrides, ForecastRequest, load_csv, run_forecast, to_monthly  # noqa: E402
from engine.forecast import run_scenarios  # noqa: E402
from engine.schema import Adjustment, ScenarioRequest  # noqa: E402


def show(title, resp):
    print(f"\n=== {title} ===")
    for name, d in resp.drivers.items():
        flag = " (override)" if d.overridden else ""
        print(f"  {name:15s} {d.value:>12,.4f}  {d.method}{flag}")
    for line, sm in resp.models.items():
        print(f"  {line}: {sm.method} ({sm.selected_by}) - {sm.reason}")
        for sc in sm.scores:
            err = f"WAPE {sc.wape:6.1%}  bias {sc.bias:+6.1%}" if sc.status == "ok" else sc.status
            print(f"      {sc.method:15s} {err}")
    if resp.uncertainty:
        u = resp.uncertainty
        t = u.totals["net_income"]
        print(f"  12-month net income  P10 {t.p10:,.0f}   P50 {t.p50:,.0f}   P90 {t.p90:,.0f}"
              f"   (chance of a loss month: {u.prob_loss_month:.0%})")
        for line, c in u.calibration.items():
            print(f"  {line} ranges caught {c.coverage:.0%} of backtest months (target 80%), widened x{c.widened_by}")
    for w in resp.warnings:
        print(f"  WARNING: {w}")
    print(f"\n  {'month':10s} {'revenue':>12s} {'gross profit':>13s} {'opex':>12s} {'net income':>12s}")
    for m in resp.months:
        print(
            f"  {m.period.isoformat()[:7]:10s} {m.revenue:>12,.0f} {m.gross_profit:>13,.0f}"
            f" {m.opex:>12,.0f} {m.net_income:>12,.0f}"
        )


def main():
    monthly = to_monthly(load_csv(ROOT / "sample_data" / "sample_financials.csv"))
    show("Baseline (12 months)", run_forecast(monthly, ForecastRequest(horizon_months=12)))

    # This is the shape the LLM layer will produce from a prompt like
    # "what if growth slows to 2% a month, opex stops growing, and we hire
    # two engineers in March?"
    hire = Adjustment(line="opex", kind="add", value=30_000, start="2026-03-01", label="Hire 2 engineers")
    scenario = ForecastRequest(
        horizon_months=12,
        overrides=DriverOverrides(revenue_growth=0.02, opex_growth=0.0),
        adjustments=[hire],
    )
    resp = run_forecast(monthly, scenario)
    show("Scenario: 2% growth, flat opex, hire 2 in March", resp)

    out = ROOT / "sample_data" / "example_response.json"
    out.write_text(resp.model_dump_json(indent=2))
    print(f"\nFull JSON response written to {out.relative_to(ROOT)}")

    comparison = run_scenarios(monthly, ScenarioRequest(scenarios=[
        {"name": "Hire 2 in March", "adjustments": [hire]},
        {"name": "Price +5% from April", "adjustments": [
            Adjustment(line="revenue", kind="pct", value=0.05, start="2026-04-01")]},
        {"name": "Growth halves", "overrides": {"revenue_growth": 0.016}},
    ]))
    print("\n=== Scenario comparison: change in 12-month net income vs baseline ===")
    for sc in comparison.scenarios:
        d = sc.delta.net_income
        print(f"  {sc.name:22s} P50 {d.p50:>+12,.0f}   (P10 {d.p10:+,.0f} .. P90 {d.p90:+,.0f})")
    out = ROOT / "sample_data" / "example_scenarios.json"
    out.write_text(comparison.model_dump_json(indent=2))
    print(f"\nFull JSON written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
