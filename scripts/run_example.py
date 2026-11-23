"""Run a baseline and a scenario forecast on the sample data and print them."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import DriverOverrides, ForecastRequest, load_csv, run_forecast, to_monthly  # noqa: E402


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
    # "what if growth slows to 2% a month and we cut opex growth to zero?"
    scenario = ForecastRequest(
        horizon_months=12,
        overrides=DriverOverrides(revenue_growth=0.02, opex_growth=0.0),
    )
    resp = run_forecast(monthly, scenario)
    show("Scenario: 2% growth, flat opex", resp)

    out = ROOT / "sample_data" / "example_response.json"
    out.write_text(resp.model_dump_json(indent=2))
    print(f"\nFull JSON response written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
