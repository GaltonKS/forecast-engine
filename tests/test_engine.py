import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import DriverOverrides, ForecastRequest, run_forecast, to_monthly  # noqa: E402
from engine.ingest import IngestError, load_csv  # noqa: E402
from engine.schema import LineItem  # noqa: E402


def synthetic(months=24, rev0=100_000, g=0.03, gm=0.6, opex0=40_000, og=0.01, other=500):
    items = []
    for m in range(months):
        period = (pd.Timestamp("2024-01-01") + pd.DateOffset(months=m)).date()
        rev = rev0 * (1 + g) ** m
        items += [
            LineItem(period=period, account="Sales", category="revenue", amount=rev),
            LineItem(period=period, account="COGS", category="cogs", amount=rev * (1 - gm)),
            LineItem(period=period, account="Opex", category="opex", amount=opex0 * (1 + og) ** m),
            LineItem(period=period, account="Interest", category="other", amount=other),
        ]
    return to_monthly(items)


def test_fit_recovers_known_drivers():
    resp = run_forecast(synthetic(), ForecastRequest())
    d = resp.drivers
    assert d["revenue_growth"].value == pytest.approx(0.03, abs=1e-9)
    assert d["gross_margin"].value == pytest.approx(0.6, abs=1e-9)
    assert d["opex_growth"].value == pytest.approx(0.01, abs=1e-9)
    assert d["other_net"].value == pytest.approx(500)
    assert resp.warnings == []


def test_forecast_arithmetic_is_consistent():
    resp = run_forecast(synthetic(), ForecastRequest(horizon_months=6))
    assert len(resp.months) == 6
    for m in resp.months:
        assert m.gross_profit == pytest.approx(m.revenue - m.cogs, abs=0.02)
        assert m.operating_income == pytest.approx(m.gross_profit - m.opex, abs=0.02)
        assert m.net_income == pytest.approx(m.operating_income + m.other_net, abs=0.02)
    assert str(resp.months[0].period) == "2026-01-01"


def test_overrides_replace_fitted_values():
    req = ForecastRequest(overrides=DriverOverrides(revenue_growth=0.0, gross_margin=0.5))
    resp = run_forecast(synthetic(), req)
    assert resp.drivers["revenue_growth"].overridden
    assert not resp.drivers["opex_growth"].overridden
    revenues = {m.revenue for m in resp.months}
    assert len(revenues) == 1  # zero growth -> flat revenue
    assert resp.months[0].cogs == pytest.approx(resp.months[0].revenue * 0.5, abs=0.02)


def test_short_history_warns_and_too_short_fails():
    assert run_forecast(synthetic(months=6), ForecastRequest()).warnings
    with pytest.raises(ValueError):
        run_forecast(synthetic(months=2), ForecastRequest())


def test_gaps_are_reported():
    items = [
        LineItem(period=f"2024-{m:02d}-01", account="Sales", category="revenue", amount=100)
        for m in (1, 2, 4, 5, 6)
    ]
    monthly = to_monthly(items)
    assert monthly.attrs["gaps"] == ["2024-03-01"]


def test_bad_csv_rows_are_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("period,account,category,amount\n2024-01-01,Sales,revenue,100\n2024-02-01,Sales,sales,100\n")
    with pytest.raises(IngestError, match="row 3"):
        load_csv(p)


def test_missing_columns_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("date,account,amount\n2024-01-01,Sales,100\n")
    with pytest.raises(IngestError, match="Missing columns"):
        load_csv(p)
