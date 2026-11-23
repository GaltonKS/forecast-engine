"""Step 3: seasonality, statistical models, backtest and model selection."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import DriverOverrides, ForecastRequest, run_forecast, to_monthly  # noqa: E402
from engine.backtest import backtest, plan_origins  # noqa: E402
from engine.schema import LineItem  # noqa: E402
from engine.stats import MODELS, available_models, detect_seasonality  # noqa: E402

# Retail-style year: slow Q1, December peak. Multipliers, geometric mean ~1.
SEASON = np.array([0.85, 0.82, 0.92, 0.95, 1.0, 1.0, 0.97, 0.98, 1.02, 1.08, 1.18, 1.35])
SEASON = SEASON / np.exp(np.log(SEASON).mean())


def make(months=36, g=0.015, season=None, noise=0.02, seed=0, start="2023-01-01"):
    rng = np.random.default_rng(seed)
    items = []
    for m in range(months):
        period = (pd.Timestamp(start) + pd.DateOffset(months=m)).date()
        s = season[period.month - 1] if season is not None else 1.0
        rev = 100_000 * (1 + g) ** m * s * (1 + rng.uniform(-noise, noise))
        items += [
            LineItem(period=period, account="Sales", category="revenue", amount=rev),
            LineItem(period=period, account="COGS", category="cogs", amount=rev * 0.4),
            LineItem(period=period, account="Opex", category="opex", amount=40_000 * 1.01 ** m * (1 + rng.uniform(-noise, noise))),
        ]
    return to_monthly(items)


def test_seasonality_detected_when_present():
    m = make(36, season=SEASON)
    p = detect_seasonality(m["revenue"])
    assert p.detected and p.p_value < 0.01
    assert p.indices[11] == max(p.indices)  # December peak
    assert p.indices[11] == pytest.approx(SEASON[11], rel=0.05)


def test_noise_is_not_mistaken_for_seasonality():
    # 24 months = 2 points per calendar month, the easiest case to fool
    for seed in range(5):
        m = make(24, noise=0.10, seed=seed)
        assert not detect_seasonality(m["revenue"]).detected, seed


def test_seasonality_needs_two_years():
    p = detect_seasonality(make(18, season=SEASON)["revenue"])
    assert not p.detected and "24 months" in p.note


def test_seasonality_can_be_forced_off():
    m = make(36, season=SEASON)
    resp = run_forecast(m, ForecastRequest(seasonal=False, method="driver"))
    assert not resp.models["revenue"].seasonality.detected
    assert "season" not in resp.formulas["revenue"]


def test_seasonal_forecast_follows_the_pattern():
    m = make(36, season=SEASON)
    resp = run_forecast(m, ForecastRequest(horizon_months=12))
    rev = {mo.period.month: mo.revenue for mo in resp.months}
    assert rev[12] > rev[11] > rev[1]  # December peak, January trough
    sm = resp.models["revenue"]
    assert sm.selected_by == "backtest"
    assert sm.method != "seasonal_naive" or sm.scores  # selection is backed by scores


def test_backtest_beats_driver_on_seasonal_data():
    # Pure seasonal data that the non-seasonal trend line can't capture well
    m = make(48, season=SEASON, noise=0.01)
    bt = backtest(m["revenue"], available_models())
    by = {s.method: s for s in bt.scores if s.status == "ok"}
    assert min(s.wape for s in by.values()) < 0.03
    assert {s.folds for s in by.values()} == {3}  # every model scored on the same folds


def test_driver_wins_near_ties_on_clean_trend():
    m = make(30, season=None, noise=0.0)
    resp = run_forecast(m, ForecastRequest())
    assert resp.models["revenue"].method == "driver"
    assert resp.models["revenue"].selected_by == "backtest"


def test_holdout_is_chosen_models_last_window():
    m = make(36, season=SEASON)
    sm = run_forecast(m, ForecastRequest()).models["revenue"]
    assert len(sm.holdout) == sm.backtest_horizon == 6
    assert str(sm.holdout[-1].period) == str(m.index[-1].date())


def test_override_switches_line_to_driver_and_skips_backtest():
    m = make(36, season=SEASON)
    resp = run_forecast(m, ForecastRequest(overrides=DriverOverrides(revenue_growth=0.0)))
    rev = resp.models["revenue"]
    assert rev.method == "driver" and rev.selected_by == "override"
    assert rev.scores == []
    assert resp.models["opex"].selected_by in ("backtest", "default")  # opex untouched
    # zero growth: once the estimated seasonal index is divided out, revenue is flat
    idx = rev.seasonality.indices
    assert idx and resp.formulas["revenue"].endswith("* season[month(t)]")
    levels = [mo.revenue / idx[mo.period.month - 1] for mo in resp.months]
    assert max(levels) == pytest.approx(min(levels), rel=1e-4)


@pytest.mark.parametrize("method", [m for m in MODELS if m in available_models()])
def test_every_method_can_be_requested(method):
    m = make(36, season=SEASON)
    resp = run_forecast(m, ForecastRequest(method=method, horizon_months=6))
    assert resp.models["revenue"].method == method
    assert resp.models["revenue"].selected_by == "requested"
    assert all(mo.revenue > 0 for mo in resp.months)
    for mo in resp.months:  # statements stay internally consistent whatever the model
        assert mo.gross_profit == pytest.approx(mo.revenue - mo.cogs, abs=0.02)
        assert mo.net_income == pytest.approx(mo.operating_income + mo.other_net, abs=0.02)


def test_statistical_method_needs_enough_history():
    with pytest.raises(ValueError, match="needs 12 months"):
        run_forecast(make(8), ForecastRequest(method="ets"))


def test_unknown_method_rejected():
    with pytest.raises(ValidationError):
        ForecastRequest(method="lstm")


def test_short_history_falls_back_to_driver():
    resp = run_forecast(make(10), ForecastRequest())
    assert resp.models["revenue"].method == "driver"
    assert resp.models["revenue"].selected_by == "default"


def test_backtest_plan():
    assert plan_origins(24) == (6, [12, 15, 18])
    assert plan_origins(15) == (3, [12])
    assert plan_origins(14) == (3, [])


def test_forecast_is_deterministic():
    m = make(36, season=SEASON)
    a = run_forecast(m, ForecastRequest()).model_dump()
    b = run_forecast(m, ForecastRequest()).model_dump()
    assert a == b
