"""Step 4: Monte Carlo ranges, dated adjustments and scenario comparison."""
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import DriverOverrides, ForecastRequest, run_forecast  # noqa: E402
from engine.forecast import run_scenarios  # noqa: E402
from engine.schema import Adjustment, LineNoise, ScenarioRequest  # noqa: E402
from engine.simulate import Z80, estimate_noise, horizon_sd  # noqa: E402
from tests.test_models import SEASON, make  # noqa: E402

HIRE = {"line": "opex", "kind": "add", "value": 30_000, "start": "2026-03-01", "label": "Hire 2"}


def test_p50_tracks_point_forecast_and_bands_are_ordered():
    resp = run_forecast(make(36, season=SEASON), ForecastRequest(horizon_months=12))
    u = resp.uncertainty
    assert u.simulations == 1000 and len(u.months) == 12
    for point, b in zip(resp.months, u.months):
        for key in ("revenue", "opex", "net_income"):
            band = getattr(b, key)
            assert band.p10 <= band.p50 <= band.p90
        assert b.revenue.p50 == pytest.approx(point.revenue, rel=0.02)


def test_ranges_widen_with_horizon():
    resp = run_forecast(make(36, noise=0.05), ForecastRequest(horizon_months=24, method="driver"))
    width = [(b.revenue.p90 - b.revenue.p10) / b.revenue.p50 for b in resp.uncertainty.months]
    assert width[-1] > width[0]


def test_same_seed_same_ranges_and_different_seed_differs():
    m = make(30)
    a = run_forecast(m, ForecastRequest(seed=1)).uncertainty
    b = run_forecast(m, ForecastRequest(seed=1)).uncertainty
    c = run_forecast(m, ForecastRequest(seed=2)).uncertainty
    assert a == b
    assert a.totals["net_income"] != c.totals["net_income"]


def test_zero_simulations_skips_ranges():
    assert run_forecast(make(24), ForecastRequest(simulations=0)).uncertainty is None


def test_noise_estimate_recovers_known_sigma():
    rng = np.random.default_rng(3)
    import pandas as pd

    idx = pd.date_range("2020-01-01", periods=12, freq="MS")
    sigmas = []
    for _ in range(200):
        y = pd.Series(1000 * 1.02 ** np.arange(12) * np.exp(rng.normal(0, 0.05, 12)), index=idx)
        sigmas.append(estimate_noise(y, None).monthly_sd)
    assert np.median(sigmas) == pytest.approx(0.05, rel=0.15)


def test_fixed_assumptions_remove_their_uncertainty():
    n = LineNoise(scale="log", monthly_sd=0.02, autocorrelation=0, level_sd=0.01, growth_sd=0.002)
    free = horizon_sd(n, 12)
    fixed = horizon_sd(n.model_copy(update={"level_fixed": True, "growth_fixed": True}), 12)
    assert np.allclose(fixed, 0.02) and free[-1] > fixed[-1]


def test_calibration_reports_backtest_coverage():
    resp = run_forecast(make(36, season=SEASON, noise=0.04), ForecastRequest())
    cal = resp.uncertainty.calibration
    assert set(cal) == {"revenue", "opex"}
    for c in cal.values():
        assert c.points == 18 and 0 <= c.coverage <= 1 and c.widened_by >= 1


def test_add_adjustment_hits_only_its_months():
    m = make(24)  # history ends 2024-12; forecast 2025-01..
    adj = Adjustment(line="opex", kind="add", value=10_000, start="2025-03-15", end="2025-05-01")
    base = run_forecast(m, ForecastRequest(horizon_months=8, simulations=0))
    resp = run_forecast(m, ForecastRequest(horizon_months=8, simulations=0, adjustments=[adj]))
    diffs = [round(a.opex - b.opex, 2) for a, b in zip(resp.months, base.months)]
    assert diffs == [0, 0, 10_000, 10_000, 10_000, 0, 0, 0]
    assert resp.months[2].net_income == pytest.approx(base.months[2].net_income - 10_000, abs=0.02)


def test_pct_revenue_adjustment_flows_through_cogs():
    m = make(24)
    adj = Adjustment(line="revenue", kind="pct", value=0.10, start="2025-01-01")
    base = run_forecast(m, ForecastRequest(horizon_months=3, simulations=0))
    resp = run_forecast(m, ForecastRequest(horizon_months=3, simulations=0, adjustments=[adj]))
    for a, b in zip(resp.months, base.months):
        assert a.revenue == pytest.approx(b.revenue * 1.1, rel=1e-6)
        assert a.cogs == pytest.approx(b.cogs * 1.1, rel=1e-6)  # margin unchanged


def test_adjustment_outside_horizon_warns():
    adj = Adjustment(line="opex", kind="add", value=1, start="2030-01-01", label="Far away")
    resp = run_forecast(make(24), ForecastRequest(horizon_months=6, adjustments=[adj]))
    assert any("Far away" in w for w in resp.warnings)


def test_pct_adjustment_cannot_go_below_minus_100():
    with pytest.raises(ValidationError):
        Adjustment(line="revenue", kind="pct", value=-1.5, start="2025-01-01")


def test_scenarios_share_models_and_random_draws():
    m = make(36, season=SEASON)
    req = ScenarioRequest(
        horizon_months=12,
        scenarios=[
            {"name": "Hire 2", "adjustments": [HIRE | {"start": "2026-03-01"}]},
            {"name": "Flat revenue", "overrides": {"revenue_growth": 0.0}},
        ],
    )
    c = run_scenarios(m, req)
    hire, flat = c.scenarios
    # a fixed cost has no uncertainty of its own: paired paths cancel exactly
    assert hire.delta_point["net_income"] == pytest.approx(-300_000, abs=1)
    assert hire.delta.net_income.p10 == pytest.approx(-300_000, abs=1)
    assert hire.delta.net_income.p90 == pytest.approx(-300_000, abs=1)
    # the baseline is untouched and the hire scenario keeps the baseline's models
    assert hire.forecast.models["revenue"].method == c.baseline.models["revenue"].method
    assert flat.forecast.models["revenue"].selected_by == "override"
    assert flat.delta.net_income.p50 < 0


def test_scenario_matches_single_forecast_with_same_inputs():
    m = make(30)
    sc = {"name": "Slow", "overrides": {"revenue_growth": 0.005}}
    c = run_scenarios(m, ScenarioRequest(scenarios=[sc]))
    single = run_forecast(m, ForecastRequest(overrides=DriverOverrides(revenue_growth=0.005)))
    assert c.scenarios[0].forecast.months == single.months
    assert c.scenarios[0].forecast.uncertainty.totals == single.uncertainty.totals


def test_scenario_request_limits():
    with pytest.raises(ValidationError):
        ScenarioRequest(scenarios=[])
    with pytest.raises(ValidationError):
        ScenarioRequest(scenarios=[{"name": f"s{i}"} for i in range(11)])


def test_z80_is_the_10_90_quantile():
    from scipy.stats import norm

    assert Z80 == pytest.approx(norm.ppf(0.9))
