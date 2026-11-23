"""Balance sheet, cash flow and runway."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import DriverOverrides, ForecastRequest, run_forecast, to_monthly  # noqa: E402
from engine import statements as st  # noqa: E402
from engine.forecast import run_scenarios  # noqa: E402
from engine.ingest import to_balances  # noqa: E402
from engine.schema import Adjustment, Drivers, LineItem, ScenarioRequest  # noqa: E402

DAYS = st.DAYS


def company(months=24, rev0=100_000, g=0.02, gm=0.6, opex0=50_000, og=0.01, dep=2_000, dso=45, dpo=30,
            capex=3_000, cash0=200_000, debt=100_000, fa0=150_000, inventory_days=0, balance_months=None):
    """Income statement + balances with known drivers, consistent month to month."""
    items, cash, fa = [], cash0, fa0
    ar_prev = ap_prev = inv_prev = None
    for m in range(months):
        p = (pd.Timestamp("2024-01-01") + pd.DateOffset(months=m)).date()
        rev = rev0 * (1 + g) ** m
        cogs = rev * (1 - gm)
        opex_cash = opex0 * (1 + og) ** m  # opex excluding depreciation
        items += [
            LineItem(period=p, account="Sales", category="revenue", amount=rev),
            LineItem(period=p, account="COGS", category="cogs", amount=cogs),
            LineItem(period=p, account="Opex", category="opex", amount=opex_cash),
            LineItem(period=p, account="Depreciation", category="depreciation", amount=dep),
        ]
        ni = rev - cogs - opex_cash - dep
        ar, ap = rev * dso / DAYS, (cogs + opex_cash) * dpo / DAYS
        inv = cogs * inventory_days / DAYS
        if m > 0:
            cash += ni + dep - (ar - ar_prev) - (inv - inv_prev) + (ap - ap_prev) - capex
            fa += capex - dep
        ar_prev, ap_prev, inv_prev = ar, ap, inv
        if balance_months is None or m >= months - balance_months:
            equity = cash + ar + inv + fa - ap - debt
            for cat, v in [("cash", cash), ("receivables", ar), ("inventory", inv), ("fixed_assets", fa),
                           ("payables", ap), ("debt", debt), ("equity", equity)]:
                items.append(LineItem(period=p, account=cat.title(), category=cat, amount=v))
    return to_monthly(items), to_balances(items)


def req(**kw):
    kw.setdefault("method", "driver")
    return ForecastRequest(**kw)


# ---------- ingest ----------

def test_depreciation_counts_as_opex_and_is_tracked():
    m, b = company(months=3)
    assert m["depreciation"].tolist() == [2000, 2000, 2000]
    assert m["opex"].iloc[0] == pytest.approx(50_000 + 2_000)
    assert list(b.columns) == ["cash", "receivables", "inventory", "fixed_assets", "other_assets",
                               "payables", "debt", "other_liability", "equity"]


def test_balances_only_for_months_provided():
    m, b = company(months=12, balance_months=2)
    assert len(m) == 12 and len(b) == 2


def test_cash_lines_take_amounts_not_percentages():
    with pytest.raises(ValidationError, match="kind 'add'"):
        Adjustment(line="equity", kind="pct", value=0.1, start="2026-01-01")


# ---------- fitting ----------

def test_fitted_drivers_recover_known_values():
    m, b = company(dso=45, dpo=30, capex=3_000, dep=2_000)
    resp = run_forecast(m, req(simulations=0), b)
    d = resp.drivers
    assert d["dso_days"].value == pytest.approx(45, rel=1e-6)
    assert d["dpo_days"].value == pytest.approx(30, rel=1e-6)
    assert d["dio_days"].value == 0 and "No inventory" in d["dio_days"].method
    assert d["depreciation_monthly"].value == pytest.approx(2_000)
    assert d["capex_monthly"].value == pytest.approx(3_000)


def test_capex_defaults_to_depreciation_without_consecutive_balances():
    m, b = company(balance_months=1)
    d = run_forecast(m, req(simulations=0), b).drivers
    assert d["capex_monthly"].value == pytest.approx(2_000) and "Assumed equal" in d["capex_monthly"].method


# ---------- the statements ----------

def test_balance_sheet_balances_and_cash_flow_reconciles():
    m, b = company()
    r = run_forecast(m, req(horizon_months=18, simulations=0), b)
    assert r.opening_balance.check == 0
    assert max(abs(x.check) for x in r.balance_sheet) < 0.02
    cash = r.opening_balance.cash
    for cf, bs, pl in zip(r.cash_flow, r.balance_sheet, r.months):
        assert cf.net_income == pytest.approx(pl.net_income, abs=0.02)
        assert cf.operating == pytest.approx(
            cf.net_income + cf.depreciation + cf.change_receivables + cf.change_inventory + cf.change_payables, abs=0.05)
        assert cf.net_change == pytest.approx(cf.operating + cf.investing + cf.financing, abs=0.05)
        cash += cf.net_change
        assert bs.cash == pytest.approx(cash, abs=0.1) and cf.ending_cash == bs.cash


def test_balances_hold_on_every_simulated_path():
    m, b = company()
    opening, _ = st.opening_balance(m, b)
    periods = pd.date_range("2026-01-01", periods=12, freq="MS")
    rng = np.random.default_rng(0)
    rev = rng.uniform(50_000, 200_000, (500, 12))
    cogs, opex = rev * rng.uniform(0.2, 0.6, (500, 12)), rng.uniform(20_000, 90_000, (500, 12))
    other = rng.normal(0, 1_000, (500, 12))
    drivers = Drivers(revenue_base=1, revenue_growth=0, gross_margin=0.5, opex_base=1, opex_growth=0,
                      dso_days=40, dio_days=10, dpo_days=25, capex_monthly=5_000, depreciation_monthly=1_500)
    adj = [Adjustment(line="equity", kind="add", value=1e5, start="2026-04-01", end="2026-04-01"),
           Adjustment(line="debt", kind="add", value=-2e3, start="2026-02-01")]
    s = st.project(opening, drivers, rev, cogs, opex, other, adj, periods)
    assert np.abs(s["check"]).max() < 1e-6


def test_working_capital_moves_cash_not_profit():
    m, b = company(dso=30)
    base = run_forecast(m, req(simulations=0), b)
    slow = run_forecast(m, req(simulations=0, overrides=DriverOverrides(dso_days=60)), b)
    assert [x.net_income for x in slow.months] == [x.net_income for x in base.months]
    tied_up = slow.balance_sheet[-1].receivables - base.balance_sheet[-1].receivables
    assert tied_up == pytest.approx(base.months[-1].revenue * 30 / DAYS, rel=1e-6)
    assert base.cash.ending_cash - slow.cash.ending_cash == pytest.approx(tied_up, abs=0.1)


def test_financing_and_capex_adjustments():
    m, b = company()
    adj = [
        Adjustment(line="equity", kind="add", value=500_000, start="2026-03-01", end="2026-03-01", label="Round"),
        Adjustment(line="debt", kind="add", value=200_000, start="2026-02-01", end="2026-02-01"),
        Adjustment(line="debt", kind="add", value=-10_000, start="2026-03-01"),
        Adjustment(line="capex", kind="add", value=80_000, start="2026-04-01", end="2026-04-01"),
    ]
    base = run_forecast(m, req(simulations=0), b)
    r = run_forecast(m, req(simulations=0, adjustments=adj), b)
    assert [x.net_income for x in r.months] == [x.net_income for x in base.months]  # cash items only
    cf = {x.period.month: x for x in r.cash_flow}
    assert cf[3].equity == 500_000 and cf[2].debt == 200_000 and cf[3].debt == -10_000
    assert cf[4].capex == pytest.approx(base.cash_flow[3].capex - 80_000)
    assert r.balance_sheet[-1].debt == pytest.approx(base.balance_sheet[-1].debt + 200_000 - 10 * 10_000)
    assert r.balance_sheet[-1].fixed_assets == pytest.approx(base.balance_sheet[-1].fixed_assets + 80_000)
    expected = 500_000 + 200_000 - 100_000 - 80_000
    assert r.cash.ending_cash - base.cash.ending_cash == pytest.approx(expected, abs=0.1)
    assert max(abs(x.check) for x in r.balance_sheet) < 0.02


# ---------- runway ----------

def test_runway_for_a_loss_making_company():
    m, b = company(rev0=50_000, opex0=60_000, g=0.0, og=0.0, cash0=1_000_000, debt=0)
    r = run_forecast(m, req(horizon_months=36, simulations=500), b)
    # burns about 33k/month (32k loss + 3k capex - 2k depreciation added back); ~240k left at the start
    assert r.cash.cash_out_period is not None and 1 <= r.cash.runway_months <= 36
    assert r.cash.note.startswith("Cash goes negative in")
    u = r.uncertainty
    assert u.prob_cash_out > 0.9
    assert u.runway.p10 <= u.runway.p50 <= r.cash.runway_months + 2
    assert all(mb.cash is not None for mb in u.months)


def test_runway_beyond_horizon_is_null():
    m, b = company()  # profitable
    r = run_forecast(m, req(simulations=200), b)
    assert r.cash.runway_months is None and r.cash.cash_out_period is None
    assert r.uncertainty.prob_cash_out == 0 and r.uncertainty.runway.p50 is None


def test_equity_raise_extends_runway_in_scenarios():
    m, b = company(rev0=50_000, opex0=60_000, g=0.0, og=0.0, cash0=1_000_000, debt=0)
    c = run_scenarios(m, ScenarioRequest(method="driver", horizon_months=36, simulations=300, scenarios=[
        {"name": "Raise", "adjustments": [{"line": "equity", "kind": "add", "value": 400_000,
                                           "start": "2026-02-01", "end": "2026-02-01"}]}]), b)
    raise_ = c.scenarios[0]
    assert raise_.forecast.cash.runway_months > c.baseline.cash.runway_months or raise_.forecast.cash.runway_months is None
    assert raise_.delta_point["ending_cash"] == pytest.approx(400_000, abs=1)
    assert raise_.delta.ending_cash.p10 == pytest.approx(400_000, abs=1)  # paired paths: exact


# ---------- missing or messy balances ----------

def test_no_balances_means_no_statements_and_cash_items_warn():
    m, _ = company()
    r = run_forecast(m, req(simulations=50, overrides=DriverOverrides(dso_days=60), adjustments=[
        Adjustment(line="equity", kind="add", value=1, start="2026-01-01")]))
    assert r.balance_sheet is None and r.cash is None and r.uncertainty.prob_cash_out is None
    assert any("need a balance sheet" in w for w in r.warnings)
    assert "dso_days" not in r.drivers


def test_balances_must_cover_the_last_actual_month():
    m, b = company()
    r = run_forecast(m, req(simulations=0), b.iloc[:-1])
    assert r.balance_sheet is None
    assert any("Upload balances for 2025-12" in w for w in r.warnings)


def test_unbalanced_input_uses_derived_equity_and_warns():
    m, b = company()
    b = b.copy()
    b.loc[b.index[-1], "equity"] += 50_000
    r = run_forecast(m, req(simulations=0), b)
    assert r.opening_balance.check == 0
    assert any("doesn't balance" in w for w in r.warnings)


def test_opening_cash_negative_means_zero_runway():
    m, b = company(cash0=-500_000)
    assert st.runway(np.array([[-1.0, -2.0]]), -1.0).tolist() == [0.0]
    r = run_forecast(m, req(simulations=0), b)
    assert r.cash.runway_months == 0 or r.opening_balance.cash >= 0
