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
            capex=3_000, cash0=200_000, debt=100_000, fa0=150_000, inventory_days=0, balance_months=None,
            interest_rate=0.0, tax_rate=0.0):
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
        interest = debt * interest_rate / 12
        tax = tax_rate * max(0.0, rev - cogs - opex_cash - dep - interest)
        if interest:
            items.append(LineItem(period=p, account="Loan interest", category="interest", amount=interest))
        if tax:
            items.append(LineItem(period=p, account="Income tax", category="tax", amount=tax))
        ni = rev - cogs - opex_cash - dep - interest - tax
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
                               "payables", "debt", "other_liability", "tax_payable", "equity"]


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
                      dso_days=40, dio_days=10, dpo_days=25, capex_monthly=5_000, depreciation_monthly=1_500,
                      interest_rate=0.08, tax_rate=0.25)
    adj = [Adjustment(line="equity", kind="add", value=1e5, start="2026-04-01", end="2026-04-01"),
           Adjustment(line="debt", kind="add", value=-2e3, start="2026-02-01"),
           Adjustment(line="capex", kind="add", value=40e3, start="2026-03-01", end="2026-03-01")]
    capex, new_dep = st.capex_and_new_depreciation(drivers, 3_000, adj, periods)
    interest = st.interest_path(opening, drivers, adj, periods)
    opex = opex + new_dep
    pretax = rev - cogs - opex + other - interest
    tax = st.tax_schedule(pretax, drivers.tax_rate)
    pl = dict(revenue=rev, cogs=cogs, opex=opex, depreciation=1_500 + new_dep, tax=tax, net_income=pretax - tax)
    s = st.project(opening, drivers, pl, capex, adj, periods)
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
    # financing doesn't touch profit; the equipment does, through depreciation from May (80k / 60 months)
    profit_change = [round(x.net_income - y.net_income, 2) for x, y in zip(r.months, base.months)]
    assert profit_change[:4] == [0, 0, 0, 0]
    assert profit_change[4:] == [pytest.approx(-80_000 / 60, abs=0.02)] * (len(profit_change) - 4)
    cf = {x.period.month: x for x in r.cash_flow}
    assert cf[3].equity == 500_000 and cf[2].debt == 200_000 and cf[3].debt == -10_000
    assert cf[4].capex == pytest.approx(base.cash_flow[3].capex - 80_000)
    assert r.balance_sheet[-1].debt == pytest.approx(base.balance_sheet[-1].debt + 200_000 - 10 * 10_000)
    months_depreciated = len(r.months) - 4
    assert r.balance_sheet[-1].fixed_assets == pytest.approx(
        base.balance_sheet[-1].fixed_assets + 80_000 - months_depreciated * 80_000 / 60, abs=0.1)
    expected = 500_000 + 200_000 - 100_000 - 80_000  # depreciation is non-cash: cash effect unchanged
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


# ---------- tax, interest, depreciation on new capex ----------

def test_fitted_interest_and_tax_rates_recover_known_values():
    m, b = company(interest_rate=0.08, tax_rate=0.25)
    d = run_forecast(m, req(simulations=0), b).drivers
    assert d["interest_rate"].value == pytest.approx(0.08, rel=1e-6)
    assert d["tax_rate"].value == pytest.approx(0.25, rel=1e-6)


def test_income_statement_lines_add_up():
    m, b = company(interest_rate=0.08, tax_rate=0.25)
    r = run_forecast(m, req(simulations=0), b)
    for x in r.months:
        assert x.interest == pytest.approx(100_000 * 0.08 / 12, abs=0.01)  # flat debt
        assert x.pretax_income == pytest.approx(x.operating_income + x.other_net - x.interest, abs=0.02)
        assert x.tax == pytest.approx(0.25 * x.pretax_income, abs=0.02)
        assert x.net_income == pytest.approx(x.pretax_income - x.tax, abs=0.02)


def test_new_loan_costs_interest_from_the_next_month():
    m, b = company(interest_rate=0.08)
    loan = Adjustment(line="debt", kind="add", value=200_000, start="2026-03-01", end="2026-03-01")
    base = run_forecast(m, req(simulations=0), b)
    r = run_forecast(m, req(simulations=0, adjustments=[loan]), b)
    extra = [round(x.interest - y.interest, 2) for x, y in zip(r.months, base.months)]
    assert extra[:3] == [0, 0, 0] and extra[3] == pytest.approx(200_000 * 0.08 / 12, abs=0.01)
    # a scenario rate applies to the whole balance
    r = run_forecast(m, req(simulations=0, overrides=DriverOverrides(interest_rate=0.12)), b)
    assert r.months[0].interest == pytest.approx(100_000 * 0.12 / 12, abs=0.01)


def test_losses_carry_forward_before_tax_is_charged():
    tax = st.tax_schedule(np.array([-100.0, 50.0, 80.0, 10.0]), 0.25)
    assert tax.tolist() == [0, 0, pytest.approx(7.5), pytest.approx(2.5)]
    paths = st.tax_schedule(np.array([[-100.0, 50.0, 80.0], [10.0, 10.0, 10.0]]), 0.5)
    assert paths.shape == (2, 3) and paths[1].tolist() == [5, 5, 5]
    assert st.tax_schedule(np.array([[1.0, 2.0]]), 0.5).shape == (1, 2)


def test_tax_is_paid_quarterly():
    m, b = company(tax_rate=0.25)
    r = run_forecast(m, req(simulations=0, horizon_months=6), b)
    owed = [x.tax_payable for x in r.balance_sheet]  # Jan..Jun 2026
    taxes = [x.tax for x in r.months]
    assert owed[0] == pytest.approx(taxes[0], abs=0.02)  # January: paid nothing owed at opening, accrues Jan
    assert owed[2] == pytest.approx(sum(taxes[:3]), abs=0.05)  # accrues through March
    assert owed[3] == pytest.approx(taxes[3], abs=0.02)  # April: Q1 paid
    cf = r.cash_flow
    assert cf[1].change_tax_payable == pytest.approx(taxes[1], abs=0.02)  # Feb: owed, not paid
    assert cf[3].change_tax_payable == pytest.approx(taxes[3] - owed[2], abs=0.05)  # Apr: Q1 paid out
    assert max(abs(x.check) for x in r.balance_sheet) < 0.02


def test_tax_works_without_a_balance_sheet():
    m, _ = company()
    base = run_forecast(m, req(simulations=0))
    r = run_forecast(m, req(simulations=0, overrides=DriverOverrides(tax_rate=0.2)))
    assert r.months[0].net_income == pytest.approx(base.months[0].net_income * 0.8, abs=0.05)
    assert r.drivers["tax_rate"].overridden


def test_interest_without_balance_sheet_stays_at_recent_average():
    m, _ = company(interest_rate=0.06)
    r = run_forecast(m, req(simulations=0, overrides=DriverOverrides(interest_rate=0.2)))
    assert all(x.interest == pytest.approx(100_000 * 0.06 / 12, abs=0.01) for x in r.months)
    assert any("interest_rate" in w and "need a balance sheet" in w for w in r.warnings)


def test_each_purchase_uses_its_own_useful_life():
    m, b = company()
    buy = Adjustment(line="capex", kind="add", value=36_000, start="2026-02-01", end="2026-02-01",
                     useful_life_months=36, label="Van")
    base = run_forecast(m, req(simulations=0), b)
    r = run_forecast(m, req(simulations=0, adjustments=[buy]), b)
    extra = [round(x.depreciation - y.depreciation, 2) for x, y in zip(r.months, base.months)]
    assert extra[:2] == [0, 0] and extra[2:] == [1000.0] * 10
    assert [round(y.opex - x.opex, 2) for x, y in zip(base.months, r.months)] == extra  # in opex too


def test_higher_regular_capex_depreciates_the_extra_only():
    m, b = company(capex=3_000)
    r = run_forecast(m, req(simulations=0, overrides=DriverOverrides(capex_monthly=5_000, useful_life_months=40)), b)
    base = run_forecast(m, req(simulations=0), b)
    extra = [x.depreciation - y.depreciation for x, y in zip(r.months, base.months)]
    # 2k/month extra, each month's purchase adds 2000/40 = 50/month from the month after
    assert extra[:4] == pytest.approx([0, 50, 100, 150])


def test_simulated_paths_include_tax_and_interest():
    m, b = company(interest_rate=0.08, tax_rate=0.25)
    r = run_forecast(m, req(simulations=500), b)
    for point, band_ in zip(r.months, r.uncertainty.months):
        assert band_.net_income.p50 == pytest.approx(point.net_income, rel=0.03)
