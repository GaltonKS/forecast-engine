"""Balance sheet and cash flow, driven by the income statement forecast.

Starting from the balance sheet at the last actual month:

    receivables[t] = revenue[t] * dso_days / DAYS
    inventory[t]   = cogs[t] * dio_days / DAYS
    payables[t]    = (cogs[t] + opex[t] - depreciation[t]) * dpo_days / DAYS
    fixed_assets[t] = fixed_assets[t-1] + capex[t] - depreciation[t]
    debt[t]        = debt[t-1] + borrowed[t] - repaid[t]
    tax_payable[t] = tax_payable[t-1] + tax[t] - paid[t]   (paid quarterly)
    equity[t]      = equity[t-1] + net_income[t] + raised[t]
    other assets and other liabilities stay at their opening values

    interest[t] = debt[t-1] * interest_rate / 12
    tax[t]      = tax_rate * taxable profit, after losses carried forward
    depreciation[t] = depreciation_monthly + straight-line depreciation of
                      capex above the fitted baseline, over its useful life

    operating = net_income + depreciation - change in receivables
                - change in inventory + change in payables + change in tax payable
    investing = -capex
    financing = borrowed - repaid + raised
    cash[t]   = cash[t-1] + operating + investing + financing

Cash is the only line not set by a driver, so assets = liabilities + equity
holds every month by construction (the `check` column shows it).

Every function works on arrays shaped (months,) or (paths, months), so the
Monte Carlo runs all paths at once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import (
    ASSET_CATEGORIES,
    LIABILITY_CATEGORIES,
    Adjustment,
    BalanceSheetMonth,
    CashFlowMonth,
    CashSummary,
    FittedDriver,
)
from .simulate import adjustment_mask

DAYS = 365 / 12  # average days per month
BALANCE_DRIVERS = (
    "dso_days", "dio_days", "dpo_days", "capex_monthly", "depreciation_monthly", "useful_life_months", "interest_rate",
)
DEFAULT_USEFUL_LIFE = 60
TAX_PAYMENT_MONTHS = (1, 4, 7, 10)  # tax owed at the end of a quarter is paid the month after
CASH_LINES = ("capex", "debt", "equity")


@dataclass
class Opening:
    period: pd.Timestamp
    values: dict[str, float]  # every balance category; equity is assets - liabilities


def opening_balance(monthly: pd.DataFrame, balances: pd.DataFrame | None) -> tuple[Opening | None, list[str]]:
    """The balance sheet at the last actual month, or None (with the reason)."""
    last = monthly.index.max()
    if balances is None:
        return None, []
    if last not in balances.index:
        latest = balances.index.max()
        return None, [
            f"No balance sheet for {last:%Y-%m}, the last month of income statement data "
            f"(latest balances: {latest:%Y-%m}). Upload balances for {last:%Y-%m} to get the "
            "balance sheet, cash flow and runway."
        ]
    row = balances.loc[last]
    values = {c: float(row[c]) for c in balances.columns}
    assets = sum(values[c] for c in ASSET_CATEGORIES)
    liabilities = sum(values[c] for c in LIABILITY_CATEGORIES)
    derived = assets - liabilities
    warnings = []
    reported = values["equity"]
    if reported != 0 and abs(reported - derived) > max(1.0, 0.005 * abs(assets)):
        warnings.append(
            f"The {last:%Y-%m} balance sheet doesn't balance: assets {assets:,.0f} - liabilities "
            f"{liabilities:,.0f} = {derived:,.0f}, but equity is {reported:,.0f}. Equity was set to "
            f"{derived:,.0f}; check for accounts missing from the upload."
        )
    values["equity"] = derived
    return Opening(last, values), warnings


def _recent(series: pd.Series, n: int = 3) -> float | None:
    s = series.dropna()
    s = s[np.isfinite(s)]
    return float(s.tail(n).mean()) if len(s) else None


def fit_balance_drivers(
    monthly: pd.DataFrame, balances: pd.DataFrame, opening: Opening
) -> dict[str, FittedDriver]:
    """Working-capital days, capex and depreciation from the latest 3 months
    that have both balances and income statement data."""
    b = balances.reindex(monthly.index)  # NaN where no balances that month
    rev, cogs = monthly["revenue"], monthly["cogs"]
    cash_costs = monthly["cogs"] + monthly["opex"] - monthly["depreciation"]
    out = {}

    def days(balance: pd.Series, base: pd.Series, name: str, base_name: str):
        if not (balances[name] != 0).any():
            return FittedDriver(value=0.0, method=f"No {name} in the balance sheet")
        v = _recent((balance / base.where(base > 0)) * DAYS)
        if v is None:
            return FittedDriver(value=0.0, method=f"No months with both {name} and {base_name}")
        return FittedDriver(value=v, method=f"{name} / monthly {base_name} x {DAYS:.1f} days, last 3 months with balances")

    out["dso_days"] = days(b["receivables"], rev, "receivables", "revenue")
    out["dio_days"] = days(b["inventory"], cogs, "inventory", "cogs")
    out["dpo_days"] = days(b["payables"], cash_costs, "payables", "cash costs (cogs + opex - depreciation)")

    dep = _recent(monthly["depreciation"]) or 0.0
    if dep > 0:
        out["depreciation_monthly"] = FittedDriver(value=dep, method="Average depreciation of the last 3 months")
    else:
        note = "No depreciation in the income statement"
        if opening.values["fixed_assets"] > 0:
            note += "; tag depreciation accounts 'depreciation' to model it"
        out["depreciation_monthly"] = FittedDriver(value=0.0, method=note)

    fa = b["fixed_assets"]
    consecutive = fa.notna() & fa.shift(1).notna()
    implied = (fa - fa.shift(1) + monthly["depreciation"])[consecutive]
    capex = _recent(implied)
    if capex is not None:
        out["capex_monthly"] = FittedDriver(
            value=max(0.0, capex),
            method="Change in fixed assets + depreciation, last 3 months with consecutive balances",
        )
    else:
        out["capex_monthly"] = FittedDriver(
            value=dep, method="Assumed equal to depreciation (fixed assets held level): no consecutive balances"
        )
    out["useful_life_months"] = FittedDriver(
        value=DEFAULT_USEFUL_LIFE,
        method=f"Default {DEFAULT_USEFUL_LIFE} months (5 years) for capex above the baseline; override per scenario or per purchase",
    )
    debt_prev = b["debt"].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = (monthly["interest"] / debt_prev.where(debt_prev > 0)) * 12
    rate = _recent(rates[monthly["interest"] > 0])
    if rate is not None:
        out["interest_rate"] = FittedDriver(
            value=rate, method="Annualised interest / previous month's debt, last 3 months with both"
        )
    elif (monthly["interest"] > 0).any():
        out["interest_rate"] = FittedDriver(value=0.0, method="Interest rows but no debt balances the month before")
    else:
        out["interest_rate"] = FittedDriver(
            value=0.0, method="No interest rows; set interest_rate to charge interest on debt"
        )
    return {k: out[k] for k in BALANCE_DRIVERS}


def fit_tax_rate(monthly: pd.DataFrame) -> FittedDriver:
    """Effective rate: tax / pre-tax profit over the last 12 months."""
    t = monthly.tail(12)
    pretax = (t["revenue"] - t["cogs"] - t["opex"] + t["other"] - t["interest"]).sum()
    tax = t["tax"].sum()
    if tax <= 0:
        return FittedDriver(value=0.0, method="No tax rows; set tax_rate to model income tax")
    if pretax <= 0:
        return FittedDriver(value=0.0, method="Tax rows, but no pre-tax profit in the last 12 months to measure a rate")
    rate = float(min(max(tax / pretax, 0.0), 0.6))
    return FittedDriver(value=rate, method=f"Tax / pre-tax profit over the last {len(t)} months (effective rate)")


def debt_path(opening: "Opening | None", adjustments: list[Adjustment], periods: pd.DatetimeIndex) -> np.ndarray:
    start = opening.values["debt"] if opening else 0.0
    return start + np.cumsum(cash_line("debt", adjustments, periods))


def interest_path(
    opening: "Opening | None", drivers, adjustments: list[Adjustment], periods: pd.DatetimeIndex,
    flat_interest: float = 0.0,
) -> np.ndarray:
    """With a balance sheet: last month's debt x rate / 12. Without one: the
    recent average interest, held flat."""
    if opening is None:
        return np.full(len(periods), flat_interest)
    debt = debt_path(opening, adjustments, periods)
    prev = np.concatenate([[opening.values["debt"]], debt[:-1]])
    return np.clip(prev, 0, None) * drivers.interest_rate / 12


def capex_and_new_depreciation(
    drivers, fitted_capex: float, adjustments: list[Adjustment], periods: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray]:
    """Total capex per month, and depreciation on capex above the fitted
    baseline (baseline capex is already reflected in historical opex).
    Each month's extra capex depreciates straight-line from the next month."""
    h = len(periods)
    capex = drivers.capex_monthly + cash_line("capex", adjustments, periods)
    new_dep = np.zeros(h)

    def depreciate(amounts: np.ndarray, life: float):
        for start in np.nonzero(amounts)[0]:
            end = min(h, start + 1 + int(np.ceil(life)))
            new_dep[start + 1:end] += amounts[start] / life

    base_extra = drivers.capex_monthly - fitted_capex
    if abs(base_extra) > 1e-9:
        depreciate(np.full(h, base_extra), drivers.useful_life_months)
    for a in adjustments:
        if a.line == "capex":
            amounts = np.where(adjustment_mask(a, periods), a.value, 0.0)
            depreciate(amounts, a.useful_life_months or drivers.useful_life_months)
    return capex, new_dep


def tax_schedule(pretax: np.ndarray, rate: float) -> np.ndarray:
    """Tax on profit with losses carried forward (no refunds). Works on
    (months,) or (paths, months); returns the same shape."""
    pretax = np.asarray(pretax, dtype=float)
    p2 = pretax.reshape(-1, pretax.shape[-1])
    tax = np.zeros_like(p2)
    if rate > 0:
        pool = np.zeros(p2.shape[0])
        for t in range(p2.shape[1]):
            pool = pool + np.clip(-p2[:, t], 0, None)
            profit = np.clip(p2[:, t], 0, None)
            used = np.minimum(pool, profit)
            pool = pool - used
            tax[:, t] = rate * (profit - used)
    return tax.reshape(pretax.shape)


def cash_line(line: str, adjustments: list[Adjustment], periods: pd.DatetimeIndex) -> np.ndarray:
    out = np.zeros(len(periods))
    for a in adjustments:
        if a.line == line:
            out[adjustment_mask(a, periods)] += a.value
    return out


def project(
    opening: Opening,
    drivers,
    pl: dict[str, np.ndarray],
    capex: np.ndarray,
    adjustments: list[Adjustment],
    periods: pd.DatetimeIndex,
) -> dict[str, np.ndarray]:
    """All balance sheet and cash flow lines, shaped like pl["revenue"].

    pl holds the finished income statement: revenue, cogs, opex (including
    all depreciation), depreciation, tax and net_income."""
    o = opening.values
    revenue, cogs, opex = pl["revenue"], pl["cogs"], pl["opex"]
    shape = revenue.shape
    dep = np.broadcast_to(pl["depreciation"], shape)
    tax = np.broadcast_to(pl["tax"], shape)
    net_income = pl["net_income"]
    capex = np.broadcast_to(capex, shape)
    borrowed = np.broadcast_to(cash_line("debt", adjustments, periods), shape)
    raised = np.broadcast_to(cash_line("equity", adjustments, periods), shape)

    receivables = revenue * drivers.dso_days / DAYS
    inventory = cogs * drivers.dio_days / DAYS
    payables = np.clip(cogs + opex - dep, 0, None) * drivers.dpo_days / DAYS

    # tax: accrues monthly, paid the month after each quarter end
    pay_month = np.isin(periods.month, TAX_PAYMENT_MONTHS)
    tax_payable = np.empty(shape)
    owed = np.full(shape[:-1], o["tax_payable"])
    for t in range(shape[-1]):
        if pay_month[t]:
            owed = np.zeros(shape[:-1])
        owed = owed + tax[..., t]
        tax_payable[..., t] = owed

    def change(x, start):
        prev = np.concatenate([np.full(shape[:-1] + (1,), start), x[..., :-1]], axis=-1)
        return x - prev

    d_rec, d_inv, d_pay = change(receivables, o["receivables"]), change(inventory, o["inventory"]), change(payables, o["payables"])
    d_tax = change(tax_payable, o["tax_payable"])
    operating = net_income + dep - d_rec - d_inv + d_pay + d_tax
    investing = -capex
    financing = borrowed + raised
    net_change = operating + investing + financing

    cash = o["cash"] + np.cumsum(net_change, axis=-1)
    fixed_assets = o["fixed_assets"] + np.cumsum(capex - dep, axis=-1)
    debt = o["debt"] + np.cumsum(borrowed, axis=-1)
    equity = o["equity"] + np.cumsum(net_income + raised, axis=-1)
    other_assets = np.full(shape, o["other_assets"])
    other_liability = np.full(shape, o["other_liability"])
    total_assets = cash + receivables + inventory + fixed_assets + other_assets
    total_liabilities = payables + debt + other_liability + tax_payable
    return dict(
        cash=cash, receivables=receivables, inventory=inventory, fixed_assets=fixed_assets,
        other_assets=other_assets, total_assets=total_assets, payables=payables, debt=debt,
        other_liability=other_liability, tax_payable=tax_payable, total_liabilities=total_liabilities, equity=equity,
        check=total_assets - total_liabilities - equity,
        net_income=net_income, depreciation=dep, change_receivables=-d_rec, change_inventory=-d_inv,
        change_payables=d_pay, change_tax_payable=d_tax, operating=operating, capex=-capex, investing=investing,
        debt_flow=borrowed, equity_flow=raised, financing=financing, net_change=net_change,
    )


def runway(cash: np.ndarray, opening_cash: float) -> np.ndarray:
    """Months until cash first goes negative, per path; inf if it never does
    within the horizon. 0 if it's already negative."""
    cash = np.atleast_2d(cash)
    neg = cash < 0
    months = np.where(neg.any(axis=1), neg.argmax(axis=1) + 1, np.inf).astype(float)
    if opening_cash < 0:
        months[:] = 0
    return months


def _r(x) -> float:
    return round(float(x), 2) + 0.0  # + 0.0 turns -0.0 into 0.0


def to_models(opening: Opening, s: dict[str, np.ndarray], periods: pd.DatetimeIndex):
    o = opening.values
    assets = sum(o[c] for c in ASSET_CATEGORIES)
    liabilities = sum(o[c] for c in LIABILITY_CATEGORIES)
    open_row = BalanceSheetMonth(
        period=opening.period.date(), **{c: _r(o[c]) for c in ASSET_CATEGORIES + LIABILITY_CATEGORIES},
        total_assets=_r(assets), total_liabilities=_r(liabilities), equity=_r(o["equity"]),
        check=_r(assets - liabilities - o["equity"]),
    )
    bs_keys = ("cash", "receivables", "inventory", "fixed_assets", "other_assets", "total_assets",
               "payables", "debt", "other_liability", "tax_payable", "total_liabilities", "equity", "check")
    balance_sheet = [
        BalanceSheetMonth(period=p.date(), **{k: _r(s[k][i]) for k in bs_keys}) for i, p in enumerate(periods)
    ]
    cash_flow = [
        CashFlowMonth(
            period=p.date(),
            net_income=_r(s["net_income"][i]), depreciation=_r(s["depreciation"][i]),
            change_receivables=_r(s["change_receivables"][i]), change_inventory=_r(s["change_inventory"][i]),
            change_payables=_r(s["change_payables"][i]), change_tax_payable=_r(s["change_tax_payable"][i]),
            operating=_r(s["operating"][i]),
            capex=_r(s["capex"][i]), investing=_r(s["investing"][i]),
            debt=_r(s["debt_flow"][i]), equity=_r(s["equity_flow"][i]), financing=_r(s["financing"][i]),
            net_change=_r(s["net_change"][i]), ending_cash=_r(s["cash"][i]),
        )
        for i, p in enumerate(periods)
    ]
    cash = s["cash"]
    low = int(np.argmin(cash))
    months = float(runway(cash, o["cash"])[0])
    out_period = None if not np.isfinite(months) or months == 0 else periods[int(months) - 1].date()
    if o["cash"] < 0:
        note = "Cash is already negative at the last actual month."
    elif out_period:
        note = f"Cash goes negative in {out_period:%Y-%m}, {int(months)} months into the forecast."
    else:
        note = f"Cash stays positive through {periods[-1]:%Y-%m}."
    summary = CashSummary(
        opening_cash=_r(o["cash"]), ending_cash=_r(cash[-1]), lowest_cash=_r(cash[low]),
        lowest_cash_period=periods[low].date(), cash_out_period=out_period,
        runway_months=None if not np.isfinite(months) else int(months), note=note,
    )
    return open_row, balance_sheet, cash_flow, summary
