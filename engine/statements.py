"""Balance sheet and cash flow, driven by the income statement forecast.

Starting from the balance sheet at the last actual month:

    receivables[t] = revenue[t] * dso_days / DAYS
    inventory[t]   = cogs[t] * dio_days / DAYS
    payables[t]    = (cogs[t] + opex[t] - depreciation[t]) * dpo_days / DAYS
    fixed_assets[t] = fixed_assets[t-1] + capex[t] - depreciation[t]
    debt[t]        = debt[t-1] + borrowed[t] - repaid[t]
    equity[t]      = equity[t-1] + net_income[t] + raised[t]
    other assets and other liabilities stay at their opening values

    operating = net_income + depreciation - change in receivables
                - change in inventory + change in payables
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
BALANCE_DRIVERS = ("dso_days", "dio_days", "dpo_days", "capex_monthly", "depreciation_monthly")
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
    return {k: out[k] for k in BALANCE_DRIVERS}


def cash_line(line: str, adjustments: list[Adjustment], periods: pd.DatetimeIndex) -> np.ndarray:
    out = np.zeros(len(periods))
    for a in adjustments:
        if a.line == line:
            out[adjustment_mask(a, periods)] += a.value
    return out


def project(
    opening: Opening,
    drivers,
    revenue: np.ndarray,
    cogs: np.ndarray,
    opex: np.ndarray,
    other: np.ndarray,
    adjustments: list[Adjustment],
    periods: pd.DatetimeIndex,
) -> dict[str, np.ndarray]:
    """All balance sheet and cash flow lines, shape like `revenue`."""
    o = opening.values
    shape = revenue.shape
    dep = np.broadcast_to(drivers.depreciation_monthly, shape)
    capex = drivers.capex_monthly + cash_line("capex", adjustments, periods)
    capex = np.broadcast_to(capex, shape)
    borrowed = np.broadcast_to(cash_line("debt", adjustments, periods), shape)
    raised = np.broadcast_to(cash_line("equity", adjustments, periods), shape)

    net_income = revenue - cogs - opex + other
    receivables = revenue * drivers.dso_days / DAYS
    inventory = cogs * drivers.dio_days / DAYS
    payables = np.clip(cogs + opex - dep, 0, None) * drivers.dpo_days / DAYS

    def change(x, start):
        prev = np.concatenate([np.full(shape[:-1] + (1,), start), x[..., :-1]], axis=-1)
        return x - prev

    d_rec, d_inv, d_pay = change(receivables, o["receivables"]), change(inventory, o["inventory"]), change(payables, o["payables"])
    operating = net_income + dep - d_rec - d_inv + d_pay
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
    total_liabilities = payables + debt + other_liability
    return dict(
        cash=cash, receivables=receivables, inventory=inventory, fixed_assets=fixed_assets,
        other_assets=other_assets, total_assets=total_assets, payables=payables, debt=debt,
        other_liability=other_liability, total_liabilities=total_liabilities, equity=equity,
        check=total_assets - total_liabilities - equity,
        net_income=net_income, depreciation=dep, change_receivables=-d_rec, change_inventory=-d_inv,
        change_payables=d_pay, operating=operating, capex=-capex, investing=investing,
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
               "payables", "debt", "other_liability", "total_liabilities", "equity", "check")
    balance_sheet = [
        BalanceSheetMonth(period=p.date(), **{k: _r(s[k][i]) for k in bs_keys}) for i, p in enumerate(periods)
    ]
    cash_flow = [
        CashFlowMonth(
            period=p.date(),
            net_income=_r(s["net_income"][i]), depreciation=_r(s["depreciation"][i]),
            change_receivables=_r(s["change_receivables"][i]), change_inventory=_r(s["change_inventory"][i]),
            change_payables=_r(s["change_payables"][i]), operating=_r(s["operating"][i]),
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
