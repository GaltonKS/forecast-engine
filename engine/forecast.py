"""Driver-based forecast: fit drivers from history, then project forward.

Deterministic on purpose: the same data and overrides always give the same
numbers, and every number can be traced to a driver and a formula.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import (
    DriverOverrides,
    Drivers,
    FittedDriver,
    ForecastMonth,
    ForecastRequest,
    ForecastResponse,
)

ENGINE_VERSION = "0.1.0"
MIN_HISTORY = 3
RECOMMENDED_HISTORY = 12

FORMULAS = {
    "revenue": "revenue[t] = revenue_base * (1 + revenue_growth) ^ t",
    "cogs": "cogs[t] = revenue[t] * (1 - gross_margin)",
    "gross_profit": "gross_profit[t] = revenue[t] - cogs[t]",
    "opex": "opex[t] = opex_base * (1 + opex_growth) ^ t",
    "operating_income": "operating_income[t] = gross_profit[t] - opex[t]",
    "net_income": "net_income[t] = operating_income[t] + other_net",
}


def _log_linear_growth(series: pd.Series) -> float:
    """Monthly compound growth from a least-squares fit of log(value) on time.

    More robust to a single noisy month than first-to-last CAGR.
    """
    s = series[series > 0]
    if len(s) < 2:
        return 0.0
    t = np.arange(len(s))
    slope = np.polyfit(t, np.log(s.to_numpy()), 1)[0]
    return float(np.exp(slope) - 1)


def fit_drivers(monthly: pd.DataFrame) -> tuple[dict[str, FittedDriver], list[str]]:
    n = len(monthly)
    if n < MIN_HISTORY:
        raise ValueError(f"Need at least {MIN_HISTORY} months of history, got {n}")

    warnings = []
    if n < RECOMMENDED_HISTORY:
        warnings.append(
            f"Only {n} months of history; growth estimates are unreliable below "
            f"{RECOMMENDED_HISTORY}. Consider overriding growth drivers."
        )
    gaps = monthly.attrs.get("gaps") or []
    if gaps:
        warnings.append(f"Months with no data were treated as zero: {', '.join(gaps)}")

    trail12 = monthly.tail(RECOMMENDED_HISTORY)
    trail6 = monthly.tail(6)
    trail3 = monthly.tail(3)
    k = len(trail12)

    rev_sum = trail6["revenue"].sum()
    gm = float((rev_sum - trail6["cogs"].sum()) / rev_sum) if rev_sum else 0.0

    drivers = {
        "revenue_base": FittedDriver(
            value=float(trail3["revenue"].mean()),
            method="Average revenue of the last 3 months",
        ),
        "revenue_growth": FittedDriver(
            value=_log_linear_growth(trail12["revenue"]),
            method=f"Log-linear trend fit over the last {k} months",
        ),
        "gross_margin": FittedDriver(
            value=gm, method="(revenue - cogs) / revenue over the last 6 months"
        ),
        "opex_base": FittedDriver(
            value=float(trail3["opex"].mean()),
            method="Average opex of the last 3 months",
        ),
        "opex_growth": FittedDriver(
            value=_log_linear_growth(trail12["opex"]),
            method=f"Log-linear trend fit over the last {k} months",
        ),
        "other_net": FittedDriver(
            value=float(trail6["other"].mean()),
            method="Average net other income of the last 6 months",
        ),
    }
    return drivers, warnings


def apply_overrides(
    fitted: dict[str, FittedDriver], overrides: DriverOverrides
) -> dict[str, FittedDriver]:
    out = dict(fitted)
    for name, value in overrides.model_dump(exclude_none=True).items():
        out[name] = FittedDriver(value=value, method="Scenario override", overridden=True)
    return out


def project(drivers: Drivers, start: pd.Timestamp, horizon: int) -> list[ForecastMonth]:
    months = []
    for t in range(1, horizon + 1):
        revenue = drivers.revenue_base * (1 + drivers.revenue_growth) ** t
        cogs = revenue * (1 - drivers.gross_margin)
        gross_profit = revenue - cogs
        opex = drivers.opex_base * (1 + drivers.opex_growth) ** t
        operating_income = gross_profit - opex
        net_income = operating_income + drivers.other_net
        months.append(
            ForecastMonth(
                period=(start + pd.DateOffset(months=t)).date(),
                revenue=round(revenue, 2),
                cogs=round(cogs, 2),
                gross_profit=round(gross_profit, 2),
                opex=round(opex, 2),
                operating_income=round(operating_income, 2),
                other_net=round(drivers.other_net, 2),
                net_income=round(net_income, 2),
            )
        )
    return months


def run_forecast(monthly: pd.DataFrame, request: ForecastRequest) -> ForecastResponse:
    fitted, warnings = fit_drivers(monthly)
    final = apply_overrides(fitted, request.overrides)
    drivers = Drivers(**{k: v.value for k, v in final.items()})
    last = monthly.index.max()
    return ForecastResponse(
        engine_version=ENGINE_VERSION,
        last_actual_period=last.date(),
        history_months=len(monthly),
        drivers=final,
        formulas=FORMULAS,
        months=project(drivers, last, request.horizon_months),
        warnings=warnings,
    )
