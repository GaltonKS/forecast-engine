"""Forecast: fit drivers, pick a model per line by backtest, project forward.

Revenue and opex each get the model that was most accurate on recent months
(driver formula, seasonal naive, ETS, ARIMA or Prophet). COGS follows revenue
through gross margin, and other income stays flat. Scenario overrides on a
line's base or growth switch that line to the driver formula, so "2% growth"
means exactly 2%.

Deterministic: the same data and request always give the same numbers.
"""
from __future__ import annotations

import pandas as pd

from .backtest import BacktestResult, backtest
from .stats import MODELS, available_models, detect_seasonality, driver_path, fit_driver, future_index
from .schema import (
    DriverOverrides,
    Drivers,
    FittedDriver,
    ForecastMonth,
    ForecastRequest,
    ForecastResponse,
    HoldoutPoint,
    ModelScore,
    Seasonality,
    SeriesModel,
)

ENGINE_VERSION = "0.2.0"
MIN_HISTORY = 3
RECOMMENDED_HISTORY = 12
LINES = ("revenue", "opex")

BASE_FORMULAS = {
    "cogs": "cogs[t] = revenue[t] * (1 - gross_margin)",
    "gross_profit": "gross_profit[t] = revenue[t] - cogs[t]",
    "operating_income": "operating_income[t] = gross_profit[t] - opex[t]",
    "net_income": "net_income[t] = operating_income[t] + other_net",
}


def fit_drivers(
    monthly: pd.DataFrame, seasonal: bool | None = None
) -> tuple[dict[str, FittedDriver], list[str], dict[str, Seasonality]]:
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

    drivers, seasonality = {}, {}
    k = min(n, 12)
    for line in LINES:
        profile = detect_seasonality(monthly[line], seasonal)
        if seasonal is True and not profile.detected:
            warnings.append(f"Seasonality requested but not applied to {line}: {profile.note}")
        seasonality[line] = Seasonality(**profile.__dict__)
        fit = fit_driver(monthly[line], profile)
        adj = "seasonally adjusted " if fit.indices else ""
        drivers[f"{line}_base"] = FittedDriver(
            value=fit.base,
            method=f"Average {adj}{line} of the last 3 months, rolled forward to the last month at {line}_growth",
        )
        drivers[f"{line}_growth"] = FittedDriver(
            value=fit.growth, method=f"Log-linear trend fit over the last {k} months of {adj}{line}"
        )

    trail6 = monthly.tail(6)
    rev_sum = trail6["revenue"].sum()
    gm = float((rev_sum - trail6["cogs"].sum()) / rev_sum) if rev_sum else 0.0
    drivers["gross_margin"] = FittedDriver(
        value=gm, method="(revenue - cogs) / revenue over the last 6 months"
    )
    drivers["other_net"] = FittedDriver(
        value=float(trail6["other"].mean()), method="Average net other income of the last 6 months"
    )
    order = ["revenue_base", "revenue_growth", "gross_margin", "opex_base", "opex_growth", "other_net"]
    return {name: drivers[name] for name in order}, warnings, seasonality


def apply_overrides(
    fitted: dict[str, FittedDriver], overrides: DriverOverrides
) -> dict[str, FittedDriver]:
    out = dict(fitted)
    for name, value in overrides.model_dump(exclude_none=True).items():
        out[name] = FittedDriver(value=value, method="Scenario override", overridden=True)
    return out


def _scores(bt: BacktestResult) -> list[ModelScore]:
    return [
        ModelScore(
            method=s.method,
            status=s.status,
            wape=None if s.wape is None else round(s.wape, 5),
            bias=None if s.bias is None else round(s.bias, 5),
            folds=s.folds,
            note=s.note,
        )
        for s in bt.scores
    ]


def _choose(
    line: str, y: pd.Series, request: ForecastRequest, overridden: set[str]
) -> tuple[str, str, str, BacktestResult | None]:
    """Returns (method, selected_by, reason, backtest)."""
    if overridden:
        names = ", ".join(sorted(overridden))
        return "driver", "override", f"Scenario overrides {names}, so the driver formula is used", None

    available = available_models()
    if request.method != "auto":
        if request.method not in available:
            raise ValueError(f"Method '{request.method}' is not installed on this server")
        need = MODELS[request.method].min_train
        if len(y) < need:
            raise ValueError(f"Method '{request.method}' needs {need} months of history, have {len(y)}")

    bt = backtest(y, available, request.seasonal)
    if request.method != "auto":
        return request.method, "requested", f"Requested method '{request.method}'", bt

    best = bt.best(prefer_order=list(MODELS))
    if best is None:
        return "driver", "default", bt.note or "No model could be backtested; using the driver formula", bt
    score = next(s for s in bt.scores if s.method == best)
    reason = (
        f"Lowest backtest error over {score.folds} origins x {bt.horizon} months "
        f"(WAPE {score.wape:.1%}); simpler models win near-ties"
    )
    return best, "backtest", reason, bt


def run_forecast(monthly: pd.DataFrame, request: ForecastRequest) -> ForecastResponse:
    fitted, warnings, seasonality = fit_drivers(monthly, request.seasonal)
    final = apply_overrides(fitted, request.overrides)
    drivers = Drivers(**{k: v.value for k, v in final.items()})
    last = monthly.index.max()
    h = request.horizon_months
    periods = future_index(monthly["revenue"], h)
    overridden = set(request.overrides.model_dump(exclude_none=True))

    paths, models, formulas = {}, {}, {}
    for line in LINES:
        y = monthly[line]
        method, selected_by, reason, bt = _choose(
            line, y, request, overridden & {f"{line}_base", f"{line}_growth"}
        )
        indices = seasonality[line].indices if seasonality[line].detected else None
        if method != "driver":
            try:
                paths[line] = MODELS[method].fn(y, h, request.seasonal)
            except Exception as e:
                warnings.append(f"{method} failed on {line} ({e.__class__.__name__}); used the driver formula")
                method, selected_by, reason = "driver", "default", f"{reason}; fell back after a fit error"
        if method == "driver":
            base, growth = getattr(drivers, f"{line}_base"), getattr(drivers, f"{line}_growth")
            paths[line] = driver_path(base, growth, indices, periods)
            season = " * season[month(t)]" if indices else ""
            formulas[line] = f"{line}[t] = {line}_base * (1 + {line}_growth) ^ t{season}"
        else:
            formulas[line] = f"{line}[t] = {MODELS[method].label} forecast (see models.{line})"
            for d in (f"{line}_base", f"{line}_growth"):
                final[d] = final[d].model_copy(
                    update={"method": f"{final[d].method} (reference only; {line} uses {method})"}
                )

        holdout = []
        if bt and method in bt.holdouts:
            holdout = [
                HoldoutPoint(period=p.date(), actual=round(a, 2), predicted=round(float(f), 2))
                for p, a, f in bt.holdouts[method]
            ]
        models[line] = SeriesModel(
            method=method,
            label=MODELS[method].label,
            selected_by=selected_by,
            reason=reason,
            seasonality=seasonality[line],
            backtest_horizon=bt.horizon if bt and bt.origins else None,
            backtest_origins=[o.date() for o in bt.origins] if bt else [],
            scores=_scores(bt) if bt else [],
            holdout=holdout,
        )
        if (paths[line] < 0).any():
            warnings.append(f"{line} forecast went negative and was floored at zero")
            paths[line] = paths[line].clip(min=0)

    months = []
    for i, period in enumerate(periods):
        revenue, opex = float(paths["revenue"][i]), float(paths["opex"][i])
        cogs = revenue * (1 - drivers.gross_margin)
        gross_profit = revenue - cogs
        operating_income = gross_profit - opex
        months.append(
            ForecastMonth(
                period=period.date(),
                revenue=round(revenue, 2),
                cogs=round(cogs, 2),
                gross_profit=round(gross_profit, 2),
                opex=round(opex, 2),
                operating_income=round(operating_income, 2),
                other_net=round(drivers.other_net, 2),
                net_income=round(operating_income + drivers.other_net, 2),
            )
        )

    return ForecastResponse(
        engine_version=ENGINE_VERSION,
        last_actual_period=last.date(),
        history_months=len(monthly),
        drivers=final,
        formulas={**formulas, **BASE_FORMULAS},
        months=months,
        models=models,
        warnings=warnings,
    )
