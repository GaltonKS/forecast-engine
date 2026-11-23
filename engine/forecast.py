"""Forecast: fit drivers, pick a model per line by backtest, project forward,
then simulate P10/P50/P90 ranges and apply dated adjustments.

Revenue and opex each get the model that was most accurate on recent months
(driver formula, seasonal naive, ETS, ARIMA or Prophet). COGS follows revenue
through gross margin, and other income stays flat. Scenario overrides on a
line's base or growth switch that line to the driver formula, so "2% growth"
means exactly 2%.

Fitting and backtesting happen once in prepare(); build() then produces the
baseline or any scenario from that, so comparing scenarios costs little more
than one forecast.

Deterministic: the same data, request and seed always give the same numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import BacktestResult, backtest
from .stats import MODELS, available_models, detect_seasonality, driver_path, fit_driver, future_index
from . import statements as st
from .simulate import (
    apply_adjustments,
    adjustment_mask,
    band,
    calibrate,
    draw,
    estimate_noise,
    line_paths,
)
from .schema import (
    Adjustment,
    Coverage,
    DriverOverrides,
    Drivers,
    FittedDriver,
    ForecastMonth,
    ForecastOptions,
    ForecastRequest,
    ForecastResponse,
    HoldoutPoint,
    LineNoise,
    ModelScore,
    MonthBands,
    ScenarioComparison,
    ScenarioDelta,
    ScenarioRequest,
    ScenarioResult,
    Seasonality,
    SeriesModel,
    RunwayBand,
    Uncertainty,
)

ENGINE_VERSION = "0.4.0"
MIN_HISTORY = 3
RECOMMENDED_HISTORY = 12
LINES = ("revenue", "opex")

BALANCE_FORMULAS = {
    "receivables": "receivables[t] = revenue[t] * dso_days / 30.4",
    "inventory": "inventory[t] = cogs[t] * dio_days / 30.4",
    "payables": "payables[t] = (cogs[t] + opex[t] - depreciation_monthly) * dpo_days / 30.4",
    "fixed_assets": "fixed_assets[t] = fixed_assets[t-1] + capex[t] - depreciation_monthly",
    "debt": "debt[t] = debt[t-1] + debt adjustments[t]",
    "equity": "equity[t] = equity[t-1] + net_income[t] + equity adjustments[t]",
    "operating_cash_flow": "net_income + depreciation - change in receivables - change in inventory + change in payables",
    "cash": "cash[t] = cash[t-1] + operating - capex + debt and equity raised",
}

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


def _select(y: pd.Series, opts: ForecastOptions) -> tuple[str, str, str, BacktestResult]:
    """Returns (method, selected_by, reason, backtest)."""
    available = available_models()
    if opts.method != "auto":
        if opts.method not in available:
            raise ValueError(f"Method '{opts.method}' is not installed on this server")
        need = MODELS[opts.method].min_train
        if len(y) < need:
            raise ValueError(f"Method '{opts.method}' needs {need} months of history, have {len(y)}")

    bt = backtest(y, available, opts.seasonal)
    if opts.method != "auto":
        return opts.method, "requested", f"Requested method '{opts.method}'", bt

    best = bt.best(prefer_order=list(MODELS))
    if best is None:
        return "driver", "default", bt.note or "No model could be backtested; using the driver formula", bt
    score = next(s for s in bt.scores if s.method == best)
    reason = (
        f"Lowest backtest error over {score.folds} origins x {bt.horizon} months "
        f"(WAPE {score.wape:.1%}); simpler models win near-ties"
    )
    return best, "backtest", reason, bt


@dataclass
class LinePlan:
    method: str
    selected_by: str
    reason: str
    bt: BacktestResult | None
    stat_path: np.ndarray | None  # point forecast when the method isn't the driver formula


@dataclass
class Prepared:
    monthly: pd.DataFrame
    opts: ForecastOptions
    fitted: dict[str, FittedDriver]
    warnings: list[str]
    seasonality: dict[str, Seasonality]
    periods: pd.DatetimeIndex
    plans: dict[str, LinePlan]
    noise: dict[str, LineNoise]
    margin_sd: float
    other_sd: float
    opening: st.Opening | None = None
    balance_fitted: dict[str, FittedDriver] = field(default_factory=dict)
    _coverage: dict[tuple[str, str], Coverage | None] = field(default_factory=dict)

    def indices(self, line: str) -> list[float] | None:
        s = self.seasonality[line]
        return s.indices if s.detected else None

    def coverage(self, line: str, method: str) -> Coverage | None:
        key = (line, method)
        if key not in self._coverage:
            bt = self.plans[line].bt
            result = None
            if bt and method in bt.fold_preds:
                def seasonal_for(train):
                    p = detect_seasonality(train, self.opts.seasonal)
                    return p.indices if p.detected else None
                result = calibrate(self.monthly[line], bt.fold_preds[method], seasonal_for)
            self._coverage[key] = result
        return self._coverage[key]


def prepare(
    monthly: pd.DataFrame,
    opts: ForecastOptions,
    skip_backtest: set[str] = frozenset(),
    balances: pd.DataFrame | None = None,
) -> Prepared:
    """Everything that depends only on the data and options, not the scenario."""
    fitted, warnings, seasonality = fit_drivers(monthly, opts.seasonal)
    h = opts.horizon_months
    periods = future_index(monthly["revenue"], h)
    plans, noise = {}, {}
    for line in LINES:
        y = monthly[line]
        idx = seasonality[line].indices if seasonality[line].detected else None
        noise[line] = estimate_noise(y, idx)
        if line in skip_backtest:
            # Driver-only backtest: cheap, and it calibrates the ranges exactly as
            # a scenario comparison would, so both give the same answer.
            plans[line] = LinePlan("driver", "override", "", backtest(y, ["driver"], opts.seasonal), None)
            continue
        method, selected_by, reason, bt = _select(y, opts)
        stat_path = None
        if method != "driver":
            try:
                stat_path = np.asarray(MODELS[method].fn(y, h, opts.seasonal), dtype=float)
            except Exception as e:
                warnings.append(f"{method} failed on {line} ({e.__class__.__name__}); used the driver formula")
                method, selected_by, reason = "driver", "default", f"{reason}; fell back after a fit error"
        plans[line] = LinePlan(method, selected_by, reason, bt, stat_path)

    trail6 = monthly.tail(6)
    rev = trail6["revenue"].to_numpy(dtype=float)
    ok = rev > 0
    margins = 1 - trail6["cogs"].to_numpy(dtype=float)[ok] / rev[ok]
    margin_sd = float(margins.std(ddof=1)) if len(margins) >= 2 else 0.0
    other_sd = float(trail6["other"].std(ddof=1)) if len(trail6) >= 2 else 0.0
    opening, bwarn = st.opening_balance(monthly, balances)
    warnings.extend(bwarn)
    balance_fitted = st.fit_balance_drivers(monthly, balances, opening) if opening else {}
    return Prepared(
        monthly, opts, fitted, warnings, seasonality, periods, plans, noise, margin_sd, other_sd,
        opening=opening, balance_fitted=balance_fitted,
    )


@dataclass
class Sims:
    totals: dict[str, np.ndarray]  # per-path horizon totals, for paired scenario deltas


def build(
    prep: Prepared, overrides: DriverOverrides, adjustments: list[Adjustment]
) -> tuple[ForecastResponse, Sims | None]:
    opts, periods, h = prep.opts, prep.periods, prep.opts.horizon_months
    overridden = set(overrides.model_dump(exclude_none=True))
    warnings = list(prep.warnings)
    if prep.opening is None:
        # Balance sheet drivers and cash lines need an uploaded balance sheet.
        ignored = sorted(overridden & set(st.BALANCE_DRIVERS))
        cash_adj = [a for a in adjustments if a.line in st.CASH_LINES]
        if ignored or cash_adj:
            what = ", ".join(ignored + sorted({a.line for a in cash_adj}))
            warnings.append(f"{what} need a balance sheet (upload balances for the last actual month); ignored")
        overrides = overrides.model_copy(update={k: None for k in st.BALANCE_DRIVERS})
        overridden -= set(st.BALANCE_DRIVERS)
    final = apply_overrides({**prep.fitted, **prep.balance_fitted}, overrides)
    drivers = Drivers(**{k: v.value for k, v in final.items()})

    paths, models, formulas, widen, noise_used, calibration = {}, {}, {}, {}, {}, {}
    for line in LINES:
        plan = prep.plans[line]
        ov = overridden & {f"{line}_base", f"{line}_growth"}
        indices = prep.indices(line)
        if ov:
            method, selected_by, bt = "driver", "override", None
            reason = f"Scenario overrides {', '.join(sorted(ov))}, so the driver formula is used"
        else:
            method, selected_by, reason, bt = plan.method, plan.selected_by, plan.reason, plan.bt

        if method == "driver":
            base, growth = getattr(drivers, f"{line}_base"), getattr(drivers, f"{line}_growth")
            paths[line] = driver_path(base, growth, indices, periods)
            season = " * season[month(t)]" if indices else ""
            formulas[line] = f"{line}[t] = {line}_base * (1 + {line}_growth) ^ t{season}"
        else:
            paths[line] = plan.stat_path.copy()
            formulas[line] = f"{line}[t] = {MODELS[method].label} forecast (see models.{line})"
            for d in (f"{line}_base", f"{line}_growth"):
                final[d] = final[d].model_copy(
                    update={"method": f"{final[d].method} (reference only; {line} uses {method})"}
                )
        if (paths[line] < 0).any():
            warnings.append(f"{line} forecast went negative and was floored at zero")
            paths[line] = paths[line].clip(min=0)

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
            seasonality=prep.seasonality[line],
            backtest_horizon=bt.horizon if bt and bt.origins else None,
            backtest_origins=[o.date() for o in bt.origins] if bt else [],
            scores=_scores(bt) if bt else [],
            holdout=holdout,
        )

        cov = prep.coverage(line, method)
        if cov:
            calibration[line] = cov
        widen[line] = cov.widened_by if cov else 1.0
        noise_used[line] = prep.noise[line].model_copy(
            update={"level_fixed": f"{line}_base" in ov, "growth_fixed": f"{line}_growth" in ov}
        )

    for adj in adjustments:
        if not adjustment_mask(adj, periods).any():
            name = f"'{adj.label}'" if adj.label else f"{adj.kind} {adj.value} on {adj.line}"
            warnings.append(f"Adjustment {name} falls outside the forecast horizon and has no effect")

    gm, other_net = drivers.gross_margin, drivers.other_net
    revenue = apply_adjustments(paths["revenue"], "revenue", adjustments, periods)
    cogs = apply_adjustments(revenue * (1 - gm), "cogs", adjustments, periods)
    opex = apply_adjustments(paths["opex"], "opex", adjustments, periods)
    other = apply_adjustments(np.full(h, other_net), "other", adjustments, periods)
    months = []
    for i, period in enumerate(periods):
        gp = revenue[i] - cogs[i]
        oi = gp - opex[i]
        months.append(
            ForecastMonth(
                period=period.date(),
                revenue=round(float(revenue[i]), 2),
                cogs=round(float(cogs[i]), 2),
                gross_profit=round(float(gp), 2),
                opex=round(float(opex[i]), 2),
                operating_income=round(float(oi), 2),
                other_net=round(float(other[i]), 2),
                net_income=round(float(oi + other[i]), 2),
            )
        )
    adj_note = " + adjustments" if adjustments else ""
    formulas = {
        **formulas,
        **BASE_FORMULAS,
        "net_income": BASE_FORMULAS["net_income"] + adj_note,
    }
    if adjustments:
        formulas["adjustments"] = "Each adjustment adds (currency/month) or scales (pct) its line from start through end"

    opening_row = balance_sheet = cash_flow = cash_summary = None
    if prep.opening:
        stm = st.project(prep.opening, drivers, revenue, cogs, opex, other, adjustments, periods)
        opening_row, balance_sheet, cash_flow, cash_summary = st.to_models(prep.opening, stm, periods)
        formulas.update(BALANCE_FORMULAS)
        if any(r.fixed_assets < 0 for r in balance_sheet):
            warnings.append("Fixed assets go negative: depreciation exceeds the asset base plus capex")

    uncertainty, sims = None, None
    if opts.simulations > 0:
        n = opts.simulations
        d = draw(n, h, opts.seed)
        rev_p = line_paths(paths["revenue"], noise_used["revenue"], d, "revenue", widen["revenue"]).clip(min=0)
        opex_p = line_paths(paths["opex"], noise_used["opex"], d, "opex", widen["opex"]).clip(min=0)
        shift_sd = 0.0 if "gross_margin" in overridden else prep.margin_sd / np.sqrt(6)
        gm_p = np.clip(gm + shift_sd * d.margin_shift + prep.margin_sd * d.margin_monthly, -1, 1)
        rev_p = apply_adjustments(rev_p, "revenue", adjustments, periods)
        cogs_p = apply_adjustments(rev_p * (1 - gm_p), "cogs", adjustments, periods)
        opex_p = apply_adjustments(opex_p, "opex", adjustments, periods)
        other_p = apply_adjustments(other_net + prep.other_sd * d.other, "other", adjustments, periods)
        gp_p = rev_p - cogs_p
        ni_p = gp_p - opex_p + other_p

        for line, cov in calibration.items():
            if cov.coverage < 0.6:
                warnings.append(
                    f"{line} ranges caught only {cov.coverage:.0%} of backtest months (target 80%); "
                    f"widened x{cov.widened_by:.2f}"
                )
        rb, gb, ob, nb = band(rev_p), band(gp_p), band(opex_p), band(ni_p)
        totals = {"revenue": rev_p.sum(1), "gross_profit": gp_p.sum(1), "opex": opex_p.sum(1), "net_income": ni_p.sum(1)}
        cb, prob_cash_out, runway_band, ending_cash = [None] * h, None, None, None
        if prep.opening:
            cash_p = st.project(prep.opening, drivers, rev_p, cogs_p, opex_p, other_p, adjustments, periods)["cash"]
            cb = band(cash_p)
            months_left = st.runway(cash_p, prep.opening.values["cash"])
            prob_cash_out = round(float(np.isfinite(months_left).mean()), 4)
            q = np.quantile(months_left, [0.1, 0.5, 0.9], method="inverted_cdf")
            runway_band = RunwayBand(**{k: (int(v) if np.isfinite(v) else None) for k, v in zip(("p10", "p50", "p90"), q)})
            ending_cash = cash_p[:, -1]
        uncertainty = Uncertainty(
            simulations=n,
            seed=opts.seed,
            months=[
                MonthBands(period=p.date(), revenue=rb[i], gross_profit=gb[i], opex=ob[i], net_income=nb[i], cash=cb[i])
                for i, p in enumerate(periods)
            ],
            prob_cash_out=prob_cash_out,
            runway=runway_band,
            totals={k: band(v) for k, v in totals.items()},
            prob_loss_month=round(float((ni_p < 0).any(axis=1).mean()), 4),
            prob_total_loss=round(float((totals["net_income"] < 0).mean()), 4),
            noise=noise_used,
            margin_sd=round(prep.margin_sd, 6),
            other_sd=round(prep.other_sd, 2),
            calibration=calibration,
        )
        if ending_cash is not None:
            totals["ending_cash"] = ending_cash
        sims = Sims(totals)

    resp = ForecastResponse(
        engine_version=ENGINE_VERSION,
        last_actual_period=prep.monthly.index.max().date(),
        history_months=len(prep.monthly),
        drivers=final,
        formulas=formulas,
        months=months,
        models=models,
        adjustments=adjustments,
        uncertainty=uncertainty,
        opening_balance=opening_row,
        balance_sheet=balance_sheet,
        cash_flow=cash_flow,
        cash=cash_summary,
        warnings=warnings,
    )
    return resp, sims


def run_forecast(
    monthly: pd.DataFrame, request: ForecastRequest, balances: pd.DataFrame | None = None
) -> ForecastResponse:
    overridden = set(request.overrides.model_dump(exclude_none=True))
    # A line whose base or growth is overridden uses the driver formula, so
    # skip its backtest; scenario runs stay fast.
    skip = {line for line in LINES if overridden & {f"{line}_base", f"{line}_growth"}}
    resp, _ = build(prepare(monthly, request, skip, balances), request.overrides, request.adjustments)
    return resp


def _totals(resp: ForecastResponse, key: str) -> float:
    return sum(getattr(m, key) for m in resp.months)


def run_scenarios(
    monthly: pd.DataFrame, request: ScenarioRequest, balances: pd.DataFrame | None = None
) -> ScenarioComparison:
    prep = prepare(monthly, request, balances=balances)
    baseline, base_sims = build(prep, DriverOverrides(), [])
    results = []
    for sc in request.scenarios:
        resp, sims = build(prep, sc.overrides, sc.adjustments)
        delta = None
        if sims and base_sims:
            delta = ScenarioDelta(
                revenue=band(sims.totals["revenue"] - base_sims.totals["revenue"]),
                net_income=band(sims.totals["net_income"] - base_sims.totals["net_income"]),
                ending_cash=band(sims.totals["ending_cash"] - base_sims.totals["ending_cash"])
                if "ending_cash" in sims.totals else None,
            )
        delta_point = {
            k: round(_totals(resp, k) - _totals(baseline, k), 2)
            for k in ("revenue", "gross_profit", "opex", "net_income")
        }
        if resp.cash and baseline.cash:
            delta_point["ending_cash"] = round(resp.cash.ending_cash - baseline.cash.ending_cash, 2)
        results.append(ScenarioResult(name=sc.name, forecast=resp, delta=delta, delta_point=delta_point))
    return ScenarioComparison(engine_version=ENGINE_VERSION, baseline=baseline, scenarios=results)
