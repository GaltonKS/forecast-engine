"""Seasonality detection and the forecasting models for one monthly series.

Every model has the same shape: take a training series (monthly, indexed by
month start) and a horizon, return that many future values. That's what lets
the backtest treat them interchangeably.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as sps

SEASON_MIN_MONTHS = 24  # two full years before seasonality is estimated
SEASON_P_VALUE = 0.01  # significance for the month-of-year effect
SEASON_MIN_AMPLITUDE = 0.03  # ignore month-of-year swings under 3%


# ---------- seasonality ----------

@dataclass
class SeasonalProfile:
    detected: bool
    strength: float | None  # share of detrended variance explained by month of year
    p_value: float | None
    indices: list[float] | None  # Jan..Dec multipliers, geometric mean 1
    note: str


def detect_seasonality(y: pd.Series, force: bool | None = None) -> SeasonalProfile:
    """Find a multiplicative month-of-year pattern.

    Detrends log(y) with a straight line, then runs a one-way ANOVA of the
    residuals grouped by calendar month. An F-test (rather than a raw
    variance ratio) matters here: with two years of data each month has only
    two points, and pure noise would otherwise look "seasonal".
    """
    if force is False:
        return SeasonalProfile(False, None, None, None, "Seasonality turned off by request")
    if len(y) < SEASON_MIN_MONTHS:
        note = f"Needs {SEASON_MIN_MONTHS} months to estimate seasonality, have {len(y)}"
        return SeasonalProfile(False, None, None, None, note)
    if (y <= 0).any():
        return SeasonalProfile(False, None, None, None, "Series has zero or negative months")

    ly = np.log(y.to_numpy(dtype=float))
    t = np.arange(len(ly))
    resid = ly - np.polyval(np.polyfit(t, ly, 1), t)
    months = y.index.month.to_numpy()
    groups = [resid[months == m] for m in range(1, 13)]

    total_var = resid.var()
    if total_var < 1e-12:
        return SeasonalProfile(False, 0.0, 1.0, None, "No variation around the trend")

    means = np.array([g.mean() for g in groups])
    means -= means.mean()
    strength = float(max(0.0, 1 - (resid - means[months - 1]).var() / total_var))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = float(sps.f_oneway(*groups).pvalue)
    indices = [round(float(v), 4) for v in np.exp(means)]
    amplitude = max(indices) / min(indices) - 1

    if force is True:
        return SeasonalProfile(True, strength, p, indices, "Seasonality turned on by request")
    if p < SEASON_P_VALUE and amplitude >= SEASON_MIN_AMPLITUDE:
        note = f"Month-of-year effect is significant (p={p:.3g}), peak-to-trough {amplitude:.0%}"
        return SeasonalProfile(True, strength, p, indices, note)
    note = f"No significant month-of-year effect (p={p:.2g})"
    return SeasonalProfile(False, strength, p, None, note)


def season_factors(indices: list[float] | None, periods: pd.DatetimeIndex) -> np.ndarray:
    if not indices:
        return np.ones(len(periods))
    return np.array([indices[d.month - 1] for d in periods])


def future_index(y: pd.Series, h: int) -> pd.DatetimeIndex:
    return pd.date_range(y.index[-1] + pd.DateOffset(months=1), periods=h, freq="MS")


# ---------- driver model (step 1 math, now seasonality-aware) ----------

def log_linear_growth(series: pd.Series) -> float:
    """Monthly compound growth from a least-squares fit of log(value) on time."""
    s = series[series > 0]
    if len(s) < 2:
        return 0.0
    t = np.arange(len(s))
    slope = np.polyfit(t, np.log(s.to_numpy(dtype=float)), 1)[0]
    return float(np.exp(slope) - 1)


@dataclass
class DriverFit:
    base: float  # deseasonalized level as of the last actual month
    growth: float  # monthly, log-linear over the last 12 months
    indices: list[float] | None


def fit_driver(y: pd.Series, profile: SeasonalProfile) -> DriverFit:
    """Base is the last 3 months averaged after rolling each forward to the
    last month at the fitted growth rate. A plain 3-month average sits one
    month behind, which made every forecast start one month of growth low."""
    idx = profile.indices if profile.detected else None
    ds = y / season_factors(idx, y.index)
    growth = log_linear_growth(ds.tail(12))
    tail = ds.tail(3).to_numpy(dtype=float)
    rolled = tail * (1 + growth) ** np.arange(len(tail) - 1, -1, -1)
    return DriverFit(base=float(rolled.mean()), growth=growth, indices=idx)


def driver_path(base: float, growth: float, indices: list[float] | None, periods: pd.DatetimeIndex) -> np.ndarray:
    t = np.arange(1, len(periods) + 1)
    return base * (1 + growth) ** t * season_factors(indices, periods)


def _driver(y: pd.Series, h: int, seasonal: bool | None) -> np.ndarray:
    fit = fit_driver(y, detect_seasonality(y, seasonal))
    return driver_path(fit.base, fit.growth, fit.indices, future_index(y, h))


# ---------- statistical models ----------

def _seasonal_naive(y: pd.Series, h: int, seasonal: bool | None) -> np.ndarray:
    """Same month last year. The benchmark every seasonal model has to beat."""
    last_year = y.to_numpy(dtype=float)[-12:]
    return np.array([last_year[i % 12] for i in range(h)])


def _use_seasonal(y: pd.Series, seasonal: bool | None) -> bool:
    return detect_seasonality(y, seasonal).detected


def _ets(y: pd.Series, h: int, seasonal: bool | None) -> np.ndarray:
    """Holt-Winters exponential smoothing with a damped trend."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    positive = bool((y > 0).all())
    seas = _use_seasonal(y, seasonal)
    model = ExponentialSmoothing(
        y.to_numpy(dtype=float),
        trend="add",
        damped_trend=True,
        seasonal=("mul" if positive else "add") if seas else None,
        seasonal_periods=12 if seas else None,
        initialization_method="estimated",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(model.fit(optimized=True).forecast(h), dtype=float)


ARIMA_ORDERS = [(0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 0)]


def _arima(y: pd.Series, h: int, seasonal: bool | None) -> np.ndarray:
    """ARIMA on log values with drift; small order grid picked by AIC.

    Seasonal fits add a (0,1,1)x12 term, the "airline" structure.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    positive = bool((y > 0).all())
    vals = np.log(y.to_numpy(dtype=float)) if positive else y.to_numpy(dtype=float)
    seas = _use_seasonal(y, seasonal)
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in ARIMA_ORDERS:
            try:
                res = SARIMAX(
                    vals,
                    order=order,
                    seasonal_order=(0, 1, 1, 12) if seas else (0, 0, 0, 0),
                    trend=None if seas else "t",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if np.isfinite(res.aic) and (best is None or res.aic < best.aic):
                best = res
    if best is None:
        raise RuntimeError("no ARIMA order converged")
    fc = np.asarray(best.forecast(h), dtype=float)
    return np.exp(fc) if positive else fc


def _quiet_prophet() -> None:
    # cmdstanpy resets its own log level on first fit, so filter instead.
    logging.getLogger("cmdstanpy").addFilter(lambda r: r.levelno >= logging.WARNING)
    logging.getLogger("prophet.plot").disabled = True  # "plotly not installed" notice


_quiet_prophet()


def prophet_available() -> bool:
    try:
        import prophet  # noqa: F401
    except ImportError:
        return False
    return True


def _prophet(y: pd.Series, h: int, seasonal: bool | None) -> np.ndarray:
    from prophet import Prophet

    seas = _use_seasonal(y, seasonal)
    m = Prophet(
        growth="linear",
        yearly_seasonality=4 if seas else False,  # low Fourier order: monthly data is short
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        uncertainty_samples=0,
    )
    m.fit(pd.DataFrame({"ds": y.index, "y": y.to_numpy(dtype=float)}))
    future = pd.DataFrame({"ds": future_index(y, h)})
    return m.predict(future)["yhat"].to_numpy(dtype=float)


# ---------- registry ----------

Forecaster = Callable[[pd.Series, int, "bool | None"], np.ndarray]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    min_train: int
    fn: Forecaster


# Simplest first: on a near-tie in the backtest, the earlier one wins.
MODELS: dict[str, ModelSpec] = {
    m.name: m
    for m in [
        ModelSpec("driver", "Driver model: trend growth x seasonal index", 3, _driver),
        ModelSpec("seasonal_naive", "Seasonal naive: same month last year", 12, _seasonal_naive),
        ModelSpec("ets", "Holt-Winters exponential smoothing (damped trend)", 12, _ets),
        ModelSpec("arima", "ARIMA on log values, order picked by AIC", 12, _arima),
        ModelSpec("prophet", "Prophet (piecewise linear trend + yearly seasonality)", 12, _prophet),
    ]
}


def available_models() -> list[str]:
    return [n for n in MODELS if n != "prophet" or prophet_available()]
