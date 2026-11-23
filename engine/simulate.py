"""Monte Carlo ranges around the point forecast, and dated adjustments.

Each simulated path multiplies a line's point forecast by noise fitted to its
own history (log scale, so the noise is a percentage):

    log(path[t] / point[t]) = level_err + growth_err * t + e[t]
    e[t] = phi * e[t-1] + sqrt(1 - phi^2) * sigma * z[t]   (AR(1), marginal sd sigma)

level_err and growth_err are drawn once per path, so they compound; e[t] is the
month-to-month wobble. The median path is the point forecast. All random draws
happen in a fixed order whatever the assumptions are, so a baseline and its
scenarios with the same seed see the same "luck" and their differences come
from the assumptions alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import Adjustment, Band, Coverage, LineNoise
from .stats import season_factors

Z80 = 1.2815515655446004  # standard normal 90th percentile: P10-P90 is +-Z80 sd
NOISE_WINDOW = 12
MIN_CALIBRATION_POINTS = 9


# ---------- noise estimation ----------

def estimate_noise(y: pd.Series, indices: list[float] | None) -> LineNoise:
    """Fit the noise model to the last 12 months around a log-linear trend."""
    w = y.tail(NOISE_WINDOW)
    ds = (w / season_factors(indices, w.index)).to_numpy(dtype=float)
    log = bool((ds > 0).all())
    v = np.log(ds) if log else ds
    n = len(v)
    t = np.arange(n)
    resid = v - np.polyval(np.polyfit(t, v, 1), t) if n >= 3 else v - v.mean()
    sigma = float(np.sqrt((resid ** 2).sum() / max(n - 2, 1)))
    phi = 0.0
    if n >= 6 and resid[:-1].std() > 0 and resid[1:].std() > 0:
        phi = float(np.clip(np.corrcoef(resid[:-1], resid[1:])[0, 1], 0.0, 0.9))
    inflate = np.sqrt((1 + phi) / (1 - phi))  # autocorrelated noise carries less information
    sxx = float(((t - t.mean()) ** 2).sum()) or 1.0
    return LineNoise(
        scale="log" if log else "linear",
        monthly_sd=sigma,
        autocorrelation=phi,
        level_sd=float(sigma / np.sqrt(3) * inflate),  # base averages 3 months
        growth_sd=float(sigma / np.sqrt(sxx) * inflate),
    )


def horizon_sd(noise: LineNoise, h: int) -> np.ndarray:
    """Analytic sd of the log (or linear) error at horizons 1..h."""
    k = np.arange(1, h + 1)
    lvl = 0.0 if noise.level_fixed else noise.level_sd
    gr = 0.0 if noise.growth_fixed else noise.growth_sd
    return np.sqrt(lvl ** 2 + (gr * k) ** 2 + noise.monthly_sd ** 2)


def calibrate(
    y: pd.Series, fold_preds: list[tuple[int, np.ndarray]], seasonal_for
) -> Coverage | None:
    """Check the 80% range against backtest folds; return how much to widen it.

    seasonal_for(train) gives the seasonal indices that would have been used.
    """
    z = []
    for origin, pred in fold_preds:
        train = y.iloc[:origin]
        actual = y.iloc[origin:origin + len(pred)].to_numpy(dtype=float)
        noise = estimate_noise(train, seasonal_for(train))
        sd = horizon_sd(noise, len(pred))
        if noise.scale == "log":
            ok = (actual > 0) & (pred > 0)
            err = np.log(np.where(ok, actual, 1) / np.where(ok, pred, 1))
        else:
            ok = np.ones(len(pred), dtype=bool)
            err = actual - pred
        ok &= sd > 0
        z.extend((np.abs(err[ok]) / sd[ok]).tolist())
    if len(z) < MIN_CALIBRATION_POINTS:
        return None
    z = np.array(z)
    coverage = float((z <= Z80).mean())
    widen = float(max(1.0, np.quantile(z, 0.8) / Z80))
    return Coverage(coverage=round(coverage, 4), points=len(z), widened_by=round(widen, 3))


# ---------- random draws ----------

@dataclass
class Draws:
    """Standard normals for every random quantity, drawn in a fixed order."""

    level: dict[str, np.ndarray]
    growth: dict[str, np.ndarray]
    monthly: dict[str, np.ndarray]
    margin_shift: np.ndarray
    margin_monthly: np.ndarray
    other: np.ndarray


def draw(n: int, h: int, seed: int) -> Draws:
    rng = np.random.default_rng(seed)
    level, growth, monthly = {}, {}, {}
    for line in ("revenue", "opex"):
        level[line] = rng.standard_normal((n, 1))
        growth[line] = rng.standard_normal((n, 1))
        monthly[line] = rng.standard_normal((n, h))
    return Draws(
        level, growth, monthly,
        margin_shift=rng.standard_normal((n, 1)),
        margin_monthly=rng.standard_normal((n, h)),
        other=rng.standard_normal((n, h)),
    )


def line_paths(point: np.ndarray, noise: LineNoise, d: Draws, line: str, widen: float) -> np.ndarray:
    h = len(point)
    k = np.arange(1, h + 1)
    lvl = 0.0 if noise.level_fixed else noise.level_sd
    gr = 0.0 if noise.growth_fixed else noise.growth_sd
    s, phi = noise.monthly_sd, noise.autocorrelation
    z = d.monthly[line]
    e = np.empty_like(z)
    e[:, 0] = s * z[:, 0]
    scale = s * np.sqrt(1 - phi ** 2)
    for i in range(1, h):
        e[:, i] = phi * e[:, i - 1] + scale * z[:, i]
    total = widen * (lvl * d.level[line] + gr * d.growth[line] * k + e)
    if noise.scale == "log":
        return point[None, :] * np.exp(total)
    return point[None, :] + total


# ---------- adjustments ----------

def adjustment_mask(adj: Adjustment, periods: pd.DatetimeIndex) -> np.ndarray:
    start = pd.Timestamp(adj.start)
    mask = periods >= start
    if adj.end:
        mask &= periods <= pd.Timestamp(adj.end)
    return np.asarray(mask)


def apply_adjustments(values: np.ndarray, line: str, adjustments: list[Adjustment], periods) -> np.ndarray:
    """values: shape (h,) or (n, h). Adjustments apply in the order given."""
    out = values.astype(float).copy()
    for adj in adjustments:
        if adj.line != line:
            continue
        m = adjustment_mask(adj, periods)
        if adj.kind == "pct":
            out[..., m] *= 1 + adj.value
        else:
            out[..., m] += adj.value
    return out


# ---------- summaries ----------

def band(x: np.ndarray, axis=0) -> list[Band] | Band:
    q = np.percentile(x, [10, 50, 90], axis=axis)
    if q.ndim == 1:
        return Band(p10=round(float(q[0]), 2), p50=round(float(q[1]), 2), p90=round(float(q[2]), 2))
    return [Band(p10=round(float(a), 2), p50=round(float(b), 2), p90=round(float(c), 2)) for a, b, c in q.T]
