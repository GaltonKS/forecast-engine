"""Rolling-origin backtest and model selection.

For each origin, every model is fitted on the months before it and asked to
predict the next few months, which we already know. The model with the
lowest error wins. All models are scored on the same origins so the
comparison is fair.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .stats import MODELS

N_FOLDS = 3
MIN_TRAIN = 12
# A more complex model has to beat the simplest one by more than this to be
# picked: relative 10% of the best error plus 0.2 percentage points.
TIE_REL = 0.10
TIE_ABS = 0.002


@dataclass
class Score:
    method: str
    status: str  # ok | skipped | failed
    wape: float | None = None  # sum|error| / sum|actual|
    bias: float | None = None  # sum(error) / sum|actual|, + means over-forecast
    folds: int = 0
    note: str | None = None


@dataclass
class BacktestResult:
    horizon: int
    origins: list[pd.Timestamp]
    scores: list[Score]
    holdouts: dict[str, list[tuple[pd.Timestamp, float, float]]] = field(default_factory=dict)
    # every fold's predictions per model: (origin position, predicted values)
    fold_preds: dict[str, list[tuple[int, np.ndarray]]] = field(default_factory=dict)
    note: str | None = None

    def best(self, prefer_order: list[str]) -> str | None:
        ok = [s for s in self.scores if s.status == "ok"]
        if not ok:
            return None
        best = min(s.wape for s in ok)
        limit = best * (1 + TIE_REL) + TIE_ABS
        by_name = {s.method: s for s in ok}
        for name in prefer_order:
            if name in by_name and by_name[name].wape <= limit:
                return name
        return min(ok, key=lambda s: s.wape).method


def plan_origins(n: int) -> tuple[int, list[int]]:
    """Backtest horizon and origin positions (index of first predicted month)."""
    horizon = 6 if n >= 24 else 3
    step = max(1, horizon // 2)
    origins = [n - horizon - i * step for i in range(N_FOLDS)]
    return horizon, sorted(o for o in origins if o >= MIN_TRAIN)


def backtest(y: pd.Series, methods: list[str], seasonal: bool | None = None) -> BacktestResult:
    n = len(y)
    horizon, origins = plan_origins(n)
    if not origins:
        return BacktestResult(
            horizon, [], [Score(m, "skipped", note="not enough history") for m in methods],
            note=f"Backtesting needs at least {MIN_TRAIN + 3} months of history, have {n}",
        )

    scores, holdouts, fold_preds = [], {}, {}
    for name in methods:
        spec = MODELS[name]
        if origins[0] < spec.min_train:
            scores.append(Score(name, "skipped", note=f"needs {spec.min_train} months before the first origin"))
            continue
        errors, actuals, last_fold, preds = [], [], [], []
        try:
            for o in origins:
                train, test = y.iloc[:o], y.iloc[o:o + horizon]
                pred = spec.fn(train, len(test), seasonal)
                if not np.all(np.isfinite(pred)):
                    raise RuntimeError("non-finite forecast")
                errors.append(pred - test.to_numpy(dtype=float))
                actuals.append(test.to_numpy(dtype=float))
                last_fold = list(zip(test.index, test.to_numpy(dtype=float), pred))
                preds.append((o, np.asarray(pred, dtype=float)))
        except Exception as e:  # a model that can't fit this series just drops out
            scores.append(Score(name, "failed", note=f"{e.__class__.__name__}: {e}"[:200]))
            continue
        err = np.concatenate(errors)
        denom = np.abs(np.concatenate(actuals)).sum()
        if denom == 0:
            scores.append(Score(name, "skipped", note="actuals are all zero"))
            continue
        scores.append(Score(name, "ok", float(np.abs(err).sum() / denom), float(err.sum() / denom), len(origins)))
        holdouts[name] = last_fold
        fold_preds[name] = preds

    return BacktestResult(horizon, [y.index[o] for o in origins], scores, holdouts, fold_preds)
