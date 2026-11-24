"""Load raw line items and normalize them into one monthly summary table."""
from __future__ import annotations

from pathlib import Path
from typing import IO

import pandas as pd
from pydantic import ValidationError

from .schema import BALANCE_CATEGORIES, PL_CATEGORIES, Category, LineItem

REQUIRED_COLUMNS = {"period", "account", "category", "amount"}


class IngestError(ValueError):
    pass


def load_csv(source: str | Path | IO) -> list[LineItem]:
    """Read a CSV with columns: period, account, category, amount.

    period: any date in the month (YYYY-MM or YYYY-MM-DD)
    category: income statement: revenue | cogs | opex | other | depreciation
              balance sheet: cash | receivables | inventory | fixed_assets |
              other_assets | payables | debt | other_liability | equity
    amount: income statement: the month's total (positive; signed for other)
            balance sheet: the month-end balance
    """
    try:
        df = pd.read_csv(source)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        raise IngestError(f"Could not read CSV: {e}") from e
    df.columns = [c.strip().lower() for c in df.columns]
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise IngestError(f"Missing columns: {sorted(missing)}")

    df["period"] = pd.to_datetime(df["period"].astype(str), errors="coerce")
    df["category"] = df["category"].astype(str).str.strip().str.lower()

    items, errors = [], []
    for i, row in df.iterrows():
        try:
            items.append(
                LineItem(
                    period=row["period"].date(),
                    account=str(row["account"]).strip(),
                    category=row["category"],
                    amount=float(row["amount"]),
                )
            )
        except (ValidationError, ValueError, AttributeError) as e:
            errors.append(f"row {i + 2}: {e.__class__.__name__}: {str(e).splitlines()[0]}")
    if errors:
        raise IngestError("Invalid rows:\n" + "\n".join(errors[:20]))
    return items


def _frame(items: list[LineItem]) -> pd.DataFrame:
    df = pd.DataFrame([i.model_dump() for i in items])
    df["category"] = df["category"].map(lambda c: c.value if isinstance(c, Category) else c)
    return df


def to_monthly(items: list[LineItem]) -> pd.DataFrame:
    """Pivot income statement items to one row per month with a column per line.

    Columns: revenue, cogs, opex, other, depreciation. Depreciation is included
    in opex (it is an operating expense) and also given on its own, because
    it's the non-cash part the cash flow statement adds back. Balance sheet
    items are ignored here; see to_balances.

    Months with no data inside the range are filled with 0 and reported,
    because a silent gap would distort growth fitting.
    """
    pl = [i for i in items if i.category.value in PL_CATEGORIES] if items else []
    if not pl:
        raise IngestError("No income statement data" if items else "No data")
    df = _frame(pl)
    monthly = df.pivot_table(
        index="period", columns="category", values="amount", aggfunc="sum", fill_value=0.0
    )
    for c in PL_CATEGORIES:
        if c not in monthly.columns:
            monthly[c] = 0.0
    monthly["opex"] = monthly["opex"] + monthly["depreciation"]
    monthly.index = pd.to_datetime(monthly.index)
    full = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    gaps = full.difference(monthly.index)
    monthly = monthly.reindex(full, fill_value=0.0)
    monthly.index.name = "period"
    monthly.attrs["gaps"] = [d.date().isoformat() for d in gaps]
    return monthly[list(PL_CATEGORIES)].sort_index()


def to_balances(items: list[LineItem]) -> pd.DataFrame | None:
    """Pivot balance sheet items to one row per month-end with a column per
    category. Only months that have balance rows are included; within such a
    month, a category with no rows is 0 (e.g. no inventory). None if there are
    no balance rows at all."""
    bs = [i for i in items if i.category.value in BALANCE_CATEGORIES]
    if not bs:
        return None
    df = _frame(bs)
    bal = df.pivot_table(index="period", columns="category", values="amount", aggfunc="sum", fill_value=0.0)
    for c in BALANCE_CATEGORIES:
        if c not in bal.columns:
            bal[c] = 0.0
    bal.index = pd.to_datetime(bal.index)
    bal.index.name = "period"
    return bal[list(BALANCE_CATEGORIES)].sort_index()
