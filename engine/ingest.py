"""Load raw line items and normalize them into one monthly summary table."""
from __future__ import annotations

from pathlib import Path
from typing import IO

import pandas as pd
from pydantic import ValidationError

from .schema import Category, LineItem

REQUIRED_COLUMNS = {"period", "account", "category", "amount"}


class IngestError(ValueError):
    pass


def load_csv(source: str | Path | IO) -> list[LineItem]:
    """Read a CSV with columns: period, account, category, amount.

    period: any date in the month (YYYY-MM or YYYY-MM-DD)
    category: revenue | cogs | opex | other
    amount: positive numbers for revenue/cogs/opex; signed for other
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


def to_monthly(items: list[LineItem]) -> pd.DataFrame:
    """Pivot line items to one row per month with a column per category.

    Months with no data inside the range are filled with 0 and reported,
    because a silent gap would distort growth fitting.
    """
    if not items:
        raise IngestError("No data")
    df = pd.DataFrame([i.model_dump() for i in items])
    df["category"] = df["category"].map(lambda c: c.value if isinstance(c, Category) else c)
    monthly = df.pivot_table(
        index="period", columns="category", values="amount", aggfunc="sum", fill_value=0.0
    )
    for c in Category:
        if c.value not in monthly.columns:
            monthly[c.value] = 0.0
    monthly.index = pd.to_datetime(monthly.index)
    full = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    gaps = full.difference(monthly.index)
    monthly = monthly.reindex(full, fill_value=0.0)
    monthly.index.name = "period"
    monthly.attrs["gaps"] = [d.date().isoformat() for d in gaps]
    return monthly[[c.value for c in Category]].sort_index()
