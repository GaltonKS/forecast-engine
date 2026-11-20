"""Data contracts for the projection engine.

Everything that crosses a boundary (CSV in, API out, LLM-parsed scenario in)
is validated by these models, so the math only ever sees clean numbers.
"""
from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Category(str, Enum):
    REVENUE = "revenue"
    COGS = "cogs"
    OPEX = "opex"
    OTHER = "other"  # net other income (+) / expense (-), e.g. interest


class LineItem(BaseModel):
    """One row of raw financial data: an account's amount for one month."""

    period: date
    account: str = Field(min_length=1)
    category: Category
    amount: float

    @field_validator("period")
    @classmethod
    def to_month_start(cls, v: date) -> date:
        return v.replace(day=1)


class Drivers(BaseModel):
    """The assumptions that drive the model. All growth rates are monthly."""

    revenue_base: float = Field(description="Starting monthly revenue level")
    revenue_growth: float = Field(description="Monthly revenue growth, 0.03 = 3%")
    gross_margin: float = Field(ge=-1, le=1, description="(revenue - cogs) / revenue")
    opex_base: float = Field(ge=0, description="Starting monthly operating expenses")
    opex_growth: float = Field(description="Monthly opex growth")
    other_net: float = Field(default=0.0, description="Flat monthly other income/expense")


class DriverOverrides(BaseModel):
    """Partial drivers. This is what a scenario (or the LLM layer) fills in."""

    revenue_base: float | None = None
    revenue_growth: float | None = None
    gross_margin: float | None = Field(default=None, ge=-1, le=1)
    opex_base: float | None = Field(default=None, ge=0)
    opex_growth: float | None = None
    other_net: float | None = None


class ForecastRequest(BaseModel):
    horizon_months: int = Field(default=12, ge=1, le=120)
    overrides: DriverOverrides = DriverOverrides()


class ForecastMonth(BaseModel):
    period: date
    revenue: float
    cogs: float
    gross_profit: float
    opex: float
    operating_income: float
    other_net: float
    net_income: float


class FittedDriver(BaseModel):
    value: float
    method: str  # human-readable explanation of how it was derived
    overridden: bool = False


class ForecastResponse(BaseModel):
    engine_version: str
    last_actual_period: date
    history_months: int
    drivers: dict[str, FittedDriver]
    formulas: dict[str, str]
    months: list[ForecastMonth]
    warnings: list[str] = []
