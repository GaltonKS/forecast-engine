"""Data contracts for the projection engine.

Everything that crosses a boundary (CSV in, API out, LLM-parsed scenario in)
is validated by these models, so the math only ever sees clean numbers.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Category(str, Enum):
    # Income statement: amount is the month's total
    REVENUE = "revenue"
    COGS = "cogs"
    OPEX = "opex"
    OTHER = "other"  # net other income (+) / expense (-), e.g. interest
    DEPRECIATION = "depreciation"  # counted as opex; tracked as the non-cash part
    # Balance sheet: amount is the month-end balance
    CASH = "cash"
    RECEIVABLES = "receivables"
    INVENTORY = "inventory"
    FIXED_ASSETS = "fixed_assets"  # net of accumulated depreciation
    OTHER_ASSETS = "other_assets"
    PAYABLES = "payables"
    DEBT = "debt"
    OTHER_LIABILITY = "other_liability"
    EQUITY = "equity"


PL_CATEGORIES = ("revenue", "cogs", "opex", "other", "depreciation")
ASSET_CATEGORIES = ("cash", "receivables", "inventory", "fixed_assets", "other_assets")
LIABILITY_CATEGORIES = ("payables", "debt", "other_liability")
BALANCE_CATEGORIES = ASSET_CATEGORIES + LIABILITY_CATEGORIES + ("equity",)


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
    # Balance sheet drivers (used only when balances are uploaded)
    dso_days: float = Field(default=0.0, ge=0, description="Receivables in days of revenue")
    dio_days: float = Field(default=0.0, ge=0, description="Inventory in days of cogs")
    dpo_days: float = Field(default=0.0, ge=0, description="Payables in days of cash costs")
    capex_monthly: float = Field(default=0.0, ge=0, description="Monthly capital spending")
    depreciation_monthly: float = Field(default=0.0, ge=0, description="Non-cash part of monthly opex")


class DriverOverrides(BaseModel):
    """Partial drivers. This is what a scenario (or the LLM layer) fills in."""

    revenue_base: float | None = None
    revenue_growth: float | None = None
    gross_margin: float | None = Field(default=None, ge=-1, le=1)
    opex_base: float | None = Field(default=None, ge=0)
    opex_growth: float | None = None
    other_net: float | None = None
    dso_days: float | None = Field(default=None, ge=0, le=365)
    dio_days: float | None = Field(default=None, ge=0, le=730)
    dpo_days: float | None = Field(default=None, ge=0, le=365)
    capex_monthly: float | None = Field(default=None, ge=0)
    depreciation_monthly: float | None = Field(default=None, ge=0)


ForecastMethod = Literal["auto", "driver", "seasonal_naive", "ets", "arima", "prophet"]


class Adjustment(BaseModel):
    """A dated change on top of the forecast: a hire, a price rise, a lost client,
    a loan or an equity raise.

    kind "add": value is currency per month (negative to subtract).
    kind "pct": value is a fraction, 0.05 = +5%, applied to that line.
    Applies to months from start through end (inclusive; null = to the horizon).

    revenue/cogs/opex/other change the income statement. capex (equipment
    bought), debt (+ borrow, - repay) and equity (+ raised) are cash flows; they
    need an uploaded balance sheet and only take kind "add".
    """

    line: Literal["revenue", "cogs", "opex", "other", "capex", "debt", "equity"]
    kind: Literal["add", "pct"]
    value: float
    start: date
    end: date | None = None
    label: str | None = Field(default=None, max_length=200)

    @field_validator("start", "end")
    @classmethod
    def to_month_start(cls, v: date | None) -> date | None:
        return v.replace(day=1) if v else v

    @field_validator("value")
    @classmethod
    def pct_floor(cls, v: float, info):
        if info.data.get("kind") == "pct" and v <= -1:
            raise ValueError("pct adjustments must be greater than -1 (-100%)")
        if info.data.get("kind") == "pct" and info.data.get("line") in ("capex", "debt", "equity"):
            raise ValueError(f"{info.data['line']} adjustments take kind 'add' (an amount), not 'pct'")
        return v


class ForecastOptions(BaseModel):
    """Settings shared by a single forecast and a scenario comparison."""

    horizon_months: int = Field(default=12, ge=1, le=120)
    method: ForecastMethod = Field(
        default="auto",
        description="Model for revenue and opex. 'auto' backtests every model and picks "
        "the most accurate per line. A revenue/opex base or growth override always "
        "uses the driver formula for that line.",
    )
    seasonal: bool | None = Field(
        default=None, description="Force month-of-year seasonality on or off; null = detect"
    )
    simulations: int = Field(
        default=1000, ge=0, le=20000,
        description="Monte Carlo paths for P10/P50/P90 ranges; 0 = no ranges",
    )
    seed: int = Field(default=0, description="Random seed; same seed + data = same ranges")


class ForecastRequest(ForecastOptions):
    overrides: DriverOverrides = DriverOverrides()
    adjustments: list[Adjustment] = Field(default=[], max_length=50)


class Scenario(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    overrides: DriverOverrides = DriverOverrides()
    adjustments: list[Adjustment] = Field(default=[], max_length=50)


class ScenarioRequest(ForecastOptions):
    """Baseline plus up to 10 named scenarios, run on the same fitted models and
    the same random draws so differences come from the assumptions, not noise."""

    scenarios: list[Scenario] = Field(min_length=1, max_length=10)


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
    models: dict[str, SeriesModel] = {}
    adjustments: list[Adjustment] = []
    uncertainty: Uncertainty | None = None
    # Balance sheet and cash flow: present when balances were uploaded for the last actual month
    opening_balance: BalanceSheetMonth | None = None
    balance_sheet: list[BalanceSheetMonth] | None = None
    cash_flow: list[CashFlowMonth] | None = None
    cash: CashSummary | None = None
    warnings: list[str] = []


class Seasonality(BaseModel):
    detected: bool
    strength: float | None = None  # share of detrended variance explained by month of year
    p_value: float | None = None
    indices: list[float] | None = None  # Jan..Dec multipliers
    note: str


class ModelScore(BaseModel):
    method: str
    status: str  # ok | skipped | failed
    wape: float | None = None  # weighted absolute % error: sum|error| / sum|actual|
    bias: float | None = None  # sum(error) / sum|actual|; positive = over-forecast
    folds: int = 0
    note: str | None = None


class HoldoutPoint(BaseModel):
    period: date
    actual: float
    predicted: float


class SeriesModel(BaseModel):
    """How one line (revenue or opex) was forecast, and how accurate it was."""

    method: str
    label: str
    selected_by: Literal["backtest", "requested", "override", "default"]
    reason: str
    seasonality: Seasonality
    backtest_horizon: int | None = None
    backtest_origins: list[date] = []
    scores: list[ModelScore] = []
    holdout: list[HoldoutPoint] = []  # chosen model's most recent backtest window


class BalanceSheetMonth(BaseModel):
    period: date
    cash: float
    receivables: float
    inventory: float
    fixed_assets: float
    other_assets: float
    total_assets: float
    payables: float
    debt: float
    other_liability: float
    total_liabilities: float
    equity: float
    check: float  # total_assets - total_liabilities - equity; always 0


class CashFlowMonth(BaseModel):
    period: date
    net_income: float
    depreciation: float  # added back: non-cash
    change_receivables: float  # cash effect: an increase in receivables is negative
    change_inventory: float
    change_payables: float
    operating: float
    capex: float  # negative: cash spent
    investing: float
    debt: float  # + borrowed, - repaid
    equity: float  # + raised
    financing: float
    net_change: float
    ending_cash: float


class CashSummary(BaseModel):
    opening_cash: float
    ending_cash: float
    lowest_cash: float
    lowest_cash_period: date
    cash_out_period: date | None  # first month cash is negative; null = never in the horizon
    runway_months: int | None  # months until cash_out_period; null = beyond the horizon
    note: str


class Band(BaseModel):
    p10: float
    p50: float
    p90: float


class MonthBands(BaseModel):
    period: date
    revenue: Band
    gross_profit: Band
    opex: Band
    net_income: Band
    cash: Band | None = None


class Coverage(BaseModel):
    """How often the 80% range caught the actual in the backtest."""

    coverage: float  # share of backtest months inside P10-P90 before widening
    points: int
    widened_by: float  # 1.0 = ranges used as estimated; >1 = widened to reach 80%


class LineNoise(BaseModel):
    scale: Literal["log", "linear"]  # log: % noise; linear: currency noise
    monthly_sd: float  # month-to-month noise around the trend (log units or currency)
    autocorrelation: float  # how much a month's miss carries into the next
    level_sd: float  # uncertainty of the starting level
    growth_sd: float  # uncertainty of the monthly growth rate
    level_fixed: bool = False  # true when the scenario sets the base
    growth_fixed: bool = False  # true when the scenario sets the growth


class RunwayBand(BaseModel):
    """Months until cash runs out across simulated paths; null = beyond the horizon.
    p10 is the pessimistic end."""

    p10: int | None
    p50: int | None
    p90: int | None


class Uncertainty(BaseModel):
    simulations: int
    seed: int
    months: list[MonthBands]
    totals: dict[str, Band]  # summed over the horizon: revenue, gross_profit, opex, net_income
    prob_loss_month: float  # share of paths with at least one month of negative net income
    prob_total_loss: float  # share of paths with negative net income over the whole horizon
    prob_cash_out: float | None = None  # share of paths where cash goes negative in the horizon
    runway: "RunwayBand | None" = None
    noise: dict[str, LineNoise]  # revenue, opex
    margin_sd: float
    other_sd: float
    calibration: dict[str, Coverage] = {}
    method: str = (
        "Each path multiplies the point forecast by noise fitted to history: a "
        "starting-level error, a growth-rate error that compounds, and month-to-month "
        "AR(1) noise. Gross margin and other income vary around their fitted values. "
        "Lines are simulated independently. Ranges are widened, never narrowed, when "
        "the backtest shows they caught fewer than 80% of actual months."
    )


class ScenarioDelta(BaseModel):
    """Scenario minus baseline over the horizon, on paired simulation paths."""

    revenue: Band
    net_income: Band
    ending_cash: Band | None = None  # when a balance sheet is uploaded


class ScenarioResult(BaseModel):
    name: str
    forecast: ForecastResponse
    delta: ScenarioDelta | None = None  # null when simulations = 0
    delta_point: dict[str, float]  # point-forecast difference in totals


class ScenarioComparison(BaseModel):
    engine_version: str
    baseline: ForecastResponse
    scenarios: list[ScenarioResult]


ForecastResponse.model_rebuild()
ScenarioResult.model_rebuild()
