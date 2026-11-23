"""What the model must produce, and what the API returns.

LLMOutput is sent to the model as a JSON schema for constrained decoding, so
it is deliberately flat and simple: every field required (nullable where it
can be empty), dates as "YYYY-MM" strings, no validators the model can't see.
It is then converted into the engine's own Scenario models, which validate
everything again.
"""
from __future__ import annotations

import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from engine.schema import ScenarioComparison, Scenario

Month = Annotated[str, Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$", description="YYYY-MM")]


class LLMOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revenue_base: float | None = Field(description="Monthly revenue level; null = keep fitted")
    revenue_growth: float | None = Field(description="MONTHLY revenue growth, 0.02 = 2%/month")
    gross_margin: float | None = Field(description="(revenue - cogs) / revenue, 0.75 = 75%")
    opex_base: float | None = Field(description="Monthly opex level")
    opex_growth: float | None = Field(description="MONTHLY opex growth")
    other_net: float | None = Field(description="Monthly other income (+) / expense (-)")


class LLMAdjustment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    line: Literal["revenue", "cogs", "opex", "other"]
    kind: Literal["add", "pct"]
    value: float = Field(description="add: currency per month (negative to subtract); pct: 0.05 = +5%")
    start: Month
    end: Month | None = Field(description="Last month it applies, inclusive; null = to the end")
    label: str = Field(max_length=120)


class LLMScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(max_length=60)
    overrides: LLMOverrides
    adjustments: list[LLMAdjustment] = Field(max_length=20)


class LLMOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: Literal["scenarios", "baseline", "unsupported"]
    explanation: str = Field(max_length=600, description="One or two sentences: how the request was read")
    assumptions: list[str] = Field(max_length=10, description="Anything not stated in the request that you had to assume")
    horizon_months: int | None = Field(ge=1, le=120, description="Only if the request names a period")
    scenarios: list[LLMScenario] = Field(max_length=5)
    unsupported_reason: str | None = Field(max_length=300)


# ---------- API side ----------

class QueryRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    horizon_months: int | None = Field(default=None, ge=1, le=120, description="Overrides the period the model reads from the prompt")
    simulations: int = Field(default=1000, ge=0, le=20000)
    seed: int = 0
    run: bool = Field(default=True, description="false = only interpret the prompt, don't run forecasts")


class Interpretation(BaseModel):
    intent: Literal["scenarios", "baseline", "unsupported"]
    explanation: str
    assumptions: list[str]
    horizon_months: int
    scenarios: list[Scenario]
    unsupported_reason: str | None = None


class LLMCall(BaseModel):
    model: str
    attempts: int
    latency_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class QueryResult(BaseModel):
    prompt: str
    interpretation: Interpretation
    warnings: list[str] = []  # from checking the model's values, plus the engine's
    summary: list[str] = []  # plain-language results, written by code from engine numbers
    llm: LLMCall
    comparison: ScenarioComparison | None = None  # null if not run or unsupported
    forecast_ids: list[uuid.UUID] = []  # saved scenario forecasts, with the prompt
