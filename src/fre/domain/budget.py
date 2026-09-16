"""Budget contracts (accounting arrives in Wave 2)."""

from enum import StrEnum

from pydantic import Field

from fre.domain.common import FrozenModel


class ReasoningTier(StrEnum):
    T0 = "T0"
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"
    T5 = "T5"


class BudgetLimits(FrozenModel):
    max_iterations: int = Field(ge=1)
    max_llm_calls: int = Field(ge=0)
    max_tool_calls: int = Field(ge=0)
    max_input_tokens: int | None = Field(default=None, ge=0)
    max_output_tokens: int | None = Field(default=None, ge=0)
    max_candidates: int = Field(ge=1)
    max_concurrent_actions: int = Field(ge=1)
    max_runtime_seconds: int | None = Field(default=None, ge=1)


class SearchPolicy(FrozenModel):
    initial_candidate_count: int = Field(ge=1)
    max_candidate_depth: int = Field(ge=1)
    max_representation_views: int = Field(ge=0)
    synthesis_order_limit: int = Field(ge=1, le=3)
    falsifier_operator_limit: int = Field(ge=0)
    independent_validation_branches: int = Field(ge=0)
    sensitivity_samples: int = Field(ge=0)


class BudgetPlan(FrozenModel):
    tier: ReasoningTier
    limits: BudgetLimits
    search: SearchPolicy
    acquisition_threshold: float
    stop_threshold: float
    policy_version: str


class BudgetRemaining(FrozenModel):
    iterations: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
