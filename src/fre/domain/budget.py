"""Deterministic allocation and executable budget contracts."""

from enum import StrEnum

from pydantic import Field, model_validator

from fre.domain.common import FrozenModel


class ReasoningTier(StrEnum):
    T0 = "T0"
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"
    T5 = "T5"


RESOURCE_NAMES = (
    "iterations",
    "llm_calls",
    "tool_calls",
    "input_tokens",
    "output_tokens",
    "candidates",
    "runtime_seconds",
)


class ResourceVector(FrozenModel):
    iterations: int = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    candidates: int = Field(default=0, ge=0)
    runtime_seconds: int = Field(default=0, ge=0)


class BudgetLimits(FrozenModel):
    max_iterations: int = Field(ge=1)
    max_llm_calls: int = Field(ge=0)
    max_tool_calls: int = Field(ge=0)
    max_input_tokens: int = Field(ge=0)
    max_output_tokens: int = Field(ge=0)
    max_candidates: int = Field(ge=1)
    max_concurrent_actions: int = Field(ge=1)
    max_runtime_seconds: int = Field(ge=1)

    def resources(self) -> ResourceVector:
        return ResourceVector(**{name: getattr(self, f"max_{name}") for name in RESOURCE_NAMES})


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


class TierDefinition(FrozenModel):
    limits: BudgetLimits
    search: SearchPolicy
    acquisition_threshold: float = Field(ge=0)
    stop_threshold: float = 0.0


class TierPolicy(FrozenModel):
    policy_version: str
    tiers: dict[ReasoningTier, TierDefinition]
    consequence_floors: dict[str, ReasoningTier]
    irreversibility_floors: dict[str, ReasoningTier]
    ambiguity_floors: dict[str, ReasoningTier]
    evidence_scarcity_floors: dict[str, ReasoningTier]
    search_space_floors: dict[str, ReasoningTier]

    @model_validator(mode="after")
    def all_tiers_present(self) -> "TierPolicy":
        if set(self.tiers) != set(ReasoningTier):
            raise ValueError("tier policy must define T0 through T5")
        return self


class DeploymentLimits(FrozenModel):
    maximum_tier: ReasoningTier = ReasoningTier.T5
    resource_ceilings: BudgetLimits | None = None


class BudgetReservation(FrozenModel):
    reservation_id: str
    action_id: str
    resources: ResourceVector


class BudgetProjection(FrozenModel):
    plan: BudgetPlan | None = None
    policy_hash: str | None = None
    committed: ResourceVector = ResourceVector()
    reservations: tuple[BudgetReservation, ...] = ()
    usage_events: tuple[ResourceVector, ...] = ()


class BudgetRemaining(FrozenModel):
    resources: ResourceVector
    active_concurrent_actions: int = Field(ge=0)
    projection_hash: str

    @property
    def iterations(self) -> int:
        return self.resources.iterations

    @property
    def llm_calls(self) -> int:
        return self.resources.llm_calls

    @property
    def tool_calls(self) -> int:
        return self.resources.tool_calls


class RateAvailability(StrEnum):
    AVAILABLE = "AVAILABLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    ZERO_RATE = "ZERO_RATE"


class BurnTrend(StrEnum):
    INCREASING = "INCREASING"
    STABLE = "STABLE"
    DECREASING = "DECREASING"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ResourceBurnRate(FrozenModel):
    availability: RateAvailability
    committed: int
    remaining: int
    consumption_per_usage_event: float | None
    previous_window_rate: float | None
    trend: BurnTrend
    projected_iterations_to_exhaustion: float | None
    ceiling_proximity: float


class BudgetBurnRate(FrozenModel):
    version: str
    window_size: int
    resources: dict[str, ResourceBurnRate]
    source_event_hash: str
    projection_hash: str


class BudgetError(ValueError):
    pass


class BudgetExceeded(BudgetError):
    pass


class InvalidBudgetRevision(BudgetError):
    pass


class ReservationConflict(BudgetError):
    pass


class InvalidReservationSettlement(BudgetError):
    pass


class DeploymentPolicyConflict(BudgetError):
    pass


class InvalidTierPolicy(BudgetError):
    pass
