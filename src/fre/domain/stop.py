"""Deterministic marginal-value stop contracts."""

from enum import StrEnum

from pydantic import Field, model_validator

from fre.domain.budget import BudgetBurnRate, BudgetRemaining
from fre.domain.common import ConfidenceAssessment, FrozenModel, ObjectRef
from fre.domain.context import ContextCompilationRequest
from fre.domain.ledger import LedgerNodeRef


class StopDisposition(StrEnum):
    CONTINUE = "CONTINUE"
    COMPLETE = "COMPLETE"
    PARTIAL_BUDGET = "PARTIAL_BUDGET"
    BLOCKED = "BLOCKED"
    FAILED_INVARIANT = "FAILED_INVARIANT"
    CANCELLED = "CANCELLED"


class StopReasonCode(StrEnum):
    CANCELLED = "CANCELLED"
    INVARIANT_FAILURE = "INVARIANT_FAILURE"
    HARD_BUDGET_EXHAUSTED = "HARD_BUDGET_EXHAUSTED"
    UNRESOLVABLE_BLOCKER = "UNRESOLVABLE_BLOCKER"
    MANDATORY_ACTION = "MANDATORY_ACTION"
    MANDATORY_VALIDATION = "MANDATORY_VALIDATION"
    RESOLVABLE_UNKNOWN = "RESOLVABLE_UNKNOWN"
    ACCEPTANCE_FAILED = "ACCEPTANCE_FAILED"
    ACCEPTANCE_COMPLETE = "ACCEPTANCE_COMPLETE"
    OPTIONAL_WORK_VALUABLE = "OPTIONAL_WORK_VALUABLE"
    OPTIONAL_WORK_UNCERTAIN = "OPTIONAL_WORK_UNCERTAIN"
    OPTIONAL_WORK_NOT_VALUABLE = "OPTIONAL_WORK_NOT_VALUABLE"
    NO_EXECUTABLE_WORK = "NO_EXECUTABLE_WORK"


class AcceptanceStatus(StrEnum):
    SATISFIED = "SATISFIED"
    FAILED = "FAILED"
    PENDING = "PENDING"


class ValidationStatus(StrEnum):
    COMPLETE = "COMPLETE"
    REQUIRED = "REQUIRED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class IntervalEstimate(FrozenModel):
    lower: float
    point: float | None = None
    upper: float
    method_version: str
    source_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def ordered(self) -> "IntervalEstimate":
        if self.lower > self.upper or (
            self.point is not None and not self.lower <= self.point <= self.upper
        ):
            raise ValueError("estimate interval is not ordered")
        return self


class MarginalValueEstimate(IntervalEstimate):
    confidence: ConfidenceAssessment
    ledger_refs: tuple[LedgerNodeRef, ...] = ()


class CostEstimateInterval(IntervalEstimate):
    pass


class StopPolicy(FrozenModel):
    version: str
    stop_threshold: float = 0.0
    continue_on_overlap: bool = True
    stability_window: int = Field(default=2, ge=1)


class StopInputs(FrozenModel):
    budget: BudgetRemaining
    acceptance: AcceptanceStatus
    validation: ValidationStatus
    cancelled: bool = False
    invariant_failure: bool = False
    required_work: bool = False
    executable_work: bool = False
    blocker_required: bool = False
    blocker_resolvable: bool = False
    mandatory_action: bool = False
    stable: bool = True
    value: MarginalValueEstimate | None = None
    cost: CostEstimateInterval | None = None
    burn_rate: BudgetBurnRate | None = None
    epistemic_trigger_refs: tuple[LedgerNodeRef, ...] = ()


class StopDecision(FrozenModel):
    disposition: StopDisposition
    policy_version: str
    reason_codes: tuple[StopReasonCode, ...]
    reasons: tuple[str, ...]
    triggering_refs: tuple[ObjectRef, ...] = ()
    ledger_trigger_refs: tuple[LedgerNodeRef, ...] = ()
    budget_projection_hash: str
    burn_rate_projection_hash: str | None = None
    marginal_value_method: str | None = None
    cost_method: str | None = None
    context_request: ContextCompilationRequest | None = None


class StopError(ValueError):
    pass


class InvalidStopTransition(StopError):
    pass


class MissingRequiredContext(StopError):
    pass


class InvalidStopPolicy(StopError):
    pass
