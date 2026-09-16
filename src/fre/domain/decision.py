"""Decision result contracts."""

from typing import Literal
from uuid import UUID

from fre.domain.common import FrozenModel


class Tradeoff(FrozenModel):
    candidate_ids: tuple[UUID, ...]
    description: str


class SensitivityFinding(FrozenModel):
    parameter: str
    description: str


class AblationFinding(FrozenModel):
    component_id: str
    status: Literal["MEASURED", "SIMULATED", "MODEL_ESTIMATED", "NOT_EVALUABLE"]
    description: str


class RegretFinding(FrozenModel):
    scenario: str
    candidate_id: UUID
    regret: float | None


class DecisionResult(FrozenModel):
    feasible_candidates: tuple[UUID, ...] = ()
    conditional_candidates: tuple[UUID, ...] = ()
    pareto_front: tuple[UUID, ...] = ()
    recommendation: UUID | None = None
    selection_policy: str | None = None
    tradeoffs: tuple[Tradeoff, ...] = ()
    sensitivity_findings: tuple[SensitivityFinding, ...] = ()
    ablation_findings: tuple[AblationFinding, ...] = ()
    regret_findings: tuple[RegretFinding, ...] = ()
    unstable_assumptions: tuple[str, ...] = ()
    status: Literal["RESOLVED", "FRONTIER", "BLOCKED", "INSUFFICIENT_EVIDENCE"]
