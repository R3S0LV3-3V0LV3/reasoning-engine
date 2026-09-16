"""Context packet contract."""

from uuid import UUID

from fre.domain.budget import BudgetRemaining
from fre.domain.common import FrozenModel
from fre.domain.problem import ConstraintSpec, ObjectiveSpec, UnknownSpec


class RejectedCandidate(FrozenModel):
    candidate_id: UUID
    reason: str
    provenance_refs: tuple[str, ...]


class ContextPacket(FrozenModel):
    run_id: UUID
    snapshot_version: int
    objective: tuple[ObjectiveSpec, ...] = ()
    hard_constraints: tuple[ConstraintSpec, ...] = ()
    decisions_fixed: tuple[str, ...] = ()
    verified_facts: tuple[str, ...] = ()
    important_assumptions: tuple[str, ...] = ()
    active_candidates: tuple[UUID, ...] = ()
    rejected_candidates: tuple[RejectedCandidate, ...] = ()
    current_frontier: tuple[UUID, ...] = ()
    unresolved_high_value_questions: tuple[UnknownSpec, ...] = ()
    budget_remaining: BudgetRemaining
    next_action_id: UUID | None = None
    compiler_version: str
    packet_hash: str
