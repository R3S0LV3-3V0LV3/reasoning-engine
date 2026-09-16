"""Formal problem contracts."""

from enum import StrEnum
from typing import Literal

from fre.domain.common import FrozenModel, JsonValue, OutputContract
from fre.domain.ledger import LedgerNodeRef
from fre.domain.semantic import EpistemicItemProvenance


class VerificationStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"


class DecisionVariable(FrozenModel):
    id: str
    name: str
    domain: JsonValue | None = None
    provenance: EpistemicItemProvenance | None = None


class FixedParameter(FrozenModel):
    id: str
    name: str
    value: JsonValue
    provenance: EpistemicItemProvenance


class ObservableSpec(FrozenModel):
    id: str
    description: str
    unit: str | None = None
    provenance: EpistemicItemProvenance | None = None


class ObjectiveSpec(FrozenModel):
    id: str
    name: str
    direction: Literal["MIN", "MAX", "TARGET", "LEXICOGRAPHIC", "QUALITATIVE", "UNRESOLVED"]
    unit: str | None = None
    priority: int | None = None
    evaluator_ref: str | None = None
    description: str
    provenance: EpistemicItemProvenance | None = None


class ConstraintSpec(FrozenModel):
    id: str
    description: str
    kind: Literal["HARD", "SOFT"]
    verification_mode: Literal["DETERMINISTIC", "MODEL", "HUMAN", "UNAVAILABLE"]
    verifier_ref: str | None = None
    verification_status: VerificationStatus = VerificationStatus.UNKNOWN
    source_refs: tuple[str, ...] = ()
    provenance: EpistemicItemProvenance | None = None


class UnknownSpec(FrozenModel):
    id: str
    description: str
    domain: JsonValue | None = None
    decision_relevance: float | None = None
    resolvable: bool | None = None
    candidate_actions: tuple[str, ...] = ()
    provenance: EpistemicItemProvenance | None = None


class AssumptionSpec(FrozenModel):
    id: str
    statement: str
    why_needed: str
    decision_relevance: float | None = None
    provenance: EpistemicItemProvenance


class AcceptanceCriterion(FrozenModel):
    id: str
    predicate_description: str
    verification_mode: str
    required: bool
    verification_status: VerificationStatus = VerificationStatus.UNKNOWN
    blocker_ref: LedgerNodeRef | None = None
    provenance: EpistemicItemProvenance | None = None


class ProblemRelationKind(StrEnum):
    DEPENDS_ON = "DEPENDS_ON"
    CAUSES = "CAUSES"
    TEMPORALLY_PRECEDES = "TEMPORALLY_PRECEDES"
    CONTRADICTS = "CONTRADICTS"
    INTERACTS_WITH = "INTERACTS_WITH"


class ProblemRelation(FrozenModel):
    source_id: str
    target_id: str
    kind: ProblemRelationKind


class ProblemBlocker(FrozenModel):
    blocker_id: str
    description: str
    ledger_ref: LedgerNodeRef
    resolvable: bool


class ContradictionDiagnostic(FrozenModel):
    left_ref: LedgerNodeRef
    right_ref: LedgerNodeRef
    material: bool
    message: str


class ProblemSpec(FrozenModel):
    decision_variables: tuple[DecisionVariable, ...] = ()
    objectives: tuple[ObjectiveSpec, ...] = ()
    constraints: tuple[ConstraintSpec, ...] = ()
    assumptions: tuple[str, ...] = ()
    assumption_items: tuple[AssumptionSpec, ...] = ()
    fixed_parameters: tuple[FixedParameter, ...] = ()
    unknowns: tuple[UnknownSpec, ...] = ()
    observables: tuple[ObservableSpec, ...] = ()
    acceptance_criteria: tuple[AcceptanceCriterion, ...] = ()
    relations: tuple[ProblemRelation, ...] = ()
    output_contract: OutputContract
