"""Formal problem contracts."""

from enum import StrEnum
from typing import Literal

from fre.domain.common import FrozenModel, JsonValue, OutputContract
from fre.domain.ledger import LedgerNodeRef
from fre.domain.semantic import EpistemicItemProvenance, SupportRef


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
    # Deprecated, decode-only (C06 / F03): see the note above `SupportRef` in
    # `fre.domain.semantic` -- a plain string never resolves to real,
    # admissible evidence. New code populates `support` instead.
    source_refs: tuple[str, ...] = ()
    support: tuple[SupportRef, ...] = ()
    provenance: EpistemicItemProvenance | None = None


class UnknownSpec(FrozenModel):
    id: str
    description: str
    domain: JsonValue | None = None
    # C06 remediation (F04): these fields were declared but never populated by
    # the M03 `UNKNOWN` branch -- always null/empty in practice, silently
    # discarding governed uncertainty metadata a proposal actually supplied.
    # `ProblemFormaliser.formalise` now carries every one of them through from
    # `ProblemItemProposal.attributes`.
    rationale: str | None = None
    impact: JsonValue | None = None
    decision_relevance: float | None = None
    resolvable: bool | None = None
    candidate_actions: tuple[str, ...] = ()
    provenance: EpistemicItemProvenance | None = None


class AssumptionSpec(FrozenModel):
    id: str
    statement: str
    why_needed: str
    decision_relevance: float | None = None
    # C06 remediation (F04): the scope an assumption is claimed to hold over
    # (e.g. a JSON-pointer-shaped path or a free-text qualifier); previously
    # dropped entirely. Populated from the proposal's `attrs.get("scope")` at
    # formalisation time (`m03_formaliser.py`'s `ASSUMPTION` branch); it has
    # no current downstream consumer anywhere in `src/fre/` (confirmed by
    # C06 cleanup, EU-22 -- unlike `UnknownSpec.rationale`/`.impact`, which
    # are now genuinely read per C08 remediation). It is intentionally
    # write-only/forward-compatible, not dead code: a future M04/M12/M09
    # consumer could use it to scope an assumption's applicability, and
    # removing it now would be premature given `AssumptionSpec` is a
    # persisted/wire type. If a real consumer is ever defined, wiring it up
    # is a feature unit, not a cleanup item.
    scope: str | None = None
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
    # `assumption_items` (below) is the source of truth for assumptions;
    # `assumptions` is an intentional, backward-compatible free-text mirror
    # of `assumption_items[*].statement`, always populated in lockstep by
    # `m03_formaliser.ProblemFormaliser.formalise` -- never independently.
    # `m12_context.py`'s compiled context packet threads the same pairing
    # one level further downstream (`assumptions.statements` /
    # `assumptions.items`). No field removal or computed-property
    # conversion is planned: both are persisted/wire-adjacent shapes with a
    # wide golden-fixture blast radius (C06 cleanup, EU-20).
    assumptions: tuple[str, ...] = ()
    assumption_items: tuple[AssumptionSpec, ...] = ()
    fixed_parameters: tuple[FixedParameter, ...] = ()
    unknowns: tuple[UnknownSpec, ...] = ()
    observables: tuple[ObservableSpec, ...] = ()
    acceptance_criteria: tuple[AcceptanceCriterion, ...] = ()
    relations: tuple[ProblemRelation, ...] = ()
    output_contract: OutputContract
