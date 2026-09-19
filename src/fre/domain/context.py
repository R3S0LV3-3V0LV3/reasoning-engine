"""Deterministic context packet, compression, and delta contracts."""

from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import Field, SerializerFunctionWrapHandler, model_serializer

from fre.domain.budget import BudgetRemaining
from fre.domain.common import FrozenModel, JsonValue
from fre.domain.ledger import LedgerNodeRef


class Availability(StrEnum):
    AVAILABLE = "AVAILABLE"
    NOT_PRODUCED = "NOT_PRODUCED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Wave3ContextAvailability(StrEnum):
    """Explicit, typed Wave 3 semantic-context availability (Phase 6/C08, F12).

    Modelled on the existing `*_projection_availability` pattern above, but
    for the compiled Wave 3 semantic snapshot as a whole (ProblemSpec state,
    blockers, and representation), not any single M04 projection kind.
    """

    AVAILABLE = "AVAILABLE"
    PARTIAL_BLOCKED = "PARTIAL_BLOCKED"
    UNAVAILABLE_BUDGET = "UNAVAILABLE_BUDGET"
    UNAVAILABLE_VALIDATION = "UNAVAILABLE_VALIDATION"
    NOT_REQUESTED = "NOT_REQUESTED"


WAVE3_CONTEXT_SCHEMA_VERSION = "1.0"


class UnresolvedUnknownRef(FrozenModel):
    """Full-fidelity mirror of one still-open `problem.UnknownSpec`.

    Every field `UnknownSpec` declares is carried through verbatim so no
    governed-uncertainty metadata is lost when it is surfaced into the
    permitted context view -- applying the C06 UNKNOWN-preservation lesson to
    M12's compiled output, not only to M03's ledger admission path.
    """

    id: str
    description: str
    domain: JsonValue | None = None
    rationale: str | None = None
    impact: JsonValue | None = None
    decision_relevance: float | None = None
    resolvable: bool | None = None
    candidate_actions: tuple[str, ...] = ()


class Wave3SemanticContext(FrozenModel):
    """Typed Wave 3 semantic-context envelope (Phase 6/C08, defect F12).

    Replaces prose/structure sniffing of the overloaded `ContextPacket.
    objective` field (`compiler_version == "2.0"` plus "is `objective` a
    dict") with an explicit, typed contract naming exactly what Wave 3
    semantic state this packet reflects and how available it is. Every ref
    field here is independently re-derived and verified by `RunReducer.apply`
    against the actual persisted run state at `ContextCompiled@2.0`
    application time (see the reducer's `ContextCompiledV2` branch) -- never
    trusted merely because it is internally self-consistent.

    `budget_plan_revision` was deliberately NOT included: `BudgetPlan` has no
    revision counter in the domain model, and `BudgetProjection` does not
    track how many times a plan has been revised either, so a "revision"
    field here could only ever be a caller-supplied, unverifiable claim --
    exactly the class of defect this phase exists to remove. `budget_
    policy_hash` is used instead: it is a value already stored verbatim on
    `BudgetProjection.policy_hash`, so the reducer check is an exact stored-
    value comparison, not a fabricated recomputation.
    """

    schema_version: str = WAVE3_CONTEXT_SCHEMA_VERSION
    availability: Wave3ContextAvailability
    availability_reason: str = Field(min_length=1)
    problem_spec_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_signature_ref: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    budget_plan_ref: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    budget_policy_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    blocker_refs: tuple[str, ...] = ()
    ledger_root: str = Field(pattern=r"^[0-9a-f]{64}$")
    ledger_version: int = Field(ge=0)
    representation_plan_ref: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    representation_artifact_refs: tuple[str, ...] = ()
    prompt_version: str | None = None
    model_identity: str | None = None
    unresolved_unknowns: tuple[UnresolvedUnknownRef, ...] = ()


class CompilerProfile(StrEnum):
    FULL = "FULL"
    STANDARD = "STANDARD"
    HANDOFF = "HANDOFF"


class ContextCompressionPolicy(FrozenModel):
    compression_policy_version: str = "1.0"
    ordered_rule_ids: tuple[str, ...] = ("C01", "C02", "C03", "C04", "C05")
    profile_rule_eligibility: dict[CompilerProfile, tuple[str, ...]] = Field(
        default_factory=lambda: {
            CompilerProfile.FULL: ("C01", "C03", "C05"),
            CompilerProfile.STANDARD: ("C01", "C02", "C03", "C04", "C05"),
            CompilerProfile.HANDOFF: ("C01", "C02", "C03", "C04", "C05"),
        }
    )
    size_metric: str = "CANONICAL_UTF8_BYTES"
    mandatory_retention_classes: tuple[str, ...] = (
        "hard_constraints",
        "stale_items",
        "unresolved_blockers",
        "rejection_reasons",
        "decision_provenance",
        "budget_remaining",
        "next_or_terminal_action",
    )


class ContextCompilationRequest(FrozenModel):
    profile: CompilerProfile
    size_target: int = Field(ge=1)
    terminal: bool = False


class ContextItem(FrozenModel):
    ref: LedgerNodeRef
    kind: str
    status: str
    content: JsonValue
    provenance_refs: tuple[str, ...] = ()
    predecessor_ref: LedgerNodeRef | None = None


class RejectedItem(FrozenModel):
    ref: str
    reason: str
    provenance_refs: tuple[str, ...] = ()


class ContextPacket(FrozenModel):
    run_id: UUID
    snapshot_version: int
    profile: CompilerProfile
    compiler_version: str
    compression_policy_version: str
    applied_rule_ids: tuple[str, ...]
    objective: JsonValue | None = None
    output_contract: JsonValue | None = None
    hard_constraints: tuple[JsonValue, ...] = ()
    ledger_items: tuple[ContextItem, ...] = ()
    rejected_items: tuple[RejectedItem, ...] = ()
    unresolved_blockers: tuple[str, ...] = ()
    budget_remaining: BudgetRemaining
    candidate_projection_availability: Availability = Availability.NOT_PRODUCED
    decision_projection_availability: Availability = Availability.NOT_PRODUCED
    acquisition_projection_availability: Availability = Availability.NOT_PRODUCED
    next_action: str | None = None
    terminal_disposition: str | None = None
    # C08 (F12) addition: typed Wave 3 semantic availability. `None` for every
    # Wave 1/2/v1-semantic packet, exactly as before this field existed.
    # `serialize_compatibly` below omits the key entirely whenever it is
    # `None`, so `packet_hash`, `canonical_json`, and every embedding
    # `RunState.state_hash` computed over a packet that never populated this
    # field are byte-for-byte unchanged by this phase (mirrors
    # `RepresentationView.serialize_compatibly`'s `builder_available` trick).
    wave3_context: Wave3SemanticContext | None = None
    packet_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_serializer(mode="wrap")
    def serialize_compatibly(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        payload: dict[str, Any] = handler(self)
        if self.wave3_context is None:
            payload.pop("wave3_context", None)
        return payload


class ContextCompilationRecord(FrozenModel):
    packet_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile: CompilerProfile
    compiler_version: str
    compression_policy_version: str
    applied_rule_ids: tuple[str, ...]
    renderer_version: str
    canonical_byte_size: int = Field(ge=0)
    json_artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    markdown_artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ContextCompilationResult(FrozenModel):
    packet: ContextPacket
    canonical_bytes: bytes
    markdown: str
    canonical_byte_size: int


class ContextOverflowDiagnostic(FrozenModel):
    required_bytes: int
    target_bytes: int
    mandatory_classes: tuple[str, ...]


class DeltaOperationType(StrEnum):
    ADD = "ADD"
    REMOVE = "REMOVE"
    REPLACE = "REPLACE"


class DeltaOperation(FrozenModel):
    operation: DeltaOperationType
    path: str
    value: JsonValue | None = None


class ContextDeltaPacket(FrozenModel):
    base_packet_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_packet_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_version: str
    compression_policy_version: str
    profile: CompilerProfile
    operations: tuple[DeltaOperation, ...]
    delta_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ContextError(ValueError):
    pass


class ContextOverflow(ContextError):
    def __init__(self, diagnostic: ContextOverflowDiagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__("mandatory context exceeds the configured byte target")


class InvalidCompilerProfile(ContextError):
    pass


class InvalidContextDelta(ContextError):
    pass


class ContextDeltaBaseMismatch(InvalidContextDelta):
    pass
