"""Typed event envelopes and payload registry."""

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Self
from uuid import UUID

from pydantic import BaseModel, Field, PrivateAttr

from fre.domain.budget import BudgetPlan, BudgetReservation, ResourceVector
from fre.domain.common import ArtifactRef, FrozenModel, JsonValue, SchemaVersion, UtcDateTime
from fre.domain.context import ContextPacket
from fre.domain.ledger import (
    ContradictionResolution,
    EpistemicStatus,
    LedgerEdge,
    LedgerNode,
    LedgerNodeRef,
)
from fre.domain.problem import ContradictionDiagnostic, ProblemBlocker, ProblemSpec
from fre.domain.representation import (
    RepresentationArtifact,
    RepresentationArtifactV2,
    RepresentationPlan,
    RepresentationPlanV2,
)
from fre.domain.semantic import SemanticModelCallRecord, SemanticModelCallRecordV2
from fre.domain.stop import StopDecision, StopDecisionRecord
from fre.domain.task import ClassificationRecord, TaskSignature


class RunCreated(FrozenModel):
    config_hash: str


class RunStatusChanged(FrozenModel):
    status: str
    reason: str


class ArtifactRegistered(FrozenModel):
    artifact: ArtifactRef
    media_type: str
    byte_size: int = Field(ge=0)


class TestValueSet(FrozenModel):
    key: str
    value: JsonValue


class LedgerNodeAdded(FrozenModel):
    node: LedgerNode


class LedgerEdgeAdded(FrozenModel):
    edge: LedgerEdge


class LedgerNodeRevised(FrozenModel):
    successor: LedgerNode


class LedgerNodeStatusChanged(FrozenModel):
    node_ref: LedgerNodeRef
    status: EpistemicStatus


class LedgerDependentsMarkedStale(FrozenModel):
    source_ref: LedgerNodeRef
    affected_refs: tuple[LedgerNodeRef, ...]


class LedgerContradictionResolved(FrozenModel):
    resolution: ContradictionResolution
    resolution_edge: LedgerEdge


class BudgetAllocated(FrozenModel):
    plan: BudgetPlan
    policy_version: str
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_artifact_ref: ArtifactRef | None = None


class BudgetRevised(BudgetAllocated):
    pass


class BudgetConsumed(FrozenModel):
    usage: ResourceVector


class BudgetReserved(FrozenModel):
    reservation: BudgetReservation


class BudgetReservationSettled(FrozenModel):
    reservation_id: str
    actual_usage: ResourceVector


class BudgetReservationReleased(FrozenModel):
    reservation_id: str


class ContextCompiled(FrozenModel):
    packet: ContextPacket
    json_artifact: ArtifactRef | None = None
    markdown_artifact: ArtifactRef | None = None
    renderer_version: str = "1.0"


class ContextCompiledV2(FrozenModel):
    """Wire identity `("ContextCompiled", "2.0")`: a packet carrying a typed
    `Wave3SemanticContext` (Phase 6/C08, F12). Same shape as `ContextCompiled`
    -- the version bump exists purely to mark, at the envelope level, that
    `RunReducer.apply` must run the additional Wave 3 ground-truth
    verification below rather than only the v1 packet-hash/terminal-artifact
    checks. Both durable JSON and Markdown artifacts must already be
    registered before this event is appended: this is the event that
    actually persists a compiled Wave 3 context packet (closing F12's
    "context is never persisted by any production path" gap), so unlike a
    purely in-memory `compile_semantic()` call, an unregistered artifact ref
    here is a real, rejected atomicity violation, not merely untested.

    Finding G (C08 remediation): `json_artifact.sha256`/`markdown_artifact.
    sha256` are verified two ways, not one. `RunReducer.apply` first checks
    EXISTENCE (the claimed sha256 is some artifact this run genuinely
    registered) and then, independently, CORRECTNESS: it recomputes the
    canonical JSON bytes and the Markdown rendering directly from `packet`
    itself (both are pure functions of the packet) and requires the claimed
    sha256 to match that recomputation exactly. This proves the referenced
    artifacts really are the correct rendering of THIS packet, not merely
    that they are some real artifact this run happened to register earlier
    (e.g. from an unrelated compilation) with a colliding claim.
    """

    packet: ContextPacket
    json_artifact: ArtifactRef
    markdown_artifact: ArtifactRef
    renderer_version: str = "1.0"


class StopDecisionRecorded(FrozenModel):
    decision: StopDecision


class StopDecisionRecordedV2(FrozenModel):
    record: StopDecisionRecord


class TerminalContextAssociated(FrozenModel):
    stop_disposition: str
    packet_hash: str


class ModelCallRecorded(FrozenModel):
    record: SemanticModelCallRecord


class ModelCallFailed(FrozenModel):
    record: SemanticModelCallRecord
    reason: str


class ModelCallRecordedV2(FrozenModel):
    record: SemanticModelCallRecordV2


class ModelCallFailedV2(FrozenModel):
    record: SemanticModelCallRecordV2
    reason: str


class TaskClassified(FrozenModel):
    signature: TaskSignature
    record: ClassificationRecord


class TaskEnvelopeBound(FrozenModel):
    """W3 final-gate fix #3: records the canonical hash of the `TaskEnvelope`
    that produced a resumable Wave 3 front-end step's persisted result.

    `Wave3Engine.classify_task`/`classify_task_semantic`/`formalise_problem`
    return their already-persisted result unchanged whenever it exists,
    without previously checking that the CURRENT call's `envelope` argument
    is the same one that produced it. A caller resuming the same `run_id`
    with a genuinely different `TaskEnvelope` (e.g. a different `task_id`)
    got back stale data silently instead of an error. `stage` is either
    `"classification"` or `"formalisation"`; `envelope_hash` is
    `canonical_hash(envelope)` at the moment that stage's result was first
    persisted.
    """

    stage: str
    envelope_hash: str


class TaskPreliminarilyClassified(FrozenModel):
    """Deterministic-only bootstrap signature used solely to seed the M02 bootstrap budget.

    Distinct from `TaskClassified`: this signature is never itself the
    authoritative classification -- it is always superseded, in the same
    atomic batch, by a `TaskClassified` built from the full (deterministic +
    model) proposal.

    C05 remediation (finding #14, documentation-only): this event, and the
    `RunState.preliminary_task_signature` field it is reduced into (see
    `runtime/reducer.py`), are intentionally write-only in production code
    today: (a) they are persisted purely for audit/forensic traceability of
    the bootstrap-budget-sizing step -- so a later investigator can see
    exactly what signature `TaskClassifier.canonical_events` used to size the
    bootstrap `BudgetAllocated`; (b) the bootstrap budget calculation itself
    reads the local `bootstrap_signature` variable directly (see
    `canonical_events`), never this persisted field -- so there is currently
    no production read path for `preliminary_task_signature`; (c) all 10
    `tests/fixtures/golden/*.json` streams include a
    `TaskPreliminarilyClassified` event, so removing this event or field
    would require regenerating every golden fixture, which is explicitly out
    of scope for this cleanup pass. If a future consumer is added for this
    field, update this note to reflect the new read site rather than
    removing it.
    """

    signature: TaskSignature


class ClassificationDiagnosticRecorded(FrozenModel):
    code: str
    message: str


class ProblemFormalised(FrozenModel):
    problem: ProblemSpec


class ProblemBlockerRecorded(FrozenModel):
    blocker: ProblemBlocker


class ProblemContradictionRecorded(FrozenModel):
    diagnostic: ContradictionDiagnostic


class RepresentationPlanSelected(FrozenModel):
    plan: RepresentationPlan


class RepresentationArtifactCompiled(FrozenModel):
    artifact: RepresentationArtifact


class RepresentationPlanSelectedV2(FrozenModel):
    plan: RepresentationPlanV2


class RepresentationArtifactCompiledV2(FrozenModel):
    artifact: RepresentationArtifactV2


EventPayload = (
    RunCreated
    | RunStatusChanged
    | ArtifactRegistered
    | TestValueSet
    | LedgerNodeAdded
    | LedgerEdgeAdded
    | LedgerNodeRevised
    | LedgerNodeStatusChanged
    | LedgerDependentsMarkedStale
    | LedgerContradictionResolved
    | BudgetAllocated
    | BudgetRevised
    | BudgetConsumed
    | BudgetReserved
    | BudgetReservationSettled
    | BudgetReservationReleased
    | ContextCompiled
    | ContextCompiledV2
    | StopDecisionRecorded
    | StopDecisionRecordedV2
    | TerminalContextAssociated
    | ModelCallRecorded
    | ModelCallFailed
    | ModelCallRecordedV2
    | ModelCallFailedV2
    | TaskClassified
    | TaskEnvelopeBound
    | TaskPreliminarilyClassified
    | ClassificationDiagnosticRecorded
    | ProblemFormalised
    | ProblemBlockerRecorded
    | ProblemContradictionRecorded
    | RepresentationPlanSelected
    | RepresentationArtifactCompiled
    | RepresentationPlanSelectedV2
    | RepresentationArtifactCompiledV2
)
EVENT_PAYLOADS: dict[tuple[str, str], type[EventPayload]] = {
    (payload.__name__, "1.0"): payload
    for payload in (
        RunCreated,
        RunStatusChanged,
        ArtifactRegistered,
        TestValueSet,
        LedgerNodeAdded,
        LedgerEdgeAdded,
        LedgerNodeRevised,
        LedgerNodeStatusChanged,
        LedgerDependentsMarkedStale,
        LedgerContradictionResolved,
        BudgetAllocated,
        BudgetRevised,
        BudgetConsumed,
        BudgetReserved,
        BudgetReservationSettled,
        BudgetReservationReleased,
        ContextCompiled,
        StopDecisionRecorded,
        TerminalContextAssociated,
        ModelCallRecorded,
        ModelCallFailed,
        TaskClassified,
        TaskEnvelopeBound,
        TaskPreliminarilyClassified,
        ClassificationDiagnosticRecorded,
        ProblemFormalised,
        ProblemBlockerRecorded,
        ProblemContradictionRecorded,
        RepresentationPlanSelected,
        RepresentationArtifactCompiled,
    )
}
EVENT_PAYLOADS.update(
    {
        ("ModelCallRecorded", "2.0"): ModelCallRecordedV2,
        ("ModelCallFailed", "2.0"): ModelCallFailedV2,
        ("StopDecisionRecorded", "2.0"): StopDecisionRecordedV2,
        ("RepresentationPlanSelected", "2.0"): RepresentationPlanSelectedV2,
        ("RepresentationArtifactCompiled", "2.0"): RepresentationArtifactCompiledV2,
        ("ContextCompiled", "2.0"): ContextCompiledV2,
    }
)
EVENT_WIRE_IDENTITIES: dict[type[FrozenModel], tuple[str, SchemaVersion]] = {
    ModelCallRecordedV2: ("ModelCallRecorded", "2.0"),
    ModelCallFailedV2: ("ModelCallFailed", "2.0"),
    StopDecisionRecordedV2: ("StopDecisionRecorded", "2.0"),
    RepresentationPlanSelectedV2: ("RepresentationPlanSelected", "2.0"),
    RepresentationArtifactCompiledV2: ("RepresentationArtifactCompiled", "2.0"),
    ContextCompiledV2: ("ContextCompiled", "2.0"),
}


def event_wire_identity(payload: FrozenModel) -> tuple[str, SchemaVersion]:
    return EVENT_WIRE_IDENTITIES.get(type(payload), (type(payload).__name__, "1.0"))


class UncommittedEvent(FrozenModel):
    event_id: UUID
    run_id: UUID
    event_type: str = Field(min_length=1)
    action_id: UUID
    module_id: str = Field(min_length=1)
    schema_version: SchemaVersion = "1.0"
    module_version: str = Field(min_length=1)
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: UtcDateTime
    payload: dict[str, JsonValue]

    # EU-02 (C04 cleanup, item #2): `validated_payload()` used to re-decode
    # `self.payload` from scratch on every call. Within a single
    # `FrontierReasoningEngine.append`, the same `StoredEvent` instance is
    # decoded once by `validate_semantic_reservation_admission` (twice,
    # actually -- once per pass) and again by `RunReducer.apply` -- all
    # identical decodes of the same immutable payload dict, since every call
    # site invokes this same method with no parameters (no strict/lenient
    # variance to preserve). `UncommittedEvent`/`StoredEvent` are frozen
    # (`FrozenModel`, `frozen=True`), but a `PrivateAttr` is exempt from that
    # field-level immutability enforcement -- it is not a pydantic "field" --
    # so it can safely hold a per-instance memo without weakening the
    # model's frozen-field guarantees. The cache is instance-scoped (not a
    # module-level/keyed cache), so it never leaks across distinct events.
    _validated_payload_cache: EventPayload | None = PrivateAttr(default=None)

    def __eq__(self, other: object) -> bool:
        # Pydantic's default `__eq__` folds `__pydantic_private__` into the
        # comparison, so two field-identical events would compare unequal
        # purely because one had `validated_payload()` called on it (and
        # therefore populated `_validated_payload_cache`) and the other
        # hadn't. That would make equality depend on incidental memoization
        # state rather than on the event's actual (field) value -- restore
        # pure field-based value equality, matching this class's pre-EU-02
        # behavior. `__hash__` is left untouched: pydantic's generated hash
        # function for a frozen model already hashes only the field values,
        # not `__pydantic_private__`, so it needs no corresponding override.
        if not isinstance(other, BaseModel):
            return NotImplemented
        if type(self) is not type(other):
            return False
        return self.__dict__ == other.__dict__

    def validated_payload(self) -> EventPayload:
        if self._validated_payload_cache is not None:
            return self._validated_payload_cache
        payload_type = EVENT_PAYLOADS.get((self.event_type, self.schema_version))
        if payload_type is None:
            raise ValueError(
                f"unregistered event type/schema: {self.event_type}@{self.schema_version}"
            )
        # Payloads are canonical JSON values, so UUIDs and other rich types are
        # deliberately decoded from their wire representations here.
        decoded = payload_type.model_validate(self.payload, strict=False)
        self._validated_payload_cache = decoded
        return decoded

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        # A copy (even a field-preserving one) must not inherit the source
        # instance's cached decode: pydantic's `model_copy` carries
        # `__pydantic_private__` (and therefore `_validated_payload_cache`)
        # over to the copy by default, which would let a copy constructed
        # with `update={...}` (e.g. a different `event_type`/`schema_version`/
        # `payload`) silently return the *original* instance's stale decoded
        # payload instead of re-decoding against its own, possibly different,
        # wire representation. Reset the cache on every copy so
        # `validated_payload()` always decodes fresh for a copy's own state.
        copied = super().model_copy(update=update, deep=deep)
        copied._validated_payload_cache = None
        return copied


class StoredEvent(UncommittedEvent):
    sequence: int = Field(ge=1)


def event_timestamp(event: UncommittedEvent) -> datetime:
    """Typing helper exposing an event's validated datetime."""
    return event.created_at
