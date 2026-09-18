"""Typed event envelopes and payload registry."""

from datetime import datetime
from uuid import UUID

from pydantic import Field

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
from fre.domain.representation import RepresentationArtifact, RepresentationPlan
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


class TaskPreliminarilyClassified(FrozenModel):
    """Deterministic-only bootstrap signature used solely to seed the M02 bootstrap budget.

    Distinct from `TaskClassified`: this signature is never itself the
    authoritative classification -- it is always superseded, in the same
    atomic batch, by a `TaskClassified` built from the full (deterministic +
    model) proposal.
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
    | StopDecisionRecorded
    | StopDecisionRecordedV2
    | TerminalContextAssociated
    | ModelCallRecorded
    | ModelCallFailed
    | ModelCallRecordedV2
    | ModelCallFailedV2
    | TaskClassified
    | TaskPreliminarilyClassified
    | ClassificationDiagnosticRecorded
    | ProblemFormalised
    | ProblemBlockerRecorded
    | ProblemContradictionRecorded
    | RepresentationPlanSelected
    | RepresentationArtifactCompiled
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
    }
)
EVENT_WIRE_IDENTITIES: dict[type[FrozenModel], tuple[str, SchemaVersion]] = {
    ModelCallRecordedV2: ("ModelCallRecorded", "2.0"),
    ModelCallFailedV2: ("ModelCallFailed", "2.0"),
    StopDecisionRecordedV2: ("StopDecisionRecorded", "2.0"),
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

    def validated_payload(self) -> EventPayload:
        payload_type = EVENT_PAYLOADS.get((self.event_type, self.schema_version))
        if payload_type is None:
            raise ValueError(
                f"unregistered event type/schema: {self.event_type}@{self.schema_version}"
            )
        # Payloads are canonical JSON values, so UUIDs and other rich types are
        # deliberately decoded from their wire representations here.
        return payload_type.model_validate(self.payload, strict=False)


class StoredEvent(UncommittedEvent):
    sequence: int = Field(ge=1)


def event_timestamp(event: UncommittedEvent) -> datetime:
    """Typing helper exposing an event's validated datetime."""
    return event.created_at
