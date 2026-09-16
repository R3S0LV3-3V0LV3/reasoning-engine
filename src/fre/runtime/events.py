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
from fre.domain.stop import StopDecision


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
    policy_hash: str
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


class TerminalContextAssociated(FrozenModel):
    stop_disposition: str
    packet_hash: str


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
    | TerminalContextAssociated
)
EVENT_PAYLOADS: dict[str, type[EventPayload]] = {
    payload.__name__: payload
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
    )
}


class UncommittedEvent(FrozenModel):
    event_id: UUID
    run_id: UUID
    event_type: str
    action_id: UUID
    module_id: str
    schema_version: SchemaVersion = "1.0"
    module_version: str
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: UtcDateTime
    payload: dict[str, JsonValue]

    def validated_payload(self) -> EventPayload:
        payload_type = EVENT_PAYLOADS.get(self.event_type)
        if payload_type is None:
            raise ValueError(f"unregistered event type: {self.event_type}")
        # Payloads are canonical JSON values, so UUIDs and other rich types are
        # deliberately decoded from their wire representations here.
        return payload_type.model_validate(self.payload, strict=False)


class StoredEvent(UncommittedEvent):
    sequence: int = Field(ge=1)


def event_timestamp(event: UncommittedEvent) -> datetime:
    """Typing helper exposing an event's validated datetime."""
    return event.created_at
