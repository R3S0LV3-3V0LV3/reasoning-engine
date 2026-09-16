"""Typed event envelopes and payload registry."""

from datetime import datetime
from uuid import UUID

from pydantic import Field

from fre.domain.common import ArtifactRef, FrozenModel, JsonValue, SchemaVersion, UtcDateTime


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


EventPayload = RunCreated | RunStatusChanged | ArtifactRegistered | TestValueSet
EVENT_PAYLOADS: dict[str, type[EventPayload]] = {
    payload.__name__: payload
    for payload in (RunCreated, RunStatusChanged, ArtifactRegistered, TestValueSet)
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
