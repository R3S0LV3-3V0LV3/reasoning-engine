"""Minimal Wave 1 SDK for persistence and replay."""

from datetime import datetime
from uuid import UUID

from fre.adapters.artifacts_local import LocalArtifactStore
from fre.adapters.storage_sqlite import SQLiteStore
from fre.domain.common import ArtifactDescriptor, FrozenModel, JsonValue, canonical_hash
from fre.ports.clock import Clock, UUIDFactory
from fre.runtime.events import RunCreated, StoredEvent, UncommittedEvent, event_wire_identity
from fre.runtime.reducer import RunReducer, RunState


class RunHandle(FrozenModel):
    run_id: UUID
    version: int


class FrontierReasoningEngine:
    """Small deterministic façade; scheduling and reasoning remain deferred."""

    def __init__(
        self,
        store: SQLiteStore,
        artifacts: LocalArtifactStore,
        clock: Clock,
        uuids: UUIDFactory,
        reducer: RunReducer | None = None,
    ) -> None:
        self.store = store
        self.artifacts = artifacts
        self.clock = clock
        self.uuids = uuids
        self.reducer = reducer or RunReducer()

    def _event(
        self,
        run_id: UUID,
        payload: FrozenModel,
        *,
        module_id: str,
        module_version: str = "1.0",
        input_value: object | None = None,
    ) -> UncommittedEvent:
        event_type, schema_version = event_wire_identity(payload)
        return UncommittedEvent(
            event_id=self.uuids.new(),
            run_id=run_id,
            event_type=event_type,
            action_id=self.uuids.new(),
            module_id=module_id,
            schema_version=schema_version,
            module_version=module_version,
            input_hash=canonical_hash(input_value),
            created_at=self.clock.now(),
            payload=payload.model_dump(mode="json"),
        )

    def create_run(self, config: object) -> RunHandle:
        run_id = self.uuids.new()
        config_hash = canonical_hash(config)
        event = self._event(
            run_id, RunCreated(config_hash=config_hash), module_id="runtime", input_value=config
        )
        stored = self.store.create_run(event)
        return RunHandle(run_id=run_id, version=stored.sequence)

    def make_event(
        self, run_id: UUID, payload: FrozenModel, *, module_id: str = "test"
    ) -> UncommittedEvent:
        return self._event(run_id, payload, module_id=module_id, input_value=payload)

    def append(
        self, run_id: UUID, expected_version: int, events: tuple[UncommittedEvent, ...]
    ) -> tuple[StoredEvent, ...]:
        # Preview the complete batch through the pure reducer before persistence.
        # SQLite expected-version append remains the race-arbitration authority.
        preview = self.inspect(run_id)
        if preview.version != expected_version:
            return self.store.append(run_id, expected_version, events)
        for offset, event in enumerate(events, 1):
            stored = StoredEvent.model_validate(
                {
                    **event.model_dump(),
                    "created_at": event.created_at,
                    "sequence": expected_version + offset,
                },
                strict=True,
            )
            preview = self.reducer.apply(preview, stored)
        return self.store.append(run_id, expected_version, events)

    def inspect(self, run_id: UUID) -> RunState:
        return self.reducer.reduce(run_id, self.store.load(run_id))

    def snapshot(self, run_id: UUID) -> str:
        return self.store.save_snapshot(self.inspect(run_id), self.reducer.version)

    def replay(self, run_id: UUID) -> RunState:
        return self.inspect(run_id)

    def replay_from_snapshot(self, run_id: UUID) -> RunState:
        return self.store.replay_from_snapshot(run_id, self.reducer)

    def verify(self, run_id: UUID) -> RunState:
        return self.store.verify_snapshot(run_id, self.reducer)

    def store_artifact(
        self,
        content: bytes,
        *,
        media_type: str,
        metadata: dict[str, JsonValue] | None = None,
    ) -> ArtifactDescriptor:
        return self.artifacts.put(content, media_type=media_type, metadata=metadata)


def ensure_utc(value: datetime) -> datetime:
    """Reject naive adapter timestamps at the SDK boundary."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value


def json_payload(value: JsonValue) -> JsonValue:
    """Explicit identity helper for typed generic event payloads."""
    return value
