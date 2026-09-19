"""Minimal Wave 1 SDK for persistence and replay."""

from datetime import datetime
from uuid import UUID

from fre.adapters.artifacts_local import LocalArtifactStore
from fre.adapters.storage_sqlite import SQLiteStore
from fre.domain.common import ArtifactDescriptor, FrozenModel, JsonValue, canonical_hash
from fre.ports.clock import Clock, UUIDFactory
from fre.prompts.schemas import default_output_schema_registry
from fre.runtime.events import RunCreated, StoredEvent, UncommittedEvent, event_wire_identity
from fre.runtime.reducer import RunReducer, RunState, validate_semantic_reservation_admission


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
        # Default reducer is fully wired with the same artifact store and a
        # fresh (content-identical, hence identically-hashed) output schema
        # registry, so the production append/replay path always exercises the
        # F09/F13 content-provenance check inside `RunReducer.apply` (finding
        # #1 in the C04 remediation round) -- not just this façade's own
        # batch-admission pre-check below.
        self.reducer = reducer or RunReducer(
            artifact_reader=artifacts.get, schema_registry=default_output_schema_registry()
        )

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
            # Finding #7 (C04 remediation): this fast path deliberately skips the
            # `validate_semantic_reservation_admission` batch-admission check and the
            # per-event reducer preview below. That is safe ONLY because
            # `preview.version` is derived from replaying the store's own current
            # event history (`self.inspect` -> `self.store.load`), so a mismatch here
            # means `expected_version` is already stale against the store's real
            # current version -- and `SQLiteStore._append_locked` independently and
            # unconditionally re-checks `expected_version` against the authoritative
            # `current_version` row under its own transaction lock, raising
            # `ConcurrentAppendError` before a single row is inserted. No batch,
            # forged or otherwise, can ever be persisted through this branch without
            # first re-deriving a fresh, correct `expected_version` and going back
            # through the validated path above on retry. If `SQLiteStore.append` ever
            # stopped performing that check unconditionally, this fast path would
            # silently become an admission-check bypass -- do not remove or weaken
            # that store-side check without re-auditing this comment.
            return self.store.append(run_id, expected_version, events)
        stored_events = tuple(
            StoredEvent.model_validate(
                {
                    **event.model_dump(),
                    "created_at": event.created_at,
                    "sequence": expected_version + offset,
                },
                strict=True,
            )
            for offset, event in enumerate(events, 1)
        )
        # Cross-event batch admission (F09): proves every semantic model-call
        # outcome in this batch is backed by its own matching budget
        # settlement, not merely a caller's claim, before any event in the
        # batch reaches the per-event reducer or the store.
        validate_semantic_reservation_admission(stored_events)
        for stored in stored_events:
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
