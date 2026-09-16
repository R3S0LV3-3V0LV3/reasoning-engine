"""Transactional SQLite event and snapshot storage."""

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from uuid import UUID

from fre.domain.common import canonical_hash, canonical_json, utc_now
from fre.runtime.events import RunCreated, StoredEvent, UncommittedEvent
from fre.runtime.reducer import RunReducer, RunState


class ConcurrentAppendError(RuntimeError):
    """The caller's expected stream version is stale."""


class SnapshotIntegrityError(RuntimeError):
    """A stored snapshot differs from its hash or event replay."""


class SQLiteStore:
    SCHEMA_VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = path
        self._connection = sqlite3.connect(path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def _migrate(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_info(version INTEGER NOT NULL);
            INSERT INTO schema_info(version)
              SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_info);
            CREATE TABLE IF NOT EXISTS runs(
              run_id TEXT PRIMARY KEY, status TEXT NOT NULL, schema_version TEXT NOT NULL,
              config_hash TEXT NOT NULL, current_version INTEGER NOT NULL,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events(
              event_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
              sequence INTEGER NOT NULL, event_type TEXT NOT NULL, action_id TEXT NOT NULL,
              module_id TEXT NOT NULL, schema_version TEXT NOT NULL,
              module_version TEXT NOT NULL, input_hash TEXT NOT NULL,
              created_at TEXT NOT NULL, payload_json TEXT NOT NULL,
              UNIQUE(run_id, sequence)
            );
            CREATE TABLE IF NOT EXISTS snapshots(
              run_id TEXT NOT NULL REFERENCES runs(run_id), sequence INTEGER NOT NULL,
              state_json TEXT NOT NULL, state_hash TEXT NOT NULL,
              reducer_version TEXT NOT NULL, created_at TEXT NOT NULL,
              PRIMARY KEY(run_id, sequence)
            );
            """
        )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def create_run(self, event: UncommittedEvent) -> StoredEvent:
        payload = event.validated_payload()
        if not isinstance(payload, RunCreated):
            raise ValueError("the first event must be RunCreated")
        config_hash = payload.config_hash
        now = event.created_at.isoformat()
        with self._connection:
            self._connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, 0, ?, ?)",
                (str(event.run_id), "CREATED", event.schema_version, config_hash, now, now),
            )
            return self._append_locked(event.run_id, 0, (event,))[0]

    def append(
        self, run_id: UUID, expected_version: int, events: tuple[UncommittedEvent, ...]
    ) -> tuple[StoredEvent, ...]:
        if not events:
            return ()
        for event in events:
            if event.run_id != run_id:
                raise ValueError("all events must belong to the target run")
            event.validated_payload()
        with self._connection:
            return self._append_locked(run_id, expected_version, events)

    def _append_locked(
        self, run_id: UUID, expected_version: int, events: tuple[UncommittedEvent, ...]
    ) -> tuple[StoredEvent, ...]:
        row = self._connection.execute(
            "SELECT current_version FROM runs WHERE run_id = ?", (str(run_id),)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown run: {run_id}")
        if row["current_version"] != expected_version:
            raise ConcurrentAppendError(
                f"expected version {expected_version}, found {row['current_version']}"
            )
        stored: list[StoredEvent] = []
        for offset, event in enumerate(events, 1):
            item = StoredEvent.model_validate(
                {
                    **event.model_dump(),
                    "created_at": event.created_at,
                    "sequence": expected_version + offset,
                },
                strict=True,
            )
            self._connection.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(item.event_id),
                    str(item.run_id),
                    item.sequence,
                    item.event_type,
                    str(item.action_id),
                    item.module_id,
                    item.schema_version,
                    item.module_version,
                    item.input_hash,
                    item.created_at.isoformat(),
                    canonical_json(item.payload).decode(),
                ),
            )
            stored.append(item)
        new_version = expected_version + len(events)
        self._connection.execute(
            "UPDATE runs SET current_version = ?, updated_at = ? WHERE run_id = ?",
            (new_version, utc_now().isoformat(), str(run_id)),
        )
        return tuple(stored)

    def load(self, run_id: UUID, after_sequence: int = 0) -> tuple[StoredEvent, ...]:
        rows = self._connection.execute(
            "SELECT * FROM events WHERE run_id = ? AND sequence > ? ORDER BY sequence",
            (str(run_id), after_sequence),
        ).fetchall()
        return tuple(self._row_to_event(row) for row in rows)

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> StoredEvent:
        return StoredEvent.model_validate(
            {
                "event_id": row["event_id"],
                "run_id": row["run_id"],
                "sequence": row["sequence"],
                "event_type": row["event_type"],
                "action_id": row["action_id"],
                "module_id": row["module_id"],
                "schema_version": row["schema_version"],
                "module_version": row["module_version"],
                "input_hash": row["input_hash"],
                "created_at": row["created_at"],
                "payload": json.loads(row["payload_json"]),
            },
            strict=False,
        )

    def save_snapshot(self, state: RunState, reducer_version: str) -> str:
        state_hash = state.state_hash
        with self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO snapshots VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(state.run_id),
                    state.version,
                    canonical_json(state).decode(),
                    state_hash,
                    reducer_version,
                    utc_now().isoformat(),
                ),
            )
        return state_hash

    def verify_snapshot(self, run_id: UUID, reducer: RunReducer) -> RunState:
        row = self._connection.execute(
            "SELECT * FROM snapshots WHERE run_id = ? ORDER BY sequence DESC LIMIT 1",
            (str(run_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"no snapshot for run: {run_id}")
        if row["reducer_version"] != reducer.version:
            raise SnapshotIntegrityError("snapshot reducer version mismatch")
        raw_state = json.loads(row["state_json"])
        if canonical_hash(raw_state) != row["state_hash"]:
            raise SnapshotIntegrityError("snapshot content hash mismatch")
        rebuilt = reducer.reduce(run_id, self.load(run_id))
        if rebuilt.version != row["sequence"] or rebuilt.state_hash != row["state_hash"]:
            raise SnapshotIntegrityError("snapshot does not match event replay")
        return rebuilt

    def execute_for_test(self, sql: str, parameters: Iterable[object] = ()) -> None:
        """Execute corruption SQL for integrity-test fixtures only."""
        self._connection.execute(sql, tuple(parameters))
