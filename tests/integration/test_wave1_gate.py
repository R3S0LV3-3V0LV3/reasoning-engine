import sqlite3
from pathlib import Path

import pytest

from fre.adapters.storage_sqlite import ConcurrentAppendError, SnapshotIntegrityError, SQLiteStore
from fre.domain.common import ArtifactRef
from fre.engine import FrontierReasoningEngine
from fre.runtime.events import ArtifactRegistered
from fre.runtime.events import TestValueSet as ValueSet


@pytest.mark.integration
def test_complete_replay_concurrency_and_deduplication_gate(
    engine: FrontierReasoningEngine, tmp_path: Path
) -> None:
    run = engine.create_run({"policy": "test"})
    artifact1 = engine.store_artifact(b"raw evidence", media_type="text/plain")
    artifact2 = engine.store_artifact(b"raw evidence", media_type="text/plain")
    assert artifact1.sha256 == artifact2.sha256
    assert artifact1.id == artifact2.id
    assert engine.artifacts.object_count() == 1

    first = engine.make_event(run.run_id, ValueSet(key="answer", value=42))
    second = engine.make_event(
        run.run_id,
        ArtifactRegistered(
            artifact=ArtifactRef(artifact_id=artifact1.id, sha256=artifact1.sha256),
            media_type=artifact1.media_type,
            byte_size=artifact1.byte_size,
        ),
    )
    committed = engine.append(run.run_id, run.version, (first, second))
    assert [event.sequence for event in committed] == [2, 3]
    state = engine.inspect(run.run_id)
    snapshot_hash = engine.snapshot(run.run_id)
    assert state.state_hash == snapshot_hash

    database = engine.store.path
    artifact_root = engine.artifacts.root
    engine.store.close()
    reopened = SQLiteStore(database)
    engine.store = reopened
    assert engine.replay(run.run_id) == state
    assert engine.verify(run.run_id).state_hash == snapshot_hash
    assert engine.artifacts.root == artifact_root

    stale = engine.make_event(run.run_id, ValueSet(key="stale", value=True))
    with pytest.raises(ConcurrentAppendError):
        engine.append(run.run_id, 1, (stale,))


@pytest.mark.integration
def test_batch_failure_rolls_back_every_event(engine: FrontierReasoningEngine) -> None:
    run = engine.create_run({})
    event = engine.make_event(run.run_id, ValueSet(key="first", value=1))
    with pytest.raises(sqlite3.IntegrityError):
        engine.append(run.run_id, run.version, (event, event))
    assert len(engine.store.load(run.run_id)) == 1


@pytest.mark.integration
def test_snapshot_tampering_is_detected(engine: FrontierReasoningEngine) -> None:
    run = engine.create_run({})
    engine.snapshot(run.run_id)
    engine.store.execute_for_test(
        "UPDATE snapshots SET state_json = ? WHERE run_id = ?",
        ('{"tampered":true}', str(run.run_id)),
    )
    with pytest.raises(SnapshotIntegrityError, match="content hash"):
        engine.verify(run.run_id)
