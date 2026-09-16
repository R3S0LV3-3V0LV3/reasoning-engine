"""Snapshot contracts."""

from uuid import UUID

from fre.domain.common import FrozenModel
from fre.runtime.reducer import RunState

RunSnapshot = RunState


class SnapshotRecord(FrozenModel):
    run_id: UUID
    sequence: int
    state: RunSnapshot
    state_hash: str
    reducer_version: str
