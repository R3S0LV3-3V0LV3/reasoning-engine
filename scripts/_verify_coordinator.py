"""Packaging smoke test for the C09 Wave 3 coordinator.

Run by `scripts/verify_package.py` inside a fresh venv that has ONLY the
built wheel (plus its locked dependencies) installed -- never the source
checkout. This proves `fre.composition.Wave3Engine.execute_front_end` (the
one authoritative, replayable Wave 3 front-end path) is importable and
runnable end-to-end as a real installed consumer would use it, not merely
from the source tree's own test suite.

Deliberately uses `allow_model=False` (the zero-call, fully deterministic
fallback configuration) so this smoke test never depends on network access
or a real model provider -- it only needs the wheel and its locked deps.
"""

import asyncio
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from fre.adapters.artifacts_local import LocalArtifactStore
from fre.adapters.storage_sqlite import SQLiteStore
from fre.adapters.testing import FakeClock, FakeUUIDFactory
from fre.composition import compose_wave3
from fre.config import Wave3Config
from fre.domain.common import OutputContract, PermissionSet
from fre.domain.semantic import StructuredModelResult
from fre.domain.task import TaskEnvelope
from fre.engine import FrontierReasoningEngine


class _NeverCalledModel:
    async def generate(self, request: object) -> StructuredModelResult:
        raise AssertionError("allow_model=False must never invoke the provider")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="fre-coordinator-smoke-") as directory:
        root = Path(directory)
        clock = FakeClock(
            datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=i) for i in range(50)
        )
        uuids = FakeUUIDFactory(UUID(int=i) for i in range(1, 200))
        store = SQLiteStore(root / "events.sqlite3")
        engine = FrontierReasoningEngine(
            store, LocalArtifactStore(root / "artifacts"), clock, uuids
        )
        wave3 = compose_wave3(engine, _NeverCalledModel(), Wave3Config())
        handle = wave3.create_run()
        task = TaskEnvelope(
            task_id=UUID(int=1),
            text="Packaging smoke test task.",
            requested_output=OutputContract(form="TEXT"),
            execution_permissions=PermissionSet(),
        )
        result = asyncio.run(
            wave3.execute_front_end(
                handle.run_id, task, allow_model=False, allow_adjudication=False
            )
        )
        assert result.task_signature is not None
        assert result.problem_spec is not None
        assert result.representation_plan is not None
        assert result.context_packet_hash is not None
        replayed = engine.replay(handle.run_id)
        assert replayed.task_signature == result.task_signature
        store.close()
    print("coordinator smoke test passed")


if __name__ == "__main__":
    main()
