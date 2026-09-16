from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from fre.adapters.artifacts_local import LocalArtifactStore
from fre.adapters.storage_sqlite import SQLiteStore
from fre.adapters.testing import FakeClock, FakeUUIDFactory
from fre.engine import FrontierReasoningEngine


@pytest.fixture
def engine(tmp_path: object) -> FrontierReasoningEngine:
    from pathlib import Path

    root = Path(str(tmp_path))
    times = (datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=index) for index in range(500))
    uuids = (UUID(int=index) for index in range(1, 1000))
    return FrontierReasoningEngine(
        SQLiteStore(root / "events.sqlite3"),
        LocalArtifactStore(root / "artifacts"),
        FakeClock(times),
        FakeUUIDFactory(uuids),
    )
