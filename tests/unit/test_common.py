from datetime import UTC, datetime
from uuid import UUID

import pytest

from fre.config import load_config
from fre.domain.common import canonical_hash, canonical_json


@pytest.mark.unit
def test_canonical_json_is_stable_and_typed(tmp_path: object) -> None:
    assert canonical_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    assert canonical_hash({"id": UUID(int=1)}) == canonical_hash({"id": str(UUID(int=1))})
    assert canonical_json(datetime(2026, 1, 1, tzinfo=UTC)) == b'"2026-01-01T00:00:00Z"'
    with pytest.raises(ValueError):
        canonical_json(float("nan"))


@pytest.mark.unit
def test_configuration_is_versioned_and_strict(tmp_path: object) -> None:
    from pathlib import Path

    path = Path(str(tmp_path)) / "config.yaml"
    path.write_text(
        'schema_version: "1.0"\ndatabase_path: db.sqlite3\nartifact_path: objects\n',
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.schema_version == "1.0"
    assert config.snapshot_interval == 50
