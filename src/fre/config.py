"""Versioned YAML configuration loading."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from fre.domain.common import SchemaVersion


class EngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: SchemaVersion = Field(default="1.0")
    database_path: Path
    artifact_path: Path
    snapshot_interval: int = Field(default=50, ge=1)


def load_config(path: Path) -> EngineConfig:
    """Load and strictly validate a YAML engine configuration."""
    value: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return EngineConfig.model_validate(value)
