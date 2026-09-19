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
    wave3: "Wave3Config" = Field(default_factory=lambda: Wave3Config())


class Wave3Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: SchemaVersion = "1.0"
    classification_policy_version: str = "wave3-m01/1.0"
    confidence_threshold: float = Field(default=0.70, ge=0, le=1)
    semantic_runtime_policy_version: str = "wave3-semantic-runtime/2.0"
    maximum_repair_attempts: int = Field(default=1, ge=0, le=1)
    reserve_input_tokens: int = Field(default=4096, ge=0)
    reserve_output_tokens: int = Field(default=2048, ge=0)
    prompt_registry_version: str = "wave3-prompts/1.0"
    output_schema_registry_version: str = "wave3-schemas/1.0"
    representation_registry_version: str = "wave3-m04-registry/1.0"
    representation_selection_policy_version: str = "wave3-m04/1.0"
    representation_tie_band: float = Field(default=0.05, ge=0, le=1)
    representation_minimum_compatibility: float = Field(default=0.20, ge=0, le=1)
    model_adjudication_enabled: bool = False
    wave3_context_compiler_version: str = "2.0"


def load_config(path: Path) -> EngineConfig:
    """Load and strictly validate a YAML engine configuration."""
    value: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return EngineConfig.model_validate(value)
