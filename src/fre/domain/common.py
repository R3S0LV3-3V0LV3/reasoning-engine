"""Common immutable types and canonical serialisation."""

import hashlib
import json
import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, PlainSerializer

type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
SchemaVersion = Annotated[str, Field(pattern=r"^[1-9]\d*\.\d+$")]
UtcDateTime = Annotated[
    AwareDatetime,
    PlainSerializer(lambda value: value.astimezone(UTC).isoformat().replace("+00:00", "Z")),
]


class FrozenModel(BaseModel):
    """Strict, immutable base for domain values."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PersistentModel(FrozenModel):
    """Metadata shared by persisted object revisions."""

    id: UUID
    revision: int = Field(default=1, ge=1)
    schema_version: SchemaVersion = "1.0"
    created_at: UtcDateTime
    created_by_action: UUID
    created_by_module: str
    provenance_refs: tuple["ObjectRef", ...] = ()


class ObjectRef(FrozenModel):
    object_type: str
    object_id: str
    revision: int | None = Field(default=None, ge=1)


class ArtifactRef(FrozenModel):
    artifact_id: UUID
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ArtifactDescriptor(PersistentModel):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: str
    byte_size: int = Field(ge=0)
    path: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class PermissionSet(FrozenModel):
    capabilities: frozenset[str] = frozenset()
    allow_network: bool = False
    allow_external_writes: bool = False
    allowed_paths: tuple[str, ...] = ()


class OutputContract(FrozenModel):
    form: str
    schema_ref: str | None = None
    requirements: tuple[str, ...] = ()


class Diagnostic(FrozenModel):
    code: str
    message: str
    severity: str = "WARNING"


class ConfidenceAssessment(FrozenModel):
    score: float | None = Field(default=None, ge=0, le=1)
    level: str | None = None
    method: str
    calibration_group: str | None = None
    evidence_independence_groups: tuple[str, ...] = ()
    explanation: str


def utc_now() -> datetime:
    return datetime.now(UTC)


def _normalise(value: Any) -> JsonValue:
    if isinstance(value, BaseModel):
        return _normalise(value.model_dump(mode="json"))
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("naive timestamps are not canonical")
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_normalise(item) for item in value), key=repr)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("NaN and infinity are not canonical JSON")
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value)!r}")


def canonical_json(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes, rejecting non-finite numbers."""
    return json.dumps(
        _normalise(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()
