"""Common immutable types and canonical serialisation."""

import hashlib
import json
import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Never, Protocol
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, PlainSerializer, model_validator

type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


class SupportsIndex(Protocol):
    def __index__(self) -> int: ...


SchemaVersion = Annotated[str, Field(pattern=r"^[1-9]\d*\.\d+$")]
UtcDateTime = Annotated[
    AwareDatetime,
    PlainSerializer(lambda value: value.astimezone(UTC).isoformat().replace("+00:00", "Z")),
]


class FrozenDict(dict[Any, Any]):
    """A recursively immutable mapping that retains normal JSON serialization."""

    def _immutable(self) -> Never:
        raise TypeError("canonical domain mappings are immutable")

    def __delitem__(self, key: Any) -> Never:
        self._immutable()

    def __setitem__(self, key: Any, value: Any) -> Never:
        self._immutable()

    def clear(self) -> Never:
        self._immutable()

    def pop(self, key: Any, default: Any = None) -> Never:
        self._immutable()

    def popitem(self) -> Never:
        self._immutable()

    def setdefault(self, key: Any, default: Any = None) -> Never:
        self._immutable()

    def update(self, *args: Any, **kwargs: Any) -> Never:
        self._immutable()


class FrozenList(list[Any]):
    """A recursively immutable list retaining list-compatible serialization."""

    def _immutable(self) -> Never:
        raise TypeError("canonical domain sequences are immutable")

    def __delitem__(self, key: Any) -> Never:
        self._immutable()

    def __setitem__(self, key: Any, value: Any) -> Never:
        self._immutable()

    def append(self, value: Any) -> Never:
        self._immutable()

    def clear(self) -> Never:
        self._immutable()

    def extend(self, values: Any) -> Never:
        self._immutable()

    def insert(self, index: SupportsIndex, value: Any) -> Never:
        self._immutable()

    def pop(self, index: SupportsIndex = -1) -> Never:
        self._immutable()

    def remove(self, value: Any) -> Never:
        self._immutable()

    def reverse(self) -> Never:
        self._immutable()

    def sort(self, *args: Any, **kwargs: Any) -> Never:
        self._immutable()


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, FrozenModel):
        return value
    if isinstance(value, dict):
        return FrozenDict({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return FrozenList(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_deep_freeze(item) for item in value)
    return value


class FrozenModel(BaseModel):
    """Strict, immutable base for domain values."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @model_validator(mode="after")
    def recursively_freeze(self) -> "FrozenModel":
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            frozen = _deep_freeze(value)
            if frozen is not value:
                object.__setattr__(self, field_name, frozen)
        return self


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
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON mapping keys must be strings")
        return {key: _normalise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalised = (_normalise(item) for item in value)
        return sorted(
            normalised,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            ),
        )
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


def canonical_unordered[CanonicalT](
    values: tuple[CanonicalT, ...],
) -> tuple[CanonicalT, ...]:
    """Normalize a semantically set-like tuple by canonical bytes and remove duplicates."""
    unique = {canonical_json(value): value for value in values}
    return tuple(unique[key] for key in sorted(unique))
