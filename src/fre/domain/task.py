"""Task ingestion and classification contracts."""

from enum import StrEnum
from uuid import UUID

from pydantic import Field

from fre.domain.common import ArtifactRef, FrozenModel, OutputContract, PermissionSet
from fre.domain.semantic import SourceAnchor


class TaskType(StrEnum):
    ANALYSIS = "ANALYSIS"
    DECISION = "DECISION"
    DESIGN = "DESIGN"
    DIAGNOSIS = "DIAGNOSIS"
    RESEARCH = "RESEARCH"


class Ordinal4(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SearchSpaceClass(StrEnum):
    CLOSED = "CLOSED"
    BOUNDED = "BOUNDED"
    OPEN = "OPEN"


class HorizonClass(StrEnum):
    IMMEDIATE = "IMMEDIATE"
    SHORT = "SHORT"
    LONG = "LONG"


class OutputForm(StrEnum):
    TEXT = "TEXT"
    STRUCTURED = "STRUCTURED"
    ARTIFACT = "ARTIFACT"


class TaskEnvelope(FrozenModel):
    task_id: UUID
    text: str
    attachments: tuple[ArtifactRef, ...] = ()
    explicit_constraints: tuple[str, ...] = ()
    requested_output: OutputContract
    user_metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    execution_permissions: PermissionSet


class TaskSignature(FrozenModel):
    task_type: TaskType
    consequence: Ordinal4
    irreversibility: Ordinal4
    ambiguity: Ordinal4
    search_space: SearchSpaceClass
    evidence_scarcity: Ordinal4
    horizon: HorizonClass
    output_form: OutputForm
    dimension_confidence: dict[str, float | None]
    evidence_refs: tuple[str, ...] = ()


class ClassificationDimensionResult(FrozenModel):
    estimated: str
    effective: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    conservative_upper: str | None = None
    source_anchors: tuple[SourceAnchor, ...] = ()
    basis: str


class ClassificationRecord(FrozenModel):
    policy_version: str
    policy_hash: str
    mode: str
    fallback_used: bool
    dimensions: dict[str, ClassificationDimensionResult]
    model_call_key: str | None = None
    diagnostics: tuple[str, ...] = ()


class ClassificationBlocked(ValueError):
    pass
