"""Task ingestion and classification contracts."""

from enum import StrEnum
from uuid import UUID

from pydantic import Field, model_validator

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
    """Complete per-axis classification provenance.

    Every material classification dimension that influences budget or
    routing carries: a validated estimate (``estimated``), the value actually
    used downstream after floors/escalation (``effective``), a
    confidence/uncertainty treatment (``confidence``), a conservative bound
    (``conservative_upper``), a rationale, resolvable support
    (``source_anchors``) or an explicit deterministic-policy basis
    (``basis``/``policy_version``), and -- when ``effective`` differs from
    ``estimated`` -- the audited reason for that deviation
    (``override_basis``).
    """

    estimated: str
    effective: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    conservative_upper: str | None = None
    source_anchors: tuple[SourceAnchor, ...] = ()
    basis: str
    rationale: str | None = None
    override_basis: str | None = None
    policy_version: str | None = None


class FloorOverrideRecord(FrozenModel):
    """Audit trail entry for a deterministic floor/escalation changing a dimension."""

    axis: str
    reason: str
    policy_version: str
    policy_hash: str
    previous_floor: str
    new_floor: str
    approving_rule: str


class ClassificationRecord(FrozenModel):
    policy_version: str
    policy_hash: str
    mode: str
    fallback_used: bool
    dimensions: dict[str, ClassificationDimensionResult]
    model_call_key: str | None = None
    diagnostics: tuple[str, ...] = ()
    floor_overrides: tuple[FloorOverrideRecord, ...] = ()

    @model_validator(mode="after")
    def _floor_overrides_are_exhaustive(self) -> "ClassificationRecord":
        deviated = {
            name
            for name, dimension in self.dimensions.items()
            if dimension.effective != dimension.estimated
        }
        audited = {record.axis for record in self.floor_overrides}
        if deviated != audited:
            raise ValueError(
                "every dimension whose effective value differs from its estimate must "
                "carry a matching, audited FloorOverrideRecord (unaudited floor change)"
            )
        return self


class ClassificationBlocked(ValueError):
    pass
