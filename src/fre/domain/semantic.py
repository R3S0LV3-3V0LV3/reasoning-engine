"""Shared Wave 3 semantic provenance and model-call contracts."""

from enum import StrEnum
from uuid import UUID

from pydantic import Field, model_validator

from fre.domain.common import ArtifactRef, FrozenModel, JsonValue, ObjectRef


class SourceKind(StrEnum):
    TASK_TEXT = "TASK_TEXT"
    TASK_FIELD = "TASK_FIELD"
    ARTIFACT = "ARTIFACT"


class SourceAnchor(FrozenModel):
    source_kind: SourceKind
    source_ref: ObjectRef | ArtifactRef
    selector: str
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    excerpt_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def valid_range(self) -> "SourceAnchor":
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("source span requires both bounds")
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_end <= self.char_start
        ):
            raise ValueError("source span must be non-empty")
        return self


class EpistemicOriginLabel(StrEnum):
    EXPLICIT_INPUT = "EXPLICIT_INPUT"
    SUPPORTED_INFERENCE = "SUPPORTED_INFERENCE"
    WORKING_ASSUMPTION = "WORKING_ASSUMPTION"
    UNRESOLVED = "UNRESOLVED"
    CONTRADICTED = "CONTRADICTED"


class EpistemicItemProvenance(FrozenModel):
    origin: EpistemicOriginLabel
    anchors: tuple[SourceAnchor, ...] = ()
    supporting_refs: tuple[str, ...] = ()
    basis: str | None = None
    policy_basis: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_origin(self) -> "EpistemicItemProvenance":
        if self.origin is EpistemicOriginLabel.EXPLICIT_INPUT and not self.anchors:
            raise ValueError("EXPLICIT_INPUT requires a SourceAnchor")
        if self.origin is EpistemicOriginLabel.SUPPORTED_INFERENCE and (
            not self.supporting_refs or not self.basis
        ):
            raise ValueError("SUPPORTED_INFERENCE requires support and basis")
        if self.origin is EpistemicOriginLabel.WORKING_ASSUMPTION and (
            not self.basis or not self.policy_basis
        ):
            raise ValueError("WORKING_ASSUMPTION requires basis and policy basis")
        return self


class StructuredModelStatus(StrEnum):
    SUCCESS = "SUCCESS"
    INVALID_STRUCTURED_OUTPUT = "INVALID_STRUCTURED_OUTPUT"
    UNAVAILABLE = "UNAVAILABLE"
    TRANSIENT_FAILURE = "TRANSIENT_FAILURE"
    PERMANENT_FAILURE = "PERMANENT_FAILURE"


class SemanticCallUsage(FrozenModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    duration_ms: int | None = Field(default=None, ge=0)


class SemanticCallCharge(FrozenModel):
    llm_calls: int = Field(default=1, ge=0, le=1)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    basis: str


class StructuredModelRequest(FrozenModel):
    role: str
    messages: tuple[dict[str, JsonValue], ...]
    output_schema_id: str
    output_schema_version: str
    max_input_tokens: int = Field(ge=0)
    max_output_tokens: int = Field(ge=0)
    idempotency_key: str = Field(pattern=r"^[0-9a-f]{64}$")


class StructuredModelResult(FrozenModel):
    status: StructuredModelStatus
    adapter_id: str
    model_id: str
    raw_response: bytes = b""
    decoded: JsonValue | None = None
    usage: SemanticCallUsage = SemanticCallUsage()
    diagnostics: tuple[str, ...] = ()


class SemanticModelCallRecord(FrozenModel):
    call_id: UUID
    idempotency_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    module_id: str
    operation: str
    module_version: str
    policy_version: str
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_id: str
    prompt_version: str
    template_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_schema_id: str
    output_schema_version: str
    output_schema_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_role: str
    adapter_id: str
    model_id: str
    status: StructuredModelStatus
    raw_artifact: ArtifactRef | None = None
    usage: SemanticCallUsage = SemanticCallUsage()
    policy_charge: SemanticCallCharge
    repair_parent_key: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    fallback_used: bool = False
