"""Constraint and objective evaluation contracts."""

from enum import StrEnum
from uuid import UUID

from fre.domain.common import ArtifactRef, ConfidenceAssessment, FrozenModel


class ConstraintStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"


class EvidenceQuality(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ConstraintEvaluation(FrozenModel):
    candidate_id: UUID
    constraint_id: str
    status: ConstraintStatus
    confidence: ConfidenceAssessment
    evidence_refs: tuple[str, ...] = ()
    verifier_version: str


class ObjectiveEstimate(FrozenModel):
    candidate_id: UUID
    objective_id: str
    point: float | None = None
    lower: float | None = None
    upper: float | None = None
    distribution_ref: ArtifactRef | None = None
    quality: EvidenceQuality
    evidence_refs: tuple[str, ...] = ()
