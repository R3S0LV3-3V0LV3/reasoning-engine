"""Epistemic graph contracts."""

from enum import StrEnum
from uuid import UUID

from pydantic import Field

from fre.domain.common import ArtifactRef, ConfidenceAssessment, FrozenModel, JsonValue


class LedgerNodeType(StrEnum):
    FACT = "FACT"
    OBSERVATION = "OBSERVATION"
    INFERENCE = "INFERENCE"
    ASSUMPTION = "ASSUMPTION"
    HYPOTHESIS = "HYPOTHESIS"
    PROPOSAL = "PROPOSAL"
    DECISION = "DECISION"
    UNKNOWN = "UNKNOWN"
    CHALLENGE = "CHALLENGE"
    EVIDENCE = "EVIDENCE"
    ARTIFACT = "ARTIFACT"


class EpistemicStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    PROVISIONAL = "PROVISIONAL"
    CONTESTED = "CONTESTED"
    REFUTED = "REFUTED"
    UNRESOLVED = "UNRESOLVED"
    STALE = "STALE"


class LedgerRelation(StrEnum):
    SUPPORTS = "SUPPORTS"
    DEPENDS_ON = "DEPENDS_ON"
    DERIVED_FROM = "DERIVED_FROM"
    CONTRADICTS = "CONTRADICTS"
    SUPERSEDES = "SUPERSEDES"
    FALSIFIES = "FALSIFIES"
    RESOLVES = "RESOLVES"
    OBSERVED_IN = "OBSERVED_IN"
    EVALUATES = "EVALUATES"
    SELECTS = "SELECTS"
    REJECTS = "REJECTS"


class LedgerNode(FrozenModel):
    node_id: UUID
    revision: int
    node_type: LedgerNodeType
    content: JsonValue
    epistemic_status: EpistemicStatus
    confidence: ConfidenceAssessment | None = None
    source_refs: tuple[ArtifactRef, ...] = ()


class LedgerEdge(FrozenModel):
    edge_id: UUID
    source_id: UUID
    target_id: UUID
    relation: LedgerRelation
    independence_group: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
