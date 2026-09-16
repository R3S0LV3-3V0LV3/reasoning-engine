"""Immutable, hash-linked epistemic ledger contracts."""

from enum import StrEnum
from uuid import UUID

from pydantic import Field, model_validator

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
    CONTESTED = "CONTESTED"


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


class ContradictionState(StrEnum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"


class LedgerNodeRef(FrozenModel):
    node_id: UUID
    revision: int = Field(ge=1)


class LedgerNode(FrozenModel):
    node_id: UUID
    revision: int = Field(ge=1)
    node_type: LedgerNodeType
    content: JsonValue
    epistemic_status: EpistemicStatus
    confidence: ConfidenceAssessment | None = None
    source_refs: tuple[ArtifactRef, ...] = ()
    schema_version: str = "1.0"
    created_at: str
    created_by_action: UUID
    producing_module: str
    provenance_refs: tuple[str, ...] = ()
    predecessor_revision_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    revision_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_predecessor(self) -> "LedgerNode":
        if (self.revision == 1) != (self.predecessor_revision_hash is None):
            raise ValueError("only the first revision may omit predecessor_revision_hash")
        return self

    @property
    def ref(self) -> LedgerNodeRef:
        return LedgerNodeRef(node_id=self.node_id, revision=self.revision)


class LedgerEdge(FrozenModel):
    edge_id: UUID
    source: LedgerNodeRef
    target: LedgerNodeRef
    relation: LedgerRelation
    independence_group: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class DependencyConfidenceEnvelope(FrozenModel):
    affected_node_ref: LedgerNodeRef
    changed_dependency_refs: tuple[LedgerNodeRef, ...]
    maximum_dependency_depth: int = Field(ge=1)
    previous_confidence_range: tuple[float, float] | None = None
    current_confidence_range: tuple[float, float] | None = None
    confidence_delta_interval: tuple[float, float] | None = None
    direct_dependencies_changed: int = Field(ge=0)
    direct_dependency_fraction: float = Field(ge=0, le=1)
    independence_groups: tuple[str, ...] = ()
    method_version: str = "dependency-envelope/1.0"
    reason_code: str = "UPSTREAM_REVISION"
    source_event_refs: tuple[str, ...] = ()


class ContradictionResolution(FrozenModel):
    contradiction_edge_ids: tuple[UUID, ...]
    resolver_ref: LedgerNodeRef
    affected_refs: tuple[LedgerNodeRef, ...]
    status_transitions: tuple[tuple[LedgerNodeRef, EpistemicStatus], ...]
    outcome: str
    action_id: UUID
    module_id: str


class LedgerProjection(FrozenModel):
    projection_version: str = "1.0"
    nodes: tuple[LedgerNode, ...] = ()
    edges: tuple[LedgerEdge, ...] = ()
    status_overlays: tuple[tuple[LedgerNodeRef, EpistemicStatus], ...] = ()
    stale_envelopes: tuple[DependencyConfidenceEnvelope, ...] = ()
    resolved_contradiction_edges: tuple[UUID, ...] = ()


class GraphValidationResult(FrozenModel):
    valid: bool
    errors: tuple[str, ...] = ()


class LedgerError(ValueError):
    """Base class for ledger invariant violations."""


class LedgerCycleError(LedgerError):
    pass


class DanglingLedgerReference(LedgerError):
    pass


class InvalidRevisionError(LedgerError):
    pass


class InvalidLedgerRelation(LedgerError):
    pass


class InvalidStatusTransition(LedgerError):
    pass


class InvalidContradictionResolution(LedgerError):
    pass


class RevisionHashMismatch(LedgerError):
    pass
