"""Representation projection contracts."""

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_serializer

from fre.domain.common import ArtifactRef, FrozenModel, JsonValue, ObjectRef


class RepresentationKind(StrEnum):
    TYPED_CONSTRAINT_SET = "TYPED_CONSTRAINT_SET"
    DECISION_TABLE = "DECISION_TABLE"
    PARETO_OBJECTIVE_MATRIX = "PARETO_OBJECTIVE_MATRIX"
    DEPENDENCY_DAG = "DEPENDENCY_DAG"
    CAUSAL_GRAPH = "CAUSAL_GRAPH"
    STATE_MACHINE = "STATE_MACHINE"
    EVENT_LOG = "EVENT_LOG"
    KNOWLEDGE_GRAPH = "KNOWLEDGE_GRAPH"
    CAUSAL_DAG = "CAUSAL_DAG"
    CSP = "CSP"
    ILP = "ILP"
    SCENARIO_TREE = "SCENARIO_TREE"
    HYPERGRAPH = "HYPERGRAPH"
    PROPERTY_GRAPH = "PROPERTY_GRAPH"
    MORPHOLOGICAL_SPACE = "MORPHOLOGICAL_SPACE"
    TEXT_FALLBACK = "TEXT_FALLBACK"
    TEXT_TABLE_FALLBACK = "TEXT_TABLE_FALLBACK"


class RepresentationScoreComponent(FrozenModel):
    feature: str
    contribution: float
    basis: str


class RepresentationView(FrozenModel):
    id: str
    kind: RepresentationKind
    role: Literal["PRIMARY", "AUXILIARY"]
    compatibility_score: float
    score_components: tuple[RepresentationScoreComponent, ...] = ()
    purpose: str = "structural projection"
    expected_value: str
    builder_ref: str
    builder_available: bool = True
    source_object_refs: tuple[ObjectRef, ...] = ()
    builder_version: str = "1.0"
    registry_version: str = "wave3-m04-registry/1.0"
    selection_policy_version: str = "wave3-m04/1.0"
    limitations: tuple[str, ...] = ()


class RepresentationPlan(FrozenModel):
    # Wave 3's original persisted 1.0 shape did not contain this binding.  Keep
    # accepting that shape so old events and snapshots remain replayable; all
    # newly selected plans supply the hash.
    problem_spec_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    views: tuple[RepresentationView, ...]
    selection_basis: tuple[str, ...] = ()
    omitted_reasons: tuple[str, ...] = ()

    @model_serializer(mode="plain")
    def serialize_compatibly(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "views": self.views,
            "selection_basis": self.selection_basis,
            "omitted_reasons": self.omitted_reasons,
        }
        if self.problem_spec_hash is not None:
            payload["problem_spec_hash"] = self.problem_spec_hash
        return payload


class RepresentationArtifact(FrozenModel):
    requested_kind: RepresentationKind
    actual_kind: RepresentationKind
    builder_id: str
    builder_version: str
    registry_version: str
    selection_policy_version: str
    problem_spec_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_snapshot_version: int = Field(ge=0)
    content: JsonValue
    projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_artifact_ref: ArtifactRef | None = None
    fallback_reason: str | None = None
    limitations: tuple[str, ...] = ()
