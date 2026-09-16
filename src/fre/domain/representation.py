"""Representation projection contracts."""

from enum import StrEnum
from typing import Literal

from fre.domain.common import FrozenModel, ObjectRef


class RepresentationKind(StrEnum):
    DECISION_TABLE = "DECISION_TABLE"
    DEPENDENCY_DAG = "DEPENDENCY_DAG"
    STATE_MACHINE = "STATE_MACHINE"
    EVENT_LOG = "EVENT_LOG"
    KNOWLEDGE_GRAPH = "KNOWLEDGE_GRAPH"
    CAUSAL_DAG = "CAUSAL_DAG"
    CSP = "CSP"
    ILP = "ILP"
    SCENARIO_TREE = "SCENARIO_TREE"
    HYPERGRAPH = "HYPERGRAPH"
    MORPHOLOGICAL_SPACE = "MORPHOLOGICAL_SPACE"
    TEXT_FALLBACK = "TEXT_FALLBACK"


class RepresentationView(FrozenModel):
    id: str
    kind: RepresentationKind
    role: Literal["PRIMARY", "AUXILIARY"]
    compatibility_score: float
    expected_value: str
    builder_ref: str
    source_object_refs: tuple[ObjectRef, ...] = ()


class RepresentationPlan(FrozenModel):
    views: tuple[RepresentationView, ...]
    selection_basis: tuple[str, ...] = ()
