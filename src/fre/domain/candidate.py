"""Immutable candidate contracts."""

from enum import StrEnum
from uuid import UUID

from fre.domain.common import FrozenModel, JsonValue


class CandidateStatus(StrEnum):
    PROPOSED = "PROPOSED"
    VIABLE = "VIABLE"
    CONDITIONAL = "CONDITIONAL"
    INVALID = "INVALID"
    REJECTED = "REJECTED"


class CandidateComponent(FrozenModel):
    id: str
    kind: str
    configuration: dict[str, JsonValue]
    dependencies: tuple[str, ...] = ()
    interfaces: tuple[str, ...] = ()


class PredictedEffect(FrozenModel):
    objective_id: str
    description: str


class Candidate(FrozenModel):
    candidate_id: UUID
    revision: int
    parent_ids: tuple[UUID, ...] = ()
    generator_operator: str
    representation_refs: tuple[str, ...] = ()
    title: str
    summary: str
    components: tuple[CandidateComponent, ...]
    assumptions: tuple[str, ...] = ()
    predicted_effects: tuple[PredictedEffect, ...] = ()
    status: CandidateStatus
    structural_fingerprint: str
