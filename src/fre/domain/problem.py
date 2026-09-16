"""Formal problem contracts."""

from typing import Literal

from fre.domain.common import FrozenModel, JsonValue, OutputContract


class DecisionVariable(FrozenModel):
    id: str
    name: str
    domain: JsonValue | None = None


class ObservableSpec(FrozenModel):
    id: str
    description: str
    unit: str | None = None


class ObjectiveSpec(FrozenModel):
    id: str
    name: str
    direction: Literal["MIN", "MAX", "TARGET", "LEXICOGRAPHIC"]
    unit: str | None = None
    priority: int | None = None
    evaluator_ref: str | None = None
    description: str


class ConstraintSpec(FrozenModel):
    id: str
    description: str
    kind: Literal["HARD", "SOFT"]
    verification_mode: Literal["DETERMINISTIC", "MODEL", "HUMAN", "UNAVAILABLE"]
    verifier_ref: str | None = None
    source_refs: tuple[str, ...] = ()


class UnknownSpec(FrozenModel):
    id: str
    description: str
    domain: JsonValue | None = None
    decision_relevance: float | None = None
    resolvable: bool | None = None
    candidate_actions: tuple[str, ...] = ()


class AcceptanceCriterion(FrozenModel):
    id: str
    predicate_description: str
    verification_mode: str
    required: bool


class ProblemSpec(FrozenModel):
    decision_variables: tuple[DecisionVariable, ...] = ()
    objectives: tuple[ObjectiveSpec, ...] = ()
    constraints: tuple[ConstraintSpec, ...] = ()
    assumptions: tuple[str, ...] = ()
    unknowns: tuple[UnknownSpec, ...] = ()
    observables: tuple[ObservableSpec, ...] = ()
    acceptance_criteria: tuple[AcceptanceCriterion, ...] = ()
    output_contract: OutputContract
