"""Strict structured-output schemas and deterministic registry."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from fre.domain.common import FrozenModel, JsonValue, canonical_hash, canonical_json
from fre.domain.semantic import EpistemicOriginLabel, SourceAnchor
from fre.domain.task import HorizonClass, Ordinal4, SearchSpaceClass, TaskType


class StrictOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ClassificationDimensionProposal(StrictOutput):
    estimate: Ordinal4
    confidence: float = Field(ge=0, le=1)
    conservative_upper: Ordinal4
    anchors: tuple[SourceAnchor, ...] = ()
    rationale: str


class ClassificationOutput(StrictOutput):
    task_type: TaskType
    consequence: ClassificationDimensionProposal
    reversibility: ClassificationDimensionProposal
    ambiguity: ClassificationDimensionProposal
    evidence_scarcity: ClassificationDimensionProposal
    search_space: SearchSpaceClass
    search_space_confidence: float = Field(ge=0, le=1)
    horizon: HorizonClass
    horizon_confidence: float = Field(ge=0, le=1)


class ProblemItemProposal(StrictOutput):
    id: str
    kind: str
    description: str
    origin: EpistemicOriginLabel
    anchors: tuple[SourceAnchor, ...] = ()
    supporting_refs: tuple[str, ...] = ()
    basis: str | None = None
    policy_basis: str | None = None
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class ProblemFormalisationOutput(StrictOutput):
    items: tuple[ProblemItemProposal, ...]


class RepresentationAdjudicationOutput(StrictOutput):
    selected_kinds: tuple[str, ...]
    explanation: str


class OutputSchemaDefinition(FrozenModel):
    schema_id: str
    schema_version: str
    model_name: str
    module_id: str
    operation: str
    schema_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class OutputSchemaError(ValueError):
    pass


class OutputSchemaRegistry:
    def __init__(self) -> None:
        self._items: dict[tuple[str, str], tuple[OutputSchemaDefinition, type[BaseModel]]] = {}

    def register(
        self, schema_id: str, version: str, model: type[BaseModel], module: str, operation: str
    ) -> None:
        schema_hash = canonical_hash(model.model_json_schema())
        definition = OutputSchemaDefinition(
            schema_id=schema_id,
            schema_version=version,
            model_name=f"{model.__module__}.{model.__qualname__}",
            module_id=module,
            operation=operation,
            schema_hash=schema_hash,
        )
        key = (schema_id, version)
        existing = self._items.get(key)
        if existing is not None and existing[0] != definition:
            raise OutputSchemaError(f"conflicting output schema: {schema_id}@{version}")
        self._items[key] = (definition, model)

    def get(self, schema_id: str, version: str) -> tuple[OutputSchemaDefinition, type[BaseModel]]:
        try:
            definition, model = self._items[(schema_id, version)]
        except KeyError as error:
            raise OutputSchemaError(f"unknown output schema: {schema_id}@{version}") from error
        if canonical_hash(model.model_json_schema()) != definition.schema_hash:
            raise OutputSchemaError("output schema integrity mismatch")
        return definition, model

    def validate(self, schema_id: str, version: str, value: Any) -> BaseModel:
        return self.get(schema_id, version)[1].model_validate_json(
            canonical_json(value), strict=True
        )


def default_output_schema_registry() -> OutputSchemaRegistry:
    registry = OutputSchemaRegistry()
    registry.register("m01.classification-output", "1.0", ClassificationOutput, "M01", "classify")
    registry.register(
        "m03.problem-formalisation-output", "1.0", ProblemFormalisationOutput, "M03", "formalise"
    )
    registry.register(
        "m04.representation-adjudication-output",
        "1.0",
        RepresentationAdjudicationOutput,
        "M04",
        "adjudicate",
    )
    return registry
