"""Strict structured-output schemas and deterministic registry."""

import hashlib
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from fre.domain.common import FrozenModel, JsonValue, canonical_json
from fre.domain.semantic import EpistemicOriginLabel, SourceAnchor
from fre.domain.task import HorizonClass, Ordinal4, SearchSpaceClass, TaskType

# Resource-exhaustion defence for structured-output schemas. These schemas are
# registered from trusted, hardcoded model definitions today, but the registry
# is a hard boundary: any future dynamic/plugin registration path must not be
# able to smuggle a pathologically large or deeply-nested schema through it.
MAX_SCHEMA_CANONICAL_BYTES = 65_536
MAX_SCHEMA_NESTING_DEPTH = 20
# `$ref` is only supported when it stays within the schema's own local
# `$defs` table. Remote/external refs and unbounded `patternProperties`
# (regex-driven, a classic ReDoS vector) are rejected outright.
_UNSUPPORTED_CONSTRUCTS = ("patternProperties",)


def canonical_schema_bytes(model: type[BaseModel]) -> bytes:
    """Deterministic canonical JSON bytes for a model's JSON-Schema representation."""
    return canonical_json(model.model_json_schema())


def canonical_schema_hash(model: type[BaseModel]) -> str:
    """Hash of `canonical_schema_bytes` (== `canonical_hash(model.model_json_schema())`)."""
    return hashlib.sha256(canonical_schema_bytes(model)).hexdigest()


def _resolve_local_ref(root: JsonValue, ref: str) -> JsonValue | None:
    """Resolve a local JSON-Schema pointer (e.g. "#/$defs/Thing") against `root`."""
    if not ref.startswith("#/"):
        return None
    node: JsonValue = root
    for part in ref[2:].split("/"):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def _schema_nesting_depth(
    value: JsonValue,
    *,
    root: JsonValue | None = None,
    depth: int = 0,
    visiting: frozenset[str] = frozenset(),
) -> int:
    # Pydantic flattens nested submodels into a shared `$defs` table and
    # `$ref`s them in, so raw dict/list nesting alone would under-count the
    # true composition depth of, e.g., a deep chain of nested models. Resolve
    # local refs to measure real depth. A handful of domain types (e.g.
    # `JsonValue`) are legitimately self-referential; `visiting` tracks refs
    # already unrolled along the current path so a direct or mutual cycle is
    # counted once (a finite, intentional recursive type) rather than
    # climbing forever. `depth` also strictly increases on every recursive
    # step regardless, so the bail-out below is a second, independent bound.
    root = value if root is None else root
    if depth > MAX_SCHEMA_NESTING_DEPTH:
        # Bail out early rather than recursing arbitrarily far on adversarial input.
        return depth
    if isinstance(value, dict):
        ref = value.get("$ref")
        if isinstance(ref, str):
            if ref in visiting:
                return depth + 1
            resolved = _resolve_local_ref(root, ref)
            if resolved is not None:
                return _schema_nesting_depth(
                    resolved, root=root, depth=depth + 1, visiting=visiting | {ref}
                )
        if not value:
            return depth
        return max(
            _schema_nesting_depth(item, root=root, depth=depth + 1, visiting=visiting)
            for item in value.values()
        )
    if isinstance(value, list):
        if not value:
            return depth
        return max(
            _schema_nesting_depth(item, root=root, depth=depth + 1, visiting=visiting)
            for item in value
        )
    return depth


def _check_supported_constructs(value: JsonValue) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _UNSUPPORTED_CONSTRUCTS:
                raise OutputSchemaError(f"unsupported schema construct: {key}")
            if key == "$ref" and isinstance(item, str) and not item.startswith("#/$defs/"):
                raise OutputSchemaError("unsupported schema construct: external $ref")
            _check_supported_constructs(item)
    elif isinstance(value, list):
        for item in value:
            _check_supported_constructs(item)


def enforce_schema_limits(model: type[BaseModel]) -> None:
    """Reject schemas that exceed size, nesting, or supported-construct limits."""
    json_schema = model.model_json_schema()
    canonical = canonical_json(json_schema)
    if len(canonical) > MAX_SCHEMA_CANONICAL_BYTES:
        raise OutputSchemaError("schema exceeds maximum canonical byte size")
    if _schema_nesting_depth(json_schema) > MAX_SCHEMA_NESTING_DEPTH:
        raise OutputSchemaError("schema exceeds maximum nesting depth")
    _check_supported_constructs(json_schema)


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
        enforce_schema_limits(model)
        schema_hash = canonical_schema_hash(model)
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
        if canonical_schema_hash(model) != definition.schema_hash:
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
