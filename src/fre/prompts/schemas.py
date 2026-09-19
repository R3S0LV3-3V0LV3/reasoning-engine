"""Strict structured-output schemas and deterministic registry."""

from functools import lru_cache
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from fre.domain.common import FrozenModel, JsonValue, canonical_hash, canonical_json
from fre.domain.semantic import EpistemicOriginLabel, SourceAnchor, SupportRef
from fre.domain.task import HorizonClass, Ordinal4, SearchSpaceClass, TaskType

# Resource-exhaustion defence for structured-output schemas. These schemas are
# registered from trusted, hardcoded model definitions today, but the registry
# is a hard boundary: any future dynamic/plugin registration path must not be
# able to smuggle a pathologically large or deeply-nested schema through it.
# Finding #10 (C04 remediation, documentation-only -- logged as a deferred
# cleanup item, see WAVE3_DEFERRED_CLEANUP_REGISTER.md): this bounds the
# canonical (un-dereferenced) JSON-Schema byte size, i.e. `$ref`s counted as
# their short pointer strings, not the fully dereferenced/expanded form a
# consumer might materialize. A schema with many references to a large shared
# `$def` could therefore expand to a much larger byte count than this cap
# once dereferenced. This is accepted as-is today because there is no
# dynamic/runtime schema registration path: every schema in
# `default_output_schema_registry` is hardcoded at startup and reviewed, so
# nothing here is attacker- or caller-controlled. Re-audit this assumption
# before adding any registration path that accepts schemas at runtime.
MAX_SCHEMA_CANONICAL_BYTES = 65_536
MAX_SCHEMA_NESTING_DEPTH = 20
# `$ref` is only supported when it stays within the schema's own local
# `$defs` table. Remote/external refs and unbounded `patternProperties`
# (regex-driven, a classic ReDoS vector) are rejected outright.
_UNSUPPORTED_CONSTRUCTS = ("patternProperties",)
# Finding #9 (C04 remediation): a string field's `"pattern"` constraint is
# itself a regex-driven ReDoS vector -- an attacker-controlled or merely
# careless catastrophically-backtracking pattern registered as an output
# schema would let a single crafted (or even just unlucky) provider response
# hang validation indefinitely. The safe default is to disallow `pattern`
# entirely. Two of the three schemas registered today (`ClassificationOutput`,
# `ProblemFormalisationOutput`, via the shared `ArtifactRef.sha256` and
# `SourceAnchor.excerpt_hash` domain fields) already carry one -- but only
# ever this one, fixed, hardcoded, linear, non-backtracking hex-digest
# pattern, never anything attacker- or caller-supplied. Rather than break
# those two schemas, the denylist allows exactly this pattern value and
# rejects every other one; any newly registered schema that needs a different
# `pattern` must add it here explicitly, as a deliberate, reviewed exception.
_ALLOWED_SCHEMA_PATTERNS = frozenset({r"^[0-9a-f]{64}$"})


def canonical_schema_bytes(model: type[BaseModel]) -> bytes:
    """Deterministic canonical JSON bytes for a model's JSON-Schema representation."""
    return canonical_json(model.model_json_schema())


@lru_cache(maxsize=None)
def canonical_schema_hash(model: type[BaseModel]) -> str:
    """Hash of a model's canonical JSON-Schema representation.

    Delegates to the shared `canonical_hash` primitive (EU-04, C04 cleanup,
    item #4) rather than hand-rolling `hashlib.sha256(canonical_json(...))`
    itself -- `canonical_hash(value) == hashlib.sha256(canonical_json(value)).hexdigest()`,
    so `canonical_hash(model.model_json_schema())` is byte-for-byte identical
    to the old hand-rolled computation for every currently-registered schema
    (confirmed by an explicit before/after parity test;
    see `test_canonical_schema_hash_matches_canonical_hash_of_the_json_schema`
    in `tests/unit/test_output_schema_binding.py`). `canonical_schema_bytes`
    is kept as a separate helper (still used directly by resource-exhaustion
    size checks in `enforce_schema_limits`) and remains equal to
    `canonical_json(model.model_json_schema())`.

    Memoized per model class (EU-01, C04 cleanup, item #1): a registered
    output-schema model's JSON-Schema shape is fixed once the class is
    defined, so re-deriving `model_json_schema()` and re-hashing it on every
    `OutputSchemaRegistry.get()` call was pure, deterministic, wasted work.
    Caching by `model` (a `type` object, hashable and stable for the
    lifetime of the process) makes repeated lookups O(1) after the first
    call without changing the returned value -- this is a pure caching
    change, not a hash-algorithm change; output is byte-identical to the
    uncached computation for every input. `OutputSchemaRegistry.get()` still
    performs its integrity comparison (`canonical_schema_hash(model) !=
    definition.schema_hash`) on every call -- only the underlying
    computation is now cheap, not the check itself.
    """
    return canonical_hash(model.model_json_schema())


def _resolve_local_ref(root: JsonValue, ref: str) -> JsonValue | None:
    """Resolve a local JSON-Schema pointer (e.g. "#/$defs/Thing") against `root`.

    EU-06 (C04 cleanup, item #6): deliberately does NOT perform RFC 6901
    `~0`/`~1` unescaping, unlike `source_anchors._resolve_pointer` (which
    does full RFC 6901 unescaping). This is not an oversight -- the two
    functions resolve pointers over disjoint, differently-sourced document
    kinds. Every `ref` this function ever sees is a `$ref` string Pydantic
    itself generated inside `model.model_json_schema()`'s own `$defs`
    table, keyed by the model's own Python class/type names -- identifiers
    that can never contain a literal `/` or `~`, so there is nothing to
    escape in the first place. `_resolve_pointer`, in contrast, resolves
    `SourceAnchor` selectors against arbitrary externally-sourced JSON
    documents, whose object keys (e.g. real-world field/column names) can
    legitimately contain `/` or `~` and therefore MUST be RFC 6901-escaped
    on the wire and unescaped here. Do not "fix" this function into RFC
    6901 compliance -- for its actual, always-Pydantic-generated input
    domain, a raw `/`-split is both correct and simpler; escaping semantics
    should track each function's own input domain, not be unified for its
    own sake.
    """
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
    _height_cache: dict[str, int] | None = None,
    _in_progress: set[str] | None = None,
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
    #
    # `_height_cache`/`_in_progress` (finding #3, C04 remediation): a `$def`
    # reused from many sibling paths (e.g. the same submodel referenced by
    # several fields, each nested tens of levels deep) would otherwise be
    # re-expanded once per reference *path*, which is exponential in the
    # number of reuse sites. `_height_cache` memoizes each ref's own
    # "intrinsic height" (its depth contribution computed once, starting
    # fresh at depth 0, independent of where it is reached from) the first
    # time it is fully resolved, so every later reference to the same ref
    # anywhere in the schema is an O(1) lookup. `_in_progress` tracks refs
    # currently being resolved *for their own height computation* so that a
    # true mutual cycle between two distinct `$def`s (A -> B -> A) cannot spin
    # forever before either is cached -- it is a stricter, computation-scoped
    # analogue of `visiting`, which only protects a single calling path.
    root = value if root is None else root
    height_cache = {} if _height_cache is None else _height_cache
    in_progress = set() if _in_progress is None else _in_progress
    if depth > MAX_SCHEMA_NESTING_DEPTH:
        # Bail out early rather than recursing arbitrarily far on adversarial input.
        return depth
    if isinstance(value, dict):
        ref = value.get("$ref")
        if isinstance(ref, str):
            if ref in visiting or ref in in_progress:
                return depth + 1
            if ref in height_cache:
                return depth + 1 + height_cache[ref]
            resolved = _resolve_local_ref(root, ref)
            if resolved is not None:
                in_progress.add(ref)
                try:
                    height = _schema_nesting_depth(
                        resolved,
                        root=root,
                        depth=0,
                        visiting=frozenset({ref}),
                        _height_cache=height_cache,
                        _in_progress=in_progress,
                    )
                finally:
                    in_progress.discard(ref)
                height_cache[ref] = height
                return depth + 1 + height
        if not value:
            return depth
        return max(
            _schema_nesting_depth(
                item,
                root=root,
                depth=depth + 1,
                visiting=visiting,
                _height_cache=height_cache,
                _in_progress=in_progress,
            )
            for item in value.values()
        )
    if isinstance(value, list):
        if not value:
            return depth
        return max(
            _schema_nesting_depth(
                item,
                root=root,
                depth=depth + 1,
                visiting=visiting,
                _height_cache=height_cache,
                _in_progress=in_progress,
            )
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
            if key == "pattern" and item not in _ALLOWED_SCHEMA_PATTERNS:
                raise OutputSchemaError(
                    "unsupported schema construct: pattern (not on the reviewed allowlist)"
                )
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


class TaskTypeProposal(StrictOutput):
    estimate: TaskType
    confidence: float = Field(ge=0, le=1)
    anchors: tuple[SourceAnchor, ...] = ()
    rationale: str


class SearchSpaceProposal(StrictOutput):
    estimate: SearchSpaceClass
    confidence: float = Field(ge=0, le=1)
    anchors: tuple[SourceAnchor, ...] = ()
    rationale: str


class HorizonProposal(StrictOutput):
    estimate: HorizonClass
    confidence: float = Field(ge=0, le=1)
    anchors: tuple[SourceAnchor, ...] = ()
    rationale: str


class ClassificationOutput(StrictOutput):
    task_type: TaskTypeProposal
    consequence: ClassificationDimensionProposal
    reversibility: ClassificationDimensionProposal
    ambiguity: ClassificationDimensionProposal
    evidence_scarcity: ClassificationDimensionProposal
    search_space: SearchSpaceProposal
    horizon: HorizonProposal


class ProblemItemProposal(StrictOutput):
    id: str
    kind: str
    description: str
    origin: EpistemicOriginLabel
    anchors: tuple[SourceAnchor, ...] = ()
    # Deprecated, decode-only (C06 / F03): a plain string never resolves to
    # real admissible evidence. New proposals must use `support` instead;
    # this field is kept only so an already-serialized proposal still decodes.
    supporting_refs: tuple[str, ...] = ()
    support: tuple[SupportRef, ...] = ()
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
