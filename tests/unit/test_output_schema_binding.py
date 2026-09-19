"""F13 decisive tests: schema binding, resource-exhaustion limits, and the
pre-invocation hash re-check in `SemanticModelRuntime`."""

import asyncio
from collections.abc import Sequence
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict, Field

from fre.domain.budget import DeploymentLimits
from fre.domain.common import JsonValue, OutputContract, PermissionSet, canonical_hash
from fre.domain.semantic import StructuredModelRequest, StructuredModelResult, StructuredModelStatus
from fre.domain.task import TaskEnvelope
from fre.engine import FrontierReasoningEngine
from fre.modules.m01_classifier import TaskClassifier
from fre.modules.m02_budget import BudgetAllocator, default_tier_policy
from fre.prompts import default_output_schema_registry, default_prompt_registry
from fre.prompts.schemas import (
    MAX_SCHEMA_CANONICAL_BYTES,
    MAX_SCHEMA_NESTING_DEPTH,
    ClassificationOutput,
    OutputSchemaError,
    OutputSchemaRegistry,
    _schema_nesting_depth,
    canonical_schema_bytes,
    canonical_schema_hash,
    enforce_schema_limits,
)
from fre.runtime.events import BudgetAllocated
from fre.semantic_runtime import (
    SemanticModelRuntime,
    SemanticRuntimePolicy,
    SemanticSchemaBindingError,
)

VALID: dict[str, JsonValue] = {
    "task_type": {"estimate": "DECISION", "confidence": 0.9, "anchors": [], "rationale": "test"},
    "consequence": {
        "estimate": "LOW",
        "confidence": 0.9,
        "conservative_upper": "MEDIUM",
        "anchors": [],
        "rationale": "test",
    },
    "reversibility": {
        "estimate": "LOW",
        "confidence": 0.9,
        "conservative_upper": "MEDIUM",
        "anchors": [],
        "rationale": "test",
    },
    "ambiguity": {
        "estimate": "LOW",
        "confidence": 0.9,
        "conservative_upper": "MEDIUM",
        "anchors": [],
        "rationale": "test",
    },
    "evidence_scarcity": {
        "estimate": "LOW",
        "confidence": 0.9,
        "conservative_upper": "MEDIUM",
        "anchors": [],
        "rationale": "test",
    },
    "search_space": {"estimate": "BOUNDED", "confidence": 0.9, "anchors": [], "rationale": "test"},
    "horizon": {"estimate": "SHORT", "confidence": 0.9, "anchors": [], "rationale": "test"},
}


class CapturingModel:
    def __init__(self, responses: Sequence[StructuredModelResult]) -> None:
        self.responses = iter(responses)
        self.requests: list[StructuredModelRequest] = []

    async def generate(self, request: StructuredModelRequest) -> StructuredModelResult:
        self.requests.append(request)
        return next(self.responses)


def _result() -> StructuredModelResult:
    return StructuredModelResult(
        status=StructuredModelStatus.SUCCESS,
        adapter_id="test",
        model_id="test",
        raw_response=b"",
        decoded=VALID,
    )


def _setup(engine: FrontierReasoningEngine) -> tuple[UUID, JsonValue]:
    handle = engine.create_run({"schema-binding": True})
    task = TaskEnvelope(
        task_id=UUID(int=999),
        text="Choose.",
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(),
    )
    signature, _ = TaskClassifier().classify(task, None)
    plan, policy_hash = BudgetAllocator().allocate(
        signature, default_tier_policy(), DeploymentLimits()
    )
    payload = BudgetAllocated(
        plan=plan, policy_version=plan.policy_version, policy_hash=policy_hash
    )
    engine.append(
        handle.run_id, handle.version, (engine.make_event(handle.run_id, payload, module_id="M02"),)
    )
    return handle.run_id, task.model_dump(mode="json")


@pytest.mark.unit
def test_structured_model_request_carries_the_canonical_schema_hash(
    engine: FrontierReasoningEngine,
) -> None:
    run_id, value = _setup(engine)
    model = CapturingModel([_result()])
    runtime = SemanticModelRuntime(
        model, engine, default_prompt_registry(), default_output_schema_registry()
    )
    asyncio.run(
        runtime.execute(
            run_id=run_id,
            module_id="M01",
            module_version="1.0",
            operation="classify",
            prompt_id="m01.classify",
            prompt_version="1.0",
            canonical_input=value,
            policy=SemanticRuntimePolicy(maximum_repair_attempts=0),
        )
    )
    assert len(model.requests) == 1
    request = model.requests[0]
    assert request.output_schema_hash == canonical_schema_hash(ClassificationOutput)
    assert request.output_schema_hash == canonical_hash(ClassificationOutput.model_json_schema())


@pytest.mark.unit
def test_stale_request_schema_hash_is_rejected_before_provider_invocation(
    engine: FrontierReasoningEngine,
) -> None:
    """If the registry's schema bytes diverge from what a request claims
    between the lookup in `execute()` and the invocation in `_invoke()`, the
    runtime must refuse to call the provider and must release the reservation
    it already took -- not silently trust the caller-carried hash."""
    run_id, value = _setup(engine)
    model = CapturingModel([_result()])
    schemas = default_output_schema_registry()
    real_get = schemas.get
    calls = {"n": 0}

    def flaky_get(schema_id: str, version: str):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        definition, model_type = real_get(schema_id, version)
        if calls["n"] > 1:
            definition = definition.model_copy(update={"schema_hash": "0" * 64})
        return definition, model_type

    schemas.get = flaky_get  # type: ignore[method-assign]
    runtime = SemanticModelRuntime(model, engine, default_prompt_registry(), schemas)
    before = engine.inspect(run_id)
    with pytest.raises(SemanticSchemaBindingError):
        asyncio.run(
            runtime.execute(
                run_id=run_id,
                module_id="M01",
                module_version="1.0",
                operation="classify",
                prompt_id="m01.classify",
                prompt_version="1.0",
                canonical_input=value,
                policy=SemanticRuntimePolicy(maximum_repair_attempts=0),
            )
        )
    assert not model.requests
    after = engine.inspect(run_id)
    assert after.budget.reservations == ()
    assert after.model_calls == before.model_calls


@pytest.mark.unit
def test_release_failure_in_schema_mismatch_path_does_not_mask_binding_error(
    engine: FrontierReasoningEngine,
) -> None:
    """EU-07 (C04 cleanup, item #7): if `_release()` itself raises while
    cleaning up after a stale-schema-hash rejection, the caller must still
    see `SemanticSchemaBindingError` -- not the release failure -- with the
    release failure surfaced as `__cause__` rather than replacing the
    propagated exception outright."""
    run_id, value = _setup(engine)
    model = CapturingModel([_result()])
    schemas = default_output_schema_registry()
    real_get = schemas.get
    calls = {"n": 0}

    def flaky_get(schema_id: str, version: str):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        definition, model_type = real_get(schema_id, version)
        if calls["n"] > 1:
            definition = definition.model_copy(update={"schema_hash": "0" * 64})
        return definition, model_type

    schemas.get = flaky_get  # type: ignore[method-assign]
    runtime = SemanticModelRuntime(model, engine, default_prompt_registry(), schemas)

    release_error = RuntimeError("simulated _release failure")

    def failing_release(run_id: object, reservation_id: str) -> None:
        raise release_error

    runtime._release = failing_release  # type: ignore[method-assign]

    with pytest.raises(SemanticSchemaBindingError) as excinfo:
        asyncio.run(
            runtime.execute(
                run_id=run_id,
                module_id="M01",
                module_version="1.0",
                operation="classify",
                prompt_id="m01.classify",
                prompt_version="1.0",
                canonical_input=value,
                policy=SemanticRuntimePolicy(maximum_repair_attempts=0),
            )
        )
    assert not model.requests
    assert excinfo.value.__cause__ is release_error


@pytest.mark.unit
def test_conflicting_registration_of_same_id_and_version_is_rejected() -> None:
    """2.1.5: a schema id+version paired with materially different bytes must
    be rejected, whether the difference comes from a different model
    entirely or from the model's shape changing under the same identity."""

    class First(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        value: int

    class Second(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        value: str

    registry = OutputSchemaRegistry()
    registry.register("dup.schema", "1.0", First, "M99", "op")
    with pytest.raises(OutputSchemaError, match="conflicting output schema"):
        registry.register("dup.schema", "1.0", Second, "M99", "op")


@pytest.mark.unit
def test_registry_get_detects_a_post_registration_hash_mismatch() -> None:
    """2.1.5: bytes whose hash no longer matches the registered definition are
    rejected the moment they are looked up again."""

    class Model(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        value: int

    registry = OutputSchemaRegistry()
    registry.register("mismatch.schema", "1.0", Model, "M99", "op")
    definition, model_type = registry.get("mismatch.schema", "1.0")
    registry._items[("mismatch.schema", "1.0")] = (
        definition.model_copy(update={"schema_hash": "1" * 64}),
        model_type,
    )
    with pytest.raises(OutputSchemaError, match="integrity mismatch"):
        registry.get("mismatch.schema", "1.0")


@pytest.mark.unit
def test_canonical_schema_hash_is_memoized_per_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EU-01 (C04 cleanup, item #1): `canonical_schema_hash` must not recompute
    the hash from scratch for the same model on every `register()`/`get()`
    call -- it is memoized per model class, so the underlying hash function
    runs at most once per model."""
    import fre.prompts.schemas as schemas_module

    class Model(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        value: int

    call_count = 0
    real_canonical_hash = schemas_module.canonical_hash

    def spy_canonical_hash(value: object) -> str:
        nonlocal call_count
        call_count += 1
        return real_canonical_hash(value)

    monkeypatch.setattr(schemas_module, "canonical_hash", spy_canonical_hash)

    registry_a = OutputSchemaRegistry()
    registry_b = OutputSchemaRegistry()
    registry_a.register("memo.schema.a", "1.0", Model, "M99", "op")
    assert call_count == 1

    # A second `register()` for the *same model class* (via a different
    # registry/schema id, to avoid the conflicting-registration guard) must
    # reuse the cached hash rather than recomputing it.
    registry_b.register("memo.schema.b", "1.0", Model, "M99", "op")
    assert call_count == 1

    hash_a = registry_a.get("memo.schema.a", "1.0")[0].schema_hash
    hash_b = registry_b.get("memo.schema.b", "1.0")[0].schema_hash
    assert hash_a == hash_b
    assert call_count == 1

    # Repeated `get()` calls (the registry's own integrity re-check) also
    # reuse the cached value rather than recomputing.
    registry_a.get("memo.schema.a", "1.0")
    registry_a.get("memo.schema.a", "1.0")
    assert call_count == 1


@pytest.mark.unit
def test_canonical_schema_hash_matches_canonical_hash_of_the_json_schema() -> None:
    """EU-04 (C04 cleanup, item #4) parity check: `canonical_schema_hash(model)`
    must be byte-for-byte identical to `canonical_hash(model.model_json_schema())`
    for every currently-registered schema, since `canonical_schema_hash` is
    defined in terms of the same `canonical_json` serialization `canonical_hash`
    uses -- there is no independent hand-rolled hashing path to diverge from
    the shared primitive."""
    registry = default_output_schema_registry()
    for schema_id, version in (
        ("m01.classification-output", "1.0"),
        ("m03.problem-formalisation-output", "1.0"),
        ("m04.representation-adjudication-output", "1.0"),
    ):
        _, model = registry.get(schema_id, version)
        assert canonical_schema_hash(model) == canonical_hash(model.model_json_schema())


@pytest.mark.unit
def test_registration_rejects_a_schema_exceeding_the_nesting_depth_limit() -> None:
    from pydantic import create_model

    current: type[BaseModel] = create_model(
        "Level0", __config__=ConfigDict(extra="forbid", strict=True), leaf=(int, 0)
    )
    for level in range(1, MAX_SCHEMA_NESTING_DEPTH + 4):
        current = create_model(
            f"Level{level}",
            __config__=ConfigDict(extra="forbid", strict=True),
            child=(current, ...),
        )
    registry = OutputSchemaRegistry()
    with pytest.raises(OutputSchemaError, match="nesting depth"):
        registry.register("deep.schema", "1.0", current, "M99", "op")


@pytest.mark.unit
def test_registration_rejects_a_schema_exceeding_the_byte_size_limit() -> None:
    from typing import Any

    from pydantic import create_model

    field_defs: dict[str, Any] = {
        f"field_{i}": (str, Field(default="", description="x" * 200)) for i in range(400)
    }
    huge: type[BaseModel] = create_model(
        "Huge", __config__=ConfigDict(extra="forbid", strict=True), **field_defs
    )
    assert len(canonical_schema_bytes(huge)) > MAX_SCHEMA_CANONICAL_BYTES
    registry = OutputSchemaRegistry()
    with pytest.raises(OutputSchemaError, match="byte size"):
        registry.register("huge.schema", "1.0", huge, "M99", "op")


@pytest.mark.unit
def test_registration_rejects_unsupported_pattern_properties_construct() -> None:
    from fre.prompts.schemas import _check_supported_constructs

    with pytest.raises(OutputSchemaError, match="patternProperties"):
        _check_supported_constructs(
            {"type": "object", "patternProperties": {"^x": {"type": "string"}}}
        )


@pytest.mark.unit
def test_registration_rejects_external_ref_construct() -> None:
    from fre.prompts.schemas import _check_supported_constructs

    with pytest.raises(OutputSchemaError, match="external \\$ref"):
        _check_supported_constructs({"$ref": "https://example.com/evil.json"})
    # A local $defs ref is fine.
    _check_supported_constructs({"$ref": "#/$defs/Thing"})


@pytest.mark.unit
def test_default_registry_schemas_satisfy_their_own_limits() -> None:
    registry = default_output_schema_registry()
    for schema_id, version in (
        ("m01.classification-output", "1.0"),
        ("m03.problem-formalisation-output", "1.0"),
        ("m04.representation-adjudication-output", "1.0"),
    ):
        definition, model = registry.get(schema_id, version)
        enforce_schema_limits(model)
        assert _schema_nesting_depth(model.model_json_schema()) <= MAX_SCHEMA_NESTING_DEPTH
        assert len(canonical_schema_bytes(model)) <= MAX_SCHEMA_CANONICAL_BYTES
        assert definition.schema_hash == canonical_schema_hash(model)


@pytest.mark.unit
def test_classification_output_schema_shape_matches_pre_eu11_baseline() -> None:
    """C05 remediation (finding #11): regression pin for the
    `TaskTypeProposal`/`SearchSpaceProposal`/`HorizonProposal` consolidation
    into a shared generic `_CategoricalProposal[EstimateT]` base.

    These are the exact `ClassificationOutput` schema hash, nesting depth,
    and canonical byte size measured against the pre-consolidation flat
    (copy-pasted) classes, captured as a baseline before the change per the
    unit's acceptance-test instructions. `canonical_schema_hash` must be
    byte-for-byte identical -- not merely "close" -- because it is exactly
    what `OutputSchemaRegistry.register()` persists as `schema_hash` and
    downstream event payloads (`SemanticModelCallRecordV2.output_schema_hash`)
    carry forward.
    """
    assert (
        canonical_schema_hash(ClassificationOutput)
        == "3621d418a2e614559e72b041ef8605a591282bb55756a513e8d1a996a25c4af9"
    )
    assert len(canonical_schema_bytes(ClassificationOutput)) == 4215
    assert _schema_nesting_depth(ClassificationOutput.model_json_schema()) == 17
    # And, redundantly, against the documented global budgets (belt-and-braces
    # with the exact-value pins above, which are the real regression guard).
    depth = _schema_nesting_depth(ClassificationOutput.model_json_schema())
    assert depth <= MAX_SCHEMA_NESTING_DEPTH
    assert len(canonical_schema_bytes(ClassificationOutput)) <= MAX_SCHEMA_CANONICAL_BYTES
