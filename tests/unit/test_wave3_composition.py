"""Configuration-to-runtime wiring tests for the Wave 3 composition root."""

import asyncio

import pytest

from fre.composition import UnsupportedWave3Configuration, compose_wave3
from fre.config import Wave3Config
from fre.domain.common import canonical_hash
from fre.domain.semantic import StructuredModelRequest, StructuredModelResult
from fre.engine import FrontierReasoningEngine
from fre.semantic_runtime import SemanticPolicyMismatchError, SemanticRuntimePolicy


class UnusedModel:
    async def generate(self, request: StructuredModelRequest) -> StructuredModelResult:
        raise AssertionError(f"unexpected model call: {request.idempotency_key}")


@pytest.mark.unit
def test_non_default_configuration_is_injected_and_persisted_exactly(
    engine: FrontierReasoningEngine,
) -> None:
    config = Wave3Config(
        confidence_threshold=0.83,
        maximum_repair_attempts=0,
        reserve_input_tokens=321,
        reserve_output_tokens=123,
        representation_tie_band=0.12,
        representation_minimum_compatibility=0.44,
        model_adjudication_enabled=True,
    )

    facade = compose_wave3(engine, UnusedModel(), config)
    effective = facade.effective_policy
    assert facade.components.classifier.policy == effective.classification
    assert facade.components.semantic_runtime.policy == effective.semantic_runtime
    assert facade.components.representation_selector.policy == effective.representation_selection
    assert effective.classification.confidence_threshold == 0.83
    assert effective.semantic_runtime.maximum_repair_attempts == 0
    assert effective.semantic_runtime.reserve_input_tokens == 321
    assert effective.semantic_runtime.reserve_output_tokens == 123
    assert effective.representation_selection.tie_band == 0.12
    assert effective.representation_selection.minimum_compatibility == 0.44
    assert effective.representation_selection.model_adjudication_enabled is True
    assert facade.components.context_compiler.compiler_version == "2.0"

    handle = facade.create_run()
    assert engine.inspect(handle.run_id).config_hash == effective.policy_hash
    assert effective.policy_hash == canonical_hash(effective)

    # Replay uses the persisted event identity, not a newly composed default policy.
    default_facade = compose_wave3(engine, UnusedModel())
    assert default_facade.effective_policy.policy_hash != effective.policy_hash
    assert engine.replay(handle.run_id).config_hash == effective.policy_hash


@pytest.mark.unit
def test_omitted_configuration_has_stable_defaults(engine: FrontierReasoningEngine) -> None:
    first = compose_wave3(engine, UnusedModel())
    second = compose_wave3(engine, UnusedModel(), Wave3Config())
    assert first.effective_policy == second.effective_policy
    assert first.effective_policy.policy_hash == second.effective_policy.policy_hash


@pytest.mark.unit
@pytest.mark.parametrize("use_override", (False, True))
def test_semantic_execution_rejects_policy_identity_mismatch_before_model_call(
    engine: FrontierReasoningEngine, use_override: bool
) -> None:
    original = compose_wave3(engine, UnusedModel(), Wave3Config(reserve_input_tokens=321))
    handle = original.create_run()
    current = compose_wave3(engine, UnusedModel())
    runtime = (
        original.components.semantic_runtime
        if use_override
        else current.components.semantic_runtime
    )
    override = SemanticRuntimePolicy(reserve_input_tokens=999) if use_override else None

    with pytest.raises(SemanticPolicyMismatchError, match=r"policy|config_hash"):
        asyncio.run(
            runtime.execute(
                run_id=handle.run_id,
                module_id="M01",
                module_version="1.0",
                operation="classify",
                prompt_id="m01.classify",
                prompt_version="1.0",
                canonical_input={},
                policy=override,
            )
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", "9.0"),
        ("classification_policy_version", "m01/future"),
        ("semantic_runtime_policy_version", "runtime/future"),
        ("representation_selection_policy_version", "m04/future"),
        ("representation_registry_version", "representations/future"),
        ("prompt_registry_version", "prompts/future"),
        ("output_schema_registry_version", "schemas/future"),
        ("wave3_context_compiler_version", "3.0"),
    ),
)
def test_unsupported_versions_are_rejected(
    engine: FrontierReasoningEngine, field: str, value: str
) -> None:
    values = Wave3Config().model_dump()
    values[field] = value
    with pytest.raises(UnsupportedWave3Configuration, match="unsupported"):
        compose_wave3(engine, UnusedModel(), Wave3Config.model_validate(values))
