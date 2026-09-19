"""The narrow composition root for configured Wave 3 services."""

from dataclasses import dataclass

from fre.config import Wave3Config
from fre.domain.common import FrozenModel, canonical_hash
from fre.engine import FrontierReasoningEngine, RunHandle
from fre.modules.m01_classifier import ClassificationPolicy, TaskClassifier
from fre.modules.m04_representation import (
    RepresentationSelectionPolicy,
    RepresentationSelector,
    default_registry,
)
from fre.modules.m12_context import Wave3ContextCompiler
from fre.ports.models import StructuredModelPort
from fre.prompts.registry import PromptRegistry, default_prompt_registry
from fre.prompts.schemas import OutputSchemaRegistry, default_output_schema_registry
from fre.runtime.wave3_context import Wave3ContextRuntime
from fre.semantic_runtime import SemanticModelRuntime, SemanticRuntimePolicy


class UnsupportedWave3Configuration(ValueError):
    """A configured version has no implementation in this build."""


class EffectiveWave3Policy(FrozenModel):
    """Complete, canonical identity of the policy graph that will execute."""

    schema_version: str
    classification: ClassificationPolicy
    semantic_runtime: SemanticRuntimePolicy
    representation_selection: RepresentationSelectionPolicy
    prompt_registry_version: str
    output_schema_registry_version: str
    context_compiler_version: str

    @property
    def policy_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class Wave3Components:
    """Configured services, all derived from one effective policy identity."""

    policy: EffectiveWave3Policy
    classifier: TaskClassifier
    semantic_runtime: SemanticModelRuntime
    representation_selector: RepresentationSelector
    context_compiler: Wave3ContextCompiler
    # C08 (F12): the real caller that persists a compiled Wave 3 semantic
    # context packet (`ContextCompiled@2.0` + its two durable artifacts,
    # atomically) instead of `context_compiler.compile_semantic()` ever being
    # invoked only to produce an in-memory-only result nothing durably reads.
    context_runtime: Wave3ContextRuntime
    prompts: PromptRegistry
    schemas: OutputSchemaRegistry


class Wave3Engine:
    """Owning façade that persists the exact effective policy at run creation."""

    def __init__(self, engine: FrontierReasoningEngine, components: Wave3Components) -> None:
        self.engine = engine
        self.components = components
        self.effective_policy = components.policy

    def create_run(self) -> RunHandle:
        return self.engine.create_run(self.effective_policy)


def _require(name: str, actual: str, supported: str) -> None:
    if actual != supported:
        raise UnsupportedWave3Configuration(
            f"unsupported {name} {actual!r}; supported version is {supported!r}"
        )


def compose_wave3(
    engine: FrontierReasoningEngine,
    model: StructuredModelPort,
    config: Wave3Config | None = None,
) -> Wave3Engine:
    """Translate configuration once and inject the resulting policies everywhere."""
    configured = config or Wave3Config()
    _require("Wave3 configuration schema", configured.schema_version, "1.0")
    _require("classification policy", configured.classification_policy_version, "wave3-m01/1.0")
    _require(
        "semantic runtime policy",
        configured.semantic_runtime_policy_version,
        "wave3-semantic-runtime/2.0",
    )
    _require(
        "representation selection policy",
        configured.representation_selection_policy_version,
        "wave3-m04/1.0",
    )
    _require(
        "representation registry",
        configured.representation_registry_version,
        "wave3-m04-registry/1.0",
    )
    _require("prompt registry", configured.prompt_registry_version, "wave3-prompts/1.0")
    _require(
        "output schema registry", configured.output_schema_registry_version, "wave3-schemas/1.0"
    )
    _require("Wave3 context compiler", configured.wave3_context_compiler_version, "2.0")

    classification = ClassificationPolicy(
        version=configured.classification_policy_version,
        confidence_threshold=configured.confidence_threshold,
    )
    semantic = SemanticRuntimePolicy(
        version=configured.semantic_runtime_policy_version,
        maximum_repair_attempts=configured.maximum_repair_attempts,
        reserve_input_tokens=configured.reserve_input_tokens,
        reserve_output_tokens=configured.reserve_output_tokens,
    )
    representation = RepresentationSelectionPolicy(
        version=configured.representation_selection_policy_version,
        registry_version=configured.representation_registry_version,
        tie_band=configured.representation_tie_band,
        minimum_compatibility=configured.representation_minimum_compatibility,
        model_adjudication_enabled=configured.model_adjudication_enabled,
    )
    effective = EffectiveWave3Policy(
        schema_version=configured.schema_version,
        classification=classification,
        semantic_runtime=semantic,
        representation_selection=representation,
        prompt_registry_version=configured.prompt_registry_version,
        output_schema_registry_version=configured.output_schema_registry_version,
        context_compiler_version=configured.wave3_context_compiler_version,
    )
    prompts = default_prompt_registry()
    schemas = default_output_schema_registry()
    context_compiler = Wave3ContextCompiler()
    components = Wave3Components(
        policy=effective,
        classifier=TaskClassifier(classification),
        semantic_runtime=SemanticModelRuntime(
            model,
            engine,
            prompts,
            schemas,
            semantic,
            execution_config_hash=effective.policy_hash,
        ),
        representation_selector=RepresentationSelector(representation, default_registry()),
        context_compiler=context_compiler,
        context_runtime=Wave3ContextRuntime(engine, context_compiler),
        prompts=prompts,
        schemas=schemas,
    )
    return Wave3Engine(engine, components)
