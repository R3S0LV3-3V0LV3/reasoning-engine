"""Combined Wave 3 SQLite, artifact, model, snapshot, and replay gate."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest

from fre.adapters.artifacts_local import LocalArtifactStore
from fre.adapters.storage_sqlite import SQLiteStore
from fre.adapters.testing import FakeClock, FakeUUIDFactory
from fre.domain.budget import DeploymentLimits
from fre.domain.common import JsonValue, OutputContract, PermissionSet
from fre.domain.context import CompilerProfile
from fre.domain.semantic import (
    SemanticCallUsage,
    StructuredModelRequest,
    StructuredModelResult,
    StructuredModelStatus,
)
from fre.domain.task import TaskEnvelope
from fre.engine import FrontierReasoningEngine
from fre.modules.m01_classifier import TaskClassifier
from fre.modules.m02_budget import BudgetAllocator, default_tier_policy
from fre.modules.m03_formaliser import ProblemFormaliser
from fre.modules.m04_representation import RepresentationSelector
from fre.modules.m12_context import Wave3ContextCompiler
from fre.prompts import default_output_schema_registry, default_prompt_registry
from fre.prompts.schemas import ClassificationOutput
from fre.runtime.budget_meter import BudgetMeter
from fre.runtime.events import (
    BudgetAllocated,
    ProblemFormalised,
    RepresentationArtifactCompiled,
    RepresentationPlanSelected,
    TaskClassified,
)
from fre.semantic_runtime import SemanticModelRuntime


class QueueModel:
    def __init__(self, responses: list[StructuredModelResult]) -> None:
        self.responses = iter(responses)
        self.calls: list[str] = []

    def generate(self, request: StructuredModelRequest) -> StructuredModelResult:
        self.calls.append(request.idempotency_key)
        return next(self.responses)


def uuids(count: int = 100) -> list[UUID]:
    return [UUID(int=index) for index in range(1, count + 1)]


@pytest.mark.integration
def test_combined_wave3_gate_three_path_replay_and_zero_model_calls(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=i) for i in range(100))
    uuid_factory = FakeUUIDFactory(uuids())
    database = tmp_path / "events.db"
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    store = SQLiteStore(database)
    engine = FrontierReasoningEngine(store, artifacts, clock, uuid_factory)
    handle = engine.create_run({"wave": 3})
    task = TaskEnvelope(
        task_id=UUID(int=900),
        text="Choose a safe option.",
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(),
    )
    fallback_signature, _ = TaskClassifier().classify(task, None)
    plan, policy_hash = BudgetAllocator().allocate(
        fallback_signature, default_tier_policy(), DeploymentLimits()
    )
    allocated = BudgetAllocated(
        plan=plan, policy_version=plan.policy_version, policy_hash=policy_hash
    )
    engine.append(
        handle.run_id,
        handle.version,
        (engine.make_event(handle.run_id, allocated, module_id="M02"),),
    )

    valid: dict[str, JsonValue] = {
        "task_type": "DECISION",
        "consequence": {
            "estimate": "LOW",
            "confidence": 0.9,
            "conservative_upper": "MEDIUM",
            "anchors": [],
            "rationale": "fixture",
        },
        "reversibility": {
            "estimate": "HIGH",
            "confidence": 0.9,
            "conservative_upper": "HIGH",
            "anchors": [],
            "rationale": "fixture",
        },
        "ambiguity": {
            "estimate": "MEDIUM",
            "confidence": 0.9,
            "conservative_upper": "MEDIUM",
            "anchors": [],
            "rationale": "fixture",
        },
        "evidence_scarcity": {
            "estimate": "MEDIUM",
            "confidence": 0.9,
            "conservative_upper": "MEDIUM",
            "anchors": [],
            "rationale": "fixture",
        },
        "search_space": "BOUNDED",
        "search_space_confidence": 0.9,
        "horizon": "SHORT",
        "horizon_confidence": 0.9,
    }
    model = QueueModel(
        [
            StructuredModelResult(
                status=StructuredModelStatus.INVALID_STRUCTURED_OUTPUT,
                adapter_id="fake",
                model_id="fixture",
                raw_response=b"{bad",
                diagnostics=("invalid",),
            ),
            StructuredModelResult(
                status=StructuredModelStatus.SUCCESS,
                adapter_id="fake",
                model_id="fixture",
                raw_response=json.dumps(valid).encode(),
                decoded=valid,
                usage=SemanticCallUsage(input_tokens=100, output_tokens=50),
            ),
        ]
    )
    runtime = SemanticModelRuntime(
        model, artifacts, uuid_factory, default_prompt_registry(), default_output_schema_registry()
    )
    state = engine.inspect(handle.run_id)
    execution = runtime.execute(
        state,
        module_id="M01",
        module_version="1.0",
        operation="classify",
        prompt_id="m01.classify",
        prompt_version="1.0",
        canonical_input=task.model_dump(mode="json"),
    )
    assert execution.repaired and len(model.calls) == 2
    proposal = cast(ClassificationOutput, execution.proposal)
    signature, classification_record = TaskClassifier().classify(
        task,
        proposal,
        model_call_key=execution.record.idempotency_key if execution.record else None,
    )
    payloads = (
        *execution.event_payloads,
        TaskClassified(signature=signature, record=classification_record),
    )
    state = engine.inspect(handle.run_id)
    events = tuple(
        engine.make_event(handle.run_id, payload, module_id="semantic-runtime")
        for payload in payloads
    )
    engine.append(handle.run_id, state.version, events)

    persisted_state = engine.inspect(handle.run_id)
    reused = runtime.execute(
        persisted_state,
        module_id="M01",
        module_version="1.0",
        operation="classify",
        prompt_id="m01.classify",
        prompt_version="1.0",
        canonical_input=task.model_dump(mode="json"),
    )
    assert reused.reused and reused.repaired and reused.proposal == proposal
    assert len(model.calls) == 2

    problem = ProblemFormaliser().formalise(task, None)
    representation_plan = RepresentationSelector().select(problem, signature, plan)
    representation_artifact = RepresentationSelector().build(
        representation_plan.views[0], problem, engine.inspect(handle.run_id).version
    )
    semantic_payloads = (
        ProblemFormalised(problem=problem),
        RepresentationPlanSelected(plan=representation_plan),
        RepresentationArtifactCompiled(artifact=representation_artifact),
    )
    state = engine.inspect(handle.run_id)
    engine.append(
        handle.run_id,
        state.version,
        tuple(
            engine.make_event(handle.run_id, payload, module_id="wave3")
            for payload in semantic_payloads
        ),
    )
    state = engine.inspect(handle.run_id)
    packet = Wave3ContextCompiler().compile_semantic(
        problem=problem,
        representation=representation_plan,
        run_id=handle.run_id,
        snapshot_version=state.version,
        ledger=state.ledger,
        budget_remaining=BudgetMeter().remaining(state.budget),
        profile=CompilerProfile.HANDOFF,
    )
    assert packet.packet.compiler_version == "2.0" and packet.packet.objective is not None
    expected_hash = engine.snapshot(handle.run_id)
    calls_before_replay = len(model.calls)
    path_a = engine.replay(handle.run_id)
    path_c = engine.replay_from_snapshot(handle.run_id)
    store.close()
    reopened = SQLiteStore(database)
    path_b = FrontierReasoningEngine(reopened, artifacts, clock, uuid_factory).replay(handle.run_id)
    assert path_a.state_hash == path_b.state_hash == path_c.state_hash == expected_hash
    assert path_a.problem_spec == problem
    assert (
        path_a.representation_artifacts[0].projection_hash
        == representation_artifact.projection_hash
    )
    assert len(model.calls) == calls_before_replay
    assert path_a.budget.committed.llm_calls == 2
    reopened.close()
