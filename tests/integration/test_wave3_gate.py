"""Combined Wave 3 SQLite, artifact, model, snapshot, and replay gate."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest

from fre.adapters.artifacts_local import LocalArtifactStore
from fre.adapters.storage_sqlite import SQLiteStore
from fre.adapters.testing import FakeClock, FakeUUIDFactory
from fre.domain.budget import BudgetRemaining, DeploymentLimits, ResourceVector
from fre.domain.common import JsonValue, OutputContract, PermissionSet
from fre.domain.context import CompilerProfile
from fre.domain.ledger import DanglingLedgerReference
from fre.domain.semantic import (
    EpistemicOriginLabel,
    SemanticCallUsage,
    StructuredModelRequest,
    StructuredModelResult,
    StructuredModelStatus,
)
from fre.domain.stop import (
    AcceptanceStatus,
    StopDisposition,
    StopInputs,
    StopPolicy,
    ValidationStatus,
)
from fre.domain.task import TaskEnvelope
from fre.engine import FrontierReasoningEngine
from fre.modules.m01_classifier import TaskClassifier
from fre.modules.m02_budget import BudgetAllocator, default_tier_policy
from fre.modules.m03_formaliser import ProblemFormaliser
from fre.modules.m04_representation import RepresentationSelector
from fre.modules.m12_context import Wave3ContextCompiler
from fre.modules.m13_stop import StopController
from fre.prompts import default_output_schema_registry, default_prompt_registry
from fre.prompts.schemas import ClassificationOutput, ProblemFormalisationOutput
from fre.runtime.budget_meter import BudgetMeter
from fre.runtime.events import (
    BudgetAllocated,
    ModelCallRecordedV2,
    ProblemBlockerRecorded,
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

    async def generate(self, request: StructuredModelRequest) -> StructuredModelResult:
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

    # C05 remediation (finding #7): a TASK_TEXT anchor into the task's own
    # narrative is plausible support for every classification axis below
    # (unlike `/requested_output`, which is only ever relevant to
    # `output_form` -- a deterministic, non-model-proposed axis -- and is now
    # rejected as irrelevant for any of these material dimensions).
    fixture_anchor: JsonValue = {
        "source_kind": "TASK_TEXT",
        "source_ref": {"object_type": "TaskEnvelope", "object_id": str(task.task_id)},
        "selector": "/text",
        "char_start": 0,
        "char_end": len(task.text),
    }
    valid: dict[str, JsonValue] = {
        "task_type": {
            "estimate": "DECISION",
            "confidence": 0.9,
            "anchors": [fixture_anchor],
            "rationale": "fixture",
        },
        "consequence": {
            "estimate": "LOW",
            "confidence": 0.9,
            "conservative_upper": "MEDIUM",
            "anchors": [fixture_anchor],
            "rationale": "fixture",
        },
        "reversibility": {
            "estimate": "HIGH",
            "confidence": 0.9,
            "conservative_upper": "HIGH",
            "anchors": [fixture_anchor],
            "rationale": "fixture",
        },
        "ambiguity": {
            "estimate": "MEDIUM",
            "confidence": 0.9,
            "conservative_upper": "MEDIUM",
            "anchors": [fixture_anchor],
            "rationale": "fixture",
        },
        "evidence_scarcity": {
            "estimate": "MEDIUM",
            "confidence": 0.9,
            "conservative_upper": "MEDIUM",
            "anchors": [fixture_anchor],
            "rationale": "fixture",
        },
        "search_space": {
            "estimate": "BOUNDED",
            "confidence": 0.9,
            "anchors": [fixture_anchor],
            "rationale": "fixture",
        },
        "horizon": {
            "estimate": "SHORT",
            "confidence": 0.9,
            "anchors": [fixture_anchor],
            "rationale": "fixture",
        },
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
        model, engine, default_prompt_registry(), default_output_schema_registry()
    )
    execution = asyncio.run(
        runtime.execute(
            run_id=handle.run_id,
            module_id="M01",
            module_version="1.0",
            operation="classify",
            prompt_id="m01.classify",
            prompt_version="1.0",
            canonical_input=task.model_dump(mode="json"),
        )
    )
    assert execution.repaired and len(model.calls) == 2
    successful_call = next(
        item for item in execution.event_payloads if isinstance(item, ModelCallRecordedV2)
    )
    assert successful_call.record.raw_artifact is not None
    assert successful_call.record.proposal_artifact is not None
    proposal = cast(ClassificationOutput, execution.proposal)
    signature, classification_record = TaskClassifier().classify(
        task,
        proposal,
        model_call_key=execution.record.idempotency_key if execution.record else None,
    )
    # C05 remediation (finding #2): `RunReducer.apply` now requires every
    # MODEL-basis dimension in a `TaskClassified` to have a matching M09
    # provenance node already applied before it -- so this hand-assembled
    # batch must include that provenance too, exactly as
    # `TaskClassifier.canonical_events` does, and in the same order.
    provenance = TaskClassifier().provenance_events(
        classification_record, created_at=clock.now(), uuids=uuid_factory
    )
    payloads = (*provenance, TaskClassified(signature=signature, record=classification_record))
    state = engine.inspect(handle.run_id)
    events = tuple(
        engine.make_event(handle.run_id, payload, module_id="semantic-runtime")
        for payload in payloads
    )
    engine.append(handle.run_id, state.version, events)

    reused = asyncio.run(
        runtime.execute(
            run_id=handle.run_id,
            module_id="M01",
            module_version="1.0",
            operation="classify",
            prompt_id="m01.classify",
            prompt_version="1.0",
            canonical_input=task.model_dump(mode="json"),
        )
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


@pytest.mark.integration
def test_reformalisation_invalidates_bound_representation_plan(
    engine: FrontierReasoningEngine,
) -> None:
    handle = engine.create_run({"reformalise": True})
    task = TaskEnvelope(
        task_id=UUID(int=901),
        text="Choose safely.",
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(),
    )
    signature, _record = TaskClassifier().classify(task, None)
    budget, policy_hash = BudgetAllocator().allocate(
        signature, default_tier_policy(), DeploymentLimits()
    )
    first = ProblemFormaliser().formalise(task, None)
    second = first.model_copy(update={"output_contract": OutputContract(form="JSON")})
    plan = RepresentationSelector().select(first, signature, budget)
    events = tuple(
        engine.make_event(handle.run_id, item, module_id="wave3")
        for item in (
            BudgetAllocated(
                plan=budget,
                policy_version=budget.policy_version,
                policy_hash=policy_hash,
            ),
            ProblemFormalised(problem=first),
            RepresentationPlanSelected(plan=plan),
        )
    )
    engine.append(handle.run_id, handle.version, events)
    state = engine.inspect(handle.run_id)
    wrong = plan.model_copy(update={"problem_spec_hash": "f" * 64})
    with pytest.raises(ValueError, match="does not bind current ProblemSpec"):
        engine.append(
            handle.run_id,
            state.version,
            (
                engine.make_event(
                    handle.run_id, RepresentationPlanSelected(plan=wrong), module_id="M04"
                ),
            ),
        )
    engine.append(
        handle.run_id,
        state.version,
        (engine.make_event(handle.run_id, ProblemFormalised(problem=second), module_id="M03"),),
    )
    expected = engine.inspect(handle.run_id)
    assert expected.representation_plan is None
    assert expected.representation_artifacts == ()
    with pytest.raises(ValueError, match="does not bind current ProblemSpec"):
        Wave3ContextCompiler().compile_semantic(
            problem=second,
            representation=plan,
            run_id=handle.run_id,
            snapshot_version=expected.version,
            ledger=expected.ledger,
            budget_remaining=BudgetMeter().remaining(expected.budget),
            profile=CompilerProfile.STANDARD,
        )
    engine.snapshot(handle.run_id)
    assert engine.replay(handle.run_id) == engine.replay_from_snapshot(handle.run_id)


@pytest.mark.integration
def test_legacy_representation_plan_event_and_snapshot_remain_replayable(
    engine: FrontierReasoningEngine,
) -> None:
    handle = engine.create_run({"legacy-representation": True})
    task = TaskEnvelope(
        task_id=UUID(int=902),
        text="Choose safely.",
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(),
    )
    signature, _record = TaskClassifier().classify(task, None)
    budget, _policy_hash = BudgetAllocator().allocate(
        signature, default_tier_policy(), DeploymentLimits()
    )
    problem = ProblemFormaliser().formalise(task, None)
    plan = RepresentationSelector().select(problem, signature, budget)
    formalised = engine.make_event(
        handle.run_id, ProblemFormalised(problem=problem), module_id="M03"
    )
    selected = engine.make_event(
        handle.run_id, RepresentationPlanSelected(plan=plan), module_id="M04"
    )
    legacy_payload = dict(selected.payload)
    legacy_plan = dict(cast(dict[str, JsonValue], legacy_payload["plan"]))
    legacy_plan.pop("problem_spec_hash")
    legacy_payload["plan"] = legacy_plan
    legacy_selected = selected.model_copy(update={"payload": legacy_payload})

    engine.append(handle.run_id, handle.version, (formalised, legacy_selected))
    replayed = engine.replay(handle.run_id)
    assert replayed.representation_plan is not None
    assert replayed.representation_plan.problem_spec_hash is None
    snapshot_hash = engine.snapshot(handle.run_id)
    assert engine.replay_from_snapshot(handle.run_id).state_hash == snapshot_hash


@pytest.mark.integration
def test_m03_contradiction_batch_is_atomic_and_replayable(
    engine: FrontierReasoningEngine,
) -> None:
    handle = engine.create_run({"m03": "atomic-contradiction"})
    task = TaskEnvelope(
        task_id=UUID(int=903),
        text="Resolve A versus B.",
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(),
    )
    proposal = ProblemFormalisationOutput.model_validate_json(
        json.dumps(
            {
                "items": [
                    {
                        "id": "left",
                        "kind": "UNKNOWN",
                        "description": "A",
                        "origin": "CONTRADICTED",
                    },
                    {
                        "id": "right",
                        "kind": "UNKNOWN",
                        "description": "B",
                        "origin": "CONTRADICTED",
                    },
                    {
                        "id": "conflict",
                        "kind": "RELATION",
                        "description": "A conflicts with B",
                        "origin": "CONTRADICTED",
                        "attributes": {
                            "source_id": "left",
                            "target_id": "right",
                            "relation_kind": "CONTRADICTS",
                        },
                    },
                ]
            }
        )
    )
    payloads = ProblemFormaliser().canonical_events(
        task,
        proposal,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        uuids=FakeUUIDFactory(uuids()),
    )
    events = tuple(
        engine.make_event(handle.run_id, payload, module_id="M03") for payload in payloads
    )
    blocker_index = next(
        index
        for index, payload in enumerate(payloads)
        if isinstance(payload, ProblemBlockerRecorded)
    )
    blocker = cast(ProblemBlockerRecorded, payloads[blocker_index]).blocker
    bad_blocker = blocker.model_copy(
        update={"ledger_ref": blocker.ledger_ref.model_copy(update={"node_id": UUID(int=9999)})}
    )
    invalid_events = list(events)
    invalid_events[blocker_index] = engine.make_event(
        handle.run_id,
        ProblemBlockerRecorded(blocker=bad_blocker),
        module_id="M03",
    )
    with pytest.raises(DanglingLedgerReference, match="unknown ledger revision"):
        engine.append(handle.run_id, handle.version, tuple(invalid_events))
    assert engine.inspect(handle.run_id).version == handle.version

    engine.append(handle.run_id, handle.version, events)
    state = engine.inspect(handle.run_id)
    assert len(state.problem_contradictions) == 1
    assert state.problem_blockers == (blocker,)
    assert state.problem_blockers[0].ledger_ref in {
        state.ledger.nodes[0].ref,
        state.ledger.nodes[1].ref,
    }
    decision = StopController().evaluate(
        StopInputs(
            budget=BudgetRemaining(
                resources=ResourceVector(iterations=1),
                active_concurrent_actions=0,
                projection_hash="0" * 64,
            ),
            acceptance=AcceptanceStatus.PENDING,
            validation=ValidationStatus.NOT_APPLICABLE,
            blocker_required=True,
            blocker_resolvable=state.problem_blockers[0].resolvable,
            epistemic_trigger_refs=(state.problem_blockers[0].ledger_ref,),
        ),
        StopPolicy(version="test/1"),
    )
    assert decision.disposition is StopDisposition.CONTINUE
    assert decision.ledger_trigger_refs == (state.problem_blockers[0].ledger_ref,)
    snapshot_hash = engine.snapshot(handle.run_id)
    assert engine.replay(handle.run_id) == engine.replay_from_snapshot(handle.run_id)
    assert state.state_hash == snapshot_hash

    resolved_proposal = ProblemFormalisationOutput.model_validate(
        {
            "items": (
                {
                    "id": "reviewed",
                    "kind": "ACCEPTANCE_CRITERION",
                    "description": "resolution is deterministically reviewed",
                    "origin": EpistemicOriginLabel.UNRESOLVED,
                    "attributes": {
                        "verification_mode": "DETERMINISTIC",
                        "required": True,
                    },
                },
            )
        }
    )
    resolved_payloads = ProblemFormaliser().canonical_events(
        task,
        resolved_proposal,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
        uuids=FakeUUIDFactory(UUID(int=index) for index in range(1000, 1010)),
    )
    resolved_events = tuple(
        engine.make_event(handle.run_id, payload, module_id="M03") for payload in resolved_payloads
    )
    engine.append(handle.run_id, state.version, resolved_events)
    resolved_state = engine.inspect(handle.run_id)
    assert resolved_state.problem_blockers == ()
    assert resolved_state.problem_contradictions == ()
