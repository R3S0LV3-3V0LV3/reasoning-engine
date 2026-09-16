"""Async lifecycle and durability tests for the provider-neutral semantic runtime."""

import asyncio
from collections.abc import Sequence
from uuid import UUID

import pytest

from fre.domain.budget import DeploymentLimits
from fre.domain.common import JsonValue, OutputContract, PermissionSet
from fre.domain.semantic import (
    SemanticAccountingCondition,
    SemanticCallUsage,
    StructuredModelRequest,
    StructuredModelResult,
    StructuredModelStatus,
)
from fre.domain.task import TaskEnvelope
from fre.engine import FrontierReasoningEngine
from fre.modules.m01_classifier import TaskClassifier
from fre.modules.m02_budget import BudgetAllocator, default_tier_policy
from fre.prompts import default_output_schema_registry, default_prompt_registry
from fre.prompts.registry import PromptDefinition
from fre.runtime.events import ArtifactRegistered, BudgetAllocated, ModelCallRecorded
from fre.semantic_runtime import (
    SemanticExecution,
    SemanticModelRuntime,
    SemanticOwnershipError,
    SemanticRuntimePolicy,
)

VALID: dict[str, JsonValue] = {
    "task_type": "DECISION",
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
    "search_space": "BOUNDED",
    "search_space_confidence": 0.9,
    "horizon": "SHORT",
    "horizon_confidence": 0.9,
}


class CapturingModel:
    def __init__(self, responses: Sequence[StructuredModelResult | BaseException]) -> None:
        self.responses = iter(responses)
        self.requests: list[StructuredModelRequest] = []

    async def generate(self, request: StructuredModelRequest) -> StructuredModelResult:
        self.requests.append(request)
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


def result(
    status: StructuredModelStatus = StructuredModelStatus.SUCCESS, *, raw: bytes = b""
) -> StructuredModelResult:
    return StructuredModelResult(
        status=status,
        adapter_id="test",
        model_id="test",
        raw_response=raw,
        decoded=VALID if status is StructuredModelStatus.SUCCESS else None,
        diagnostics=("invalid fixture",)
        if status is StructuredModelStatus.INVALID_STRUCTURED_OUTPUT
        else (),
    )


def setup(engine: FrontierReasoningEngine, *, calls: int | None = None) -> tuple[UUID, JsonValue]:
    handle = engine.create_run({"semantic": True})
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
    if calls is not None:
        plan = plan.model_copy(
            update={
                "limits": plan.limits.model_copy(update={"max_llm_calls": calls}),
                "search": plan.search.model_copy(
                    update={"independent_validation_branches": min(calls, 1)}
                ),
            }
        )
    payload = BudgetAllocated(
        plan=plan, policy_version=plan.policy_version, policy_hash=policy_hash
    )
    engine.append(
        handle.run_id, handle.version, (engine.make_event(handle.run_id, payload, module_id="M02"),)
    )
    return handle.run_id, task.model_dump(mode="json")


def run(
    runtime: SemanticModelRuntime,
    run_id: UUID,
    value: JsonValue,
    *,
    policy: SemanticRuntimePolicy | None = None,
    module: str = "M01",
    repair_version: str | None = None,
    allow_repair: bool = True,
) -> SemanticExecution:
    return asyncio.run(
        runtime.execute(
            run_id=run_id,
            module_id=module,
            module_version="1.0",
            operation="classify",
            prompt_id="m01.classify",
            prompt_version="1.0",
            canonical_input=value,
            policy=policy,
            repair_prompt_version=repair_version,
            allow_repair=allow_repair,
        )
    )


@pytest.mark.unit
@pytest.mark.parametrize("status", tuple(StructuredModelStatus))
def test_async_statuses_persist_and_release_reservation(
    engine: FrontierReasoningEngine, status: StructuredModelStatus
) -> None:
    run_id, value = setup(engine)
    model = CapturingModel([result(status)])
    execution = run(
        SemanticModelRuntime(
            model, engine, default_prompt_registry(), default_output_schema_registry()
        ),
        run_id,
        value,
        policy=SemanticRuntimePolicy(maximum_repair_attempts=0),
    )
    assert execution.record is not None and execution.record.status is status
    assert engine.inspect(run_id).budget.reservations == ()
    assert len(model.requests) == 1


@pytest.mark.unit
def test_repair_executes_distinct_render_and_validated_artifact_reuses(
    engine: FrontierReasoningEngine,
) -> None:
    run_id, value = setup(engine)
    model = CapturingModel(
        [
            result(StructuredModelStatus.INVALID_STRUCTURED_OUTPUT, raw=b"```json\n{bad\n```"),
            result(raw=b"provider envelope, not the proposal"),
        ]
    )
    runtime = SemanticModelRuntime(
        model, engine, default_prompt_registry(), default_output_schema_registry()
    )
    execution = run(runtime, run_id, value)
    assert execution.repaired and execution.proposal is not None
    repair_content = str(model.requests[1].messages[0]["content"])
    assert "parent_call_identity" in repair_content
    assert "invalid_raw_response_artifact" in repair_content
    assert "schema_validation_diagnostics" in repair_content
    assert model.requests[0].idempotency_key != model.requests[1].idempotency_key
    assert execution.record is not None
    assert execution.record.raw_artifact != execution.record.proposal_artifact
    reused = run(runtime, run_id, value)
    assert reused.reused and reused.proposal == execution.proposal and len(model.requests) == 2


@pytest.mark.unit
def test_zero_repairs_and_repair_budget_refusal_preserve_initial_record(
    engine: FrontierReasoningEngine,
) -> None:
    run_id, value = setup(engine, calls=1)
    model = CapturingModel([result(StructuredModelStatus.INVALID_STRUCTURED_OUTPUT)])
    execution = run(
        SemanticModelRuntime(
            model, engine, default_prompt_registry(), default_output_schema_registry()
        ),
        run_id,
        value,
    )
    assert not execution.repaired and execution.cause == "REPAIR_BUDGET_UNAVAILABLE"
    assert len(engine.inspect(run_id).model_calls) == 1 and len(model.requests) == 1

    other_engine = engine
    other_run, other_value = setup(other_engine)
    other_model = CapturingModel([result(StructuredModelStatus.INVALID_STRUCTURED_OUTPUT)])
    zero = run(
        SemanticModelRuntime(
            other_model, other_engine, default_prompt_registry(), default_output_schema_registry()
        ),
        other_run,
        other_value,
        policy=SemanticRuntimePolicy(maximum_repair_attempts=0),
    )
    assert not zero.repaired and len(other_model.requests) == 1


@pytest.mark.unit
def test_empty_raw_success_overreported_usage_and_interruption_recovery(
    engine: FrontierReasoningEngine,
) -> None:
    run_id, value = setup(engine)
    over = result().model_copy(
        update={"usage": SemanticCallUsage(input_tokens=9999, output_tokens=9999)}
    )
    model = CapturingModel([over])
    execution = run(
        SemanticModelRuntime(
            model, engine, default_prompt_registry(), default_output_schema_registry()
        ),
        run_id,
        value,
    )
    assert execution.record is not None
    assert (
        execution.record.accounting_condition
        is SemanticAccountingCondition.USAGE_EXCEEDS_RESERVATION
    )
    committed = engine.inspect(run_id).budget.committed
    assert committed.llm_calls == 1
    assert committed.input_tokens == 4096
    assert committed.output_tokens == 2048
    assert engine.inspect(run_id).budget.reservations == ()

    cancelled_run, cancelled_value = setup(engine)
    cancelled = CapturingModel([asyncio.CancelledError()])
    runtime = SemanticModelRuntime(
        cancelled, engine, default_prompt_registry(), default_output_schema_registry()
    )
    with pytest.raises(asyncio.CancelledError):
        run(runtime, cancelled_run, cancelled_value)
    assert engine.inspect(cancelled_run).budget.reservations == ()


@pytest.mark.unit
def test_cross_module_ownership_rejected_before_spending(engine: FrontierReasoningEngine) -> None:
    run_id, value = setup(engine)
    model = CapturingModel([result()])
    runtime = SemanticModelRuntime(
        model, engine, default_prompt_registry(), default_output_schema_registry()
    )
    before = engine.inspect(run_id)
    with pytest.raises(SemanticOwnershipError):
        run(runtime, run_id, value, module="M03")
    after = engine.inspect(run_id)
    assert not model.requests and after.version == before.version


@pytest.mark.unit
def test_concurrent_same_identity_cannot_oversubscribe_or_duplicate_call(
    engine: FrontierReasoningEngine,
) -> None:
    run_id, value = setup(engine)

    class BlockingModel(CapturingModel):
        entered = asyncio.Event()
        proceed = asyncio.Event()

        async def generate(self, request: StructuredModelRequest) -> StructuredModelResult:
            self.requests.append(request)
            self.entered.set()
            await self.proceed.wait()
            return result()

    model = BlockingModel([])
    runtime = SemanticModelRuntime(
        model, engine, default_prompt_registry(), default_output_schema_registry()
    )

    async def concurrent() -> tuple[object, object]:
        async def execute() -> SemanticExecution:
            return await runtime.execute(
                run_id=run_id,
                module_id="M01",
                module_version="1.0",
                operation="classify",
                prompt_id="m01.classify",
                prompt_version="1.0",
                canonical_input=value,
            )

        first = asyncio.create_task(execute())
        await model.entered.wait()
        second = asyncio.create_task(execute())
        await asyncio.sleep(0)
        model.proceed.set()
        return await asyncio.gather(first, second)

    asyncio.run(concurrent())
    assert len(model.requests) == 1
    assert engine.inspect(run_id).budget.reservations == ()


@pytest.mark.unit
def test_repair_prompt_version_changes_repair_identity(engine: FrontierReasoningEngine) -> None:
    run_id, value = setup(engine)
    defaults = default_prompt_registry()
    repair = defaults.get("m01.classify.repair", "1.0")
    alternate = PromptDefinition.create(
        **{
            **repair.model_dump(exclude={"template_hash", "prompt_version"}),
            "prompt_version": "2.0",
            "template": repair.template + " Repair v2.",
        }
    )
    defaults.register(alternate)
    first_model = CapturingModel(
        [result(StructuredModelStatus.INVALID_STRUCTURED_OUTPUT), result()]
    )
    first = SemanticModelRuntime(first_model, engine, defaults, default_output_schema_registry())
    original = run(first, run_id, value)
    assert original.record is not None
    assert original.record.prompt_version == "1.0"
    second_run, second_value = setup(engine)
    second_model = CapturingModel(
        [result(StructuredModelStatus.INVALID_STRUCTURED_OUTPUT), result()]
    )
    changed = run(
        SemanticModelRuntime(second_model, engine, defaults, default_output_schema_registry()),
        second_run,
        second_value,
        repair_version="2.0",
    )
    assert changed.record is not None and changed.record.prompt_version == "2.0"
    assert first_model.requests[1].idempotency_key != second_model.requests[1].idempotency_key


@pytest.mark.unit
def test_durable_initial_failure_resumes_and_repair_version_does_not_cross_reuse(
    engine: FrontierReasoningEngine,
) -> None:
    run_id, value = setup(engine)
    prompts = default_prompt_registry()
    repair = prompts.get("m01.classify.repair", "1.0")
    prompts.register(
        PromptDefinition.create(
            **{
                **repair.model_dump(exclude={"template_hash", "prompt_version"}),
                "prompt_version": "2.0",
                "template": repair.template + " Repair v2.",
            }
        )
    )
    initial = CapturingModel([result(StructuredModelStatus.INVALID_STRUCTURED_OUTPUT)])
    runtime = SemanticModelRuntime(initial, engine, prompts, default_output_schema_registry())
    failed = run(
        runtime,
        run_id,
        value,
        allow_repair=False,
    )
    assert failed.record is not None

    repair_v1 = CapturingModel([result()])
    repaired = run(
        SemanticModelRuntime(repair_v1, engine, prompts, default_output_schema_registry()),
        run_id,
        value,
    )
    assert repaired.repaired and len(repair_v1.requests) == 1

    repair_v2 = CapturingModel([result()])
    changed = run(
        SemanticModelRuntime(repair_v2, engine, prompts, default_output_schema_registry()),
        run_id,
        value,
        repair_version="2.0",
    )
    assert changed.repaired and len(repair_v2.requests) == 1
    assert changed.record is not None and changed.record.prompt_version == "2.0"


@pytest.mark.unit
def test_recovery_releases_only_explicit_abandoned_reservations(
    engine: FrontierReasoningEngine,
) -> None:
    run_id, _ = setup(engine)
    from fre.domain.budget import BudgetReservation, ResourceVector
    from fre.runtime.events import BudgetReserved

    state = engine.inspect(run_id)
    reservations = (
        BudgetReservation(
            reservation_id="abandoned", action_id="semantic", resources=ResourceVector()
        ),
        BudgetReservation(
            reservation_id="live", action_id="other-worker", resources=ResourceVector()
        ),
    )
    engine.append(
        run_id,
        state.version,
        tuple(
            engine.make_event(run_id, BudgetReserved(reservation=item), module_id="test")
            for item in reservations
        ),
    )
    runtime = SemanticModelRuntime(
        CapturingModel([]), engine, default_prompt_registry(), default_output_schema_registry()
    )
    assert runtime.recover_outstanding_reservations(run_id, ("abandoned", "missing")) == (
        "abandoned",
    )
    assert tuple(item.reservation_id for item in engine.inspect(run_id).budget.reservations) == (
        "live",
    )


@pytest.mark.unit
def test_legacy_raw_only_successful_call_remains_replayable(
    engine: FrontierReasoningEngine,
) -> None:
    source_run, value = setup(engine)
    execution = run(
        SemanticModelRuntime(
            CapturingModel([result(raw=b"legacy")]),
            engine,
            default_prompt_registry(),
            default_output_schema_registry(),
        ),
        source_run,
        value,
    )
    assert execution.record is not None and execution.record.raw_artifact is not None
    raw_ref = execution.record.raw_artifact
    legacy = execution.record.model_copy(update={"proposal_artifact": None})

    replay_run, _ = setup(engine)
    state = engine.inspect(replay_run)
    payloads = (
        ArtifactRegistered(
            artifact=raw_ref,
            media_type="application/octet-stream",
            byte_size=len(b"legacy"),
        ),
        ModelCallRecorded(record=legacy),
    )
    engine.append(
        replay_run,
        state.version,
        tuple(engine.make_event(replay_run, item, module_id="legacy") for item in payloads),
    )
    assert engine.inspect(replay_run).model_calls == (legacy,)
