from uuid import UUID

import pytest

from fre.domain.budget import BudgetReservation, DeploymentLimits, ResourceVector
from fre.domain.common import FrozenModel
from fre.domain.ledger import (
    ContradictionResolution,
    EpistemicStatus,
    LedgerEdge,
    LedgerNode,
    LedgerNodeType,
    LedgerRelation,
)
from fre.domain.task import (
    HorizonClass,
    Ordinal4,
    OutputForm,
    SearchSpaceClass,
    TaskEnvelope,
    TaskSignature,
    TaskType,
)
from fre.engine import FrontierReasoningEngine
from fre.modules.m02_budget import BudgetAllocator, default_tier_policy
from fre.modules.m09_ledger import make_node
from fre.runtime.events import (
    BudgetAllocated,
    BudgetConsumed,
    BudgetReserved,
    LedgerContradictionResolved,
    LedgerEdgeAdded,
    LedgerNodeAdded,
    LedgerNodeRevised,
    LedgerNodeStatusChanged,
    RunStatusChanged,
    UncommittedEvent,
)
from fre.runtime.events import (
    TestValueSet as ValueSet,
)


def task_signature() -> TaskSignature:
    return TaskSignature(
        task_type=TaskType.ANALYSIS,
        consequence=Ordinal4.LOW,
        irreversibility=Ordinal4.LOW,
        ambiguity=Ordinal4.LOW,
        search_space=SearchSpaceClass.CLOSED,
        evidence_scarcity=Ordinal4.LOW,
        horizon=HorizonClass.IMMEDIATE,
        output_form=OutputForm.TEXT,
        dimension_confidence={},
    )


def assert_rejected_atomically(
    engine: FrontierReasoningEngine,
    run_id: UUID,
    invalid: UncommittedEvent,
) -> None:
    before = engine.inspect(run_id)
    before_events = engine.store.load(run_id)
    snapshot_hash = engine.snapshot(run_id)
    prefix = (
        engine.make_event(run_id, ValueSet(key="valid-1", value=1)),
        engine.make_event(run_id, ValueSet(key="valid-2", value=2)),
    )
    with pytest.raises(ValueError):
        engine.append(run_id, before.version, (*prefix, invalid))
    assert engine.inspect(run_id) == before
    assert engine.store.load(run_id) == before_events
    assert engine.verify(run_id).state_hash == snapshot_hash


def ledger_node(value: int, status: EpistemicStatus = EpistemicStatus.SUPPORTED) -> LedgerNode:
    return make_node(
        node_id=UUID(int=500 + value),
        revision=1,
        node_type=LedgerNodeType.FACT,
        content={"value": value},
        status=status,
        created_at="2026-01-01T00:00:00Z",
        action_id=UUID(int=600 + value),
        module_id="M09",
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "violation",
    ["dangling", "cycle", "revision_hash", "status", "resolves", "incomplete_resolution"],
)
def test_ledger_invariant_batches_are_pre_reduced_atomically(
    engine: FrontierReasoningEngine, violation: str
) -> None:
    run = engine.create_run({"violation": violation})
    left, right = ledger_node(1), ledger_node(2)
    if violation == "dangling":
        bad_edge = LedgerEdge(
            edge_id=UUID(int=701),
            source=left.ref,
            target=right.ref,
            relation=LedgerRelation.SUPPORTS,
        )
        invalid = engine.make_event(run.run_id, LedgerEdgeAdded(edge=bad_edge), module_id="M09")
    else:
        payload: FrozenModel
        initial = [
            engine.make_event(run.run_id, LedgerNodeAdded(node=left), module_id="M09"),
            engine.make_event(run.run_id, LedgerNodeAdded(node=right), module_id="M09"),
        ]
        if violation in {"cycle", "incomplete_resolution"}:
            forward = LedgerEdge(
                edge_id=UUID(int=702),
                source=left.ref,
                target=right.ref,
                relation=LedgerRelation.SUPPORTS
                if violation == "cycle"
                else LedgerRelation.CONTRADICTS,
            )
            initial.append(
                engine.make_event(run.run_id, LedgerEdgeAdded(edge=forward), module_id="M09")
            )
        engine.append(run.run_id, run.version, tuple(initial))
        if violation == "cycle":
            payload = LedgerEdgeAdded(
                edge=LedgerEdge(
                    edge_id=UUID(int=703),
                    source=right.ref,
                    target=left.ref,
                    relation=LedgerRelation.SUPPORTS,
                )
            )
        elif violation == "revision_hash":
            bad_successor = make_node(
                node_id=left.node_id,
                revision=2,
                node_type=left.node_type,
                content={"bad": True},
                status=EpistemicStatus.SUPPORTED,
                created_at="2026-01-01T00:00:01Z",
                action_id=UUID(int=704),
                module_id="M09",
                predecessor_revision_hash="f" * 64,
            )
            payload = LedgerNodeRevised(successor=bad_successor)
        elif violation == "status":
            refuted = ledger_node(3, EpistemicStatus.REFUTED)
            state = engine.inspect(run.run_id)
            engine.append(
                run.run_id,
                state.version,
                (engine.make_event(run.run_id, LedgerNodeAdded(node=refuted), module_id="M09"),),
            )
            payload = LedgerNodeStatusChanged(
                node_ref=refuted.ref, status=EpistemicStatus.SUPPORTED
            )
        elif violation == "resolves":
            payload = LedgerEdgeAdded(
                edge=LedgerEdge(
                    edge_id=UUID(int=705),
                    source=left.ref,
                    target=right.ref,
                    relation=LedgerRelation.RESOLVES,
                )
            )
        else:
            payload = LedgerContradictionResolved(
                resolution=ContradictionResolution(
                    contradiction_edge_ids=(UUID(int=702),),
                    resolver_ref=left.ref,
                    affected_refs=(right.ref,),
                    status_transitions=(),
                    outcome="incomplete",
                    action_id=UUID(int=706),
                    module_id="M09",
                ),
                resolution_edge=LedgerEdge(
                    edge_id=UUID(int=707),
                    source=left.ref,
                    target=right.ref,
                    relation=LedgerRelation.RESOLVES,
                ),
            )
        invalid = engine.make_event(run.run_id, payload, module_id="M09")
    assert_rejected_atomically(engine, run.run_id, invalid)


@pytest.mark.integration
@pytest.mark.parametrize("violation", ["overspend", "reservation"])
def test_budget_invariant_batches_are_pre_reduced_atomically(
    engine: FrontierReasoningEngine, violation: str
) -> None:
    run = engine.create_run({"violation": violation})
    plan, policy_hash = BudgetAllocator().allocate(
        task_signature(), default_tier_policy(), DeploymentLimits()
    )
    allocation = engine.make_event(
        run.run_id,
        BudgetAllocated(plan=plan, policy_version=plan.policy_version, policy_hash=policy_hash),
        module_id="M02",
    )
    engine.append(run.run_id, run.version, (allocation,))
    if violation == "overspend":
        budget_payload: FrozenModel = BudgetConsumed(usage=ResourceVector(iterations=2))
    else:
        reservation = BudgetReservation(
            reservation_id="existing", action_id="a1", resources=ResourceVector(iterations=1)
        )
        state = engine.inspect(run.run_id)
        engine.append(
            run.run_id,
            state.version,
            (engine.make_event(run.run_id, BudgetReserved(reservation=reservation)),),
        )
        budget_payload = BudgetReserved(
            reservation=BudgetReservation(
                reservation_id="conflict", action_id="a2", resources=ResourceVector(iterations=1)
            )
        )
    invalid = engine.make_event(run.run_id, budget_payload, module_id="budget-meter")
    assert_rejected_atomically(engine, run.run_id, invalid)


@pytest.mark.integration
def test_terminal_without_context_batch_is_pre_reduced_atomically(
    engine: FrontierReasoningEngine,
) -> None:
    run = engine.create_run({"violation": "terminal"})
    invalid = engine.make_event(
        run.run_id,
        RunStatusChanged(status="BLOCKED", reason="missing context"),
        module_id="runtime",
    )
    assert_rejected_atomically(engine, run.run_id, invalid)


@pytest.mark.integration
def test_forged_semantic_success_batch_is_pre_reduced_atomically(
    engine: FrontierReasoningEngine,
) -> None:
    """F09: a `ModelCallRecordedV2` forged without a matching same-batch
    settlement must leave no committed prefix -- exercised as the final event
    of an otherwise-valid batch, mirroring the ledger/budget/terminal cases
    above."""
    import asyncio

    from fre.domain.common import JsonValue, OutputContract, PermissionSet
    from fre.domain.semantic import (
        SemanticModelCallRecordV2,
        StructuredModelRequest,
        StructuredModelResult,
        StructuredModelStatus,
    )
    from fre.modules.m01_classifier import TaskClassifier
    from fre.prompts import default_output_schema_registry, default_prompt_registry
    from fre.runtime.events import ModelCallRecordedV2
    from fre.semantic_runtime import SemanticModelRuntime, SemanticRuntimePolicy

    # C05 remediation (finding #10): this fixture must decode as a genuine
    # SUCCESS against the real, current `ClassificationOutput` schema (nested
    # `TaskTypeProposal`/`SearchSpaceProposal`/`HorizonProposal` objects, not
    # flat scalars with sibling `*_confidence` keys) -- otherwise this test
    # silently stops exercising a genuine SUCCESS decode and instead degrades
    # to INVALID_STRUCTURED_OUTPUT, which is a different code path than the
    # one this test's name and docstring claim to cover.
    valid: dict[str, JsonValue] = {
        "task_type": {
            "estimate": "DECISION",
            "confidence": 0.9,
            "anchors": [],
            "rationale": "test",
        },
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
        "search_space": {
            "estimate": "BOUNDED",
            "confidence": 0.9,
            "anchors": [],
            "rationale": "test",
        },
        "horizon": {
            "estimate": "SHORT",
            "confidence": 0.9,
            "anchors": [],
            "rationale": "test",
        },
    }

    class CapturingModel:
        def __init__(self) -> None:
            self.requests: list[StructuredModelRequest] = []

        async def generate(self, request: StructuredModelRequest) -> StructuredModelResult:
            self.requests.append(request)
            return StructuredModelResult(
                status=StructuredModelStatus.SUCCESS,
                adapter_id="test",
                model_id="test",
                raw_response=b"",
                decoded=valid,
            )

    run = engine.create_run({"violation": "semantic-forgery"})
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
    engine.append(
        run.run_id,
        run.version,
        (
            engine.make_event(
                run.run_id,
                BudgetAllocated(
                    plan=plan, policy_version=plan.policy_version, policy_hash=policy_hash
                ),
                module_id="M02",
            ),
        ),
    )
    runtime = SemanticModelRuntime(
        CapturingModel(), engine, default_prompt_registry(), default_output_schema_registry()
    )
    genuine = asyncio.run(
        runtime.execute(
            run_id=run.run_id,
            module_id="M01",
            module_version="1.0",
            operation="classify",
            prompt_id="m01.classify",
            prompt_version="1.0",
            canonical_input=task.model_dump(mode="json"),
            policy=SemanticRuntimePolicy(maximum_repair_attempts=0),
        )
    ).record
    assert isinstance(genuine, SemanticModelCallRecordV2)
    # Pin the fixture to a genuine SUCCESS decode (finding #10): before the
    # fixture was updated to the nested proposal shape, this record silently
    # decoded as INVALID_STRUCTURED_OUTPUT instead, which is a different code
    # path than the one this test claims ("forged semantic *success* batch").
    assert genuine.status is StructuredModelStatus.SUCCESS
    forged = genuine.model_copy(
        update={
            "call_id": engine.uuids.new(),
            "idempotency_key": "a" * 64,
            "reservation_id": "never-reserved",
        }
    )
    invalid = engine.make_event(
        run.run_id, ModelCallRecordedV2(record=forged), module_id="attacker"
    )
    assert_rejected_atomically(engine, run.run_id, invalid)
