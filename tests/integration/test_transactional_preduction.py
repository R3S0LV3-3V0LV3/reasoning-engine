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
