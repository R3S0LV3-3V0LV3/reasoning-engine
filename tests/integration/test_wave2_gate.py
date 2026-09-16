from uuid import UUID

import pytest

from fre.adapters.storage_sqlite import SQLiteStore
from fre.domain.budget import BudgetReservation, DeploymentLimits, ResourceVector
from fre.domain.context import CompilerProfile
from fre.domain.ledger import (
    EpistemicStatus,
    LedgerEdge,
    LedgerNodeType,
    LedgerRelation,
)
from fre.domain.stop import AcceptanceStatus, StopInputs, StopPolicy, ValidationStatus
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
from fre.modules.m12_context import ContextCompiler
from fre.modules.m13_stop import StopController
from fre.runtime.budget_meter import BudgetMeter
from fre.runtime.events import (
    BudgetAllocated,
    BudgetConsumed,
    BudgetReservationSettled,
    BudgetReserved,
    ContextCompiled,
    LedgerEdgeAdded,
    LedgerNodeAdded,
    LedgerNodeRevised,
)
from fre.runtime.reducer import RunReducer
from fre.runtime.wave2 import Wave2Runtime


def signature() -> TaskSignature:
    return TaskSignature(
        task_type=TaskType.ANALYSIS,
        consequence=Ordinal4.MEDIUM,
        irreversibility=Ordinal4.MEDIUM,
        ambiguity=Ordinal4.MEDIUM,
        search_space=SearchSpaceClass.BOUNDED,
        evidence_scarcity=Ordinal4.MEDIUM,
        horizon=HorizonClass.SHORT,
        output_form=OutputForm.STRUCTURED,
        dimension_confidence={},
    )


@pytest.mark.integration
def test_combined_wave2_lifecycle_and_differential_replay(engine: FrontierReasoningEngine) -> None:
    run = engine.create_run({"wave": 2})
    allocator = BudgetAllocator()
    plan, policy_hash = allocator.allocate(signature(), default_tier_policy(), DeploymentLimits())
    allocation = engine.make_event(
        run.run_id,
        BudgetAllocated(plan=plan, policy_version=plan.policy_version, policy_hash=policy_hash),
        module_id="M02",
    )
    engine.append(run.run_id, run.version, (allocation,))

    upstream = make_node(
        node_id=UUID(int=501),
        revision=1,
        node_type=LedgerNodeType.FACT,
        content={"claim": "source"},
        status=EpistemicStatus.SUPPORTED,
        created_at="2026-01-01T00:00:00Z",
        action_id=UUID(int=601),
        module_id="M09",
    )
    dependent = make_node(
        node_id=UUID(int=502),
        revision=1,
        node_type=LedgerNodeType.INFERENCE,
        content={"claim": "derived"},
        status=EpistemicStatus.PROVISIONAL,
        created_at="2026-01-01T00:00:01Z",
        action_id=UUID(int=602),
        module_id="M09",
    )
    relation = LedgerEdge(
        edge_id=UUID(int=701),
        source=upstream.ref,
        target=dependent.ref,
        relation=LedgerRelation.SUPPORTS,
    )
    successor = make_node(
        node_id=upstream.node_id,
        revision=2,
        node_type=LedgerNodeType.FACT,
        content={"claim": "source revised"},
        status=EpistemicStatus.SUPPORTED,
        created_at="2026-01-01T00:00:02Z",
        action_id=UUID(int=603),
        module_id="M09",
        predecessor_revision_hash=upstream.revision_hash,
    )
    state = engine.inspect(run.run_id)
    ledger_events = tuple(
        engine.make_event(run.run_id, payload, module_id="M09")
        for payload in (
            LedgerNodeAdded(node=upstream),
            LedgerNodeAdded(node=dependent),
            LedgerEdgeAdded(edge=relation),
            LedgerNodeRevised(successor=successor),
        )
    )
    engine.append(run.run_id, state.version, ledger_events)
    state = engine.inspect(run.run_id)
    assert state.ledger.stale_envelopes

    budget_events = tuple(
        engine.make_event(run.run_id, payload, module_id="budget-meter")
        for payload in (
            BudgetConsumed(usage=ResourceVector(iterations=1)),
            BudgetReserved(
                reservation=BudgetReservation(
                    reservation_id="r1", action_id="a1", resources=ResourceVector(llm_calls=1)
                )
            ),
            BudgetReservationSettled(reservation_id="r1", actual_usage=ResourceVector(llm_calls=1)),
        )
    )
    engine.append(run.run_id, state.version, budget_events)
    state = engine.inspect(run.run_id)
    remaining = BudgetMeter().remaining(state.budget)
    checkpoint = ContextCompiler().compile(
        run_id=run.run_id,
        snapshot_version=state.version,
        ledger=state.ledger,
        budget_remaining=remaining,
        profile=CompilerProfile.STANDARD,
    )
    context_event = engine.make_event(
        run.run_id, ContextCompiled(packet=checkpoint.packet), module_id="M12"
    )
    engine.append(run.run_id, state.version, (context_event,))

    state = engine.inspect(run.run_id)
    decision = StopController().evaluate(
        StopInputs(
            budget=BudgetMeter().remaining(state.budget),
            acceptance=AcceptanceStatus.SATISFIED,
            validation=ValidationStatus.COMPLETE,
        ),
        StopPolicy(version="stop/1.0"),
    )
    runtime = Wave2Runtime(engine)
    runtime.record_decision(run.run_id, decision)
    terminal_hash = runtime.finalize(run.run_id, decision)
    expected = engine.inspect(run.run_id)
    assert expected.terminal_context_packet_hash == terminal_hash
    assert expected.status == "COMPLETE"
    snapshot_hash = engine.snapshot(run.run_id)

    events = engine.store.load(run.run_id)
    pure = RunReducer().reduce(run.run_id, events)
    database = engine.store.path
    engine.store.close()
    engine.store = SQLiteStore(database)
    persisted = engine.replay(run.run_id)
    snapshotted = engine.verify(run.run_id)
    assert pure.model_dump_json() == persisted.model_dump_json() == snapshotted.model_dump_json()
    assert pure.state_hash == persisted.state_hash == snapshot_hash
    assert pure.ledger == persisted.ledger
    assert pure.budget == persisted.budget
    assert BudgetMeter().burn_rate(pure.budget) == BudgetMeter().burn_rate(persisted.budget)
    assert pure.context_packets[-1].packet_hash == persisted.context_packets[-1].packet_hash
    assert pure.stop_decisions[-1] == persisted.stop_decisions[-1]


@pytest.mark.integration
def test_terminal_status_without_context_is_rejected_atomically(
    engine: FrontierReasoningEngine,
) -> None:
    from fre.runtime.events import RunStatusChanged

    run = engine.create_run({"wave": 2})
    terminal = engine.make_event(
        run.run_id,
        RunStatusChanged(status="BLOCKED", reason="fixture"),
        module_id="runtime",
    )
    with pytest.raises(ValueError, match="terminal context"):
        engine.append(run.run_id, run.version, (terminal,))
    assert engine.inspect(run.run_id).version == run.version


@pytest.mark.integration
def test_wave1_state_hash_shape_remains_compatible(engine: FrontierReasoningEngine) -> None:
    from fre.domain.common import canonical_hash

    run = engine.create_run({"compatibility": "wave1"})
    state = engine.inspect(run.run_id)
    wave1_payload = {
        "run_id": str(state.run_id),
        "version": state.version,
        "status": state.status,
        "config_hash": state.config_hash,
        "artifacts": [],
        "values": {},
    }
    assert state.state_hash == canonical_hash(wave1_payload)
    engine.snapshot(run.run_id)
    assert engine.verify(run.run_id) == state
