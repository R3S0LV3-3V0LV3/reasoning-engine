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
from fre.domain.stop import (
    AcceptanceStatus,
    StopDecisionRecord,
    StopInputs,
    StopPolicy,
    ValidationStatus,
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
from fre.modules.m12_context import ContextCompiler
from fre.modules.m13_stop import StopController
from fre.runtime.budget_meter import BudgetMeter
from fre.runtime.events import (
    BudgetAllocated,
    BudgetConsumed,
    BudgetReservationSettled,
    BudgetReserved,
    ContextCompiled,
    EventPayload,
    LedgerDependentsMarkedStale,
    LedgerEdgeAdded,
    LedgerNodeAdded,
    LedgerNodeRevised,
    StopDecisionRecordedV2,
    StoredEvent,
)
from fre.runtime.events import TestValueSet as ValueSet
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
    with pytest.raises(ValueError, match="must be recorded before finalization"):
        runtime.finalize(run.run_id, decision)
    runtime.record_decision(run.run_id, decision)
    terminal_hash = runtime.finalize(run.run_id, decision)
    expected = engine.inspect(run.run_id)
    assert expected.terminal_context_packet_hash == terminal_hash
    assert expected.status == "COMPLETE"
    terminal_event = next(
        event
        for event in reversed(engine.store.load(run.run_id))
        if event.event_type == "ContextCompiled"
    )
    terminal_payload = terminal_event.validated_payload()
    assert isinstance(terminal_payload, ContextCompiled)
    assert terminal_payload.json_artifact is not None
    assert terminal_payload.json_artifact.sha256 != terminal_payload.packet.packet_hash
    snapshot_hash = engine.snapshot(run.run_id)

    events = engine.store.load(run.run_id)
    pure = RunReducer().reduce(run.run_id, events)
    database = engine.store.path
    engine.store.close()
    engine.store = SQLiteStore(database)
    persisted = engine.replay(run.run_id)
    snapshotted = engine.verify(run.run_id)
    snapshot_assisted = engine.replay_from_snapshot(run.run_id)
    assert pure.model_dump_json() == persisted.model_dump_json() == snapshotted.model_dump_json()
    assert snapshot_assisted.model_dump_json() == pure.model_dump_json()
    assert pure.state_hash == persisted.state_hash == snapshot_hash
    assert pure.ledger == persisted.ledger
    assert pure.budget == persisted.budget
    assert BudgetMeter().burn_rate(pure.budget) == BudgetMeter().burn_rate(persisted.budget)
    assert pure.context_packets[-1].packet_hash == persisted.context_packets[-1].packet_hash
    assert pure.stop_decisions[-1] == persisted.stop_decisions[-1]


@pytest.mark.integration
def test_explicit_staleness_event_marks_exact_dependents(
    engine: FrontierReasoningEngine,
) -> None:
    run = engine.create_run({"staleness": "explicit"})
    source = make_node(
        node_id=UUID(int=801),
        revision=1,
        node_type=LedgerNodeType.FACT,
        content={"claim": "source"},
        status=EpistemicStatus.SUPPORTED,
        created_at="2026-01-01T00:00:00Z",
        action_id=UUID(int=802),
        module_id="M09",
    )
    dependent = make_node(
        node_id=UUID(int=803),
        revision=1,
        node_type=LedgerNodeType.INFERENCE,
        content={"claim": "dependent"},
        status=EpistemicStatus.SUPPORTED,
        created_at="2026-01-01T00:00:01Z",
        action_id=UUID(int=804),
        module_id="M09",
    )
    relation = LedgerEdge(
        edge_id=UUID(int=805),
        source=source.ref,
        target=dependent.ref,
        relation=LedgerRelation.SUPPORTS,
    )
    payloads = (
        LedgerNodeAdded(node=source),
        LedgerNodeAdded(node=dependent),
        LedgerEdgeAdded(edge=relation),
        LedgerDependentsMarkedStale(source_ref=source.ref, affected_refs=(dependent.ref,)),
    )
    events = tuple(engine.make_event(run.run_id, payload, module_id="M09") for payload in payloads)
    engine.append(run.run_id, run.version, events)
    state = engine.inspect(run.run_id)
    assert state.ledger.status_overlays == ((dependent.ref, EpistemicStatus.STALE),)

    invalid = engine.make_event(
        run.run_id,
        LedgerDependentsMarkedStale(source_ref=source.ref, affected_refs=()),
        module_id="M09",
    )
    with pytest.raises(ValueError, match="does not match ledger projection"):
        engine.append(run.run_id, state.version, (invalid,))


@pytest.mark.integration
@pytest.mark.parametrize("marker_source", ["predecessor", "successor"])
def test_revision_stale_markers_survive_sqlite_and_snapshot_replay(
    engine: FrontierReasoningEngine, marker_source: str
) -> None:
    run = engine.create_run({"marker": marker_source})
    source = make_node(
        node_id=UUID(int=811),
        revision=1,
        node_type=LedgerNodeType.FACT,
        content="source",
        status=EpistemicStatus.SUPPORTED,
        created_at="2026-01-01T00:00:00Z",
        action_id=UUID(int=812),
        module_id="M09",
    )
    dependent = make_node(
        node_id=UUID(int=813),
        revision=1,
        node_type=LedgerNodeType.INFERENCE,
        content="dependent",
        status=EpistemicStatus.SUPPORTED,
        created_at="2026-01-01T00:00:01Z",
        action_id=UUID(int=814),
        module_id="M09",
    )
    successor = make_node(
        node_id=source.node_id,
        revision=2,
        node_type=source.node_type,
        content="revised",
        status=EpistemicStatus.SUPPORTED,
        created_at="2026-01-01T00:00:02Z",
        action_id=UUID(int=815),
        module_id="M09",
        predecessor_revision_hash=source.revision_hash,
    )
    payloads = (
        LedgerNodeAdded(node=source),
        LedgerNodeAdded(node=dependent),
        LedgerEdgeAdded(
            edge=LedgerEdge(
                edge_id=UUID(int=816),
                source=source.ref,
                target=dependent.ref,
                relation=LedgerRelation.SUPPORTS,
            )
        ),
        LedgerNodeRevised(successor=successor),
        LedgerDependentsMarkedStale(
            source_ref=source.ref if marker_source == "predecessor" else successor.ref,
            affected_refs=(dependent.ref,),
        ),
    )
    events = tuple(engine.make_event(run.run_id, item, module_id="M09") for item in payloads)
    engine.append(run.run_id, run.version, events)
    expected = engine.inspect(run.run_id)
    engine.snapshot(run.run_id)
    path = engine.store.path
    engine.store.close()
    engine.store = SQLiteStore(path)
    assert engine.replay(run.run_id) == expected
    assert engine.replay_from_snapshot(run.run_id) == expected


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
@pytest.mark.parametrize("mutation", ["budget", "ledger", "state"])
def test_finalization_rejects_stop_decisions_staled_by_authoritative_changes(
    engine: FrontierReasoningEngine, mutation: str
) -> None:
    run = engine.create_run({"stale-stop": mutation})
    plan, policy_hash = BudgetAllocator().allocate(
        signature(), default_tier_policy(), DeploymentLimits()
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
    if mutation == "budget":
        payload: EventPayload = BudgetConsumed(usage=ResourceVector(iterations=1))
    elif mutation == "ledger":
        payload = LedgerNodeAdded(
            node=make_node(
                node_id=UUID(int=850),
                revision=1,
                node_type=LedgerNodeType.FACT,
                content="late fact",
                status=EpistemicStatus.SUPPORTED,
                created_at="2026-01-01T00:00:00Z",
                action_id=UUID(int=851),
                module_id="M09",
            )
        )
    else:
        payload = ValueSet(key="late", value=True)
    current = engine.inspect(run.run_id)
    engine.append(
        run.run_id,
        current.version,
        (engine.make_event(run.run_id, payload, module_id="late-change"),),
    )
    with pytest.raises(ValueError, match="stale"):
        runtime.finalize(run.run_id, decision)


def test_equal_stop_decision_can_be_recomputed_against_fresh_state(
    engine: FrontierReasoningEngine,
) -> None:
    run = engine.create_run({"stale-stop": "duplicate"})
    plan, policy_hash = BudgetAllocator().allocate(
        signature(), default_tier_policy(), DeploymentLimits()
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
    first_sequence = runtime.record_decision(run.run_id, decision)
    current = engine.inspect(run.run_id)
    engine.append(
        run.run_id,
        current.version,
        (engine.make_event(run.run_id, ValueSet(key="late", value=True)),),
    )

    second_sequence = runtime.record_decision(run.run_id, decision)
    rebound = engine.inspect(run.run_id)
    assert second_sequence > first_sequence
    assert len(rebound.stop_decision_records) == 2
    assert rebound.stop_decision_records[0].decision == decision
    assert rebound.stop_decision_records[1].decision == decision
    assert (
        rebound.stop_decision_records[0].evaluated_state_version
        < rebound.stop_decision_records[1].evaluated_state_version
    )
    assert runtime.finalize(run.run_id, decision)


def test_stop_decision_v2_rejects_tampered_or_duplicate_state_binding(
    engine: FrontierReasoningEngine,
) -> None:
    run = engine.create_run({"stop-binding": "tamper"})
    plan, policy_hash = BudgetAllocator().allocate(
        signature(), default_tier_policy(), DeploymentLimits()
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
    state = engine.inspect(run.run_id)
    decision = StopController().evaluate(
        StopInputs(
            budget=BudgetMeter().remaining(state.budget),
            acceptance=AcceptanceStatus.SATISFIED,
            validation=ValidationStatus.COMPLETE,
        ),
        StopPolicy(version="stop/1.0"),
    )
    valid_record = StopDecisionRecord(
        decision=decision,
        evaluated_state_version=state.version,
        evaluated_state_hash=state.state_hash,
        budget_projection_hash=decision.budget_projection_hash,
    )
    tampered = valid_record.model_copy(update={"evaluated_state_hash": "f" * 64})
    with pytest.raises(ValueError, match="state hash"):
        engine.append(
            run.run_id,
            state.version,
            (
                engine.make_event(
                    run.run_id, StopDecisionRecordedV2(record=tampered), module_id="M13"
                ),
            ),
        )
    assert engine.inspect(run.run_id).version == state.version

    event = engine.make_event(
        run.run_id, StopDecisionRecordedV2(record=valid_record), module_id="M13"
    )
    engine.append(run.run_id, state.version, (event,))
    recorded = engine.inspect(run.run_id)
    with pytest.raises(ValueError, match="state version"):
        engine.append(
            run.run_id,
            recorded.version,
            (
                engine.make_event(
                    run.run_id, StopDecisionRecordedV2(record=valid_record), module_id="M13"
                ),
            ),
        )


def test_stop_decision_v2_binding_survives_snapshot_and_replay(
    engine: FrontierReasoningEngine,
) -> None:
    run = engine.create_run({"stop-binding": "snapshot"})
    plan, policy_hash = BudgetAllocator().allocate(
        signature(), default_tier_policy(), DeploymentLimits()
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
    state = engine.inspect(run.run_id)
    decision = StopController().evaluate(
        StopInputs(
            budget=BudgetMeter().remaining(state.budget),
            acceptance=AcceptanceStatus.SATISFIED,
            validation=ValidationStatus.COMPLETE,
        ),
        StopPolicy(version="stop/1.0"),
    )
    Wave2Runtime(engine).record_decision(run.run_id, decision)
    wire_event = engine.store.load(run.run_id)[-1]
    assert wire_event.event_type == "StopDecisionRecorded"
    assert wire_event.schema_version == "2.0"
    assert isinstance(wire_event.validated_payload(), StopDecisionRecordedV2)
    expected = engine.inspect(run.run_id)
    engine.snapshot(run.run_id)

    assert engine.replay(run.run_id).stop_decision_records == expected.stop_decision_records
    assert engine.verify(run.run_id).stop_decision_records == expected.stop_decision_records
    assert (
        engine.replay_from_snapshot(run.run_id).stop_decision_records
        == expected.stop_decision_records
    )


def test_finalize_accepts_stop_decision_recorded_inside_multi_event_batch(
    engine: FrontierReasoningEngine,
) -> None:
    """Wave2Runtime.finalize compares the sequence of the scanned
    StopDecisionRecordedV2 event against ``latest_record.evaluated_state_version
    + 1``. That relationship is not an assumption of "one event per version" --
    it is guaranteed by RunReducer.apply's own invariants for any event that was
    actually applied (event.sequence == pre-apply state.version + 1, and the
    reducer separately requires record.evaluated_state_version == that same
    pre-apply state.version). It therefore holds regardless of what else is
    batched alongside the decision event, as long as the decision is evaluated
    against the state as it will exist *after* any earlier same-batch events
    are applied. This test appends an unrelated event and the stop decision
    together in a single multi-event ``append()`` call and confirms finalize()
    still recognizes and accepts the binding -- guarding SQLiteStore's
    per-event (not per-batch) sequence assignment in ``_append_locked``."""
    run = engine.create_run({"multi-event-batch": "stop-decision"})
    plan, policy_hash = BudgetAllocator().allocate(
        signature(), default_tier_policy(), DeploymentLimits()
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
    base_state = engine.inspect(run.run_id)
    side_event = engine.make_event(
        run.run_id, ValueSet(key="batched", value=True), module_id="test"
    )
    # Preview the side event to compute the state the decision must be bound
    # to: the state as it will exist after THIS batch's first event applies,
    # not the state before the batch started.
    previewed_side_event = StoredEvent.model_validate(
        {
            **side_event.model_dump(),
            "created_at": side_event.created_at,
            "sequence": base_state.version + 1,
        },
        strict=True,
    )
    intermediate_state = RunReducer().apply(base_state, previewed_side_event)
    decision = StopController().evaluate(
        StopInputs(
            budget=BudgetMeter().remaining(intermediate_state.budget),
            acceptance=AcceptanceStatus.SATISFIED,
            validation=ValidationStatus.COMPLETE,
        ),
        StopPolicy(version="stop/1.0"),
    )
    remaining = BudgetMeter().remaining(intermediate_state.budget)
    record = StopDecisionRecord(
        decision=decision,
        evaluated_state_version=intermediate_state.version,
        evaluated_state_hash=intermediate_state.state_hash,
        budget_projection_hash=remaining.projection_hash,
    )
    decision_event = engine.make_event(
        run.run_id, StopDecisionRecordedV2(record=record), module_id="M13"
    )
    engine.append(run.run_id, base_state.version, (side_event, decision_event))

    state_after_batch = engine.inspect(run.run_id)
    assert state_after_batch.stop_decision_records[-1] == record
    assert state_after_batch.values == {"batched": True}

    runtime = Wave2Runtime(engine)
    terminal_hash = runtime.finalize(run.run_id, decision)
    final_state = engine.inspect(run.run_id)
    assert final_state.terminal_context_packet_hash == terminal_hash
    assert final_state.status == decision.disposition.value


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


@pytest.mark.integration
def test_empty_v2_stop_decision_records_preserve_historic_v1_snapshot_shape(
    engine: FrontierReasoningEngine,
) -> None:
    """Reducer "2.0" with an unpopulated stop_decision_records projection must
    hash identically to a Wave 1/2 run that never knew the field existed --
    RunState.snapshot_payload() pops the key only when it is empty."""
    run = engine.create_run({"snapshot-matrix": "empty-v2-records"})
    state = engine.inspect(run.run_id)
    assert state.stop_decision_records == ()
    payload = state.snapshot_payload()
    assert "stop_decision_records" not in payload
    assert (
        state.state_hash
        == RunReducer().reduce(run.run_id, engine.store.load(run.run_id)).state_hash
    )


@pytest.mark.integration
def test_nonempty_v2_stop_decision_records_are_never_silently_dropped(
    engine: FrontierReasoningEngine,
) -> None:
    """Guards RunState.snapshot_payload(): the ``stop_decision_records`` pop
    must stay conditional on emptiness alone. If someone makes it unconditional
    (dropping populated v2 records the same way empty ones are dropped for v1
    compatibility), this test must fail -- the key would vanish and the hash
    would stop depending on the recorded decision content."""
    run = engine.create_run({"snapshot-matrix": "nonempty-v2-records"})
    plan, policy_hash = BudgetAllocator().allocate(
        signature(), default_tier_policy(), DeploymentLimits()
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
    empty_state = engine.inspect(run.run_id)
    empty_hash = empty_state.state_hash
    decision = StopController().evaluate(
        StopInputs(
            budget=BudgetMeter().remaining(empty_state.budget),
            acceptance=AcceptanceStatus.SATISFIED,
            validation=ValidationStatus.COMPLETE,
        ),
        StopPolicy(version="stop/1.0"),
    )
    Wave2Runtime(engine).record_decision(run.run_id, decision)
    populated_state = engine.inspect(run.run_id)
    assert populated_state.stop_decision_records != ()
    payload = populated_state.snapshot_payload()
    assert "stop_decision_records" in payload
    assert payload["stop_decision_records"] == [
        record.model_dump(mode="json") for record in populated_state.stop_decision_records
    ]
    # The hash must move once the (non-empty) field is included -- if the pop
    # became unconditional, populated_state.state_hash would collapse back to
    # something computed without this field and could coincide with a state
    # that never had a decision recorded at all in degenerate cases; comparing
    # directly against the empty-records hash before this decision was
    # recorded is the sharpest, least coincidence-prone check available.
    assert populated_state.state_hash != empty_hash
    engine.snapshot(run.run_id)
    assert engine.verify(run.run_id).state_hash == populated_state.state_hash
    assert engine.replay_from_snapshot(run.run_id).state_hash == populated_state.state_hash


@pytest.mark.integration
def test_snapshot_assisted_path_replays_only_post_snapshot_events(
    engine: FrontierReasoningEngine,
) -> None:
    from fre.runtime.events import TestValueSet

    run = engine.create_run({"snapshot": "path-c"})
    before = engine.make_event(run.run_id, TestValueSet(key="before", value=1))
    engine.append(run.run_id, run.version, (before,))
    engine.snapshot(run.run_id)
    checkpoint = engine.verify(run.run_id)
    after = engine.make_event(run.run_id, TestValueSet(key="after", value=2))
    engine.append(run.run_id, checkpoint.version, (after,))
    full = engine.replay(run.run_id)
    assisted = engine.replay_from_snapshot(run.run_id)
    assert checkpoint.values == {"before": 1}
    assert assisted.model_dump_json() == full.model_dump_json()
    assert assisted.state_hash == full.state_hash


@pytest.mark.integration
@pytest.mark.parametrize(
    ("fixture_name", "expected_hash"),
    (
        ("wave1_history.json", "11c0e48f4e2ac232a4f6160d63c69933e1dbd5f9aa79a7bbdda292a17fd3ecc3"),
        ("wave2_history.json", "ca12efe637299d1ef1f425146cad2cee028b5a50412d47c091b2b1cb90335ac8"),
    ),
)
def test_historic_fixture_replays_and_snapshot_verifies(
    tmp_path: object, fixture_name: str, expected_hash: str
) -> None:
    import json
    from pathlib import Path

    from fre.domain.common import canonical_hash
    from fre.runtime.events import StoredEvent
    from fre.runtime.reducer import RunReducer, RunState

    fixture = json.loads(Path("tests/fixtures", fixture_name).read_text(encoding="utf-8"))
    assert fixture["state_hash"] == expected_hash
    events = tuple(StoredEvent.model_validate(item, strict=False) for item in fixture["events"])
    run_id = events[0].run_id
    state = RunReducer().reduce(run_id, events)
    assert state.snapshot_payload() == fixture["snapshot"]
    assert state.state_hash == fixture["state_hash"]
    historic_snapshot = RunState.model_validate(fixture["snapshot"], strict=False)
    assert canonical_hash(fixture["snapshot"]) == fixture["state_hash"]
    assert historic_snapshot.state_hash == fixture["state_hash"]
    store = SQLiteStore(Path(str(tmp_path)) / "historic.sqlite3")
    first = fixture["events"][0]
    store.execute_for_test(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            str(run_id),
            fixture["snapshot"]["status"],
            "1.0",
            first["payload"]["config_hash"],
            fixture["snapshot"]["version"],
            first["created_at"],
            fixture["events"][-1]["created_at"],
        ),
    )
    for event in fixture["events"]:
        store.execute_for_test(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event["event_id"],
                event["run_id"],
                event["sequence"],
                event["event_type"],
                event["action_id"],
                event["module_id"],
                event["schema_version"],
                event["module_version"],
                event["input_hash"],
                event["created_at"],
                json.dumps(event["payload"], sort_keys=True, separators=(",", ":")),
            ),
        )
    store.execute_for_test(
        "INSERT INTO snapshots VALUES (?, ?, ?, ?, ?, ?)",
        (
            str(run_id),
            fixture["snapshot"]["version"],
            json.dumps(fixture["snapshot"], sort_keys=True, separators=(",", ":")),
            fixture["state_hash"],
            fixture["reducer_version"],
            fixture["events"][-1]["created_at"],
        ),
    )
    assert store.verify_snapshot(run_id, RunReducer()).state_hash == fixture["state_hash"]
    assert store.load_snapshot(run_id, RunReducer()).state_hash == fixture["state_hash"]
    assert store.replay_from_snapshot(run_id, RunReducer()).state_hash == fixture["state_hash"]


@pytest.mark.integration
def test_historic_budget_replay_ignores_changed_live_policy(
    engine: FrontierReasoningEngine,
) -> None:
    run = engine.create_run({"policy": "A"})
    allocator = BudgetAllocator()
    policy_a = default_tier_policy()
    plan_a, hash_a = allocator.allocate(signature(), policy_a, DeploymentLimits())
    allocated_event = engine.make_event(
        run.run_id,
        BudgetAllocated(plan=plan_a, policy_version=policy_a.policy_version, policy_hash=hash_a),
        module_id="M02",
    )
    engine.append(run.run_id, run.version, (allocated_event,))
    policy_b = policy_a.model_copy(update={"policy_version": "policy-B/2.0"})
    assert allocator.policy_hash(policy_b, DeploymentLimits()) != hash_a
    database = engine.store.path
    engine.store.close()
    engine.store = SQLiteStore(database)
    replayed = engine.replay(run.run_id)
    assert replayed.budget.plan == plan_a
    assert replayed.budget.policy_hash == hash_a


@pytest.mark.integration
def test_terminal_association_must_match_latest_stop_decision(
    engine: FrontierReasoningEngine,
) -> None:
    from fre.domain.common import ArtifactRef
    from fre.domain.stop import StopDisposition
    from fre.runtime.events import (
        ContextCompiled,
        RunStatusChanged,
        StopDecisionRecorded,
        TerminalContextAssociated,
    )

    run = engine.create_run({"terminal": "mismatch"})
    allocator = BudgetAllocator()
    plan, policy_hash = allocator.allocate(signature(), default_tier_policy(), DeploymentLimits())
    allocation = engine.make_event(
        run.run_id,
        BudgetAllocated(plan=plan, policy_version=plan.policy_version, policy_hash=policy_hash),
        module_id="M02",
    )
    engine.append(run.run_id, run.version, (allocation,))
    state = engine.inspect(run.run_id)
    decision = StopController().evaluate(
        StopInputs(
            budget=BudgetMeter().remaining(state.budget),
            acceptance=AcceptanceStatus.SATISFIED,
            validation=ValidationStatus.COMPLETE,
        ),
        StopPolicy(version="stop/1.0"),
    )
    packet = (
        ContextCompiler()
        .compile(
            run_id=run.run_id,
            snapshot_version=state.version,
            ledger=state.ledger,
            budget_remaining=BudgetMeter().remaining(state.budget),
            profile=CompilerProfile.HANDOFF,
            terminal_disposition=StopDisposition.COMPLETE,
        )
        .packet
    )
    json_artifact = engine.store_artifact(b"{}", media_type="application/json")
    markdown_artifact = engine.store_artifact(b"# context", media_type="text/markdown")
    events = tuple(
        engine.make_event(run.run_id, payload, module_id="wave2-runtime")
        for payload in (
            StopDecisionRecorded(decision=decision),
            ContextCompiled(
                packet=packet,
                json_artifact=ArtifactRef(
                    artifact_id=json_artifact.id, sha256=json_artifact.sha256
                ),
                markdown_artifact=ArtifactRef(
                    artifact_id=markdown_artifact.id, sha256=markdown_artifact.sha256
                ),
            ),
            TerminalContextAssociated(
                stop_disposition=StopDisposition.BLOCKED,
                packet_hash=packet.packet_hash,
            ),
        )
    )
    version = state.version
    event_count = len(engine.store.load(run.run_id))
    with pytest.raises(ValueError, match="latest recorded stop decision"):
        engine.append(run.run_id, version, events)
    assert engine.inspect(run.run_id).version == version
    assert len(engine.store.load(run.run_id)) == event_count

    mismatched_status_events = tuple(
        engine.make_event(run.run_id, payload, module_id="wave2-runtime")
        for payload in (
            StopDecisionRecorded(decision=decision),
            ContextCompiled(
                packet=packet,
                json_artifact=ArtifactRef(
                    artifact_id=json_artifact.id, sha256=json_artifact.sha256
                ),
                markdown_artifact=ArtifactRef(
                    artifact_id=markdown_artifact.id, sha256=markdown_artifact.sha256
                ),
            ),
            TerminalContextAssociated(
                stop_disposition=StopDisposition.COMPLETE,
                packet_hash=packet.packet_hash,
            ),
            RunStatusChanged(status="CANCELLED", reason="mismatched terminal status"),
        )
    )
    with pytest.raises(ValueError, match="terminal context association"):
        engine.append(run.run_id, version, mismatched_status_events)
    assert engine.inspect(run.run_id).version == version
    assert len(engine.store.load(run.run_id)) == event_count
