"""Wave 3 golden fixtures A-L (C10, defect F14).

Every fixture below is a real, checked, `golden`-marked test that drives its
scenario through `fre.composition.Wave3Engine` -- the C09 coordinator -- and
asserts the coordinator's persisted, replayable output against a checked-in
evidence file under `tests/fixtures/golden/`. None of these bypass the
coordinator to hand-wire M01-M04/M12/M13 directly; that is the whole point
of C09 existing (see `fre.composition`'s module docstring) and is exactly
what `docs/wave3-requirements-matrix.md` cites as evidence for the rows this
file backs.

Fixture map (letter -> scenario -> matrix rows it backs):
  A - deterministic/no-model path                      -> WAVE3-M01-05, WAVE3-M04-04, WAVE3-C09-01
  B - full semantic classification+formalisation        -> WAVE3-M01-01, WAVE3-M03-01, WAVE3-C09-01
  C - provider usage exceeds reservation (C02)           -> WAVE3-SEM-02
  D - schema-invalid output exhausts repair (C04)        -> WAVE3-SEM-03
  E - resolved SUPPORTED_INFERENCE, exact-text anchor    -> WAVE3-M03-02
  F - fictitious/dangling support is rejected            -> WAVE3-M03-03
  G - UNKNOWN + assumption metadata preserved            -> WAVE3-M03-04
  H - material contradiction + blocker propagation       -> WAVE3-M03-05, WAVE3-M09-01, WAVE3-M12-03
  I - deterministic M04 selection outside the tie band   -> WAVE3-M04-01
  J - M04 adjudication inside the tie band                -> WAVE3-M04-02, WAVE3-M04-03
  K - persisted M12 v2 context + zero-call replay        -> WAVE3-M12-01, WAVE3-C09-02
  L - equal stop decision recomputed, then finalized      -> WAVE3-M13-01, WAVE3-M13-02
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import UUID

import pytest
from _common import (
    QueueModel,
    assert_matches_golden,
    classification_response,
    formalisation_response,
    golden_record,
    invalid_response,
    make_engine,
    make_task,
)

from fre.composition import compose_wave3
from fre.config import Wave3Config
from fre.domain.common import JsonValue, OutputContract, PermissionSet
from fre.domain.representation_registry import default_registry_v2
from fre.domain.semantic import (
    SemanticCallUsage,
    SemanticModelCallRecordV2,
    StructuredModelResult,
    StructuredModelStatus,
)
from fre.domain.stop import AcceptanceStatus, StopDisposition
from fre.domain.task import TaskEnvelope
from fre.modules.source_anchors import UnresolvedSupportReference

TASK_TEXT = "Choose a safe option."


def _anchor() -> JsonValue:
    return {
        "source_kind": "TASK_TEXT",
        "source_ref": {"object_type": "TaskEnvelope", "object_id": str(UUID(int=900))},
        "selector": "/text",
        "char_start": 0,
        "char_end": len(TASK_TEXT),
    }


# ---------------------------------------------------------------------------
# A - deterministic/no-model path
# ---------------------------------------------------------------------------


@pytest.mark.golden
def test_golden_a_deterministic_no_model_path(tmp_path: Path) -> None:
    """`allow_model=False` end-to-end: the front end completes with purely
    deterministic fallbacks and issues zero provider calls."""
    engine = make_engine(tmp_path)

    class RefusingModel:
        async def generate(self, request: object) -> StructuredModelResult:  # pragma: no cover
            raise AssertionError("fixture A must never call the provider")

    wave3 = compose_wave3(engine, RefusingModel(), Wave3Config())
    handle = wave3.create_run()
    task = make_task()

    result = asyncio.run(
        wave3.execute_front_end(handle.run_id, task, allow_model=False, allow_adjudication=False)
    )
    assert result.classification_record is not None
    assert result.classification_record.mode == "DETERMINISTIC_FALLBACK"
    assert result.classification_record.fallback_used is True
    assert result.blocked is False

    record = golden_record(engine, handle.run_id, fixture="A")
    assert record["model_calls"] == 0
    assert_matches_golden("a_deterministic_no_model", record)


# ---------------------------------------------------------------------------
# B - full semantic classification + formalisation
# ---------------------------------------------------------------------------


@pytest.mark.golden
def test_golden_b_full_semantic_classification_and_formalisation(tmp_path: Path) -> None:
    """Both M01 and M03 resolve real MODEL proposals through the coordinator."""
    engine = make_engine(tmp_path)
    model = QueueModel(
        [
            classification_response(),
            formalisation_response(
                [
                    {
                        "id": "obj-1",
                        "kind": "OBJECTIVE",
                        "description": "Pick the safest option",
                        "origin": "EXPLICIT_INPUT",
                        "anchors": [_anchor()],
                        "attributes": {"direction": "MIN"},
                    }
                ]
            ),
        ]
    )
    wave3 = compose_wave3(engine, model, Wave3Config())
    handle = wave3.create_run()
    task = make_task()

    result = asyncio.run(
        wave3.execute_front_end(handle.run_id, task, allow_model=True, allow_adjudication=True)
    )
    assert result.classification_record is not None
    assert result.classification_record.mode == "HYBRID"
    assert result.problem_spec is not None
    assert len(model.calls) == 2

    record = golden_record(engine, handle.run_id, fixture="B")
    assert_matches_golden("b_full_semantic_pipeline", record)


# ---------------------------------------------------------------------------
# C - provider usage exceeds reservation (proves C02 end-to-end)
# ---------------------------------------------------------------------------


@pytest.mark.golden
def test_golden_c_provider_overage_is_charged_at_reservation_cap(tmp_path: Path) -> None:
    """A provider that reports usage exceeding the reservation is charged at
    the reservation cap (never trusted at face value), classification falls
    through to the honest deterministic path for that call, and the rest of
    the front end still completes through the coordinator."""
    engine = make_engine(tmp_path)
    over_usage_response = classification_response(
        usage=SemanticCallUsage(input_tokens=10_000_000, output_tokens=10_000_000)
    )
    model = QueueModel([over_usage_response, formalisation_response([])])
    wave3 = compose_wave3(engine, model, Wave3Config())
    handle = wave3.create_run()
    task = make_task()

    result = asyncio.run(
        wave3.execute_front_end(handle.run_id, task, allow_model=True, allow_adjudication=False)
    )
    assert result.classification_record is not None
    assert result.classification_record.fallback_used is True
    state = engine.inspect(handle.run_id)
    over_call_any = next(
        call
        for call in state.model_calls
        if call.module_id == "M01" and call.operation == "classify"
    )
    assert isinstance(over_call_any, SemanticModelCallRecordV2)
    over_call = over_call_any
    assert over_call.accounting_condition is not None
    assert (
        over_call.charged_usage.input_tokens
        <= wave3.components.policy.semantic_runtime.reserve_input_tokens
    )

    record = golden_record(
        engine,
        handle.run_id,
        fixture="C",
        accounting_condition=str(over_call.accounting_condition),
    )
    assert_matches_golden("c_provider_overage", record)


# ---------------------------------------------------------------------------
# D - schema-invalid output exhausts repair (proves C04)
# ---------------------------------------------------------------------------


@pytest.mark.golden
def test_golden_d_schema_invalid_output_falls_back_after_exhausted_repair(
    tmp_path: Path,
) -> None:
    """Two consecutive invalid structured responses exhaust the one allowed
    repair attempt; classification falls back rather than trusting garbage,
    and the rest of the front end still completes."""
    engine = make_engine(tmp_path)
    model = QueueModel([invalid_response(), invalid_response(), formalisation_response([])])
    wave3 = compose_wave3(engine, model, Wave3Config())
    handle = wave3.create_run()
    task = TaskEnvelope(
        task_id=UUID(int=900),
        text=TASK_TEXT,
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(allow_network=True, allow_external_writes=True),
    )

    result = asyncio.run(
        wave3.execute_front_end(handle.run_id, task, allow_model=True, allow_adjudication=False)
    )
    assert result.classification_record is not None
    assert result.classification_record.fallback_used is True
    assert len(model.calls) == 3

    record = golden_record(engine, handle.run_id, fixture="D")
    assert_matches_golden("d_schema_invalid_repair_exhausted", record)


# ---------------------------------------------------------------------------
# E - resolved SUPPORTED_INFERENCE with an exact-text anchor
# ---------------------------------------------------------------------------


@pytest.mark.golden
def test_golden_e_resolved_supported_inference_with_exact_text_anchor(tmp_path: Path) -> None:
    """A SUPPORTED_INFERENCE item whose `support` is a real, valid, exact-text
    `SourceAnchor` into the task text resolves through the coordinator's own
    `formalise_problem`, and stays INFERENCE/PROVISIONAL (never silently
    promoted to FACT/SUPPORTED merely because support is present)."""
    engine = make_engine(tmp_path)
    model = QueueModel(
        [
            classification_response(),
            formalisation_response(
                [
                    {
                        "id": "inferred-1",
                        "kind": "UNKNOWN",
                        "description": "the option is judged safe",
                        "origin": "SUPPORTED_INFERENCE",
                        "basis": "quoting the task text directly",
                        "anchors": [_anchor()],
                        "support": [_anchor()],
                    }
                ]
            ),
        ]
    )
    wave3 = compose_wave3(engine, model, Wave3Config())
    handle = wave3.create_run()
    task = make_task()

    asyncio.run(wave3.classify_task_semantic(handle.run_id, task, allow_model=True))
    problem = asyncio.run(wave3.formalise_problem(handle.run_id, task, allow_model=True))
    assert problem is not None
    state = engine.inspect(handle.run_id)
    inference_nodes = [
        node
        for node in state.ledger.nodes
        if isinstance(node.content, dict) and node.content.get("id") == "inferred-1"
    ]
    assert len(inference_nodes) == 1
    assert inference_nodes[0].node_type.value == "INFERENCE"
    assert inference_nodes[0].epistemic_status.value == "PROVISIONAL"

    record = golden_record(
        engine,
        handle.run_id,
        fixture="E",
        inference_node_type=inference_nodes[0].node_type.value,
        inference_node_status=inference_nodes[0].epistemic_status.value,
    )
    assert_matches_golden("e_resolved_supported_inference", record)


# ---------------------------------------------------------------------------
# F - fictitious / dangling support is rejected
# ---------------------------------------------------------------------------


@pytest.mark.golden
def test_golden_f_fictitious_support_reference_is_rejected(tmp_path: Path) -> None:
    """A `support` reference naming an item that does not exist anywhere
    (not this proposal, not the run's real ledger) is rejected by the
    coordinator's own `formalise_problem` call -- fictitious support can
    never resolve. No partial M03 state is persisted by the rejected call."""
    engine = make_engine(tmp_path)
    model = QueueModel(
        [
            classification_response(),
            formalisation_response(
                [
                    {
                        "id": "fictitious-support-item",
                        "kind": "UNKNOWN",
                        "description": "supported by nothing real",
                        "origin": "SUPPORTED_INFERENCE",
                        "basis": "a made-up citation",
                        "support": [{"item_id": "totally-fictitious-item-nobody-emitted"}],
                    }
                ]
            ),
        ]
    )
    wave3 = compose_wave3(engine, model, Wave3Config())
    handle = wave3.create_run()
    task = make_task()

    asyncio.run(wave3.classify_task_semantic(handle.run_id, task, allow_model=True))
    with pytest.raises(UnresolvedSupportReference):
        asyncio.run(wave3.formalise_problem(handle.run_id, task, allow_model=True))
    after = engine.inspect(handle.run_id)
    # The real M03 semantic call itself (budget reservation/settlement) is a
    # legitimate, separate persisted transaction that lands regardless -- but
    # the rejected `ProblemFormaliser.canonical_events` batch itself (the
    # ProblemSpec/ledger/blocker state actually built from the fictitious
    # support) must persist NOTHING: `formalise_problem` raises before ever
    # calling `engine.append` for that batch.
    assert after.problem_spec is None
    assert after.problem_blockers == ()
    assert after.problem_contradictions == ()

    record = golden_record(engine, handle.run_id, fixture="F", rejected=True)
    assert_matches_golden("f_fictitious_support_rejected", record)


# ---------------------------------------------------------------------------
# G - UNKNOWN + assumption metadata preservation
# ---------------------------------------------------------------------------


@pytest.mark.golden
def test_golden_g_unknown_and_assumption_metadata_is_preserved(tmp_path: Path) -> None:
    """A non-material UNKNOWN and an ASSUMPTION item both survive into the
    persisted `ProblemSpec` with their typed metadata intact, and produce no
    blocker (non-material)."""
    engine = make_engine(tmp_path)
    model = QueueModel(
        [
            classification_response(),
            formalisation_response(
                [
                    {
                        "id": "unknown-1",
                        "kind": "UNKNOWN",
                        "description": "exact future demand is unknown",
                        "origin": "UNRESOLVED",
                        "attributes": {"material": False, "resolvable": True},
                    },
                    {
                        "id": "assumption-1",
                        "kind": "ASSUMPTION",
                        "description": "supply remains stable",
                        "origin": "WORKING_ASSUMPTION",
                        "basis": "standard planning assumption",
                        "policy_basis": "wave3-golden-fixture-g/1.0",
                        "attributes": {"why_needed": "bounds the decision", "scope": "this run"},
                    },
                ]
            ),
        ]
    )
    wave3 = compose_wave3(engine, model, Wave3Config())
    handle = wave3.create_run()
    task = make_task()

    result = asyncio.run(
        wave3.execute_front_end(handle.run_id, task, allow_model=True, allow_adjudication=False)
    )
    assert result.problem_spec is not None
    assert {item.id for item in result.problem_spec.unknowns} == {"unknown-1"}
    assert {item.id for item in result.problem_spec.assumption_items} == {"assumption-1"}
    assert result.blocked is False

    record = golden_record(engine, handle.run_id, fixture="G")
    assert_matches_golden("g_unknown_and_assumption_preserved", record)


# ---------------------------------------------------------------------------
# H - material contradiction + blocker propagation
# ---------------------------------------------------------------------------


@pytest.mark.golden
def test_golden_h_material_contradiction_and_blocker_propagation(tmp_path: Path) -> None:
    """A real `CONTRADICTS`-relation contradiction becomes a material M09
    blocker; the coordinator's own result reports it, M12 availability is
    independently recomputed as PARTIAL_BLOCKED (never trusted from a
    self-report), and `evaluate_stop` recomputes CONTINUE bound to the real
    blocker's ledger reference."""
    engine = make_engine(tmp_path)
    contradiction_items: list[dict[str, JsonValue]] = [
        {"id": "left", "kind": "UNKNOWN", "description": "A", "origin": "CONTRADICTED"},
        {"id": "right", "kind": "UNKNOWN", "description": "B", "origin": "CONTRADICTED"},
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
        # A material, genuinely non-resolvable UNKNOWN: the contradiction's own
        # blocker is always `resolvable=True` by construction (see
        # `ProblemFormaliser._contradiction_and_blocker_events`), so this
        # second, independent blocker is what actually makes the pipeline
        # `blocked` -- proving both material-contradiction detection AND
        # material-blocker propagation in the same fixture.
        {
            "id": "unresolved-1",
            "kind": "UNKNOWN",
            "description": "What is the counterparty's true intent?",
            "origin": "UNRESOLVED",
            "attributes": {"material": True},
        },
    ]
    model = QueueModel([classification_response(), formalisation_response(contradiction_items)])
    wave3 = compose_wave3(engine, model, Wave3Config())
    handle = wave3.create_run()
    task = make_task()

    result = asyncio.run(
        wave3.execute_front_end(handle.run_id, task, allow_model=True, allow_adjudication=False)
    )
    assert result.blocked is True
    assert len(result.problem_blockers) == 2
    state = engine.inspect(handle.run_id)
    assert len(state.problem_contradictions) == 1
    material_blockers = tuple(b for b in state.problem_blockers if not b.resolvable)
    assert len(material_blockers) == 1
    packet = state.context_packets[-1]
    assert packet.wave3_context is not None
    assert packet.wave3_context.availability.value == "PARTIAL_BLOCKED"

    # A material, non-resolvable blocker (the UNRESOLVED UNKNOWN) makes M13's
    # own disposition BLOCKED rather than CONTINUE -- a stronger, more
    # decisive signal than the resolvable-only contradiction blocker alone
    # would have produced.
    decision = wave3.evaluate_stop(handle.run_id, acceptance=AcceptanceStatus.PENDING)
    assert decision.disposition is StopDisposition.BLOCKED
    assert set(decision.ledger_trigger_refs) == {b.ledger_ref for b in state.problem_blockers}

    record = golden_record(
        engine,
        handle.run_id,
        fixture="H",
        stop_decision_disposition=decision.disposition.value,
    )
    assert_matches_golden("h_material_contradiction_blocker", record)


# ---------------------------------------------------------------------------
# I - deterministic M04 selection outside the tie band
# ---------------------------------------------------------------------------


@pytest.mark.golden
def test_golden_i_deterministic_m04_selection_outside_tie_band(tmp_path: Path) -> None:
    """With the default (narrow) `representation_tie_band`, a single scored
    OBJECTIVE item selects a representation deterministically, with no tie
    and no adjudication call ever attempted."""
    engine = make_engine(tmp_path)
    model = QueueModel(
        [
            classification_response(),
            formalisation_response(
                [
                    {
                        "id": "obj-1",
                        "kind": "OBJECTIVE",
                        "description": "minimize cost",
                        "origin": "EXPLICIT_INPUT",
                        "anchors": [_anchor()],
                        "attributes": {"direction": "MIN"},
                    }
                ]
            ),
        ]
    )
    wave3 = compose_wave3(engine, model, Wave3Config())
    handle = wave3.create_run()
    task = make_task()

    result = asyncio.run(
        wave3.execute_front_end(handle.run_id, task, allow_model=True, allow_adjudication=True)
    )
    assert result.representation_plan is not None
    assert result.representation_plan.tie_triggered is False
    assert result.representation_plan.adjudication_record_ref is None
    assert len(model.calls) == 2, "no adjudication call may be attempted outside a genuine tie"

    record = golden_record(engine, handle.run_id, fixture="I")
    assert_matches_golden("i_deterministic_m04_selection", record)


# ---------------------------------------------------------------------------
# J - M04 adjudication inside the tie band + fallback-builder identity
# ---------------------------------------------------------------------------


@pytest.mark.golden
def test_golden_j_adjudication_inside_tie_band_and_fallback_builder_identity(
    tmp_path: Path,
) -> None:
    """A wide `representation_tie_band` forces a genuine tie between two real
    (non-fallback) candidates; a real M04 adjudication semantic call
    resolves it through the coordinator's own `select_representation`, and
    the persisted plan carries a verified `adjudication_record_ref` bound to
    that real call."""
    engine = make_engine(tmp_path)
    config = Wave3Config(representation_tie_band=1.0, model_adjudication_enabled=True)
    tie_items: list[dict[str, JsonValue]] = [
        {
            "id": f"obj-{i}",
            "kind": "OBJECTIVE",
            "description": f"objective {i}",
            "origin": "EXPLICIT_INPUT",
            "anchors": [_anchor()],
            "attributes": {"direction": "MIN"},
        }
        for i in range(2)
    ]
    tie_items.append(
        {
            "id": "constraint-1",
            "kind": "CONSTRAINT",
            "description": "must stay within budget",
            "origin": "EXPLICIT_INPUT",
            "anchors": [_anchor()],
            "attributes": {"constraint_kind": "HARD"},
        }
    )
    model = QueueModel([classification_response(), formalisation_response(tie_items)])
    wave3 = compose_wave3(engine, model, config)
    handle = wave3.create_run()
    task = TaskEnvelope(
        task_id=UUID(int=900),
        text=TASK_TEXT,
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(allow_external_writes=True),
    )

    signature = asyncio.run(wave3.classify_task_semantic(handle.run_id, task, allow_model=True))
    problem = asyncio.run(wave3.formalise_problem(handle.run_id, task, allow_model=True))
    state = engine.inspect(handle.run_id)
    assert state.budget.plan is not None
    deterministic = wave3.components.representation_selector.select_bound(
        problem,
        signature,
        state.budget.plan,
        state.version,
        default_registry_v2(),
        wave3.components.policy.representation_selection,
    )
    candidates = tuple(view for view in deterministic.views if view.builder_available)
    assert deterministic.tie_triggered, "fixture must produce a genuine tie"
    assert len(candidates) >= 2, "fixture must surface >=2 candidates to adjudicate between"

    model.responses = iter(
        [
            StructuredModelResult(
                status=StructuredModelStatus.SUCCESS,
                adapter_id="fake",
                model_id="fixture",
                raw_response=json.dumps(
                    {"selected_kinds": [candidates[0].kind.value], "explanation": "fixture"}
                ).encode(),
                decoded={"selected_kinds": [candidates[0].kind.value], "explanation": "fixture"},
                usage=SemanticCallUsage(input_tokens=50, output_tokens=20),
            )
        ]
    )
    plan = asyncio.run(wave3.select_representation(handle.run_id, allow_adjudication=True))
    assert plan.tie_triggered is True
    assert plan.adjudication_record_ref is not None
    final_state = engine.inspect(handle.run_id)
    assert any(
        call.idempotency_key == plan.adjudication_record_ref for call in final_state.model_calls
    )
    wave3.compile_context(handle.run_id)

    record = golden_record(
        engine,
        handle.run_id,
        fixture="J",
        selected_candidate_kind=candidates[0].kind.value,
    )
    assert_matches_golden("j_adjudication_inside_tie_band", record)


# ---------------------------------------------------------------------------
# K - persisted M12 v2 context + zero-call replay
# ---------------------------------------------------------------------------


@pytest.mark.golden
def test_golden_k_persisted_m12_context_and_zero_call_replay(tmp_path: Path) -> None:
    """A completed run's typed Wave 3 context packet is durably persisted;
    re-attaching a fresh `Wave3Engine` (simulating a process restart) against
    the SAME storage and re-invoking `execute_front_end` makes zero further
    provider calls and returns the identical packet hash and state hash."""
    from fre.adapters.storage_sqlite import SQLiteStore

    engine = make_engine(tmp_path)

    class RefusingModel:
        async def generate(self, request: object) -> StructuredModelResult:  # pragma: no cover
            raise AssertionError("a completed run must never re-invoke the provider on replay")

    wave3 = compose_wave3(engine, RefusingModel(), Wave3Config())
    handle = wave3.create_run()
    task = make_task()

    first = asyncio.run(
        wave3.execute_front_end(handle.run_id, task, allow_model=False, allow_adjudication=False)
    )
    assert first.context_packet_hash is not None
    first_hash = engine.snapshot(handle.run_id)
    engine.store.close()

    reopened = SQLiteStore(tmp_path / "events.db")
    engine2 = type(engine)(reopened, engine.artifacts, engine.clock, engine.uuids)
    wave3_resumed = compose_wave3(engine2, RefusingModel(), Wave3Config())
    second = asyncio.run(
        wave3_resumed.execute_front_end(
            handle.run_id, task, allow_model=False, allow_adjudication=False
        )
    )
    assert second == first
    second_hash = engine2.snapshot(handle.run_id)
    assert second_hash == first_hash

    record = golden_record(engine2, handle.run_id, fixture="K", replay_state_hash=second_hash)
    assert_matches_golden("k_persisted_context_zero_call_replay", record)
    reopened.close()


# ---------------------------------------------------------------------------
# L - equal stop decision recomputed at fresh state, then finalized
# ---------------------------------------------------------------------------


@pytest.mark.golden
def test_golden_l_equal_stop_decision_recomputed_then_finalized(tmp_path: Path) -> None:
    """Two `evaluate_stop` calls separated only by an unrelated context
    re-compilation produce the identical disposition/reason set (recomputed
    from the inputs that actually matter -- blockers/budget -- not an
    incidental version counter), and the second decision is then genuinely
    finalized through `finalize_if_terminal`."""
    engine = make_engine(tmp_path)
    model = QueueModel([classification_response(), formalisation_response([])])
    wave3 = compose_wave3(engine, model, Wave3Config())
    handle = wave3.create_run()
    task = make_task()
    asyncio.run(wave3.execute_front_end(handle.run_id, task, allow_model=True))

    first = wave3.evaluate_stop(handle.run_id, acceptance=AcceptanceStatus.SATISFIED)
    wave3.compile_context(handle.run_id)
    second = wave3.evaluate_stop(handle.run_id, acceptance=AcceptanceStatus.SATISFIED)
    assert first.disposition == second.disposition == StopDisposition.COMPLETE
    assert first.reason_codes == second.reason_codes

    terminal_hash = wave3.finalize_if_terminal(handle.run_id, second)
    assert terminal_hash is not None
    state = engine.inspect(handle.run_id)
    assert state.terminal_context_packet_hash == terminal_hash

    record = golden_record(
        engine,
        handle.run_id,
        fixture="L",
        first_decision_reason_codes=sorted(code.value for code in first.reason_codes),
        second_decision_reason_codes=sorted(code.value for code in second.reason_codes),
        terminal_hash=terminal_hash,
    )
    assert_matches_golden("l_equal_stop_decision_finalized", record)
