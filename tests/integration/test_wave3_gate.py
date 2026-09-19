"""Combined Wave 3 coordinator gate.

C09 (F08) remediation: this file used to hand-wire M01/M02/M03/M04/M12
directly (constructing `TaskClassifier()`, `BudgetAllocator()`,
`ProblemFormaliser()`, `RepresentationSelector()`, `Wave3ContextCompiler()`,
`StopController()` inline, never going through `compose_wave3`). That manual
wiring is no longer the product path -- `fre.composition.Wave3Engine` is. The
tests below drive `Wave3Engine.execute_front_end` (and its individual
resumable steps) exclusively, matching how a real caller would use it, and
exercise the coordinator's decisive properties: the normal semantic path, the
zero-call fallback path, M04 tie-band adjudication, a provider-overage path,
a schema-invalid path, blocker propagation into M12 availability,
interruption/resume at each module boundary, a stable M13 decision recomputed
after unrelated state, zero-provider-invocation replay of a completed run,
tampered-state rejection, and duplicate-request idempotency.
"""

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
from fre.composition import compose_wave3
from fre.config import Wave3Config
from fre.domain.common import JsonValue, OutputContract, PermissionSet
from fre.domain.context import Wave3ContextAvailability
from fre.domain.representation import RepresentationArtifactV2
from fre.domain.semantic import (
    SemanticCallUsage,
    SemanticModelCallRecordV2,
    StructuredModelRequest,
    StructuredModelResult,
    StructuredModelStatus,
)
from fre.domain.stop import AcceptanceStatus, StopDisposition
from fre.domain.task import TaskEnvelope
from fre.engine import FrontierReasoningEngine
from fre.prompts.schemas import ClassificationOutput
from fre.runtime.events import RepresentationArtifactCompiledV2
from fre.semantic_runtime import SemanticExecution


class QueueModel:
    """A structured-model port whose responses are drained one call at a time."""

    def __init__(self, responses: list[StructuredModelResult]) -> None:
        self.responses = iter(responses)
        self.calls: list[str] = []

    async def generate(self, request: StructuredModelRequest) -> StructuredModelResult:
        self.calls.append(request.idempotency_key)
        return next(self.responses)


def uuids(count: int = 400) -> list[UUID]:
    return [UUID(int=index) for index in range(1, count + 1)]


def make_engine(tmp_path: Path) -> FrontierReasoningEngine:
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=i) for i in range(2000))
    return FrontierReasoningEngine(
        SQLiteStore(tmp_path / "events.db"),
        LocalArtifactStore(tmp_path / "artifacts"),
        clock,
        FakeUUIDFactory(uuids()),
    )


def make_task(task_id: int = 900, text: str = "Choose a safe option.") -> TaskEnvelope:
    return TaskEnvelope(
        task_id=UUID(int=task_id),
        text=text,
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(),
    )


def classification_response(*, usage: SemanticCallUsage | None = None) -> StructuredModelResult:
    usage = usage or SemanticCallUsage(input_tokens=100, output_tokens=50)
    anchor: JsonValue = {
        "source_kind": "TASK_TEXT",
        "source_ref": {"object_type": "TaskEnvelope", "object_id": str(UUID(int=900))},
        "selector": "/text",
        "char_start": 0,
        "char_end": len("Choose a safe option."),
    }
    valid: dict[str, JsonValue] = {
        "task_type": {
            "estimate": "DECISION",
            "confidence": 0.9,
            "anchors": [anchor],
            "rationale": "fixture",
        },
        "consequence": {
            "estimate": "LOW",
            "confidence": 0.9,
            "conservative_upper": "MEDIUM",
            "anchors": [anchor],
            "rationale": "fixture",
        },
        "reversibility": {
            "estimate": "HIGH",
            "confidence": 0.9,
            "conservative_upper": "HIGH",
            "anchors": [anchor],
            "rationale": "fixture",
        },
        "ambiguity": {
            "estimate": "MEDIUM",
            "confidence": 0.9,
            "conservative_upper": "MEDIUM",
            "anchors": [anchor],
            "rationale": "fixture",
        },
        "evidence_scarcity": {
            "estimate": "MEDIUM",
            "confidence": 0.9,
            "conservative_upper": "MEDIUM",
            "anchors": [anchor],
            "rationale": "fixture",
        },
        "search_space": {
            "estimate": "BOUNDED",
            "confidence": 0.9,
            "anchors": [anchor],
            "rationale": "fixture",
        },
        "horizon": {
            "estimate": "SHORT",
            "confidence": 0.9,
            "anchors": [anchor],
            "rationale": "fixture",
        },
    }
    return StructuredModelResult(
        status=StructuredModelStatus.SUCCESS,
        adapter_id="fake",
        model_id="fixture",
        raw_response=json.dumps(valid).encode(),
        decoded=valid,
        usage=usage,
    )


def unavailable_response() -> StructuredModelResult:
    return StructuredModelResult(
        status=StructuredModelStatus.UNAVAILABLE,
        adapter_id="fake",
        model_id="fixture",
        raw_response=b"",
        diagnostics=("no model configured",),
    )


def invalid_response() -> StructuredModelResult:
    return StructuredModelResult(
        status=StructuredModelStatus.INVALID_STRUCTURED_OUTPUT,
        adapter_id="fake",
        model_id="fixture",
        raw_response=b"{bad",
        diagnostics=("invalid",),
    )


def formalisation_response(
    items: "list[dict[str, JsonValue]] | None" = None,
) -> StructuredModelResult:
    payload: JsonValue = {"items": [dict(item) for item in items] if items is not None else []}
    return StructuredModelResult(
        status=StructuredModelStatus.SUCCESS,
        adapter_id="fake",
        model_id="fixture",
        raw_response=json.dumps(payload).encode(),
        decoded=payload,
        usage=SemanticCallUsage(input_tokens=100, output_tokens=50),
    )


@pytest.mark.integration
def test_coordinator_normal_semantic_path_and_replay(tmp_path: Path) -> None:
    """M01 and M03 both resolve real MODEL proposals; the full front end
    persists, replays across three independent paths with an identical hash,
    and a completed run's re-invocation makes zero further provider calls."""
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
                        "anchors": [
                            {
                                "source_kind": "TASK_TEXT",
                                "source_ref": {
                                    "object_type": "TaskEnvelope",
                                    "object_id": str(UUID(int=900)),
                                },
                                "selector": "/text",
                                "char_start": 0,
                                "char_end": len("Choose a safe option."),
                            }
                        ],
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
    assert result.task_signature is not None
    assert result.classification_record is not None
    assert result.classification_record.mode == "HYBRID"
    assert result.problem_spec is not None
    assert result.representation_plan is not None
    assert result.context_packet_hash is not None
    assert result.blocked is False
    assert len(model.calls) == 2

    expected_hash = engine.snapshot(handle.run_id)
    path_a = engine.replay(handle.run_id)
    path_c = engine.replay_from_snapshot(handle.run_id)
    engine.store.close()
    reopened = SQLiteStore(tmp_path / "events.db")
    path_b = FrontierReasoningEngine(reopened, engine.artifacts, engine.clock, engine.uuids).replay(
        handle.run_id
    )
    assert path_a.state_hash == path_b.state_hash == path_c.state_hash == expected_hash
    reopened.close()

    # Completed-run replay: re-invoking the coordinator on an already
    # completed run must invoke the provider zero additional times.
    calls_before = len(model.calls)
    engine2 = FrontierReasoningEngine(
        SQLiteStore(tmp_path / "events.db"), engine.artifacts, engine.clock, engine.uuids
    )
    wave3_resumed = compose_wave3(engine2, model, Wave3Config())
    resumed_result = asyncio.run(
        wave3_resumed.execute_front_end(
            handle.run_id, task, allow_model=True, allow_adjudication=True
        )
    )
    assert len(model.calls) == calls_before
    assert resumed_result == result
    engine2.store.close()


@pytest.mark.integration
def test_coordinator_zero_call_fallback_is_first_class(tmp_path: Path) -> None:
    """`allow_model=False` never touches the provider, and the run still
    completes end-to-end using purely deterministic fallbacks."""
    engine = make_engine(tmp_path)

    class RefusingModel:
        async def generate(self, request: StructuredModelRequest) -> StructuredModelResult:
            raise AssertionError("allow_model=False must never call the provider")

    wave3 = compose_wave3(engine, RefusingModel(), Wave3Config())
    handle = wave3.create_run()
    task = make_task()

    result = asyncio.run(
        wave3.execute_front_end(handle.run_id, task, allow_model=False, allow_adjudication=False)
    )
    assert result.classification_record is not None
    assert result.classification_record.mode == "DETERMINISTIC_FALLBACK"
    assert result.classification_record.fallback_used is True
    assert result.problem_spec is not None
    assert result.representation_plan is not None
    assert result.representation_plan.fallback_used in (True, False)
    assert result.context_packet_hash is not None


@pytest.mark.integration
def test_coordinator_m04_tie_band_adjudication(tmp_path: Path) -> None:
    """A wide `representation_tie_band` forces a genuine tie; a real M04
    adjudication semantic call resolves it, and the persisted plan carries a
    verified `adjudication_record_ref` bound to that real call.

    This is also the decisive fail-before/pass-after test for PR #24 finding
    H: `select_representation` now delegates to `RepresentationSelector.
    select_with_adjudication_bound` instead of duplicating its logic inline,
    and that shared helper used to bind its final plan to the
    `source_snapshot_version` captured BEFORE the real adjudication call's
    `await` -- stale the moment that call's own budget-reservation/
    settlement events advanced the run's version. With that bug present,
    this exact test fails with "representation plan source_snapshot_version
    drifted before persistence" (verified by temporarily reverting the fix);
    with the fix (re-deriving the plan from a fresh `runtime.engine.
    inspect(run_id)` after the call lands), it passes."""
    engine = make_engine(tmp_path)
    config = Wave3Config(representation_tie_band=1.0, model_adjudication_enabled=True)
    anchor: JsonValue = {
        "source_kind": "TASK_TEXT",
        "source_ref": {"object_type": "TaskEnvelope", "object_id": str(UUID(int=900))},
        "selector": "/text",
        "char_start": 0,
        "char_end": len("Choose a safe option."),
    }
    # Two OBJECTIVE items score PARETO_OBJECTIVE_MATRIX; one CONSTRAINT item
    # scores TYPED_CONSTRAINT_SET -- both REAL, non-fallback kinds, so the
    # forced tie (via `representation_tie_band=1.0`) is between two genuine
    # candidates. Deliberately avoids a tie that includes
    # TEXT_TABLE_FALLBACK here so this test can isolate the coordinator's own
    # adjudication wiring: `RepresentationSelector.select_bound` can
    # legitimately set `fallback_used=True` while `tie_triggered=True`
    # (TEXT_TABLE_FALLBACK selected as the second, tied candidate) -- this
    # WAS a latent, independently-reviewed gap in `RunReducer.apply`'s tie
    # recomputation (PR #24 finding A: it wrongly assumed
    # `fallback_used=True` always implies "no real tie" and rejected such an
    # otherwise legitimate plan). That coupling has since been removed (see
    # `fre.runtime.reducer`'s `RepresentationPlanSelectedV2` branch) and is
    # covered by its own decisive tests in `tests/unit/test_c09_remediation.
    # py` (including a coordinator-level `execute_front_end` exercise of
    # exactly this coincidence) -- this test is unaffected either way and is
    # left focused on the adjudication wiring alone.
    tie_items: list[dict[str, JsonValue]] = [
        {
            "id": f"obj-{i}",
            "kind": "OBJECTIVE",
            "description": f"objective {i}",
            "origin": "EXPLICIT_INPUT",
            "anchors": [anchor],
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
            "anchors": [anchor],
            "attributes": {"constraint_kind": "HARD"},
        }
    )
    model = QueueModel([classification_response(), formalisation_response(tie_items)])
    wave3 = compose_wave3(engine, model, config)
    handle = wave3.create_run()
    # `allow_external_writes=True` floors consequence to HIGH regardless of
    # the model's own (LOW) proposal, which floors the M02 tier to T3 --
    # necessary for `max_representation_views >= 2` so a tie can ever surface
    # more than one selectable view to adjudicate between.
    task = TaskEnvelope(
        task_id=UUID(int=900),
        text="Choose a safe option.",
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(allow_external_writes=True),
    )

    signature = asyncio.run(wave3.classify_task_semantic(handle.run_id, task, allow_model=True))
    problem = asyncio.run(wave3.formalise_problem(handle.run_id, task, allow_model=True))
    state = engine.inspect(handle.run_id)
    assert state.budget.plan is not None

    from fre.domain.representation_registry import default_registry_v2

    deterministic = wave3.components.representation_selector.select_bound(
        problem,
        signature,
        state.budget.plan,
        state.version,
        default_registry_v2(),
        wave3.components.policy.representation_selection,
    )
    candidates = tuple(view for view in deterministic.views if view.builder_available)
    assert deterministic.tie_triggered, "fixture must produce a genuine tie for this test"
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


@pytest.mark.integration
def test_coordinator_provider_overage_path(tmp_path: Path) -> None:
    """A provider that reports usage exceeding the reservation is charged at
    the reservation cap (never trusted at face value); classification falls
    through to no-proposal handling for that dimension set, and the pipeline
    still completes."""
    engine = make_engine(tmp_path)
    over_usage_response = classification_response(
        usage=SemanticCallUsage(input_tokens=10_000_000, output_tokens=10_000_000)
    )
    model = QueueModel([over_usage_response, formalisation_response([])])
    wave3 = compose_wave3(engine, model, Wave3Config())
    handle = wave3.create_run()
    task = make_task()

    signature = asyncio.run(wave3.classify_task_semantic(handle.run_id, task, allow_model=True))
    assert signature is not None
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
    assert (
        over_call.charged_usage.output_tokens
        <= wave3.components.policy.semantic_runtime.reserve_output_tokens
    )
    # An overage response is treated as no usable proposal by the semantic
    # runtime (`SemanticExecution.proposal is None` when `over`); the
    # classification is therefore the honest deterministic fallback.
    assert state.classification_record is not None
    assert state.classification_record.fallback_used is True


@pytest.mark.integration
def test_coordinator_schema_invalid_path_falls_back_after_exhausted_repair(
    tmp_path: Path,
) -> None:
    """An invalid structured response, followed by a repair attempt that is
    also invalid, exhausts the one allowed repair attempt; classification
    falls back to the deterministic path rather than trusting garbage."""
    engine = make_engine(tmp_path)
    model = QueueModel([invalid_response(), invalid_response()])
    wave3 = compose_wave3(engine, model, Wave3Config())
    handle = wave3.create_run()
    # A task with elevated execution permissions floors classification into a
    # tier whose bootstrap budget allows a second (repair) LLM call -- the
    # default all-LOW-floor task only bootstraps a one-call tier, which
    # cannot itself demonstrate the repair path (see PermissionSet below).
    task = TaskEnvelope(
        task_id=UUID(int=900),
        text="Choose a safe option.",
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(allow_network=True, allow_external_writes=True),
    )

    signature = asyncio.run(wave3.classify_task_semantic(handle.run_id, task, allow_model=True))
    assert signature is not None
    assert len(model.calls) == 2
    state = engine.inspect(handle.run_id)
    assert state.classification_record is not None
    assert state.classification_record.fallback_used is True


@pytest.mark.integration
def test_coordinator_schema_repair_success_path_reuses_identical_repaired_proposal(
    tmp_path: Path,
) -> None:
    """Independent-review remediation (PR #24 finding G.1): restores test
    coverage the `test_wave3_gate.py` rewrite dropped for the SUCCESSFUL
    schema-repair-then-use path (as opposed to the exhausted-repair fallback
    path already covered above): an invalid structured response followed by
    a valid repair response must be repaired, consumed for real into
    classification (not silently discarded in favour of the deterministic
    fallback), and a subsequent identical request must reuse that exact
    repaired proposal -- not merely report `reused=True`."""
    engine = make_engine(tmp_path)
    model = QueueModel([invalid_response(), classification_response()])
    wave3 = compose_wave3(engine, model, Wave3Config())
    handle = wave3.create_run()
    task = TaskEnvelope(
        task_id=UUID(int=900),
        text="Choose a safe option.",
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(allow_network=True, allow_external_writes=True),
    )

    signature = asyncio.run(wave3.classify_task_semantic(handle.run_id, task, allow_model=True))
    assert signature is not None
    assert len(model.calls) == 2
    state = engine.inspect(handle.run_id)
    assert state.classification_record is not None
    # The repaired proposal was genuinely consumed: classification is HYBRID
    # (a real proposal was used), never the deterministic fallback.
    assert state.classification_record.mode == "HYBRID"
    assert state.classification_record.fallback_used is False
    repaired_calls = [
        call
        for call in state.model_calls
        if call.module_id == "M01" and call.repair_parent_key is not None
    ]
    assert len(repaired_calls) == 1
    assert repaired_calls[0].status == StructuredModelStatus.SUCCESS

    # A subsequent, identical request (same canonical input/identity) must
    # resolve to the exact SAME repaired proposal via `_reuse` -- not just a
    # `reused=True` flag with a possibly-different value.
    async def repeat_request() -> SemanticExecution:
        return await wave3.components.semantic_runtime.execute(
            run_id=handle.run_id,
            module_id="M01",
            module_version="1.0",
            operation="classify",
            prompt_id="m01.classify",
            prompt_version="1.0",
            canonical_input=task.model_dump(mode="json"),
        )

    second = asyncio.run(repeat_request())
    assert second.reused is True
    assert second.repaired is True
    assert second.proposal is not None
    assert repaired_calls[0].proposal_artifact is not None
    expected_bytes = engine.artifacts.get(repaired_calls[0].proposal_artifact.sha256)
    expected_proposal = ClassificationOutput.model_validate_json(expected_bytes)
    assert second.proposal == expected_proposal
    assert len(model.calls) == 2  # no additional provider invocation


@pytest.mark.integration
def test_coordinator_contradiction_and_unknown_blocker_propagation(tmp_path: Path) -> None:
    """A material UNRESOLVED UNKNOWN becomes a real M09 blocker; the
    coordinator's result reports it, and M12 availability is independently
    recomputed (never trusted from a self-report) as PARTIAL_BLOCKED."""
    engine = make_engine(tmp_path)
    model = QueueModel(
        [
            classification_response(),
            formalisation_response(
                [
                    {
                        "id": "unresolved-1",
                        "kind": "UNKNOWN",
                        "description": "What is the counterparty's true intent?",
                        "origin": "UNRESOLVED",
                        "attributes": {"material": True},
                    }
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
    assert result.blocked is True
    assert len(result.problem_blockers) == 1
    state = engine.inspect(handle.run_id)
    packet = state.context_packets[-1]
    assert packet.wave3_context is not None
    assert packet.wave3_context.availability is Wave3ContextAvailability.PARTIAL_BLOCKED


@pytest.mark.integration
def test_coordinator_m03_contradiction_atomicity_stop_wiring_and_blocker_resolution(
    tmp_path: Path,
) -> None:
    """Independent-review remediation (PR #24 finding G.2): restores coverage
    dropped from the pre-coordinator `test_m03_contradiction_batch_is_atomic_
    and_replayable`, adapted to drive through the coordinator instead of
    hand-wiring modules: a real `CONTRADICTS`-relation contradiction
    (produced via a genuine M03 semantic proposal, not a bare UNKNOWN),
    `Wave3Engine.evaluate_stop` (the coordinator's own M13 wiring, deriving
    `blocker_required`/`blocker_resolvable`/`epistemic_trigger_refs` from the
    run's real `problem_blockers`) recomputing a CONTINUE decision against
    that real blocker, and a subsequent blocker-resolution M03 batch (built
    through the SAME composed `wave3.components.formaliser` the coordinator
    itself uses, not a disconnected `ProblemFormaliser()`) that genuinely
    clears `problem_blockers`/`problem_contradictions`."""
    from fre.domain.problem import ProblemBlocker as _ProblemBlocker
    from fre.domain.semantic import EpistemicOriginLabel
    from fre.prompts.schemas import ProblemFormalisationOutput

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
    ]
    model = QueueModel([classification_response(), formalisation_response(contradiction_items)])
    wave3 = compose_wave3(engine, model, Wave3Config())
    handle = wave3.create_run()
    task = make_task()

    asyncio.run(wave3.classify_task_semantic(handle.run_id, task, allow_model=True))
    problem = asyncio.run(wave3.formalise_problem(handle.run_id, task, allow_model=True))
    assert problem is not None

    # Atomicity: the batch that admitted the contradiction/blocker is a
    # single already-applied append; a hand-forged, self-inconsistent
    # variant of one of its own events (a blocker citing a ledger node that
    # was never actually admitted) is independently rejected by the reducer,
    # proving the real admitted batch was not merely accepted by convention.
    state = engine.inspect(handle.run_id)
    assert len(state.problem_contradictions) == 1
    assert len(state.problem_blockers) == 1
    blocker = state.problem_blockers[0]
    assert blocker.ledger_ref in {node.ref for node in state.ledger.nodes}
    bad_blocker = blocker.model_copy(
        update={"ledger_ref": blocker.ledger_ref.model_copy(update={"node_id": UUID(int=999_999)})}
    )
    from fre.runtime.events import ProblemBlockerRecorded

    with pytest.raises(ValueError):
        engine.append(
            handle.run_id,
            state.version,
            (
                engine.make_event(
                    handle.run_id, ProblemBlockerRecorded(blocker=bad_blocker), module_id="M03"
                ),
            ),
        )
    assert engine.inspect(handle.run_id).version == state.version

    # M13 wiring: `evaluate_stop` (the coordinator's OWN method, not a
    # hand-wired `StopController()`) derives its inputs from this run's real
    # `problem_blockers` and recomputes a CONTINUE decision bound to the
    # real blocker's ledger ref.
    decision = wave3.evaluate_stop(handle.run_id, acceptance=AcceptanceStatus.PENDING)
    assert decision.disposition is StopDisposition.CONTINUE
    assert decision.ledger_trigger_refs == (blocker.ledger_ref,)

    # Blocker resolution: a second M03 batch (built through the SAME
    # composed `wave3.components.formaliser`, using the run's current,
    # already-applied ledger state -- exactly the discipline
    # `formalise_problem` itself follows) genuinely clears the blocker and
    # contradiction, not merely appends unrelated events alongside them.
    resolved_proposal = ProblemFormalisationOutput.model_validate(
        {
            "items": (
                {
                    "id": "reviewed",
                    "kind": "ACCEPTANCE_CRITERION",
                    "description": "resolution is deterministically reviewed",
                    "origin": EpistemicOriginLabel.UNRESOLVED,
                    "attributes": {"verification_mode": "DETERMINISTIC", "required": True},
                },
            )
        }
    )
    current = engine.inspect(handle.run_id)
    known_ledger_refs = frozenset((node.node_id, node.revision) for node in current.ledger.nodes)
    resolved_events = wave3.components.formaliser.canonical_events(
        task,
        resolved_proposal,
        created_at=engine.clock.now(),
        uuids=engine.uuids,
        available_artifacts=frozenset(),
        known_ledger_refs=known_ledger_refs,
    )
    stored = tuple(
        engine.make_event(handle.run_id, payload, module_id="M03") for payload in resolved_events
    )
    engine.append(handle.run_id, current.version, stored)

    resolved_state = engine.inspect(handle.run_id)
    assert resolved_state.problem_blockers == ()
    assert resolved_state.problem_contradictions == ()
    assert isinstance(blocker, _ProblemBlocker)  # sanity: blocker was a real typed value


@pytest.mark.integration
def test_coordinator_interruption_and_resume_at_every_module_boundary(tmp_path: Path) -> None:
    """Simulate an interruption after each module's persisted step by driving
    the coordinator step-by-step across separate `Wave3Engine` instances
    (fresh process re-attachment), and confirm the final state exactly
    matches a single uninterrupted run -- with no duplicate bootstrap budget,
    no duplicate model calls, and no duplicate event batches."""
    interrupted_engine = make_engine(tmp_path)
    model = QueueModel([classification_response(), formalisation_response([])])
    wave3 = compose_wave3(interrupted_engine, model, Wave3Config())
    handle = wave3.create_run()
    task = make_task()

    # Boundary 1: interrupt after the M01/M02 bootstrap budget, before final
    # classification.
    wave3._bootstrap_budget(handle.run_id, task, None, None)
    state = interrupted_engine.inspect(handle.run_id)
    assert state.budget.plan is not None
    assert state.task_signature is None
    bootstrap_events = len(interrupted_engine.store.load(handle.run_id))

    # Resume: a fresh coordinator instance over the SAME store must not
    # re-bootstrap the budget.
    resumed_engine = FrontierReasoningEngine(
        interrupted_engine.store,
        interrupted_engine.artifacts,
        interrupted_engine.clock,
        interrupted_engine.uuids,
    )
    wave3_resumed = compose_wave3(resumed_engine, model, Wave3Config())
    signature = asyncio.run(
        wave3_resumed.classify_task_semantic(handle.run_id, task, allow_model=True)
    )
    assert signature is not None
    assert len(model.calls) == 1
    state_after_classify = resumed_engine.inspect(handle.run_id)
    assert state_after_classify.task_signature is not None
    # No second BudgetAllocated: only new events are the classification tail.
    assert not any(
        event.event_type == "BudgetAllocated@1.0"
        for event in resumed_engine.store.load(handle.run_id)[bootstrap_events:]
    )

    # Boundary 2: interrupt after M03, resume into M04 + M12.
    problem = asyncio.run(wave3_resumed.formalise_problem(handle.run_id, task, allow_model=True))
    assert problem is not None
    plan = asyncio.run(wave3_resumed.select_representation(handle.run_id))
    assert plan is not None
    packet_hash = wave3_resumed.compile_context(handle.run_id)

    # Compare against a fresh, uninterrupted run over an independent store.
    control_root = tmp_path / "control"
    control_root.mkdir()
    control_engine = make_engine(control_root)
    control_model = QueueModel([classification_response(), formalisation_response([])])
    control_wave3 = compose_wave3(control_engine, control_model, Wave3Config())
    control_handle = control_wave3.create_run()
    control_result = asyncio.run(
        control_wave3.execute_front_end(control_handle.run_id, task, allow_model=True)
    )
    resumed_state = resumed_engine.inspect(handle.run_id)
    assert resumed_state.task_signature == control_result.task_signature
    assert resumed_state.problem_spec == control_result.problem_spec
    assert packet_hash is not None


@pytest.mark.integration
def test_coordinator_m13_decision_recomputed_equal_after_unrelated_state(tmp_path: Path) -> None:
    """Two `evaluate_stop` calls separated only by an unrelated context
    re-compilation must produce the identical disposition/reason set, proving
    the decision is recomputed from the inputs that actually matter to it
    (blockers/budget), not from an incidental version counter."""
    engine = make_engine(tmp_path)
    model = QueueModel([classification_response(), formalisation_response([])])
    wave3 = compose_wave3(engine, model, Wave3Config())
    handle = wave3.create_run()
    task = make_task()
    asyncio.run(wave3.execute_front_end(handle.run_id, task, allow_model=True))

    first = wave3.evaluate_stop(handle.run_id, acceptance=AcceptanceStatus.SATISFIED)

    # Unrelated state change: re-run compile_context (a no-op append given
    # idempotency) plus a fresh state read; the run's real referenced state
    # (blockers/budget) has not changed.
    wave3.compile_context(handle.run_id)
    second = wave3.evaluate_stop(handle.run_id, acceptance=AcceptanceStatus.SATISFIED)

    assert first.disposition == second.disposition == StopDisposition.COMPLETE
    assert first.reason_codes == second.reason_codes


@pytest.mark.integration
def test_compile_context_recompiles_after_representation_selection_lands(
    tmp_path: Path,
) -> None:
    """Independent-review remediation (PR #24 finding B): `compile_context`
    called BEFORE `select_representation`, then again AFTER, must produce a
    NEW packet reflecting the representation plan -- not silently reuse the
    stale, pre-representation packet. Before the fix, the dedup match key
    only compared `problem_spec_ref`/`ledger_root`/`budget_plan_ref`, none of
    which `select_representation` touches, so the second call wrongly
    returned the first (representation-less) packet's hash unchanged."""
    engine = make_engine(tmp_path)
    model = QueueModel([classification_response(), formalisation_response([])])
    wave3 = compose_wave3(engine, model, Wave3Config())
    handle = wave3.create_run()
    task = make_task()
    asyncio.run(wave3.classify_task_semantic(handle.run_id, task, allow_model=True))
    asyncio.run(wave3.formalise_problem(handle.run_id, task, allow_model=True))

    first_packet_hash = wave3.compile_context(handle.run_id)
    state_before = engine.inspect(handle.run_id)
    first_packet = state_before.context_packets[-1]
    assert first_packet.packet_hash == first_packet_hash
    # Sanity: no representation plan yet, so this packet cannot reflect one.
    assert state_before.representation_plan_v2 is None

    asyncio.run(wave3.select_representation(handle.run_id))

    second_packet_hash = wave3.compile_context(handle.run_id)
    assert second_packet_hash != first_packet_hash, (
        "compile_context must not reuse a packet compiled before representation selection landed"
    )
    state_after = engine.inspect(handle.run_id)
    second_packet = state_after.context_packets[-1]
    assert second_packet.packet_hash == second_packet_hash
    assert state_after.representation_plan_v2 is not None

    # Idempotency is still intact: a THIRD call with nothing new must reuse
    # the second (now current) packet, not compile a fourth.
    third_packet_hash = wave3.compile_context(handle.run_id)
    assert third_packet_hash == second_packet_hash
    packets_after_third = engine.inspect(handle.run_id).context_packets
    assert len(packets_after_third) == len(state_after.context_packets)


@pytest.mark.integration
def test_formalise_problem_uses_fresh_ledger_state_after_the_semantic_await(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Independent-review remediation (PR #24 finding C): `known_ledger_refs`
    must be rebuilt from a FRESH `self.engine.inspect(run_id)` call taken
    AFTER the M03 semantic call's `await` completes, never from the `state`
    captured before it -- a concurrent ledger mutation landing during that
    await window must be visible. Simulated here by having the fake model
    provider itself append a `LedgerNodeAdded` event to the SAME run as a
    side effect of answering the request (modelling some other concurrent
    writer), then asserting the coordinator's own call into
    `ProblemFormaliser.canonical_events` receives `known_ledger_refs`
    including that concurrently-added node -- which is only possible if it
    was read after, not before, the await."""
    from datetime import UTC
    from datetime import datetime as _datetime

    from fre.domain.ledger import EpistemicStatus, LedgerNodeType
    from fre.modules.m03_formaliser import ProblemFormaliser
    from fre.modules.m09_ledger import make_node
    from fre.runtime.events import LedgerNodeAdded

    engine = make_engine(tmp_path)
    concurrent_node_id = UUID(int=555_555)

    class LedgerMutatingModel:
        """Answers exactly like `QueueModel`, but appends a real
        `LedgerNodeAdded` event to the run as a side effect of the FIRST
        call it answers -- simulating a concurrent writer mutating the
        ledger while the (real) semantic call this test drives is in
        flight, before this test's own code regains control."""

        def __init__(self, responses: list[StructuredModelResult]) -> None:
            self.responses = iter(responses)
            self.calls: list[str] = []
            self._mutated = False

        async def generate(self, request: StructuredModelRequest) -> StructuredModelResult:
            self.calls.append(request.idempotency_key)
            # Gate the mutation to the M03 formalisation request specifically
            # (identified by its bound output schema) -- classification (M01)
            # is called first, and it must NOT trigger this, or the "concurrent"
            # mutation would land before `formalise_problem` even starts,
            # proving nothing about its post-await re-inspection.
            if not self._mutated and request.output_schema_id == "m03.problem-formalisation-output":
                self._mutated = True
                node = make_node(
                    node_id=concurrent_node_id,
                    revision=1,
                    node_type=LedgerNodeType.FACT,
                    content={"fact": "concurrently observed"},
                    status=EpistemicStatus.SUPPORTED,
                    created_at=_datetime(2026, 1, 1, tzinfo=UTC),
                    action_id=UUID(int=777_777),
                    module_id="CONCURRENT_WRITER",
                )
                mutation_state = engine.inspect(handle.run_id)
                engine.append(
                    handle.run_id,
                    mutation_state.version,
                    (
                        engine.make_event(
                            handle.run_id,
                            LedgerNodeAdded(node=node),
                            module_id="CONCURRENT_WRITER",
                        ),
                    ),
                )
            return next(self.responses)

    model = LedgerMutatingModel([classification_response(), formalisation_response([])])
    wave3 = compose_wave3(engine, model, Wave3Config())
    handle = wave3.create_run()
    task = make_task()
    asyncio.run(wave3.classify_task_semantic(handle.run_id, task, allow_model=True))

    captured_refs: dict[str, frozenset[tuple[UUID, int]] | None] = {"known_ledger_refs": None}
    original_canonical_events = ProblemFormaliser.canonical_events

    def spying_canonical_events(
        self: ProblemFormaliser, *args: object, **kwargs: object
    ) -> tuple[object, ...]:
        captured_refs["known_ledger_refs"] = cast(
            "frozenset[tuple[UUID, int]] | None", kwargs.get("known_ledger_refs")
        )
        return original_canonical_events(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ProblemFormaliser, "canonical_events", spying_canonical_events)
    problem = asyncio.run(wave3.formalise_problem(handle.run_id, task, allow_model=True))

    assert problem is not None
    known_refs = captured_refs["known_ledger_refs"]
    assert known_refs is not None
    assert (concurrent_node_id, 1) in known_refs, (
        "known_ledger_refs must reflect the ledger node the provider concurrently "
        "added during the semantic call's await window, not a pre-await snapshot"
    )


@pytest.mark.integration
def test_select_representation_resume_after_crash_never_repeats_the_adjudication_call(
    tmp_path: Path,
) -> None:
    """Independent-review remediation (PR #24 finding E): if a real M04
    adjudication semantic call succeeds and its budget settlement lands, but
    the process then crashes before the final `RepresentationPlanSelectedV2`
    append, resuming `select_representation` on a fresh coordinator instance
    over the same store must NOT re-issue a duplicate real adjudication
    call -- `SemanticModelRuntime.execute`'s own identity-keyed `_reuse`
    mechanism (the same one `classify_task`'s bootstrap/finalize split
    relies on being safe to resume across) must serve the already-settled
    call. Simulated by driving the semantic call directly to the point where
    its own events have landed, then handing the coordinator a fresh engine
    instance (module a fresh process re-attachment) to complete the pending
    `select_representation`."""
    interrupted_engine = make_engine(tmp_path)
    config = Wave3Config(representation_tie_band=1.0, model_adjudication_enabled=True)
    anchor: JsonValue = {
        "source_kind": "TASK_TEXT",
        "source_ref": {"object_type": "TaskEnvelope", "object_id": str(UUID(int=900))},
        "selector": "/text",
        "char_start": 0,
        "char_end": len("Choose a safe option."),
    }
    tie_items: list[dict[str, JsonValue]] = [
        {
            "id": f"obj-{i}",
            "kind": "OBJECTIVE",
            "description": f"objective {i}",
            "origin": "EXPLICIT_INPUT",
            "anchors": [anchor],
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
            "anchors": [anchor],
            "attributes": {"constraint_kind": "HARD"},
        }
    )
    model = QueueModel([classification_response(), formalisation_response(tie_items)])
    wave3 = compose_wave3(interrupted_engine, model, config)
    handle = wave3.create_run()
    task = TaskEnvelope(
        task_id=UUID(int=900),
        text="Choose a safe option.",
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(allow_external_writes=True),
    )

    signature = asyncio.run(wave3.classify_task_semantic(handle.run_id, task, allow_model=True))
    problem = asyncio.run(wave3.formalise_problem(handle.run_id, task, allow_model=True))
    state = interrupted_engine.inspect(handle.run_id)
    assert state.budget.plan is not None

    from fre.domain.representation_registry import default_registry_v2

    deterministic = wave3.components.representation_selector.select_bound(
        problem,
        signature,
        state.budget.plan,
        state.version,
        default_registry_v2(),
        wave3.components.policy.representation_selection,
    )
    candidates = tuple(view for view in deterministic.views if view.builder_available)
    assert deterministic.tie_triggered
    assert len(candidates) >= 2

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

    # Drive the adjudication semantic call directly to completion (its own
    # BudgetReserved/settlement/ModelCallRecordedV2 events land for real),
    # WITHOUT ever letting `select_representation` append its own final
    # `RepresentationPlanSelectedV2` batch -- simulating a crash exactly
    # between those two points.
    execution = asyncio.run(
        wave3.components.semantic_runtime.execute(
            run_id=handle.run_id,
            module_id="M04",
            module_version="1.0",
            operation="adjudicate",
            prompt_id="m04.adjudicate",
            prompt_version="1.0",
            canonical_input={
                "problem_spec_hash": deterministic.problem_spec_hash,
                "candidates": [
                    {
                        "kind": view.kind.value,
                        "compatibility_score": view.compatibility_score,
                        "score_components": [
                            item.model_dump(mode="json") for item in view.score_components
                        ],
                        "purpose": view.purpose,
                        "limitations": list(view.limitations),
                    }
                    for view in candidates
                ],
                "problem_summary": {
                    "objectives": [item.model_dump(mode="json") for item in problem.objectives],
                    "constraints": [item.model_dump(mode="json") for item in problem.constraints],
                    "unknowns": [item.model_dump(mode="json") for item in problem.unknowns],
                    "relations": [item.model_dump(mode="json") for item in problem.relations],
                },
                "view_limit": state.budget.plan.search.max_representation_views,
            },
        )
    )
    assert execution.record is not None
    assert execution.reused is False
    calls_after_simulated_crash = len(model.calls)
    assert calls_after_simulated_crash == 3  # classification + formalisation + this adjudication
    state_after_crash = interrupted_engine.inspect(handle.run_id)
    assert state_after_crash.representation_plan_v2 is None  # never persisted -- the "crash"

    # Resume: a FRESH coordinator instance over the SAME store completes
    # `select_representation` from here.
    resumed_engine = FrontierReasoningEngine(
        interrupted_engine.store,
        interrupted_engine.artifacts,
        interrupted_engine.clock,
        interrupted_engine.uuids,
    )
    wave3_resumed = compose_wave3(resumed_engine, model, config)
    plan = asyncio.run(wave3_resumed.select_representation(handle.run_id, allow_adjudication=True))

    assert plan.tie_triggered is True
    assert plan.adjudication_record_ref == execution.record.idempotency_key
    assert len(model.calls) == calls_after_simulated_crash, (
        "resuming select_representation must not repeat the real adjudication "
        "model call -- it must reuse the already-settled one"
    )
    final_state = resumed_engine.inspect(handle.run_id)
    assert final_state.representation_plan_v2 == plan


@pytest.mark.integration
def test_coordinator_tampered_representation_artifact_is_rejected(tmp_path: Path) -> None:
    """A representation artifact whose declared `content_hash` does not match
    its actually-stored bytes is rejected by the reducer before persistence,
    even though it is self-consistent in every other field."""
    engine = make_engine(tmp_path)
    model = QueueModel([classification_response(), formalisation_response([])])
    wave3 = compose_wave3(engine, model, Wave3Config())
    handle = wave3.create_run()
    task = make_task()
    asyncio.run(wave3.classify_task_semantic(handle.run_id, task, allow_model=True))
    asyncio.run(wave3.formalise_problem(handle.run_id, task, allow_model=True))
    asyncio.run(wave3.select_representation(handle.run_id))

    state = engine.inspect(handle.run_id)
    real_artifact = state.representation_artifacts_v2[0]
    tampered: RepresentationArtifactV2 = real_artifact.model_copy(
        update={"content_hash": "0" * 64, "determinism_hash": "0" * 64}
    )
    event = engine.make_event(
        handle.run_id, RepresentationArtifactCompiledV2(artifact=tampered), module_id="M04"
    )
    with pytest.raises(ValueError) as excinfo:
        engine.append(handle.run_id, state.version, (event,))
    assert "content_hash" in str(excinfo.value) or "bound plan" in str(excinfo.value)


@pytest.mark.integration
def test_coordinator_duplicate_request_idempotency(tmp_path: Path) -> None:
    """Two identical semantic-call requests (same canonical input, same
    identity) issued for the same run resolve to exactly one real provider
    invocation; the second is served from the persisted, replayed record."""
    engine = make_engine(tmp_path)
    model = QueueModel([classification_response()])
    wave3 = compose_wave3(engine, model, Wave3Config())
    handle = wave3.create_run()
    task = make_task()
    wave3._bootstrap_budget(handle.run_id, task, None, None)

    async def duplicate_calls() -> tuple[bool, bool]:
        first = await wave3.components.semantic_runtime.execute(
            run_id=handle.run_id,
            module_id="M01",
            module_version="1.0",
            operation="classify",
            prompt_id="m01.classify",
            prompt_version="1.0",
            canonical_input=task.model_dump(mode="json"),
        )
        second = await wave3.components.semantic_runtime.execute(
            run_id=handle.run_id,
            module_id="M01",
            module_version="1.0",
            operation="classify",
            prompt_id="m01.classify",
            prompt_version="1.0",
            canonical_input=task.model_dump(mode="json"),
        )
        return first.reused, second.reused

    first_reused, second_reused = asyncio.run(duplicate_calls())
    assert first_reused is False
    assert second_reused is True
    assert len(model.calls) == 1
