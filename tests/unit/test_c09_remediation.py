"""Decisive C09 remediation tests: independent-review findings on PR #24
(the C09 coordinator sequencing PR), findings A, D, and I.

Each test below is a fail-before/pass-after decisive test for one specific,
independently-reviewed defect in `fre.runtime.reducer.RunReducer.apply`
(finding A) or `fre.composition.Wave3Engine`'s precondition guards
(finding D), or restores dropped legacy-path coverage that is still reachable
in production (finding I). See the PR body for the full remediation summary
(A-J).
"""

from uuid import UUID

import pytest

from fre.composition import Wave3Engine, compose_wave3
from fre.config import Wave3Config
from fre.domain.budget import DeploymentLimits
from fre.domain.common import OutputContract, PermissionSet
from fre.domain.problem import DecisionVariable, ProblemSpec
from fre.domain.representation_registry import default_registry_v2
from fre.domain.semantic import StructuredModelRequest, StructuredModelResult
from fre.domain.stop import AcceptanceStatus, StopDisposition
from fre.domain.task import TaskEnvelope
from fre.engine import FrontierReasoningEngine
from fre.modules.m01_classifier import TaskClassifier
from fre.modules.m02_budget import BudgetAllocator, default_tier_policy
from fre.modules.m04_representation import RepresentationSelectionPolicy, RepresentationSelector

# ---------------------------------------------------------------------------
# Finding A: `RunReducer.apply`'s recomputation of `tie_triggered` wrongly
# coupled it to `fallback_used`, assuming `fallback_used=True` always implies
# "no real tie". `RepresentationSelector.select_bound` can legitimately set
# BOTH flags `True` simultaneously (a genuine tie between a real candidate
# and `TEXT_TABLE_FALLBACK` itself). Before the fix, the reducer wrongly
# rejected such an otherwise entirely legitimate, self-consistent plan.
# ---------------------------------------------------------------------------


class UnusedModel:
    async def generate(self, request: StructuredModelRequest) -> StructuredModelResult:
        raise AssertionError(f"unexpected model call: {request.idempotency_key}")


def _tie_and_fallback_problem() -> ProblemSpec:
    """A ProblemSpec whose real `default_registry_v2()` scoring produces
    DECISION_TABLE at 0.45 and TEXT_TABLE_FALLBACK (always present, weight
    0.20) as the top two compatible candidates -- a genuine 0.25 gap. With
    `representation_tie_band >= 0.25`, this is a real tie whose second-place
    (tied) candidate is the typed fallback itself, exactly the coincidence
    `RepresentationSelector.select_bound` handles by setting BOTH
    `tie_triggered=True` and `fallback_used=True`."""
    return ProblemSpec(
        output_contract=OutputContract(form="TEXT"),
        decision_variables=(DecisionVariable(id="dv-1", name="option", domain=["a", "b"]),),
    )


def _wide_permission_envelope() -> TaskEnvelope:
    """Floors consequence to HIGH (via `allow_external_writes=True`), which
    floors the M02 tier to T3 -- necessary for `max_representation_views >=
    2` so the tied second candidate can ever be admitted as an AUXILIARY
    view (and therefore ever surface `fallback_used=True` from that branch
    of `select_bound`, not merely the "zero compatible candidates" branch)."""
    return TaskEnvelope(
        task_id=UUID(int=900),
        text="Choose an option.",
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(allow_external_writes=True),
    )


def test_select_bound_produces_the_genuine_tie_and_fallback_coincidence() -> None:
    """Fixture sanity: confirms the constructed scenario actually produces
    `tie_triggered=True` AND `fallback_used=True` simultaneously from the
    real, unmodified `RepresentationSelector.select_bound` -- so the
    decisive tests below are exercising a real production scenario, not a
    hand-forged plan that could never actually be selected."""
    signature, _ = TaskClassifier().classify(_wide_permission_envelope(), None)
    budget, _ = BudgetAllocator().allocate(signature, default_tier_policy(), DeploymentLimits())
    assert budget.search.max_representation_views >= 2
    policy = RepresentationSelectionPolicy(tie_band=0.3)
    selector = RepresentationSelector(policy)
    problem = _tie_and_fallback_problem()
    plan = selector.select_bound(problem, signature, budget, 0, default_registry_v2(), policy)
    assert plan.tie_triggered is True
    assert plan.fallback_used is True


@pytest.mark.unit
def test_reducer_accepts_plan_with_genuine_tie_and_fallback_used_together(
    engine: FrontierReasoningEngine,
) -> None:
    """Decisive test for finding A: `engine.append` (which previews every
    event through `RunReducer.apply`) must ACCEPT a bound v2 representation
    plan whose real, independently-recomputed scores show a genuine tie
    (`tie_triggered=True`) even though the tied candidate happens to be the
    typed fallback (`fallback_used=True`). Before the fix, the reducer's
    recomputation coupled the two flags (`if plan.fallback_used:
    recomputed_tie = False`) and rejected this exact, legitimate plan with
    "tie_triggered does not match an independent recomputation" -- this test
    fails on that code and passes with the fix that recomputes
    `tie_triggered` purely from `candidate_scores`/`tie_band`."""
    from fre.domain.problem import ProblemSpec as _ProblemSpec
    from fre.runtime.events import ProblemFormalised, RepresentationPlanSelectedV2

    handle = engine.create_run({"c09": "finding-a"})
    problem = _tie_and_fallback_problem()
    signature, _ = TaskClassifier().classify(_wide_permission_envelope(), None)
    budget, _ = BudgetAllocator().allocate(signature, default_tier_policy(), DeploymentLimits())
    policy = RepresentationSelectionPolicy(tie_band=0.3)
    selector = RepresentationSelector(policy)

    state = engine.inspect(handle.run_id)
    engine.append(
        handle.run_id,
        state.version,
        (engine.make_event(handle.run_id, ProblemFormalised(problem=problem), module_id="M03"),),
    )
    state = engine.inspect(handle.run_id)
    assert isinstance(state.problem_spec, _ProblemSpec)

    plan = selector.select_bound(
        problem, signature, budget, state.version, default_registry_v2(), policy
    )
    assert plan.tie_triggered is True and plan.fallback_used is True

    engine.append(
        handle.run_id,
        state.version,
        (
            engine.make_event(
                handle.run_id, RepresentationPlanSelectedV2(plan=plan), module_id="M04"
            ),
        ),
    )
    final_state = engine.inspect(handle.run_id)
    assert final_state.representation_plan_v2 == plan


@pytest.mark.integration
def test_coordinator_execute_front_end_succeeds_through_tie_fallback_coincidence(
    engine: FrontierReasoningEngine,
) -> None:
    """Coordinator-level decisive test for finding A: with
    `representation_tie_band` configured wide enough to reach the
    fallback-tie coincidence (see `SCORE_WEIGHTS` in
    `fre.domain.representation_registry`: DECISION_TABLE=0.45 vs.
    TEXT_TABLE_FALLBACK=0.20 is a 0.25 gap), `execute_front_end` must
    complete successfully end-to-end for a `ProblemSpec` that triggers it,
    never raise from the reducer's own tie-recomputation check."""
    import asyncio

    config = Wave3Config(representation_tie_band=0.3, model_adjudication_enabled=False)
    wave3 = compose_wave3(engine, UnusedModel(), config)
    handle = wave3.create_run()
    task = _wide_permission_envelope()

    signature = asyncio.run(wave3.classify_task_semantic(handle.run_id, task, allow_model=False))
    assert signature is not None
    problem = _tie_and_fallback_problem()
    from fre.runtime.events import ProblemFormalised

    state = engine.inspect(handle.run_id)
    engine.append(
        handle.run_id,
        state.version,
        (engine.make_event(handle.run_id, ProblemFormalised(problem=problem), module_id="M03"),),
    )

    plan = asyncio.run(wave3.select_representation(handle.run_id, allow_adjudication=False))
    assert plan.tie_triggered is True
    assert plan.fallback_used is True

    packet_hash = wave3.compile_context(handle.run_id)
    assert packet_hash is not None


# ---------------------------------------------------------------------------
# EU-28 (Wave 3 post-freeze cleanup, C07 item #28): `RunReducer.apply`'s
# `RepresentationPlanSelectedV2` branch only independently re-verifies a
# bound v2 plan's `input_hash` when `state.task_signature`/
# `state.budget.plan` are BOTH already populated (see the "Finding G"
# comment on that branch) -- a real, documented, and (per this pass)
# deliberately NOT reducer-fixed verification gap. The practical exposure is
# closed one layer up: `select_representation` (below) is the only place in
# `composition.py` that ever constructs and appends a
# `RepresentationPlanSelectedV2` event, and it hard-guards on both fields
# being present before it will do so. This test pins that guard.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_select_representation_rejects_run_missing_classification_or_budget(
    engine: FrontierReasoningEngine,
) -> None:
    """Composition-level half of EU-28's pin: `select_representation` must
    always reject -- before ever calling into selection or appending a
    `RepresentationPlanSelectedV2` event -- a run whose `task_signature`
    and/or `budget.plan` are absent, regardless of whether a `ProblemSpec`
    has already been formalised. This is the guard that makes the reducer's
    own `input_hash` verification gap (see `reducer.py`'s
    `RepresentationPlanSelectedV2` branch, "Finding G") unreachable through
    the real orchestrator; see the companion reducer-level pin test in
    `tests/unit/test_c07_remediation.py` for the other half."""
    import asyncio

    from fre.runtime.events import ProblemFormalised

    wave3 = compose_wave3(engine, UnusedModel(), Wave3Config())
    handle = wave3.create_run()
    problem = ProblemSpec(output_contract=OutputContract(form="TEXT"))
    state = engine.inspect(handle.run_id)
    engine.append(
        handle.run_id,
        state.version,
        (engine.make_event(handle.run_id, ProblemFormalised(problem=problem), module_id="M03"),),
    )
    state = engine.inspect(handle.run_id)
    assert state.problem_spec is not None
    assert state.task_signature is None
    assert state.budget.plan is None

    with pytest.raises(ValueError, match="authoritative classification and an allocated budget"):
        asyncio.run(wave3.select_representation(handle.run_id, allow_adjudication=False))


# ---------------------------------------------------------------------------
# Finding D: `compile_context`/`evaluate_stop`/`finalize_if_terminal` must
# raise a clear `ValueError` naming the missing prerequisite when called out
# of sequence, rather than silently proceeding on incomplete state (or
# surfacing only as an unrelated-looking downstream exception).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compile_context_rejects_run_with_no_formalised_problem_spec(
    engine: FrontierReasoningEngine,
) -> None:
    wave3 = compose_wave3(engine, UnusedModel(), Wave3Config())
    handle = wave3.create_run()
    with pytest.raises(ValueError, match="formalised ProblemSpec"):
        wave3.compile_context(handle.run_id)


@pytest.mark.unit
def test_evaluate_stop_rejects_run_with_no_budget_allocated(
    engine: FrontierReasoningEngine,
) -> None:
    wave3 = compose_wave3(engine, UnusedModel(), Wave3Config())
    handle = wave3.create_run()
    with pytest.raises(ValueError, match="M01/M02 classification and budget allocation"):
        wave3.evaluate_stop(handle.run_id)


@pytest.mark.unit
def test_evaluate_stop_rejects_run_with_budget_but_no_formalised_problem(
    engine: FrontierReasoningEngine,
) -> None:
    import asyncio

    wave3 = compose_wave3(engine, UnusedModel(), Wave3Config())
    handle = wave3.create_run()
    task = TaskEnvelope(
        task_id=UUID(int=1),
        text="task",
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(),
    )
    asyncio.run(wave3.classify_task_semantic(handle.run_id, task, allow_model=False))
    with pytest.raises(ValueError, match="M03 problem formalisation"):
        wave3.evaluate_stop(handle.run_id)


def _completed_run(
    engine: FrontierReasoningEngine, wave3_config: Wave3Config | None = None
) -> tuple[Wave3Engine, UUID]:
    import asyncio

    wave3 = compose_wave3(engine, UnusedModel(), wave3_config or Wave3Config())
    handle = wave3.create_run()
    task = TaskEnvelope(
        task_id=UUID(int=1),
        text="task",
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(),
    )
    asyncio.run(
        wave3.execute_front_end(handle.run_id, task, allow_model=False, allow_adjudication=False)
    )
    return wave3, handle.run_id


@pytest.mark.unit
def test_finalize_if_terminal_rejects_a_decision_never_recorded_via_evaluate_stop(
    engine: FrontierReasoningEngine,
) -> None:
    """A real, valid, terminal `StopDecision` obtained on one run must be
    rejected when applied to an UNRELATED run that never recorded it via
    `evaluate_stop` -- proving the guard checks a real, matching recorded
    decision on THIS run, not merely that a well-formed decision object was
    handed in."""
    wave3, run_id = _completed_run(engine)
    decision = wave3.evaluate_stop(run_id, acceptance=AcceptanceStatus.SATISFIED)
    assert decision.disposition is not StopDisposition.CONTINUE

    other_wave3, other_run_id = _completed_run(engine)
    with pytest.raises(ValueError, match=r"already been.*recorded via evaluate_stop"):
        other_wave3.finalize_if_terminal(other_run_id, decision)


@pytest.mark.unit
def test_finalize_if_terminal_is_a_pure_noop_for_a_continue_disposition(
    engine: FrontierReasoningEngine,
) -> None:
    """CONTINUE dispositions never need a recorded decision at all -- the
    guard above must not regress this pre-existing, unconditional shortcut."""
    wave3, run_id = _completed_run(engine)
    continue_decision = wave3.evaluate_stop(run_id, acceptance=AcceptanceStatus.PENDING)
    assert continue_decision.disposition is StopDisposition.CONTINUE
    assert wave3.finalize_if_terminal(run_id, continue_decision) is None


# ---------------------------------------------------------------------------
# Finding I: the pre-coordinator test suite carried two decisive scenarios
# for the still-live LEGACY v1 `RepresentationPlanSelected`/`RepresentationPlan`
# path (`fre.modules.m04_representation.RepresentationSelector.select`/
# `.build`, unchanged by C07 -- see that module's own docstring: "All new
# selection goes through `default_registry_v2` instead", meaning v1 remains
# a live, decode-compatible surface, not a removed one) that the
# `test_wave3_gate.py` rewrite for the coordinator dropped entirely:
#
#   1. Reformalising a run (`ProblemFormalised` with a NEW `ProblemSpec`)
#      invalidates an already-bound v1 `RepresentationPlanSelected` (the
#      reducer clears `representation_plan` back to `None`), and a stale
#      plan bound to the OLD `ProblemSpec` is rejected outright if applied
#      after the reformalisation.
#   2. A legacy pre-schema `RepresentationPlanSelected` (a plan with
#      `problem_spec_hash=None`, from before that field existed) still
#      replays/applies without the binding check firing at all -- the
#      reducer's guard is deliberately gated on `payload.plan.
#      problem_spec_hash is not None`.
#
# `fre.runtime.reducer.RunReducer.apply`'s `RepresentationPlanSelected`
# branch (as opposed to the NEW `RepresentationPlanSelectedV2` branch the
# coordinator exclusively uses) is UNCHANGED and still fully reachable by any
# caller emitting v1 events directly through `engine.append` -- nothing in
# the C09 coordinator removes or deprecates the v1 surface. Both scenarios
# are therefore still real and relevant, not obsolete, and are restored here.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reformalisation_invalidates_bound_legacy_v1_representation_plan(
    engine: FrontierReasoningEngine,
) -> None:
    from fre.domain.budget import BudgetPlan
    from fre.modules.m03_formaliser import ProblemFormaliser
    from fre.runtime.events import BudgetAllocated, ProblemFormalised, RepresentationPlanSelected

    handle = engine.create_run({"c09-finding-i": "reformalise"})
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
    assert isinstance(budget, BudgetPlan)
    first = ProblemFormaliser().formalise(task, None)
    second = first.model_copy(update={"output_contract": OutputContract(form="JSON")})
    plan = RepresentationSelector().select(first, signature, budget)

    events = tuple(
        engine.make_event(handle.run_id, item, module_id="wave3")
        for item in (
            BudgetAllocated(
                plan=budget, policy_version=budget.policy_version, policy_hash=policy_hash
            ),
            ProblemFormalised(problem=first),
            RepresentationPlanSelected(plan=plan),
        )
    )
    engine.append(handle.run_id, handle.version, events)
    state = engine.inspect(handle.run_id)
    assert state.representation_plan == plan

    # A plan bound to the OLD ProblemSpec is rejected once forged to claim a
    # mismatched hash.
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

    # Reformalising with a DIFFERENT ProblemSpec invalidates the already-
    # bound plan back to None.
    engine.append(
        handle.run_id,
        state.version,
        (engine.make_event(handle.run_id, ProblemFormalised(problem=second), module_id="M03"),),
    )
    reformalised = engine.inspect(handle.run_id)
    assert reformalised.representation_plan is None
    assert reformalised.problem_spec == second

    # The old plan (bound to the FIRST ProblemSpec) is rejected if applied now.
    with pytest.raises(ValueError, match="does not bind current ProblemSpec"):
        engine.append(
            handle.run_id,
            reformalised.version,
            (
                engine.make_event(
                    handle.run_id, RepresentationPlanSelected(plan=plan), module_id="M04"
                ),
            ),
        )


@pytest.mark.unit
def test_legacy_pre_schema_representation_plan_replays_without_problem_spec_hash(
    engine: FrontierReasoningEngine,
) -> None:
    """A `RepresentationPlanSelected` from before `problem_spec_hash` existed
    on `RepresentationPlan` (`problem_spec_hash=None`) must still apply/replay
    without the C07-era binding check firing -- decode compatibility for
    already-persisted Wave 3 history, deliberately distinct from a plan that
    HAS the field populated but disagrees with it (which IS rejected, per the
    test above)."""
    from fre.modules.m03_formaliser import ProblemFormaliser
    from fre.runtime.events import ProblemFormalised, RepresentationPlanSelected

    handle = engine.create_run({"c09-finding-i": "legacy-pre-schema"})
    task = TaskEnvelope(
        task_id=UUID(int=902),
        text="Choose safely, legacy shape.",
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(),
    )
    signature, _record = TaskClassifier().classify(task, None)
    budget, _policy_hash = BudgetAllocator().allocate(
        signature, default_tier_policy(), DeploymentLimits()
    )
    problem = ProblemFormaliser().formalise(task, None)
    plan = RepresentationSelector().select(problem, signature, budget)
    legacy_plan = plan.model_copy(update={"problem_spec_hash": None})
    assert legacy_plan.problem_spec_hash is None

    events = tuple(
        engine.make_event(handle.run_id, item, module_id="wave3")
        for item in (
            ProblemFormalised(problem=problem),
            RepresentationPlanSelected(plan=legacy_plan),
        )
    )
    engine.append(handle.run_id, handle.version, events)
    state = engine.inspect(handle.run_id)
    assert state.representation_plan == legacy_plan

    replayed = engine.replay(handle.run_id)
    assert replayed.representation_plan == legacy_plan
    assert replayed.state_hash == state.state_hash
