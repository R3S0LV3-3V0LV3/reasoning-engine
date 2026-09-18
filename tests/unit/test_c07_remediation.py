"""Decisive C07 remediation tests: M04 selection-contract reproducibility and
artifact admission (defects F10, F11).

Each test below maps directly onto a bullet in the C07 validation strategy in
FRE_WAVE3_C01_C10_EXECUTION_COMPLETION_AND_VALIDATION_REGISTER.md.
"""

import hashlib
from collections.abc import Callable
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

from fre.domain.budget import BudgetPlan, DeploymentLimits
from fre.domain.common import (
    ArtifactRef,
    FrozenModel,
    OutputContract,
    PermissionSet,
    canonical_hash,
)
from fre.domain.problem import ConstraintSpec, DecisionVariable, ProblemSpec, UnknownSpec
from fre.domain.representation import (
    RepresentationArtifactV2,
    RepresentationKind,
    RepresentationPlanV2,
)
from fre.domain.task import TaskEnvelope, TaskSignature
from fre.engine import FrontierReasoningEngine
from fre.modules.m01_classifier import TaskClassifier
from fre.modules.m02_budget import BudgetAllocator, default_tier_policy
from fre.modules.m04_representation import (
    ORPHAN_KIND_REASONS,
    RepresentationDefinition,
    RepresentationSelectionPolicy,
    RepresentationSelector,
    default_registry,
    default_registry_v2,
)
from fre.prompts.schemas import RepresentationAdjudicationOutput
from fre.runtime.events import (
    ArtifactRegistered,
    ProblemFormalised,
    RepresentationArtifactCompiledV2,
    RepresentationPlanSelectedV2,
)
from fre.runtime.reducer import RunReducer, RunState


def envelope(text: str = "Compare causal and dependency structure.") -> TaskEnvelope:
    return TaskEnvelope(
        task_id=UUID(int=1),
        text=text,
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(),
    )


def context(
    problem: ProblemSpec | None = None,
) -> tuple[ProblemSpec, TaskSignature, BudgetPlan]:
    problem = problem or ProblemSpec(output_contract=OutputContract(form="TEXT"))
    signature, _ = TaskClassifier().classify(envelope(), None)
    budget, _ = BudgetAllocator().allocate(signature, default_tier_policy(), DeploymentLimits())
    return problem, signature, budget


def bytes_writer(store: dict[str, bytes]) -> Callable[[bytes], ArtifactRef]:
    def _write(content: bytes) -> ArtifactRef:
        digest = hashlib.sha256(content).hexdigest()
        store[digest] = content
        return ArtifactRef(artifact_id=UUID(int=len(store) + 1), sha256=digest)

    return _write


# ---------------------------------------------------------------------------
# F11: registry incompleteness -- every RepresentationKind is either a real
# candidate or an explicitly de-registered orphan.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_v2_registry_declares_every_representation_kind() -> None:
    registry = default_registry_v2()
    assert {definition.kind for definition in registry} == set(RepresentationKind)
    orphans = {
        definition.kind: definition for definition in registry if not definition.builder_available
    }
    assert set(orphans) == set(ORPHAN_KIND_REASONS)
    for kind, definition in orphans.items():
        assert definition.unavailable_reason == ORPHAN_KIND_REASONS[kind]
        assert definition.score_weight == 0.0


@pytest.mark.unit
def test_legacy_v1_registry_is_unchanged_and_stays_twelve_kinds() -> None:
    """The legacy registry/version strings are pinned by `composition.py` and
    the m12 golden fixtures; C07 must not perturb them."""
    registry = default_registry()
    assert len(registry) == 12
    assert RepresentationKind.KNOWLEDGE_GRAPH not in {item.kind for item in registry}


# ---------------------------------------------------------------------------
# F10: fallback-attribution bug. Both a construction-time (type-level) proof
# and a `build()`/`build_bound()` regression test.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fallback_records_actual_builder_not_requested_builder() -> None:
    """The F10 headline case. Before the fix, `build()` always set
    `builder_id`/`builder_version` from the REQUESTED view, even when a
    fallback ran -- this assertion (`builder_id` resolves to the fallback's
    OWN registered identity, distinct from what was requested) is false
    against that pre-fix code and true after it."""
    problem, signature, budget = context()
    problem = problem.model_copy(update={"unknowns": (UnknownSpec(id="u", description="demand"),)})
    unavailable = RepresentationDefinition(
        kind=RepresentationKind.SCENARIO_TREE,
        builder_id="scenario-builder-that-never-runs",
        builder_version="9.9",
        priority=0,
        purpose="unknown projection",
        limitations=("does not generate scenarios",),
        builder_available=False,
    )
    selector = RepresentationSelector()
    plan = selector.select(problem, signature, budget, (unavailable,))
    view = plan.views[0]
    assert view.builder_ref == "scenario-builder-that-never-runs"
    artifact = selector.build(view, problem, 3)
    assert artifact.actual_kind is RepresentationKind.TEXT_TABLE_FALLBACK
    assert artifact.requested_kind is RepresentationKind.SCENARIO_TREE
    # The exact F10 exploit shape: builder identity must NOT equal what was
    # requested once a fallback actually ran.
    assert artifact.requested_builder_id == "scenario-builder-that-never-runs"
    assert artifact.requested_builder_version == "9.9"
    assert artifact.builder_id != "scenario-builder-that-never-runs"
    assert artifact.builder_id == "m04.text_table_fallback"
    assert artifact.builder_version == "1.0"


@pytest.mark.unit
def test_no_fallback_keeps_requested_and_actual_builder_identical() -> None:
    problem, signature, budget = context(
        ProblemSpec(
            output_contract=OutputContract(form="TEXT"),
            constraints=(
                ConstraintSpec(
                    id="c1", description="x", kind="HARD", verification_mode="DETERMINISTIC"
                ),
            ),
        )
    )
    selector = RepresentationSelector()
    plan = selector.select(problem, signature, budget)
    artifact = selector.build(plan.views[0], problem, 1)
    assert artifact.fallback_reason is None
    assert artifact.builder_id == artifact.requested_builder_id
    assert artifact.builder_version == artifact.requested_builder_version


@pytest.mark.unit
def test_v2_artifact_cannot_be_constructed_with_forged_attribution() -> None:
    """Type-level proof (Objective 4): unlike v1's plain dataclass-shaped
    fields, `RepresentationArtifactV2` makes the F10 exploit shape
    structurally unconstructable, not merely discouraged by a wrapper's
    calling convention."""
    base: dict[str, Any] = dict(
        plan_hash="a" * 64,
        registry_hash="b" * 64,
        registry_version="wave3-m04-registry/2.0",
        selection_policy_version="wave3-m04/2.0",
        problem_spec_hash="c" * 64,
        source_snapshot_version=1,
        requested_kind=RepresentationKind.SCENARIO_TREE,
        actual_kind=RepresentationKind.TEXT_TABLE_FALLBACK,
        content_hash="d" * 64,
        determinism_hash="e" * 64,
        physical_artifact_ref=ArtifactRef(artifact_id=UUID(int=1), sha256="d" * 64),
        fallback_reason="requested builder unavailable or unsupported",
    )
    with pytest.raises(ValidationError, match="must differ"):
        RepresentationArtifactV2(
            **base,
            requested_builder_id="same-builder",
            requested_builder_version="1.0",
            actual_builder_id="same-builder",
            actual_builder_version="1.0",
        )
    consistent = {**base, "actual_kind": RepresentationKind.SCENARIO_TREE}
    del consistent["fallback_reason"]
    with pytest.raises(ValidationError, match="fallback_reason is only valid"):
        RepresentationArtifactV2(
            **consistent,
            requested_builder_id="a",
            requested_builder_version="1.0",
            actual_builder_id="a",
            actual_builder_version="1.0",
            fallback_reason="unexpected",
        )


# ---------------------------------------------------------------------------
# Tie-band boundary semantics preserved exactly (0.05), at all three call
# sites sharing the one `tie_triggered` decision.
# ---------------------------------------------------------------------------


def _two_candidate_registry(gap: float) -> tuple[RepresentationDefinition, ...]:
    high = RepresentationDefinition(
        kind=RepresentationKind.TYPED_CONSTRAINT_SET,
        builder_id="m04.typed_constraint_set",
        priority=0,
        purpose="p",
        limitations=(),
        score_weight=0.70,
    )
    low = RepresentationDefinition(
        kind=RepresentationKind.DECISION_TABLE,
        builder_id="m04.decision_table",
        priority=1,
        purpose="p",
        limitations=(),
        score_weight=round(0.70 - gap, 6),
    )
    return (high, low)


def _tied_problem() -> ProblemSpec:
    return ProblemSpec(
        output_contract=OutputContract(form="TEXT"),
        constraints=(
            ConstraintSpec(
                id="c1", description="x", kind="HARD", verification_mode="DETERMINISTIC"
            ),
        ),
        decision_variables=(DecisionVariable(id="d1", name="option"),),
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("gap", "expect_tied"),
    [(0.03, True), (0.05, True), (0.07, False)],
    ids=["below-band", "exactly-at-band", "above-band"],
)
def test_tie_band_boundary_is_consistent_at_all_call_sites(gap: float, expect_tied: bool) -> None:
    problem = _tied_problem()
    signature, _ = TaskClassifier().classify(envelope(), None)
    registry = _two_candidate_registry(gap)
    selector = RepresentationSelector()

    # Call site 1: admit-second-view (limit >= 2).
    budget_two, _ = BudgetAllocator().allocate(signature, default_tier_policy(), DeploymentLimits())
    budget_two = budget_two.model_copy(
        update={"search": budget_two.search.model_copy(update={"max_representation_views": 2})}
    )
    plan_two = selector.select_bound(problem, signature, budget_two, 1, registry)
    assert plan_two.tie_triggered is expect_tied
    assert (len(plan_two.views) == 2) is expect_tied

    # Call site 2: omitted_reasons entry (limit == 1).
    budget_one = budget_two.model_copy(
        update={"search": budget_two.search.model_copy(update={"max_representation_views": 1})}
    )
    plan_one = selector.select_bound(problem, signature, budget_one, 1, registry)
    assert plan_one.tie_triggered is expect_tied
    assert bool(plan_one.omitted_reasons) is expect_tied

    # Call site 3: `select_with_adjudication_bound`'s ambiguity gate reads the
    # very same `tie_triggered` flag computed once above.
    policy = RepresentationSelectionPolicy(model_adjudication_enabled=True)
    plan_gated = selector.select_bound(problem, signature, budget_two, 1, registry, policy)
    assert plan_gated.tie_triggered is expect_tied


# ---------------------------------------------------------------------------
# Reproducibility: selection is deterministic for identical registry+inputs.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_selection_is_deterministic_for_identical_registry_and_inputs() -> None:
    problem, signature, budget = context(_tied_problem())
    selector = RepresentationSelector()
    first = selector.select_bound(problem, signature, budget, 4)
    second = selector.select_bound(problem, signature, budget, 4)
    assert first == second
    assert first.plan_hash == second.plan_hash
    assert first.registry_hash == canonical_hash(default_registry_v2())


# ---------------------------------------------------------------------------
# Objective 5: new APIs emit bound v2 only; legacy v1 stays decode-only.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_new_selection_api_never_returns_a_v1_plan() -> None:
    problem, signature, budget = context()
    plan = RepresentationSelector().select_bound(problem, signature, budget, 0)
    assert isinstance(plan, RepresentationPlanV2)


@pytest.mark.unit
def test_v2_plan_event_rejects_a_v1_shaped_plan() -> None:
    problem, signature, budget = context()
    legacy_plan = RepresentationSelector().select(problem, signature, budget)
    with pytest.raises(ValidationError):
        RepresentationPlanSelectedV2.model_validate({"plan": legacy_plan.model_dump(mode="json")})


# ---------------------------------------------------------------------------
# Reducer-enforced binding: plan/artifact identity, bytes/hash provenance,
# and adjudication linkage. Uses the real `engine` fixture (SQLite + a real
# content-addressed artifact store) so bytes genuinely exist or don't.
# ---------------------------------------------------------------------------


def _bound_plan_and_artifact(
    engine: FrontierReasoningEngine, run_id: UUID, problem: ProblemSpec, store: dict[str, bytes]
) -> tuple[RepresentationPlanV2, RepresentationArtifactV2, ArtifactRef]:
    signature, _ = TaskClassifier().classify(envelope(), None)
    budget, _ = BudgetAllocator().allocate(signature, default_tier_policy(), DeploymentLimits())
    selector = RepresentationSelector()
    state = engine.inspect(run_id)
    plan = selector.select_bound(problem, signature, budget, state.version)
    artifact = selector.build_bound(plan.views[0], problem, plan, bytes_writer(store))
    return plan, artifact, artifact.physical_artifact_ref


def _append(
    engine: FrontierReasoningEngine, run_id: UUID, payload: FrozenModel, module_id: str = "wave3"
) -> None:
    state = engine.inspect(run_id)
    engine.append(run_id, state.version, (engine.make_event(run_id, payload, module_id=module_id),))


@pytest.mark.integration
def test_bound_plan_and_artifact_round_trip_through_the_real_engine(
    engine: FrontierReasoningEngine,
) -> None:
    handle = engine.create_run({"c07": "happy-path"})
    problem = ProblemSpec(output_contract=OutputContract(form="TEXT"))
    store: dict[str, bytes] = {}
    _append(engine, handle.run_id, ProblemFormalised(problem=problem), module_id="M03")
    plan, artifact, ref = _bound_plan_and_artifact(engine, handle.run_id, problem, store)
    real_bytes = store[ref.sha256]
    engine.store_artifact(real_bytes, media_type="application/json")
    _append(engine, handle.run_id, RepresentationPlanSelectedV2(plan=plan), module_id="M04")
    _append(
        engine,
        handle.run_id,
        ArtifactRegistered(artifact=ref, media_type="application/json", byte_size=len(real_bytes)),
    )
    _append(
        engine, handle.run_id, RepresentationArtifactCompiledV2(artifact=artifact), module_id="M04"
    )
    state = engine.inspect(handle.run_id)
    assert state.representation_plan_v2 == plan
    assert state.representation_artifacts_v2 == (artifact,)
    # v1 fields remain untouched by v2 traffic.
    assert state.representation_plan is None
    assert state.representation_artifacts == ()
    engine.snapshot(handle.run_id)
    replayed_from_snapshot = engine.replay_from_snapshot(handle.run_id)
    assert replayed_from_snapshot.state_hash == engine.replay(handle.run_id).state_hash


@pytest.mark.integration
def test_reject_missing_artifact_bytes(engine: FrontierReasoningEngine) -> None:
    handle = engine.create_run({"c07": "missing-bytes"})
    problem = ProblemSpec(output_contract=OutputContract(form="TEXT"))
    store: dict[str, bytes] = {}
    _append(engine, handle.run_id, ProblemFormalised(problem=problem), module_id="M03")
    plan, artifact, ref = _bound_plan_and_artifact(engine, handle.run_id, problem, store)
    _append(engine, handle.run_id, RepresentationPlanSelectedV2(plan=plan), module_id="M04")
    # Register the sha256 (so the "is it registered" check passes) but never
    # actually write real bytes for it to the artifact store.
    _append(
        engine,
        handle.run_id,
        ArtifactRegistered(
            artifact=ref, media_type="application/json", byte_size=len(store[ref.sha256])
        ),
    )
    with pytest.raises(ValueError, match=r"bytes could not be read|bytes are missing"):
        _append(
            engine,
            handle.run_id,
            RepresentationArtifactCompiledV2(artifact=artifact),
            module_id="M04",
        )


@pytest.mark.integration
def test_reject_forged_content_hash_trusting_computed_bytes_instead(
    engine: FrontierReasoningEngine,
) -> None:
    """Objective 4's decisive test: a caller-asserted `content_hash` that
    disagrees with the actual stored bytes is rejected outright -- the
    recomputed hash is trusted, never the claim."""
    handle = engine.create_run({"c07": "forged-hash"})
    problem = ProblemSpec(output_contract=OutputContract(form="TEXT"))
    store: dict[str, bytes] = {}
    _append(engine, handle.run_id, ProblemFormalised(problem=problem), module_id="M03")
    plan, artifact, ref = _bound_plan_and_artifact(engine, handle.run_id, problem, store)
    real_bytes = store[ref.sha256]
    engine.store_artifact(real_bytes, media_type="application/json")
    _append(engine, handle.run_id, RepresentationPlanSelectedV2(plan=plan), module_id="M04")
    _append(
        engine,
        handle.run_id,
        ArtifactRegistered(artifact=ref, media_type="application/json", byte_size=len(real_bytes)),
    )
    forged = artifact.model_copy(
        update={
            "content_hash": "0" * 64,
            "determinism_hash": canonical_hash(
                {
                    "registry_hash": artifact.registry_hash,
                    "actual_kind": artifact.actual_kind,
                    "problem_spec_hash": artifact.problem_spec_hash,
                    "actual_builder_id": artifact.actual_builder_id,
                    "actual_builder_version": artifact.actual_builder_version,
                    "content_hash": "0" * 64,
                }
            ),
        }
    )
    with pytest.raises(ValueError, match="content_hash does not match"):
        _append(
            engine,
            handle.run_id,
            RepresentationArtifactCompiledV2(artifact=forged),
            module_id="M04",
        )


@pytest.mark.unit
def test_reducer_rejects_forged_artifact_binding_when_called_directly() -> None:
    """Bypasses `engine.append`/`FrontierReasoningEngine` entirely and calls
    `RunReducer.apply` directly with a hand-built forged binding, per the
    C04/C05/C06 lesson that a wrapper's calling convention alone is not proof
    of an invariant -- the reducer itself must refuse it."""
    from datetime import UTC, datetime

    from fre.runtime.events import StoredEvent, event_wire_identity

    problem = ProblemSpec(output_contract=OutputContract(form="TEXT"))
    signature, _ = TaskClassifier().classify(envelope(), None)
    budget, _ = BudgetAllocator().allocate(signature, default_tier_policy(), DeploymentLimits())
    selector = RepresentationSelector()

    reducer = RunReducer(artifact_reader=lambda sha: b"real bytes on disk")
    run_id = UUID(int=42)
    state = reducer.initial(run_id)

    def apply_payload(state: RunState, payload: FrozenModel, module_id: str = "test") -> RunState:
        event_type, schema_version = event_wire_identity(payload)
        event = StoredEvent(
            event_id=UUID(int=state.version + 100),
            run_id=run_id,
            event_type=event_type,
            action_id=UUID(int=state.version + 200),
            module_id=module_id,
            schema_version=schema_version,
            module_version="1.0",
            input_hash="0" * 64,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            payload=payload.model_dump(mode="json"),
            sequence=state.version + 1,
        )
        return reducer.apply(state, event)

    from fre.runtime.events import RunCreated

    state = apply_payload(state, RunCreated(config_hash="c"))
    state = apply_payload(state, ProblemFormalised(problem=problem))
    plan = selector.select_bound(problem, signature, budget, state.version)
    state = apply_payload(state, RepresentationPlanSelectedV2(plan=plan))

    real_sha = hashlib.sha256(b"real bytes on disk").hexdigest()
    forged_artifact = RepresentationArtifactV2(
        plan_hash=plan.plan_hash,
        registry_hash=plan.registry_hash,
        registry_version=plan.registry_version,
        selection_policy_version=plan.selection_policy_version,
        problem_spec_hash=plan.problem_spec_hash,
        source_snapshot_version=plan.source_snapshot_version,
        requested_kind=plan.views[0].kind,
        actual_kind=plan.views[0].kind,
        requested_builder_id=plan.views[0].builder_ref,
        requested_builder_version=plan.views[0].builder_version,
        actual_builder_id=plan.views[0].builder_ref,
        actual_builder_version=plan.views[0].builder_version,
        physical_artifact_ref=ArtifactRef(artifact_id=UUID(int=1), sha256=real_sha),
        # Forged: does not match sha256(b"real bytes on disk").
        content_hash="f" * 64,
        determinism_hash="e" * 64,
    )
    state_before_registration = apply_payload(
        state,
        ArtifactRegistered(
            artifact=ArtifactRef(artifact_id=UUID(int=1), sha256=real_sha),
            media_type="application/json",
            byte_size=len(b"real bytes on disk"),
        ),
    )
    with pytest.raises(ValueError, match="content_hash does not match"):
        apply_payload(
            state_before_registration,
            RepresentationArtifactCompiledV2(artifact=forged_artifact),
        )


# ---------------------------------------------------------------------------
# Mismatched plan/problem/source revision bindings are rejected.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_reject_mismatched_source_snapshot_version(engine: FrontierReasoningEngine) -> None:
    handle = engine.create_run({"c07": "stale-revision"})
    problem = ProblemSpec(output_contract=OutputContract(form="TEXT"))
    store: dict[str, bytes] = {}
    _append(engine, handle.run_id, ProblemFormalised(problem=problem), module_id="M03")
    plan, _artifact, ref = _bound_plan_and_artifact(engine, handle.run_id, problem, store)
    engine.store_artifact(store[ref.sha256], media_type="application/json")
    stale_plan = plan.model_copy(
        update={"source_snapshot_version": plan.source_snapshot_version + 5}
    )
    stale_plan = stale_plan.model_copy(
        update={
            "plan_hash": canonical_hash(stale_plan.model_dump(mode="json", exclude={"plan_hash"}))
        }
    )
    with pytest.raises(ValueError, match="does not match the run state"):
        _append(
            engine, handle.run_id, RepresentationPlanSelectedV2(plan=stale_plan), module_id="M04"
        )


@pytest.mark.integration
def test_reject_artifact_referencing_unapplied_plan(engine: FrontierReasoningEngine) -> None:
    handle = engine.create_run({"c07": "dangling-plan-ref"})
    problem = ProblemSpec(output_contract=OutputContract(form="TEXT"))
    store: dict[str, bytes] = {}
    _append(engine, handle.run_id, ProblemFormalised(problem=problem), module_id="M03")
    _plan, artifact, ref = _bound_plan_and_artifact(engine, handle.run_id, problem, store)
    engine.store_artifact(store[ref.sha256], media_type="application/json")
    # Note: no RepresentationPlanSelectedV2 is ever applied.
    with pytest.raises(ValueError, match="does not bind an already-applied bound plan"):
        _append(
            engine,
            handle.run_id,
            RepresentationArtifactCompiledV2(artifact=artifact),
            module_id="M04",
        )


# ---------------------------------------------------------------------------
# Objective 3: semantic adjudication must be charged through the C04 runtime
# and bound to a real, applied provenance record -- never a bare claim.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_reject_adjudication_reference_outside_declared_tie_band(
    engine: FrontierReasoningEngine,
) -> None:
    handle = engine.create_run({"c07": "adjudication-outside-band"})
    problem = _tied_problem()
    signature, _ = TaskClassifier().classify(envelope(), None)
    budget, _ = BudgetAllocator().allocate(signature, default_tier_policy(), DeploymentLimits())
    _append(engine, handle.run_id, ProblemFormalised(problem=problem), module_id="M03")
    state = engine.inspect(handle.run_id)
    selector = RepresentationSelector()
    plan = selector.select_bound(
        problem, signature, budget, state.version, _two_candidate_registry(0.5)
    )
    assert not plan.tie_triggered
    forged = plan.model_copy(
        update={"adjudication_record_ref": "a" * 64},
    )
    forged = forged.model_copy(
        update={"plan_hash": canonical_hash(forged.model_dump(mode="json", exclude={"plan_hash"}))}
    )
    with pytest.raises(ValueError, match="outside its own declared tie band"):
        _append(engine, handle.run_id, RepresentationPlanSelectedV2(plan=forged), module_id="M04")


@pytest.mark.integration
def test_reject_adjudication_reference_without_matching_model_call(
    engine: FrontierReasoningEngine,
) -> None:
    handle = engine.create_run({"c07": "adjudication-unlinked"})
    problem = _tied_problem()
    signature, _ = TaskClassifier().classify(envelope(), None)
    budget, _ = BudgetAllocator().allocate(signature, default_tier_policy(), DeploymentLimits())
    _append(engine, handle.run_id, ProblemFormalised(problem=problem), module_id="M03")
    state = engine.inspect(handle.run_id)
    selector = RepresentationSelector()
    plan = selector.select_bound(
        problem, signature, budget, state.version, _two_candidate_registry(0.0)
    )
    assert plan.tie_triggered
    forged = plan.model_copy(update={"adjudication_record_ref": "a" * 64})
    forged = forged.model_copy(
        update={"plan_hash": canonical_hash(forged.model_dump(mode="json", exclude={"plan_hash"}))}
    )
    with pytest.raises(ValueError, match="no matching semantic model-call record"):
        _append(engine, handle.run_id, RepresentationPlanSelectedV2(plan=forged), module_id="M04")


@pytest.mark.unit
def test_select_bound_rejects_adjudication_ref_outside_tie_band_eagerly() -> None:
    problem, signature, budget = context()
    selector = RepresentationSelector()
    with pytest.raises(ValueError, match="genuinely tied"):
        selector.select_bound(problem, signature, budget, 0, adjudication_record_ref="a" * 64)


@pytest.mark.unit
def test_apply_adjudication_v2_never_silently_swallows_an_invalid_proposal() -> None:
    problem, signature, budget = context(_tied_problem())
    selector = RepresentationSelector()
    deterministic = selector.select_bound(
        problem, signature, budget, 0, _two_candidate_registry(0.0)
    )
    allowed = frozenset(view.kind for view in deterministic.views)

    none_outcome = selector.apply_adjudication_v2(
        deterministic, None, allowed_kinds=allowed, view_limit=2, adjudication_record_ref=None
    )
    assert none_outcome.plan == deterministic
    assert none_outcome.diagnostic is not None and "NO_PROPOSAL" in none_outcome.diagnostic

    invalid_kind_outcome = selector.apply_adjudication_v2(
        deterministic,
        RepresentationAdjudicationOutput(
            selected_kinds=("NOT_A_REAL_REPRESENTATION_KIND",), explanation="invalid"
        ),
        allowed_kinds=allowed,
        view_limit=2,
        adjudication_record_ref=None,
    )
    assert invalid_kind_outcome.diagnostic is not None
    assert "INVALID_KIND" in invalid_kind_outcome.diagnostic

    valid_outcome = selector.apply_adjudication_v2(
        deterministic,
        RepresentationAdjudicationOutput(
            selected_kinds=tuple(view.kind.value for view in reversed(deterministic.views)),
            explanation="prefer the other view",
        ),
        allowed_kinds=allowed,
        view_limit=2,
        adjudication_record_ref="a" * 64,
    )
    assert valid_outcome.diagnostic is None
    assert valid_outcome.plan.views[0].kind is deterministic.views[-1].kind
