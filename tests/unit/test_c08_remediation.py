"""Decisive C08 remediation tests: M12 typed Wave 3 context + persistence
(defect F12, downstream F04).

Each test below maps directly onto a bullet in the C08 validation strategy in
FRE_WAVE3_C01_C10_EXECUTION_COMPLETION_AND_VALIDATION_REGISTER.md.
"""

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID

import pytest

from fre.domain.budget import (
    BudgetLimits,
    BudgetPlan,
    BudgetRemaining,
    ReasoningTier,
    ResourceVector,
    SearchPolicy,
)
from fre.domain.common import ArtifactRef, FrozenModel, OutputContract, canonical_hash
from fre.domain.context import (
    CompilerProfile,
    ContextCompilationResult,
    Wave3ContextAvailability,
    Wave3SemanticContext,
)
from fre.domain.ledger import EpistemicStatus, LedgerNodeRef, LedgerNodeType, LedgerProjection
from fre.domain.problem import ProblemBlocker, ProblemSpec, UnknownSpec
from fre.domain.representation import (
    RepresentationArtifact,
    RepresentationKind,
    RepresentationPlan,
    RepresentationView,
)
from fre.domain.task import (
    ClassificationRecord,
    HorizonClass,
    Ordinal4,
    OutputForm,
    SearchSpaceClass,
    TaskSignature,
    TaskType,
)
from fre.engine import FrontierReasoningEngine
from fre.modules.m09_ledger import make_node
from fre.modules.m12_context import Wave3ContextCompiler, derive_wave3_availability
from fre.runtime.events import (
    ArtifactRegistered,
    BudgetAllocated,
    ContextCompiledV2,
    LedgerNodeAdded,
    ProblemBlockerRecorded,
    ProblemFormalised,
    RepresentationArtifactCompiled,
    RepresentationPlanSelected,
    RunCreated,
    StoredEvent,
    TaskClassified,
    event_wire_identity,
)
from fre.runtime.reducer import RunReducer, RunState
from fre.runtime.wave3_context import Wave3ContextRuntime


def _problem(*, unknown_resolvable: bool = True) -> ProblemSpec:
    return ProblemSpec(
        unknowns=(
            UnknownSpec(
                id="unknown-1",
                description="future demand",
                rationale="observed later",
                impact={"severity": "medium"},
                decision_relevance=0.5,
                resolvable=unknown_resolvable,
                candidate_actions=("wait", "survey"),
            ),
        ),
        output_contract=OutputContract(form="JSON"),
    )


def _representation(problem: ProblemSpec) -> RepresentationPlan:
    return RepresentationPlan(
        problem_spec_hash=canonical_hash(problem),
        views=(
            RepresentationView(
                id="view-1",
                kind=RepresentationKind.DECISION_TABLE,
                role="PRIMARY",
                compatibility_score=0.9,
                expected_value="clear trade-offs",
                builder_ref="m04.decision-table",
            ),
        ),
    )


def _representation_artifact(problem: ProblemSpec) -> RepresentationArtifact:
    return RepresentationArtifact(
        requested_kind=RepresentationKind.DECISION_TABLE,
        actual_kind=RepresentationKind.DECISION_TABLE,
        builder_id="m04.decision-table",
        builder_version="1.0",
        registry_version="wave3-m04-registry/1.0",
        selection_policy_version="wave3-m04/1.0",
        problem_spec_hash=canonical_hash(problem),
        source_snapshot_version=1,
        content={"rows": []},
        projection_hash="a" * 64,
    )


def _blocker(*, resolvable: bool) -> ProblemBlocker:
    return ProblemBlocker(
        blocker_id="blocker-1",
        description="Demand must be observed",
        ledger_ref=LedgerNodeRef(node_id=UUID(int=9), revision=1),
        resolvable=resolvable,
    )


def _remaining(iterations: int = 3) -> BudgetRemaining:
    return BudgetRemaining(
        resources=ResourceVector(iterations=iterations),
        active_concurrent_actions=0,
        projection_hash="0" * 64,
    )


def _exhausted_remaining() -> BudgetRemaining:
    return BudgetRemaining(
        resources=ResourceVector(),
        active_concurrent_actions=0,
        projection_hash="0" * 64,
    )


def _budget_plan() -> BudgetPlan:
    return BudgetPlan(
        tier=ReasoningTier.T1,
        limits=BudgetLimits(
            max_iterations=10,
            max_llm_calls=5,
            max_tool_calls=5,
            max_input_tokens=1000,
            max_output_tokens=1000,
            max_candidates=5,
            max_concurrent_actions=1,
            max_runtime_seconds=60,
        ),
        search=SearchPolicy(
            initial_candidate_count=1,
            max_candidate_depth=1,
            max_representation_views=1,
            synthesis_order_limit=1,
            falsifier_operator_limit=0,
            independent_validation_branches=0,
            sensitivity_samples=0,
        ),
        acquisition_threshold=0.5,
        stop_threshold=0.5,
        policy_version="1.0",
    )


def _task_signature() -> TaskSignature:
    return TaskSignature(
        task_type=TaskType.ANALYSIS,
        consequence=Ordinal4.MEDIUM,
        irreversibility=Ordinal4.MEDIUM,
        ambiguity=Ordinal4.MEDIUM,
        search_space=SearchSpaceClass.BOUNDED,
        evidence_scarcity=Ordinal4.MEDIUM,
        horizon=HorizonClass.SHORT,
        output_form=OutputForm.TEXT,
        dimension_confidence={},
    )


def _payload_applier(
    reducer: RunReducer, run_id: UUID
) -> Callable[[RunState, FrozenModel], RunState]:
    def apply_payload(state: RunState, payload: FrozenModel, module_id: str = "test") -> RunState:
        event_type, schema_version = event_wire_identity(payload)
        event = StoredEvent(
            event_id=UUID(int=state.version + 10_000),
            run_id=run_id,
            event_type=event_type,
            action_id=UUID(int=state.version + 20_000),
            module_id=module_id,
            schema_version=schema_version,
            module_version="1.0",
            input_hash="0" * 64,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            payload=payload.model_dump(mode="json"),
            sequence=state.version + 1,
        )
        return reducer.apply(state, event)

    return apply_payload


def _full_state(
    run_id: UUID,
    *,
    unknown_resolvable: bool = True,
    blocker_resolvable: bool = True,
    with_representation: bool = True,
    with_artifact: bool = True,
    with_task_signature: bool = True,
    with_budget: bool = True,
) -> tuple[RunReducer, RunState, Callable[[RunState, FrozenModel], RunState], ProblemSpec]:
    reducer = RunReducer()
    apply_payload = _payload_applier(reducer, run_id)
    problem = _problem(unknown_resolvable=unknown_resolvable)
    state = reducer.initial(run_id)
    state = apply_payload(state, RunCreated(config_hash="c"))
    state = apply_payload(state, ProblemFormalised(problem=problem))
    blocker = _blocker(resolvable=blocker_resolvable)
    state = apply_payload(
        state,
        LedgerNodeAdded(
            node=make_node(
                node_id=blocker.ledger_ref.node_id,
                revision=1,
                node_type=LedgerNodeType.FACT,
                content={"fact": "demand unresolved"},
                status=EpistemicStatus.UNRESOLVED,
                created_at="2026-01-01T00:00:00Z",
                action_id=UUID(int=90),
                module_id="M09",
            )
        ),
    )
    state = apply_payload(state, ProblemBlockerRecorded(blocker=blocker))
    if with_task_signature:
        state = apply_payload(
            state,
            TaskClassified(
                signature=_task_signature(),
                record=ClassificationRecord(
                    policy_version="1.0",
                    policy_hash="b" * 64,
                    mode="DETERMINISTIC",
                    fallback_used=False,
                    dimensions={},
                ),
            ),
        )
    if with_budget:
        plan = _budget_plan()
        state = apply_payload(
            state,
            BudgetAllocated(plan=plan, policy_version=plan.policy_version, policy_hash="d" * 64),
        )
    if with_representation:
        representation = _representation(problem)
        state = apply_payload(state, RepresentationPlanSelected(plan=representation))
        if with_artifact:
            state = apply_payload(
                state, RepresentationArtifactCompiled(artifact=_representation_artifact(problem))
            )
    return reducer, state, apply_payload, problem


def _compile_matching_packet(
    state: RunState, problem: ProblemSpec, *, profile: CompilerProfile = CompilerProfile.STANDARD
) -> ContextCompilationResult:
    compiler = Wave3ContextCompiler()
    return compiler.compile_semantic(
        problem=problem,
        representation=state.representation_plan,
        problem_blockers=state.problem_blockers,
        representation_artifacts=state.representation_artifacts,
        task_signature=state.task_signature,
        budget_plan=state.budget.plan,
        budget_policy_hash=state.budget.policy_hash,
        run_id=state.run_id,
        snapshot_version=state.version,
        ledger=state.ledger,
        budget_remaining=_remaining(),
        profile=profile,
    )


def _wave3(compiled: ContextCompilationResult) -> Wave3SemanticContext:
    wave3 = compiled.packet.wave3_context
    assert wave3 is not None
    return wave3


def _register_and_apply(
    state: RunState,
    apply_payload: Callable[[RunState, FrozenModel], RunState],
    compiled: ContextCompilationResult,
) -> RunState:
    json_ref = ArtifactRef(
        artifact_id=UUID(int=101), sha256=hashlib.sha256(compiled.canonical_bytes).hexdigest()
    )
    markdown_ref = ArtifactRef(
        artifact_id=UUID(int=102),
        sha256=hashlib.sha256(compiled.markdown.encode()).hexdigest(),
    )
    state = apply_payload(
        state,
        ArtifactRegistered(artifact=json_ref, media_type="application/json", byte_size=1),
    )
    state = apply_payload(
        state,
        ArtifactRegistered(artifact=markdown_ref, media_type="text/markdown", byte_size=1),
    )
    state = apply_payload(
        state,
        ContextCompiledV2(
            packet=compiled.packet, json_artifact=json_ref, markdown_artifact=markdown_ref
        ),
    )
    return state


# ---------------------------------------------------------------------------
# derive_wave3_availability: pure-function decisive coverage
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_availability_missing_representation_is_not_requested() -> None:
    availability, _ = derive_wave3_availability(
        problem_blockers=(), representation=None, budget_remaining=_remaining()
    )
    assert availability is Wave3ContextAvailability.NOT_REQUESTED


@pytest.mark.unit
def test_availability_material_blocker_is_partial_blocked_even_with_representation() -> None:
    problem = _problem()
    availability, reason = derive_wave3_availability(
        problem_blockers=(_blocker(resolvable=False),),
        representation=_representation(problem),
        representation_artifacts=(_representation_artifact(problem),),
        budget_remaining=_remaining(),
    )
    assert availability is Wave3ContextAvailability.PARTIAL_BLOCKED
    assert "blocker-1" in reason


@pytest.mark.unit
def test_availability_resolvable_blocker_is_not_material() -> None:
    problem = _problem()
    availability, _ = derive_wave3_availability(
        problem_blockers=(_blocker(resolvable=True),),
        representation=_representation(problem),
        representation_artifacts=(_representation_artifact(problem),),
        budget_remaining=_remaining(),
    )
    assert availability is Wave3ContextAvailability.AVAILABLE


@pytest.mark.unit
def test_availability_exhausted_budget_is_unavailable_budget() -> None:
    problem = _problem()
    availability, _ = derive_wave3_availability(
        problem_blockers=(),
        representation=_representation(problem),
        representation_artifacts=(_representation_artifact(problem),),
        budget_remaining=_exhausted_remaining(),
    )
    assert availability is Wave3ContextAvailability.UNAVAILABLE_BUDGET


@pytest.mark.unit
def test_availability_representation_without_validated_artifact_is_unavailable_validation() -> None:
    problem = _problem()
    availability, _ = derive_wave3_availability(
        problem_blockers=(),
        representation=_representation(problem),
        representation_artifacts=(),
        budget_remaining=_remaining(),
    )
    assert availability is Wave3ContextAvailability.UNAVAILABLE_VALIDATION


# ---------------------------------------------------------------------------
# Compiler-level: wave3_context is fully populated and self-consistent
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compile_semantic_populates_typed_wave3_context() -> None:
    problem = _problem()
    representation = _representation(problem)
    artifact = _representation_artifact(problem)
    blocker = _blocker(resolvable=True)
    compiled = Wave3ContextCompiler().compile_semantic(
        problem=problem,
        representation=representation,
        problem_blockers=(blocker,),
        representation_artifacts=(artifact,),
        run_id=UUID(int=1),
        snapshot_version=4,
        ledger=LedgerProjection(),
        budget_remaining=_remaining(),
        profile=CompilerProfile.STANDARD,
    )
    wave3 = _wave3(compiled)
    assert wave3 is not None
    assert wave3.availability is Wave3ContextAvailability.AVAILABLE
    assert wave3.problem_spec_ref == canonical_hash(problem)
    assert wave3.blocker_refs == ("blocker-1",)
    assert wave3.representation_plan_ref == canonical_hash(representation)
    assert wave3.representation_artifact_refs == (canonical_hash(artifact),)
    assert wave3.ledger_root == canonical_hash(LedgerProjection())
    assert wave3.ledger_version == 4
    assert len(wave3.unresolved_unknowns) == 1
    assert wave3.unresolved_unknowns[0].candidate_actions == ("wait", "survey")


@pytest.mark.unit
def test_compilation_is_deterministic_for_identical_state() -> None:
    problem = _problem()
    representation = _representation(problem)
    artifact = _representation_artifact(problem)

    def _compile() -> ContextCompilationResult:
        return Wave3ContextCompiler().compile_semantic(
            problem=problem,
            representation=representation,
            problem_blockers=(_blocker(resolvable=True),),
            representation_artifacts=(artifact,),
            run_id=UUID(int=1),
            snapshot_version=4,
            ledger=LedgerProjection(),
            budget_remaining=_remaining(),
            profile=CompilerProfile.STANDARD,
        )

    first = _compile()
    second = _compile()
    assert first.canonical_bytes == second.canonical_bytes
    assert first.packet.packet_hash == second.packet.packet_hash
    assert first.packet.wave3_context == second.packet.wave3_context


@pytest.mark.unit
def test_v1_packet_serialization_omits_wave3_context_and_is_not_falsely_complete() -> None:
    problem = _problem()
    v1_compiled = Wave3ContextCompiler().compile_semantic(
        problem=problem,
        representation=None,
        problem_blockers=(),
        run_id=UUID(int=1),
        snapshot_version=1,
        ledger=LedgerProjection(),
        budget_remaining=_remaining(),
        profile=CompilerProfile.STANDARD,
    )
    # `representation=None` still yields a typed (NOT_REQUESTED) wave3_context
    # here because it went through the C08-aware Wave3ContextCompiler. A
    # genuine pre-C08 v1 packet (e.g. plain `ContextCompiler.compile()`, or an
    # old decoded snapshot) never sets the field at all.
    from fre.domain.context import ContextPacket

    v1_dict = v1_compiled.packet.model_dump(mode="json")
    v1_dict.pop("wave3_context", None)
    decoded = ContextPacket.model_validate(v1_dict, strict=False)
    assert decoded.wave3_context is None
    dumped = decoded.model_dump(mode="json")
    assert "wave3_context" not in dumped


# ---------------------------------------------------------------------------
# Reducer-level: EVERY wave3_context ref is checked against real applied state
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reducer_accepts_a_genuine_matching_wave3_context() -> None:
    run_id = UUID(int=1)
    _, state, apply_payload, problem = _full_state(run_id)
    compiled = _compile_matching_packet(state, problem)
    final = _register_and_apply(state, apply_payload, compiled)
    final_wave3 = final.context_packets[-1].wave3_context
    assert final.context_packets[-1].packet_hash == compiled.packet.packet_hash
    assert final_wave3 is not None
    assert final_wave3.availability is Wave3ContextAvailability.AVAILABLE


@pytest.mark.unit
def test_reducer_rejects_dangling_problem_spec_ref() -> None:
    run_id = UUID(int=1)
    _, state, apply_payload, problem = _full_state(run_id)
    compiled = _compile_matching_packet(state, problem)
    tampered = compiled.packet.model_copy(
        update={"wave3_context": _wave3(compiled).model_copy(update={"problem_spec_ref": "f" * 64})}
    )
    rehashed = tampered.model_copy(
        update={"packet_hash": canonical_hash(tampered.model_dump(exclude={"packet_hash"}))}
    )
    with pytest.raises(ValueError, match="problem_spec_ref"):
        _register_and_apply(state, apply_payload, compiled.model_copy(update={"packet": rehashed}))


@pytest.mark.unit
def test_reducer_rejects_dangling_blocker_ref() -> None:
    run_id = UUID(int=1)
    _, state, apply_payload, problem = _full_state(run_id)
    compiled = _compile_matching_packet(state, problem)
    tampered_context = _wave3(compiled).model_copy(
        update={"blocker_refs": (*_wave3(compiled).blocker_refs, "ghost-blocker")}
    )
    tampered = compiled.packet.model_copy(update={"wave3_context": tampered_context})
    rehashed = tampered.model_copy(
        update={"packet_hash": canonical_hash(tampered.model_dump(exclude={"packet_hash"}))}
    )
    with pytest.raises(ValueError, match="blocker_refs"):
        _register_and_apply(state, apply_payload, compiled.model_copy(update={"packet": rehashed}))


@pytest.mark.unit
def test_reducer_rejects_dangling_representation_plan_ref() -> None:
    run_id = UUID(int=1)
    _, state, apply_payload, problem = _full_state(run_id)
    compiled = _compile_matching_packet(state, problem)
    tampered_context = _wave3(compiled).model_copy(update={"representation_plan_ref": "e" * 64})
    tampered = compiled.packet.model_copy(update={"wave3_context": tampered_context})
    rehashed = tampered.model_copy(
        update={"packet_hash": canonical_hash(tampered.model_dump(exclude={"packet_hash"}))}
    )
    with pytest.raises(ValueError, match="representation_plan_ref"):
        _register_and_apply(state, apply_payload, compiled.model_copy(update={"packet": rehashed}))


@pytest.mark.unit
def test_reducer_rejects_dangling_representation_artifact_ref() -> None:
    run_id = UUID(int=1)
    _, state, apply_payload, problem = _full_state(run_id)
    compiled = _compile_matching_packet(state, problem)
    tampered_context = _wave3(compiled).model_copy(
        update={
            "representation_artifact_refs": (
                *_wave3(compiled).representation_artifact_refs,
                "a" * 64,
            )
        }
    )
    tampered = compiled.packet.model_copy(update={"wave3_context": tampered_context})
    rehashed = tampered.model_copy(
        update={"packet_hash": canonical_hash(tampered.model_dump(exclude={"packet_hash"}))}
    )
    with pytest.raises(ValueError, match="representation_artifact_refs"):
        _register_and_apply(state, apply_payload, compiled.model_copy(update={"packet": rehashed}))


@pytest.mark.unit
def test_reducer_rejects_dangling_ledger_root() -> None:
    run_id = UUID(int=1)
    _, state, apply_payload, problem = _full_state(run_id)
    compiled = _compile_matching_packet(state, problem)
    tampered_context = _wave3(compiled).model_copy(update={"ledger_root": "b" * 64})
    tampered = compiled.packet.model_copy(update={"wave3_context": tampered_context})
    rehashed = tampered.model_copy(
        update={"packet_hash": canonical_hash(tampered.model_dump(exclude={"packet_hash"}))}
    )
    with pytest.raises(ValueError, match="ledger_root"):
        _register_and_apply(state, apply_payload, compiled.model_copy(update={"packet": rehashed}))


@pytest.mark.unit
def test_reducer_rejects_dangling_task_signature_ref() -> None:
    run_id = UUID(int=1)
    _, state, apply_payload, problem = _full_state(run_id)
    compiled = _compile_matching_packet(state, problem)
    assert _wave3(compiled).task_signature_ref is not None
    tampered_context = _wave3(compiled).model_copy(update={"task_signature_ref": "c" * 64})
    tampered = compiled.packet.model_copy(update={"wave3_context": tampered_context})
    rehashed = tampered.model_copy(
        update={"packet_hash": canonical_hash(tampered.model_dump(exclude={"packet_hash"}))}
    )
    with pytest.raises(ValueError, match="task_signature_ref"):
        _register_and_apply(state, apply_payload, compiled.model_copy(update={"packet": rehashed}))


@pytest.mark.unit
def test_reducer_rejects_dangling_budget_plan_ref() -> None:
    run_id = UUID(int=1)
    _, state, apply_payload, problem = _full_state(run_id)
    compiled = _compile_matching_packet(state, problem)
    assert _wave3(compiled).budget_plan_ref is not None
    tampered_context = _wave3(compiled).model_copy(update={"budget_plan_ref": "9" * 64})
    tampered = compiled.packet.model_copy(update={"wave3_context": tampered_context})
    rehashed = tampered.model_copy(
        update={"packet_hash": canonical_hash(tampered.model_dump(exclude={"packet_hash"}))}
    )
    with pytest.raises(ValueError, match="budget_plan_ref"):
        _register_and_apply(state, apply_payload, compiled.model_copy(update={"packet": rehashed}))


@pytest.mark.unit
def test_reducer_rejects_wrong_budget_policy_hash() -> None:
    run_id = UUID(int=1)
    _, state, apply_payload, problem = _full_state(run_id)
    compiled = _compile_matching_packet(state, problem)
    assert _wave3(compiled).budget_policy_hash is not None
    tampered_context = _wave3(compiled).model_copy(update={"budget_policy_hash": "1" * 64})
    tampered = compiled.packet.model_copy(update={"wave3_context": tampered_context})
    rehashed = tampered.model_copy(
        update={"packet_hash": canonical_hash(tampered.model_dump(exclude={"packet_hash"}))}
    )
    with pytest.raises(ValueError, match="budget_policy_hash"):
        _register_and_apply(state, apply_payload, compiled.model_copy(update={"packet": rehashed}))


@pytest.mark.unit
def test_reducer_rejects_forged_unresolved_unknown_metadata() -> None:
    run_id = UUID(int=1)
    _, state, apply_payload, problem = _full_state(run_id)
    compiled = _compile_matching_packet(state, problem)
    real = _wave3(compiled).unresolved_unknowns[0]
    forged = real.model_copy(update={"resolvable": not real.resolvable})
    tampered_context = _wave3(compiled).model_copy(update={"unresolved_unknowns": (forged,)})
    tampered = compiled.packet.model_copy(update={"wave3_context": tampered_context})
    rehashed = tampered.model_copy(
        update={"packet_hash": canonical_hash(tampered.model_dump(exclude={"packet_hash"}))}
    )
    with pytest.raises(ValueError, match="unresolved_unknowns"):
        _register_and_apply(state, apply_payload, compiled.model_copy(update={"packet": rehashed}))


@pytest.mark.unit
def test_reducer_rejects_forged_availability_disagreeing_with_recomputation() -> None:
    run_id = UUID(int=1)
    _, state, apply_payload, problem = _full_state(run_id, blocker_resolvable=False)
    compiled = _compile_matching_packet(state, problem)
    assert _wave3(compiled).availability is Wave3ContextAvailability.PARTIAL_BLOCKED
    tampered_context = _wave3(compiled).model_copy(
        update={
            "availability": Wave3ContextAvailability.AVAILABLE,
            "availability_reason": "claims everything is fine",
        }
    )
    tampered = compiled.packet.model_copy(update={"wave3_context": tampered_context})
    rehashed = tampered.model_copy(
        update={"packet_hash": canonical_hash(tampered.model_dump(exclude={"packet_hash"}))}
    )
    with pytest.raises(ValueError, match="availability"):
        _register_and_apply(state, apply_payload, compiled.model_copy(update={"packet": rehashed}))


@pytest.mark.unit
def test_reducer_rejects_unregistered_context_artifacts() -> None:
    run_id = UUID(int=1)
    _, state, apply_payload, problem = _full_state(run_id)
    compiled = _compile_matching_packet(state, problem)
    json_ref = ArtifactRef(artifact_id=UUID(int=201), sha256="9" * 64)
    markdown_ref = ArtifactRef(artifact_id=UUID(int=202), sha256="8" * 64)
    with pytest.raises(ValueError, match="not registered"):
        apply_payload(
            state,
            ContextCompiledV2(
                packet=compiled.packet, json_artifact=json_ref, markdown_artifact=markdown_ref
            ),
        )


@pytest.mark.unit
def test_reducer_rejects_v2_event_wrapping_a_v1_style_packet_with_no_wave3_context() -> None:
    run_id = UUID(int=1)
    _, state, apply_payload, problem = _full_state(run_id, with_representation=False)
    v1_compiled = Wave3ContextCompiler().compile_semantic(
        problem=problem,
        representation=None,
        problem_blockers=state.problem_blockers,
        run_id=run_id,
        snapshot_version=state.version,
        ledger=state.ledger,
        budget_remaining=_remaining(),
        profile=CompilerProfile.STANDARD,
    )
    stripped = v1_compiled.packet.model_dump(mode="json")
    stripped.pop("wave3_context", None)
    from fre.domain.context import ContextPacket

    v1_only_packet = ContextPacket.model_validate(stripped, strict=False)
    rehashed = v1_only_packet.model_copy(
        update={"packet_hash": canonical_hash(v1_only_packet.model_dump(exclude={"packet_hash"}))}
    )
    json_ref = ArtifactRef(artifact_id=UUID(int=301), sha256="1" * 64)
    markdown_ref = ArtifactRef(artifact_id=UUID(int=302), sha256="2" * 64)
    state = apply_payload(
        state, ArtifactRegistered(artifact=json_ref, media_type="application/json", byte_size=1)
    )
    state = apply_payload(
        state, ArtifactRegistered(artifact=markdown_ref, media_type="text/markdown", byte_size=1)
    )
    with pytest.raises(ValueError, match="wave3_context"):
        apply_payload(
            state,
            ContextCompiledV2(
                packet=rehashed, json_artifact=json_ref, markdown_artifact=markdown_ref
            ),
        )


@pytest.mark.unit
def test_reducer_rejects_invalid_schema_version() -> None:
    run_id = UUID(int=1)
    _, state, apply_payload, problem = _full_state(run_id)
    compiled = _compile_matching_packet(state, problem)
    tampered_context = _wave3(compiled).model_copy(update={"schema_version": "9.9"})
    tampered = compiled.packet.model_copy(update={"wave3_context": tampered_context})
    rehashed = tampered.model_copy(
        update={"packet_hash": canonical_hash(tampered.model_dump(exclude={"packet_hash"}))}
    )
    with pytest.raises(ValueError, match="schema_version"):
        _register_and_apply(state, apply_payload, compiled.model_copy(update={"packet": rehashed}))


# ---------------------------------------------------------------------------
# Real-engine integration: persistence, replay, and delta extension
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_wave3_context_runtime_persists_and_replay_reproduces_it(
    engine: FrontierReasoningEngine,
) -> None:
    handle = engine.create_run({"config": "c08"})
    problem = _problem()
    blocker = _blocker(resolvable=True)
    plan = _budget_plan()
    events = (
        engine.make_event(handle.run_id, ProblemFormalised(problem=problem), module_id="M03"),
        engine.make_event(
            handle.run_id,
            LedgerNodeAdded(
                node=make_node(
                    node_id=blocker.ledger_ref.node_id,
                    revision=1,
                    node_type=LedgerNodeType.FACT,
                    content={"fact": "demand unresolved"},
                    status=EpistemicStatus.UNRESOLVED,
                    created_at="2026-01-01T00:00:00Z",
                    action_id=UUID(int=90),
                    module_id="M09",
                )
            ),
            module_id="M09",
        ),
        engine.make_event(handle.run_id, ProblemBlockerRecorded(blocker=blocker), module_id="M03"),
        engine.make_event(
            handle.run_id,
            BudgetAllocated(plan=plan, policy_version=plan.policy_version, policy_hash="d" * 64),
            module_id="M02",
        ),
    )
    engine.append(handle.run_id, handle.version, events)
    runtime = Wave3ContextRuntime(engine)
    packet_hash = runtime.compile_and_persist(handle.run_id, profile=CompilerProfile.STANDARD)

    engine.snapshot(handle.run_id)
    replayed = engine.replay_from_snapshot(handle.run_id)
    assert replayed.context_packets[-1].packet_hash == packet_hash
    verified = engine.verify(handle.run_id)
    assert verified.state_hash == replayed.state_hash
    verified_wave3 = verified.context_packets[-1].wave3_context
    replayed_wave3 = replayed.context_packets[-1].wave3_context
    assert verified_wave3 is not None
    assert replayed_wave3 is not None
    assert verified_wave3.availability == replayed_wave3.availability
    assert verified.context_packets[-1].packet_hash == packet_hash


@pytest.mark.unit
def test_generate_delta_covers_the_new_typed_field() -> None:
    from fre.modules.m12_context import apply_delta, generate_delta

    run_id = UUID(int=1)
    _, state, _, problem = _full_state(run_id)
    v2 = _compile_matching_packet(state, problem)
    v1 = Wave3ContextCompiler().compile_semantic(
        problem=problem,
        representation=None,
        problem_blockers=state.problem_blockers,
        run_id=run_id,
        snapshot_version=state.version,
        ledger=state.ledger,
        budget_remaining=_remaining(),
        profile=CompilerProfile.STANDARD,
    )
    delta = generate_delta(v1.packet, v2.packet)
    assert any(op.path == "/wave3_context" for op in delta.operations)
    assert apply_delta(v1.packet, delta) == v2.packet
