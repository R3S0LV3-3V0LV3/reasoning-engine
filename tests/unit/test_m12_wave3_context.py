"""Golden coverage for the compiler 2.0 semantic context projection."""

from pathlib import Path
from uuid import UUID

import pytest

from fre.domain.budget import BudgetRemaining, ResourceVector
from fre.domain.common import OutputContract, canonical_hash
from fre.domain.context import CompilerProfile, ContextCompilationResult
from fre.domain.ledger import EpistemicStatus, LedgerNodeRef, LedgerNodeType, LedgerProjection
from fre.domain.problem import (
    AcceptanceCriterion,
    AssumptionSpec,
    ConstraintSpec,
    DecisionVariable,
    FixedParameter,
    ObjectiveSpec,
    ObservableSpec,
    ProblemBlocker,
    ProblemRelation,
    ProblemRelationKind,
    ProblemSpec,
    UnknownSpec,
)
from fre.domain.representation import RepresentationKind, RepresentationPlan, RepresentationView
from fre.domain.semantic import EpistemicItemProvenance, EpistemicOriginLabel
from fre.modules.m09_ledger import EpistemicLedger, make_node
from fre.modules.m12_context import Wave3ContextCompiler, apply_delta, generate_delta

GOLDENS = Path(__file__).parents[1] / "fixtures" / "m12_wave3"


def semantic_context() -> tuple[ProblemSpec, RepresentationPlan, tuple[ProblemBlocker, ...]]:
    provenance = EpistemicItemProvenance(
        origin=EpistemicOriginLabel.WORKING_ASSUMPTION,
        basis="deterministic fixture",
        policy_basis="test policy",
    )
    problem = ProblemSpec(
        decision_variables=(DecisionVariable(id="dv-1", name="option", domain=["a", "b"]),),
        objectives=(
            ObjectiveSpec(id="obj-1", name="cost", direction="MIN", description="Minimize cost"),
        ),
        constraints=(
            ConstraintSpec(
                id="hard-1",
                description="cost <= 10",
                kind="HARD",
                verification_mode="DETERMINISTIC",
                verifier_ref="verify.cost",
            ),
            ConstraintSpec(
                id="soft-1",
                description="prefer option a",
                kind="SOFT",
                verification_mode="HUMAN",
            ),
        ),
        assumptions=("Demand is stable",),
        assumption_items=(
            AssumptionSpec(
                id="assume-1",
                statement="Supply is stable",
                why_needed="Bound the choice",
                provenance=provenance,
            ),
        ),
        fixed_parameters=(
            FixedParameter(id="fixed-1", name="limit", value=10, provenance=provenance),
        ),
        unknowns=(UnknownSpec(id="unknown-1", description="future demand", resolvable=True),),
        observables=(ObservableSpec(id="obs-1", description="quoted cost", unit="USD"),),
        acceptance_criteria=(
            AcceptanceCriterion(
                id="accept-1",
                predicate_description="cost is within limit",
                verification_mode="DETERMINISTIC",
                required=True,
            ),
        ),
        relations=(
            ProblemRelation(
                source_id="obj-1", target_id="hard-1", kind=ProblemRelationKind.DEPENDS_ON
            ),
        ),
        output_contract=OutputContract(form="JSON", requirements=("include rationale",)),
    )
    representation = RepresentationPlan(
        problem_spec_hash=canonical_hash(problem),
        views=(
            RepresentationView(
                id="view-1",
                kind=RepresentationKind.DECISION_TABLE,
                role="PRIMARY",
                compatibility_score=0.9,
                purpose="compare options",
                expected_value="clear trade-offs",
                builder_ref="m04.decision-table",
            ),
        ),
        selection_basis=("decision variables available",),
    )
    blockers = (
        ProblemBlocker(
            blocker_id="blocker-1",
            description="Demand must be observed",
            ledger_ref=LedgerNodeRef(node_id=UUID(int=9), revision=1),
            resolvable=True,
        ),
    )
    return problem, representation, blockers


def remaining(iterations: int = 0) -> BudgetRemaining:
    return BudgetRemaining(
        resources=ResourceVector(iterations=iterations),
        active_concurrent_actions=0,
        projection_hash="0" * 64,
    )


def ledger() -> LedgerProjection:
    node = make_node(
        node_id=UUID(int=8),
        revision=1,
        node_type=LedgerNodeType.FACT,
        content={"fact": "current"},
        status=EpistemicStatus.SUPPORTED,
        created_at="2026-01-01T00:00:00Z",
        action_id=UUID(int=80),
        module_id="M09",
    )
    return EpistemicLedger().append_node(LedgerProjection(), node)


def compile_profile(profile: CompilerProfile) -> ContextCompilationResult:
    problem, representation, blockers = semantic_context()
    return Wave3ContextCompiler().compile_semantic(
        problem=problem,
        representation=representation,
        problem_blockers=blockers,
        run_id=UUID(int=1),
        snapshot_version=7,
        ledger=ledger(),
        budget_remaining=remaining(3),
        profile=profile,
        unresolved_blockers=("Demand must be observed",),
        next_action="observe demand",
    )


@pytest.mark.parametrize("profile", tuple(CompilerProfile))
def test_semantic_json_and_markdown_exact_goldens(profile: CompilerProfile) -> None:
    result = compile_profile(profile)
    stem = profile.value.lower()
    assert result.canonical_bytes == (GOLDENS / f"{stem}.json").read_bytes()
    assert result.markdown == (GOLDENS / f"{stem}.md").read_text()

    semantic = result.packet.objective
    assert isinstance(semantic, dict)
    assert tuple(semantic) == (
        "objectives",
        "hard_constraints",
        "soft_preferences",
        "decision_variables",
        "fixed_parameters",
        "unknowns",
        "observables",
        "assumptions",
        "acceptance_criteria",
        "output_requirements",
        "relations",
        "explicit_blockers",
        "representation",
    )
    assert len({item.ref for item in result.packet.ledger_items}) == len(result.packet.ledger_items)


def test_semantic_delta_reconstructs_exact_packet_and_stale_views_are_rejected() -> None:
    full = compile_profile(CompilerProfile.FULL)
    handoff = compile_profile(CompilerProfile.HANDOFF)
    assert apply_delta(full.packet, generate_delta(full.packet, handoff.packet)) == handoff.packet

    problem, representation, blockers = semantic_context()
    stale = representation.model_copy(update={"problem_spec_hash": None})
    with pytest.raises(ValueError, match="does not bind current ProblemSpec"):
        Wave3ContextCompiler().compile_semantic(
            problem=problem,
            representation=stale,
            problem_blockers=blockers,
            run_id=UUID(int=1),
            snapshot_version=7,
            ledger=LedgerProjection(),
            budget_remaining=remaining(),
            profile=CompilerProfile.STANDARD,
        )

    omitted = Wave3ContextCompiler().compile_semantic(
        problem=problem,
        representation=None,
        problem_blockers=blockers,
        run_id=UUID(int=1),
        snapshot_version=7,
        ledger=LedgerProjection(),
        budget_remaining=remaining(),
        profile=CompilerProfile.STANDARD,
    )
    assert isinstance(omitted.packet.objective, dict)
    assert omitted.packet.objective["representation"] is None
