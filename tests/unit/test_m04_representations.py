"""M04 compatibility, adjudication, and structural projection regressions."""

import asyncio
import json
from typing import cast
from uuid import UUID

import pytest

from fre.domain.budget import BudgetPlan, DeploymentLimits
from fre.domain.common import JsonValue, OutputContract, PermissionSet, canonical_hash
from fre.domain.problem import ProblemRelation, ProblemRelationKind, ProblemSpec, UnknownSpec
from fre.domain.representation import RepresentationKind, RepresentationPlan
from fre.domain.task import TaskEnvelope, TaskSignature
from fre.modules.m01_classifier import TaskClassifier
from fre.modules.m02_budget import BudgetAllocator, default_tier_policy
from fre.modules.m04_representation import (
    RepresentationDefinition,
    RepresentationSelectionPolicy,
    RepresentationSelector,
    default_registry,
)
from fre.prompts.schemas import RepresentationAdjudicationOutput
from fre.semantic_runtime import SemanticExecution, SemanticModelRuntime


def envelope() -> TaskEnvelope:
    return TaskEnvelope(
        task_id=UUID(int=1),
        text="Compare causal and dependency structure.",
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(),
    )


def context() -> tuple[ProblemSpec, TaskSignature, BudgetPlan]:
    problem = ProblemSpec(output_contract=OutputContract(form="TEXT"))
    signature, _ = TaskClassifier().classify(envelope(), None)
    budget, _ = BudgetAllocator().allocate(signature, default_tier_policy(), DeploymentLimits())
    return problem, signature, budget


@pytest.mark.unit
def test_sparse_problem_falls_back_and_registry_order_is_irrelevant() -> None:
    problem, signature, budget = context()
    selector = RepresentationSelector()
    registry = default_registry()
    first = selector.select(problem, signature, budget, registry)
    reversed_result = selector.select(problem, signature, budget, tuple(reversed(registry)))
    assert first == reversed_result
    assert tuple(view.kind for view in first.views) == (RepresentationKind.TEXT_TABLE_FALLBACK,)
    assert all(
        component.contribution >= 0.20
        for view in first.views
        for component in view.score_components
    )
    specialized_only = tuple(
        definition
        for definition in registry
        if definition.kind is not RepresentationKind.TEXT_TABLE_FALLBACK
    )
    assert (
        selector.select(problem, signature, budget, specialized_only).views[0].kind
        is RepresentationKind.TEXT_TABLE_FALLBACK
    )


@pytest.mark.unit
def test_registry_builder_availability_produces_typed_fallback() -> None:
    problem, signature, budget = context()
    problem = problem.model_copy(update={"unknowns": (UnknownSpec(id="u", description="demand"),)})
    unavailable = RepresentationDefinition(
        kind=RepresentationKind.SCENARIO_TREE,
        builder_id="scenario",
        priority=0,
        purpose="unknown projection",
        limitations=("does not generate scenarios",),
        builder_available=False,
    )
    plan = RepresentationSelector().select(problem, signature, budget, (unavailable,))
    assert not plan.views[0].builder_available
    artifact = RepresentationSelector().build(plan.views[0], problem, 3)
    assert artifact.requested_kind is RepresentationKind.SCENARIO_TREE
    assert artifact.actual_kind is RepresentationKind.TEXT_TABLE_FALLBACK
    assert artifact.fallback_reason and "requested SCENARIO_TREE" in artifact.limitations[-1]


@pytest.mark.unit
def test_legacy_view_validation_preserves_canonical_hash() -> None:
    problem, signature, budget = context()
    current = RepresentationSelector().select(problem, signature, budget).model_dump(mode="json")
    legacy_hash = canonical_hash(current)

    restored = RepresentationPlan.model_validate_json(json.dumps(current))

    assert restored.views[0].builder_available is True
    assert "builder_available" not in restored.model_dump(mode="json")["views"][0]
    assert canonical_hash(restored) == legacy_hash


@pytest.mark.unit
def test_unavailable_builder_remains_explicit_in_wire_representation() -> None:
    problem, signature, budget = context()
    plan = RepresentationSelector().select(problem, signature, budget)
    unavailable = plan.views[0].model_copy(update={"builder_available": False})
    plan = plan.model_copy(update={"views": (unavailable,)})

    assert plan.model_dump(mode="json")["views"][0]["builder_available"] is False


@pytest.mark.unit
def test_all_registered_builders_are_deterministic_and_do_not_run_wave4_algorithms() -> None:
    problem, _, _ = context()
    problem = problem.model_copy(
        update={
            "unknowns": (UnknownSpec(id="u", description="demand"),),
            "relations": tuple(
                ProblemRelation(source_id="a", target_id="b", kind=kind)
                for kind in ProblemRelationKind
            ),
        }
    )
    selector = RepresentationSelector()
    for definition in default_registry():
        view = selector.select(
            problem,
            context()[1],
            context()[2],
            (definition,),
            RepresentationSelectionPolicy(minimum_compatibility=0),
        ).views[0]
        artifact = selector.build(view, problem, 7)
        assert artifact.projection_hash == selector.build(view, problem, 7).projection_hash
        body = cast(dict[str, JsonValue], artifact.content)
        assert body["projection_only"] is True
        assert "solutions" not in body and "recommendation" not in body
        if definition.kind is RepresentationKind.SCENARIO_TREE:
            assert body["unknowns"] and body["scenarios"] == []
        if definition.kind is RepresentationKind.PARETO_OBJECTIVE_MATRIX:
            assert body["pareto_front"] is None


class StubRuntime:
    def __init__(self, proposal: RepresentationAdjudicationOutput | None) -> None:
        self.proposal = proposal
        self.inputs: list[JsonValue] = []

    async def execute(self, **kwargs: object) -> SemanticExecution:
        self.inputs.append(cast(JsonValue, kwargs["canonical_input"]))
        return SemanticExecution(proposal=self.proposal, cause="BUDGET_UNAVAILABLE")


def tied_problem() -> ProblemSpec:
    problem, _, _ = context()
    return problem.model_copy(
        update={
            "relations": (
                ProblemRelation(source_id="a", target_id="b", kind=ProblemRelationKind.DEPENDS_ON),
                ProblemRelation(source_id="a", target_id="b", kind=ProblemRelationKind.CAUSES),
            )
        }
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "proposal",
    [
        None,
        RepresentationAdjudicationOutput(selected_kinds=("ILP",), explanation="invalid"),
    ],
)
def test_adjudication_refusal_or_invalidity_preserves_deterministic_plan(
    proposal: RepresentationAdjudicationOutput | None,
) -> None:
    _, signature, budget = context()
    selector = RepresentationSelector()
    policy = RepresentationSelectionPolicy(model_adjudication_enabled=True)
    expected = selector.select(tied_problem(), signature, budget, policy=policy)
    runtime = StubRuntime(proposal)
    actual = asyncio.run(
        selector.select_with_adjudication(
            tied_problem(),
            signature,
            budget,
            runtime=cast(SemanticModelRuntime, runtime),
            run_id=UUID(int=1),
            policy=policy,
        )
    )
    assert actual == expected and len(runtime.inputs) == 1


@pytest.mark.unit
def test_adjudication_is_disabled_by_default_and_valid_enabled_reorder_is_bounded() -> None:
    _, signature, budget = context()
    selector = RepresentationSelector()
    deterministic = selector.select(tied_problem(), signature, budget)
    proposal = RepresentationAdjudicationOutput(
        selected_kinds=tuple(view.kind.value for view in reversed(deterministic.views)),
        explanation="prefer causal view",
    )
    assert (
        selector.apply_adjudication(
            deterministic,
            proposal,
            allowed_kinds=frozenset(view.kind for view in deterministic.views),
            view_limit=2,
        )
        == deterministic
    )
    runtime = StubRuntime(proposal)
    enabled = RepresentationSelectionPolicy(model_adjudication_enabled=True)
    adjudicated = asyncio.run(
        selector.select_with_adjudication(
            tied_problem(),
            signature,
            budget,
            runtime=cast(SemanticModelRuntime, runtime),
            run_id=UUID(int=1),
            policy=enabled,
        )
    )
    assert adjudicated.views[0].kind is deterministic.views[-1].kind
    unavailable = asyncio.run(
        selector.select_with_adjudication(
            tied_problem(), signature, budget, runtime=None, run_id=UUID(int=1), policy=enabled
        )
    )
    assert unavailable == selector.select(tied_problem(), signature, budget, policy=enabled)
