"""M04 deterministic-first representation selection and structural builders."""

from collections.abc import Callable

from pydantic import Field

from fre.domain.budget import BudgetPlan
from fre.domain.common import FrozenModel, JsonValue, canonical_hash
from fre.domain.problem import ProblemRelationKind, ProblemSpec
from fre.domain.representation import (
    RepresentationArtifact,
    RepresentationKind,
    RepresentationPlan,
    RepresentationScoreComponent,
    RepresentationView,
)
from fre.domain.task import TaskSignature
from fre.prompts.schemas import RepresentationAdjudicationOutput
from fre.semantic_runtime import SemanticModelRuntime


class RepresentationDefinition(FrozenModel):
    kind: RepresentationKind
    builder_id: str
    builder_version: str = "1.0"
    priority: int = Field(ge=0)
    purpose: str
    limitations: tuple[str, ...]
    builder_available: bool = True


class RepresentationSelectionPolicy(FrozenModel):
    version: str = "wave3-m04/1.0"
    registry_version: str = "wave3-m04-registry/1.0"
    tie_band: float = Field(default=0.05, ge=0, le=1)
    minimum_compatibility: float = Field(default=0.20, ge=0, le=1)
    model_adjudication_enabled: bool = False


def default_registry() -> tuple[RepresentationDefinition, ...]:
    kinds = (
        RepresentationKind.TYPED_CONSTRAINT_SET,
        RepresentationKind.DECISION_TABLE,
        RepresentationKind.PARETO_OBJECTIVE_MATRIX,
        RepresentationKind.DEPENDENCY_DAG,
        RepresentationKind.CAUSAL_GRAPH,
        RepresentationKind.STATE_MACHINE,
        RepresentationKind.EVENT_LOG,
        RepresentationKind.PROPERTY_GRAPH,
        RepresentationKind.HYPERGRAPH,
        RepresentationKind.SCENARIO_TREE,
        RepresentationKind.MORPHOLOGICAL_SPACE,
        RepresentationKind.TEXT_TABLE_FALLBACK,
    )
    return tuple(
        RepresentationDefinition(
            kind=kind,
            builder_id=f"m04.{kind.value.lower()}",
            priority=index,
            purpose=f"Structural {kind.value.lower()} view",
            limitations=("Projection only; does not solve or infer truth.",),
        )
        for index, kind in enumerate(kinds)
    )


class RepresentationSelector:
    def __init__(
        self,
        policy: RepresentationSelectionPolicy | None = None,
        registry: tuple[RepresentationDefinition, ...] | None = None,
    ) -> None:
        self.policy = policy or RepresentationSelectionPolicy()
        self.registry = registry or default_registry()

    @staticmethod
    def apply_adjudication(
        deterministic: RepresentationPlan,
        proposal: RepresentationAdjudicationOutput | None,
        *,
        allowed_kinds: frozenset[RepresentationKind],
        view_limit: int,
        policy: RepresentationSelectionPolicy | None = None,
    ) -> RepresentationPlan:
        """Accept only a bounded reorder/subset of deterministic compatible views."""
        policy = policy or RepresentationSelectionPolicy()
        if not policy.model_adjudication_enabled or proposal is None:
            return deterministic
        try:
            selected = tuple(RepresentationKind(item) for item in proposal.selected_kinds)
        except ValueError:
            return deterministic
        if (
            not selected
            or len(selected) > view_limit
            or len(selected) != len(set(selected))
            or any(kind not in allowed_kinds for kind in selected)
        ):
            return deterministic
        by_kind = {view.kind: view for view in deterministic.views}
        if any(kind not in by_kind for kind in selected):
            return deterministic
        views = tuple(
            by_kind[kind].model_copy(update={"role": "PRIMARY" if index == 0 else "AUXILIARY"})
            for index, kind in enumerate(selected)
        )
        return deterministic.model_copy(
            update={
                "views": views,
                "selection_basis": (*deterministic.selection_basis, "bounded model adjudication"),
            }
        )

    async def select_with_adjudication(
        self,
        problem: ProblemSpec,
        signature: TaskSignature,
        budget: BudgetPlan,
        *,
        runtime: SemanticModelRuntime | None,
        run_id: object,
        registry: tuple[RepresentationDefinition, ...] | None = None,
        policy: RepresentationSelectionPolicy | None = None,
    ) -> RepresentationPlan:
        """Select deterministically, then optionally adjudicate a qualified tie.

        Runtime refusal, invalid output, and unavailability all leave the deterministic
        result untouched.  Only the already selected, buildable, budget-bounded tie
        candidates are disclosed to the semantic runtime.
        """
        policy = policy or self.policy
        deterministic = self.select(problem, signature, budget, registry or self.registry, policy)
        candidates = tuple(view for view in deterministic.views if view.builder_available)
        ambiguous = (
            len(candidates) > 1
            and abs(candidates[0].compatibility_score - candidates[1].compatibility_score)
            <= policy.tie_band
        )
        if not policy.model_adjudication_enabled or not ambiguous or runtime is None:
            return deterministic
        execution = await runtime.execute(
            run_id=run_id,
            module_id="M04",
            module_version="1.0",
            operation="adjudicate",
            prompt_id="m04.adjudicate",
            prompt_version="1.0",
            canonical_input={
                "problem_spec_hash": canonical_hash(problem),
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
                "view_limit": budget.search.max_representation_views,
            },
        )
        proposal = (
            execution.proposal
            if isinstance(execution.proposal, RepresentationAdjudicationOutput)
            else None
        )
        return self.apply_adjudication(
            deterministic,
            proposal,
            allowed_kinds=frozenset(view.kind for view in candidates),
            view_limit=budget.search.max_representation_views,
            policy=policy,
        )

    def select(
        self,
        problem: ProblemSpec,
        signature: TaskSignature,
        budget: BudgetPlan,
        registry: tuple[RepresentationDefinition, ...] | None = None,
        policy: RepresentationSelectionPolicy | None = None,
    ) -> RepresentationPlan:
        del signature
        registry = registry or self.registry
        policy = policy or self.policy
        scores = [
            (definition, *self._score_details(definition.kind, problem)) for definition in registry
        ]
        scores.sort(key=lambda item: (-item[1], item[0].priority, item[0].kind.value))
        compatible = [item for item in scores if item[1] >= policy.minimum_compatibility]
        if not compatible:
            fallback = next(
                (item for item in scores if item[0].kind is RepresentationKind.TEXT_TABLE_FALLBACK),
                None,
            )
            if fallback is None:
                definition = next(
                    item
                    for item in default_registry()
                    if item.kind is RepresentationKind.TEXT_TABLE_FALLBACK
                )
                score, components = self._score_details(definition.kind, problem)
                fallback = (definition, score, components)
            compatible = [fallback]
        limit = budget.search.max_representation_views
        if limit < 1:
            return RepresentationPlan(
                problem_spec_hash=canonical_hash(problem),
                views=(),
                selection_basis=(policy.version,),
                omitted_reasons=("view budget is zero",),
            )
        selected = compatible[:1]
        if (
            limit >= 2
            and len(compatible) > 1
            and abs(compatible[0][1] - compatible[1][1]) <= policy.tie_band
        ):
            selected.append(compatible[1])
        views = tuple(
            RepresentationView(
                id=f"view-{index + 1}",
                kind=definition.kind,
                role="PRIMARY" if index == 0 else "AUXILIARY",
                compatibility_score=score,
                score_components=components,
                purpose=definition.purpose,
                expected_value=definition.purpose,
                builder_ref=definition.builder_id,
                builder_available=definition.builder_available,
                builder_version=definition.builder_version,
                registry_version=policy.registry_version,
                selection_policy_version=policy.version,
                limitations=definition.limitations,
            )
            for index, (definition, score, components) in enumerate(selected)
        )
        omitted: tuple[str, ...] = ()
        if (
            limit == 1
            and len(compatible) > 1
            and abs(compatible[0][1] - compatible[1][1]) <= policy.tie_band
        ):
            omitted = (f"{compatible[1][0].kind}: omitted by one-view budget",)
        return RepresentationPlan(
            problem_spec_hash=canonical_hash(problem),
            views=views,
            selection_basis=(
                policy.version,
                policy.registry_version,
                "deterministic compatibility scoring",
            ),
            omitted_reasons=omitted,
        )

    @staticmethod
    def _score(kind: RepresentationKind, problem: ProblemSpec) -> float:
        return RepresentationSelector._score_details(kind, problem)[0]

    @staticmethod
    def _score_details(
        kind: RepresentationKind, problem: ProblemSpec
    ) -> tuple[float, tuple[RepresentationScoreComponent, ...]]:
        relations = {relation.kind for relation in problem.relations}
        score = 0.0
        feature = ""
        basis = ""
        if kind is RepresentationKind.TYPED_CONSTRAINT_SET and problem.constraints:
            score += 0.65
            feature, basis = "constraints", "one or more typed constraints"
        if kind is RepresentationKind.DECISION_TABLE and problem.decision_variables:
            score += 0.45
            feature, basis = "decision_variables", "one or more decision variables"
        if kind is RepresentationKind.PARETO_OBJECTIVE_MATRIX and len(problem.objectives) > 1:
            score += 0.70
            feature, basis = "objectives", "multiple objectives"
        if (
            kind is RepresentationKind.DEPENDENCY_DAG
            and ProblemRelationKind.DEPENDS_ON in relations
        ):
            score += 0.70
            feature, basis = "dependency_relations", "DEPENDS_ON relation present"
        if kind is RepresentationKind.CAUSAL_GRAPH and ProblemRelationKind.CAUSES in relations:
            score += 0.70
            feature, basis = "causal_relations", "CAUSES relation present"
        if (
            kind in {RepresentationKind.EVENT_LOG, RepresentationKind.STATE_MACHINE}
            and ProblemRelationKind.TEMPORALLY_PRECEDES in relations
        ):
            score += 0.65
            feature, basis = "temporal_relations", "TEMPORALLY_PRECEDES relation present"
        if kind is RepresentationKind.SCENARIO_TREE and problem.unknowns:
            score += 0.45
            feature, basis = "unknowns", "one or more unresolved unknowns"
        if kind is RepresentationKind.TEXT_TABLE_FALLBACK:
            score = 0.20
            feature, basis = "fallback", "baseline typed fallback"
        score = min(score, 1.0)
        components = (
            (RepresentationScoreComponent(feature=feature, contribution=score, basis=basis),)
            if score
            else ()
        )
        return score, components

    def build(
        self,
        view: RepresentationView,
        problem: ProblemSpec,
        snapshot_version: int,
    ) -> RepresentationArtifact:
        supported = view.kind in _BUILDERS
        available = view.builder_available and supported
        actual = view.kind if available else RepresentationKind.TEXT_TABLE_FALLBACK
        content = _BUILDERS[actual](problem)
        problem_hash = canonical_hash(problem)
        preimage = {
            "requested_kind": view.kind,
            "actual_kind": actual,
            "builder_id": view.builder_ref,
            "builder_version": view.builder_version,
            "registry_version": view.registry_version,
            "selection_policy_version": view.selection_policy_version,
            "problem_spec_hash": problem_hash,
            "source_snapshot_version": snapshot_version,
            "content": content,
        }
        return RepresentationArtifact(
            requested_kind=view.kind,
            actual_kind=actual,
            builder_id=view.builder_ref,
            builder_version=view.builder_version,
            registry_version=view.registry_version,
            selection_policy_version=view.selection_policy_version,
            problem_spec_hash=problem_hash,
            source_snapshot_version=snapshot_version,
            content=content,
            projection_hash=canonical_hash(preimage),
            fallback_reason=None if available else "requested builder unavailable or unsupported",
            limitations=view.limitations
            if available
            else (*view.limitations, f"requested {view.kind.value}; emitted typed fallback"),
        )

    @staticmethod
    def is_stale(artifact: RepresentationArtifact, problem: ProblemSpec) -> bool:
        return artifact.problem_spec_hash != canonical_hash(problem)


def _base(problem: ProblemSpec, kind: RepresentationKind) -> dict[str, JsonValue]:
    return {"kind": kind.value, "projection_only": True}


def _typed_constraints(problem: ProblemSpec) -> JsonValue:
    return {
        **_base(problem, RepresentationKind.TYPED_CONSTRAINT_SET),
        "constraints": [item.model_dump(mode="json") for item in problem.constraints],
        "acceptance_criteria": [
            item.model_dump(mode="json") for item in problem.acceptance_criteria
        ],
    }


def _objectives(problem: ProblemSpec) -> JsonValue:
    return {
        **_base(problem, RepresentationKind.PARETO_OBJECTIVE_MATRIX),
        "objectives": [item.model_dump(mode="json") for item in problem.objectives],
        "pareto_front": None,
    }


def _relations(problem: ProblemSpec, kind: RepresentationKind) -> JsonValue:
    allowed = {
        RepresentationKind.DEPENDENCY_DAG: {ProblemRelationKind.DEPENDS_ON},
        RepresentationKind.CAUSAL_GRAPH: {ProblemRelationKind.CAUSES},
        RepresentationKind.STATE_MACHINE: {ProblemRelationKind.TEMPORALLY_PRECEDES},
        RepresentationKind.EVENT_LOG: {ProblemRelationKind.TEMPORALLY_PRECEDES},
        RepresentationKind.PROPERTY_GRAPH: {
            ProblemRelationKind.DEPENDS_ON,
            ProblemRelationKind.CAUSES,
            ProblemRelationKind.INTERACTS_WITH,
        },
        RepresentationKind.HYPERGRAPH: {ProblemRelationKind.INTERACTS_WITH},
    }[kind]
    return {
        **_base(problem, kind),
        "relations": [
            item.model_dump(mode="json") for item in problem.relations if item.kind in allowed
        ],
    }


def _scenario_tree(problem: ProblemSpec) -> JsonValue:
    return {
        **_base(problem, RepresentationKind.SCENARIO_TREE),
        "unknowns": [item.model_dump(mode="json") for item in problem.unknowns],
        "scenarios": [],
    }


def _simple(problem: ProblemSpec, kind: RepresentationKind, field: str) -> JsonValue:
    values: list[JsonValue]
    if field == "variables":
        values = [item.model_dump(mode="json") for item in problem.decision_variables]
    else:
        values = [item.model_dump(mode="json") for item in problem.unknowns]
    return {**_base(problem, kind), field: values}


def _fallback(problem: ProblemSpec) -> JsonValue:
    return {
        **_base(problem, RepresentationKind.TEXT_TABLE_FALLBACK),
        "objectives": [item.model_dump(mode="json") for item in problem.objectives],
        "constraints": [item.model_dump(mode="json") for item in problem.constraints],
        "variables": [item.model_dump(mode="json") for item in problem.decision_variables],
        "unknowns": [item.model_dump(mode="json") for item in problem.unknowns],
        "relations": [item.model_dump(mode="json") for item in problem.relations],
    }


_BUILDERS: dict[RepresentationKind, Callable[[ProblemSpec], JsonValue]] = {
    RepresentationKind.TYPED_CONSTRAINT_SET: _typed_constraints,
    RepresentationKind.DECISION_TABLE: lambda p: _simple(
        p, RepresentationKind.DECISION_TABLE, "variables"
    ),
    RepresentationKind.PARETO_OBJECTIVE_MATRIX: _objectives,
    RepresentationKind.DEPENDENCY_DAG: lambda p: _relations(p, RepresentationKind.DEPENDENCY_DAG),
    RepresentationKind.CAUSAL_GRAPH: lambda p: _relations(p, RepresentationKind.CAUSAL_GRAPH),
    RepresentationKind.STATE_MACHINE: lambda p: _relations(p, RepresentationKind.STATE_MACHINE),
    RepresentationKind.EVENT_LOG: lambda p: _relations(p, RepresentationKind.EVENT_LOG),
    RepresentationKind.PROPERTY_GRAPH: lambda p: _relations(p, RepresentationKind.PROPERTY_GRAPH),
    RepresentationKind.HYPERGRAPH: lambda p: _relations(p, RepresentationKind.HYPERGRAPH),
    RepresentationKind.SCENARIO_TREE: _scenario_tree,
    RepresentationKind.MORPHOLOGICAL_SPACE: lambda p: _simple(
        p, RepresentationKind.MORPHOLOGICAL_SPACE, "unknowns"
    ),
    RepresentationKind.TEXT_TABLE_FALLBACK: _fallback,
}
