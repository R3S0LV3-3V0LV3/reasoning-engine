"""M04 deterministic-first representation selection and structural builders."""

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
    @staticmethod
    def apply_adjudication(
        deterministic: RepresentationPlan,
        proposal: RepresentationAdjudicationOutput | None,
        *,
        allowed_kinds: frozenset[RepresentationKind],
        view_limit: int,
    ) -> RepresentationPlan:
        """Accept only a bounded reorder/subset of deterministic compatible views."""
        if proposal is None:
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

    def select(
        self,
        problem: ProblemSpec,
        signature: TaskSignature,
        budget: BudgetPlan,
        registry: tuple[RepresentationDefinition, ...] | None = None,
        policy: RepresentationSelectionPolicy | None = None,
    ) -> RepresentationPlan:
        del signature
        registry = registry or default_registry()
        policy = policy or RepresentationSelectionPolicy()
        scores = [(definition, self._score(definition.kind, problem)) for definition in registry]
        scores.sort(key=lambda item: (-item[1], item[0].priority, item[0].kind.value))
        compatible = [item for item in scores if item[1] >= policy.minimum_compatibility]
        if not compatible:
            compatible = [
                next(
                    item
                    for item in scores
                    if item[0].kind is RepresentationKind.TEXT_TABLE_FALLBACK
                )
            ]
        limit = budget.search.max_representation_views
        if limit < 1:
            return RepresentationPlan(
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
                score_components=(
                    RepresentationScoreComponent(
                        feature="problem_structure",
                        contribution=score,
                        basis="wave3-m04 deterministic rules",
                    ),
                ),
                purpose=definition.purpose,
                expected_value=definition.purpose,
                builder_ref=definition.builder_id,
                builder_version=definition.builder_version,
                registry_version=policy.registry_version,
                selection_policy_version=policy.version,
                limitations=definition.limitations,
            )
            for index, (definition, score) in enumerate(selected)
        )
        omitted: tuple[str, ...] = ()
        if (
            limit == 1
            and len(compatible) > 1
            and abs(compatible[0][1] - compatible[1][1]) <= policy.tie_band
        ):
            omitted = (f"{compatible[1][0].kind}: omitted by one-view budget",)
        return RepresentationPlan(
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
        relations = {relation.kind for relation in problem.relations}
        score = 0.20
        if kind is RepresentationKind.TYPED_CONSTRAINT_SET and problem.constraints:
            score += 0.65
        if kind is RepresentationKind.DECISION_TABLE and problem.decision_variables:
            score += 0.45
        if kind is RepresentationKind.PARETO_OBJECTIVE_MATRIX and len(problem.objectives) > 1:
            score += 0.70
        if (
            kind is RepresentationKind.DEPENDENCY_DAG
            and ProblemRelationKind.DEPENDS_ON in relations
        ):
            score += 0.70
        if kind is RepresentationKind.CAUSAL_GRAPH and ProblemRelationKind.CAUSES in relations:
            score += 0.70
        if (
            kind in {RepresentationKind.EVENT_LOG, RepresentationKind.STATE_MACHINE}
            and ProblemRelationKind.TEMPORALLY_PRECEDES in relations
        ):
            score += 0.65
        if kind is RepresentationKind.SCENARIO_TREE and problem.unknowns:
            score += 0.45
        if kind is RepresentationKind.TEXT_TABLE_FALLBACK:
            score = 0.20
        return min(score, 1.0)

    def build(
        self,
        view: RepresentationView,
        problem: ProblemSpec,
        snapshot_version: int,
        *,
        builder_available: bool = True,
    ) -> RepresentationArtifact:
        actual = view.kind if builder_available else RepresentationKind.TEXT_TABLE_FALLBACK
        content: JsonValue = {
            "kind": actual.value,
            "objectives": [item.id for item in problem.objectives],
            "constraints": [item.id for item in problem.constraints],
            "variables": [item.id for item in problem.decision_variables],
            "relations": [relation.model_dump(mode="json") for relation in problem.relations],
        }
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
            fallback_reason=None if builder_available else "specialized builder unavailable",
            limitations=view.limitations,
        )

    @staticmethod
    def is_stale(artifact: RepresentationArtifact, problem: ProblemSpec) -> bool:
        return artifact.problem_spec_hash != canonical_hash(problem)
