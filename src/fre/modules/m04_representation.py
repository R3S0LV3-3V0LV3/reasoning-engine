"""M04 deterministic-first representation selection and structural builders."""

from collections.abc import Callable
from typing import NamedTuple

from pydantic import Field

from fre.domain.budget import BudgetPlan
from fre.domain.common import ArtifactRef, FrozenModel, JsonValue, canonical_hash, canonical_json
from fre.domain.problem import ProblemRelationKind, ProblemSpec
from fre.domain.representation import (
    RepresentationArtifact,
    RepresentationArtifactV2,
    RepresentationCandidateScore,
    RepresentationKind,
    RepresentationPlan,
    RepresentationPlanV2,
    RepresentationScoreComponent,
    RepresentationView,
)
from fre.domain.task import TaskSignature
from fre.prompts.schemas import RepresentationAdjudicationOutput
from fre.semantic_runtime import SemanticModelRuntime

# F11 remediation: `RepresentationKind` declares 17 members but the legacy v1
# registry below (`default_registry`, unchanged for decode/replay parity with
# already-selected v1 plans and the golden `composition.py`/m12 fixtures that
# pin its exact version strings) only ever declared 12. The five orphans were
# in the vocabulary but in neither the registry nor `_BUILDERS`: an event
# hand-built with one of these kinds would silently decode (the enum accepts
# it) yet never appear as a selectable or buildable candidate anywhere, with
# no record of that being intentional.
#
# Choice made here (documented, not an oversight): explicitly DE-REGISTER all
# five in the new versioned registry (`default_registry_v2`, Objective 1)
# rather than implement fake builders for them, because implementing them for
# real would mean either (a) genuine constraint-solving/graph-inference
# machinery (`CSP`, `ILP`, `KNOWLEDGE_GRAPH`, `CAUSAL_DAG`) that this module's
# own contract explicitly forbids -- M04 is a deterministic-first STRUCTURAL
# projector that "does not solve or infer truth"
# (`test_all_registered_builders_are_deterministic_and_do_not_run_wave4_algorithms`)
# -- or (b) a redundant alias (`TEXT_FALLBACK` duplicates `TEXT_TABLE_FALLBACK`,
# the one fallback kind that IS implemented). A honest, versioned "not
# supported by this registry, for this reason" entry is smaller and more
# truthful than a hollow builder that returns an empty stub just to claim
# coverage.
ORPHAN_KIND_REASONS: dict[RepresentationKind, str] = {
    RepresentationKind.KNOWLEDGE_GRAPH: (
        "requires knowledge-graph inference machinery outside M04's projection-only contract"
    ),
    RepresentationKind.CAUSAL_DAG: (
        "duplicates CAUSAL_GRAPH's structural projection; no distinct DAG-solving builder exists "
        "or is in scope for M04"
    ),
    RepresentationKind.CSP: (
        "constraint-solving is explicitly out of scope for M04's deterministic-first projection "
        "contract (see the Wave 4 solving boundary)"
    ),
    RepresentationKind.ILP: (
        "integer-linear-programming solving is explicitly out of scope for M04's projection-only "
        "contract (see the Wave 4 solving boundary)"
    ),
    RepresentationKind.TEXT_FALLBACK: (
        "legacy alias superseded by TEXT_TABLE_FALLBACK, the one implemented fallback kind"
    ),
}

REGISTRY_VERSION_V2 = "wave3-m04-registry/2.0"
SELECTION_POLICY_VERSION_V2 = "wave3-m04/2.0"


class RepresentationDefinition(FrozenModel):
    kind: RepresentationKind
    builder_id: str
    builder_version: str = "1.0"
    priority: int = Field(ge=0)
    purpose: str
    limitations: tuple[str, ...]
    builder_available: bool = True
    # Objective 1: declared, versioned candidate contract fields. All are
    # additive with safe defaults so the legacy v1 registry/tests (which never
    # set them) are unaffected.
    supported_operations: tuple[str, ...] = ("PROJECT",)
    hard_preconditions: tuple[str, ...] = ()
    incompatible_with: tuple[RepresentationKind, ...] = ()
    # Declared replacement for the inline numeric literals previously buried
    # in `_score_details` ("Scoring is hardcoded, not declared"): the maximum
    # compatibility contribution this kind earns when its structural trigger
    # condition (still code -- which structural feature signals which kind is
    # domain logic, not a number) is met. `0.0` for a kind with no declared
    # trigger (e.g. an orphan, or a kind that has never had one, such as
    # PROPERTY_GRAPH/HYPERGRAPH/MORPHOLOGICAL_SPACE in the legacy registry)
    # means it can never outscore a real candidate.
    # `None` (the default) means "use the standard declared weight for this
    # kind" (`_SCORE_WEIGHTS`, resolved in `_score_details`) -- preserving the
    # exact pre-C07 hardcoded behaviour for any `RepresentationDefinition`
    # built without explicitly overriding it (including every ad-hoc
    # definition existing tests already construct). Passing an explicit
    # value overrides that default, e.g. to pin an exact tie-band boundary
    # in a test.
    score_weight: float | None = Field(default=None, ge=0, le=1)
    cost_weight: float = Field(default=1.0, ge=0)
    expected_benefit_weight: float = Field(default=1.0, ge=0)
    fallback_kind: RepresentationKind = RepresentationKind.TEXT_TABLE_FALLBACK
    unavailable_reason: str | None = None


class RepresentationSelectionPolicy(FrozenModel):
    version: str = "wave3-m04/1.0"
    registry_version: str = "wave3-m04-registry/1.0"
    tie_band: float = Field(default=0.05, ge=0, le=1)
    minimum_compatibility: float = Field(default=0.20, ge=0, le=1)
    model_adjudication_enabled: bool = False


# Per-kind declared score weight, preserving the exact numeric literals the
# pre-C07 `_score_details` had hardcoded inline, now the single declared
# source both the legacy v1 registry and the new v2 registry read from --
# neither hand-rolls its own copy of these numbers.
_SCORE_WEIGHTS: dict[RepresentationKind, float] = {
    RepresentationKind.TYPED_CONSTRAINT_SET: 0.65,
    RepresentationKind.DECISION_TABLE: 0.45,
    RepresentationKind.PARETO_OBJECTIVE_MATRIX: 0.70,
    RepresentationKind.DEPENDENCY_DAG: 0.70,
    RepresentationKind.CAUSAL_GRAPH: 0.70,
    RepresentationKind.STATE_MACHINE: 0.65,
    RepresentationKind.EVENT_LOG: 0.65,
    RepresentationKind.SCENARIO_TREE: 0.45,
    RepresentationKind.TEXT_TABLE_FALLBACK: 0.20,
}


def default_registry() -> tuple[RepresentationDefinition, ...]:
    """Legacy v1 registry -- unchanged.

    Left exactly as it was (still 12 kinds, still the exact version strings
    `composition.py` and the m12 golden fixtures pin) so every already-
    selected v1 `RepresentationPlan`/`RepresentationArtifact` remains
    decode/replay-compatible. All new selection goes through
    `default_registry_v2` instead (Objective 1/5).
    """
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
            score_weight=_SCORE_WEIGHTS.get(kind, 0.0),
        )
        for index, kind in enumerate(kinds)
    )


def default_registry_v2() -> tuple[RepresentationDefinition, ...]:
    """Versioned v2 registry declaring the FULL `RepresentationKind` vocabulary.

    Every kind in `RepresentationKind` is either a real, buildable candidate
    (identical builder wiring to `default_registry`, for the twelve legacy
    kinds) or an explicitly de-registered orphan (`builder_available=False`,
    `unavailable_reason` set, `score_weight=0.0` so it can never be selected)
    -- see `ORPHAN_KIND_REASONS` above for the per-kind rationale (F11).
    """
    live_kinds = (
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
    orphan_kinds = tuple(ORPHAN_KIND_REASONS)
    assert frozenset(live_kinds) | frozenset(orphan_kinds) == frozenset(RepresentationKind)
    definitions = [
        RepresentationDefinition(
            kind=kind,
            builder_id=f"m04.{kind.value.lower()}",
            priority=index,
            purpose=f"Structural {kind.value.lower()} view",
            limitations=("Projection only; does not solve or infer truth.",),
            score_weight=_SCORE_WEIGHTS.get(kind, 0.0),
        )
        for index, kind in enumerate(live_kinds)
    ]
    definitions.extend(
        RepresentationDefinition(
            kind=kind,
            builder_id=f"m04.{kind.value.lower()}",
            priority=len(live_kinds) + index,
            purpose=f"Declared but de-registered {kind.value.lower()} candidate",
            limitations=("Not implemented by M04; see unavailable_reason.",),
            builder_available=False,
            score_weight=0.0,
            unavailable_reason=ORPHAN_KIND_REASONS[kind],
        )
        for index, kind in enumerate(orphan_kinds)
    )
    return tuple(definitions)


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
            (definition, *self._score_details(definition, problem)) for definition in registry
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
                score, components = self._score_details(definition, problem)
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
    def _score(definition: RepresentationDefinition, problem: ProblemSpec) -> float:
        return RepresentationSelector._score_details(definition, problem)[0]

    @staticmethod
    def _score_details(
        definition: RepresentationDefinition, problem: ProblemSpec
    ) -> tuple[float, tuple[RepresentationScoreComponent, ...]]:
        """Score one candidate against `problem`.

        Which structural feature signals which kind remains code (domain
        logic that a plain declared number cannot express), but the
        magnitude each trigger is worth is read from
        `definition.score_weight` -- a declared, versioned registry value --
        rather than a literal buried in this function (the previously
        "hardcoded, not declared" gap). `default_registry`/`default_registry_v2`
        populate `score_weight` from the single shared `_SCORE_WEIGHTS`
        table, so both registries score identically for every kind they both
        carry.
        """
        kind = definition.kind
        weight = (
            definition.score_weight
            if definition.score_weight is not None
            else _SCORE_WEIGHTS.get(kind, 0.0)
        )
        relations = {relation.kind for relation in problem.relations}
        score = 0.0
        feature = ""
        basis = ""
        if kind is RepresentationKind.TYPED_CONSTRAINT_SET and problem.constraints:
            score += weight
            feature, basis = "constraints", "one or more typed constraints"
        if kind is RepresentationKind.DECISION_TABLE and problem.decision_variables:
            score += weight
            feature, basis = "decision_variables", "one or more decision variables"
        if kind is RepresentationKind.PARETO_OBJECTIVE_MATRIX and len(problem.objectives) > 1:
            score += weight
            feature, basis = "objectives", "multiple objectives"
        if (
            kind is RepresentationKind.DEPENDENCY_DAG
            and ProblemRelationKind.DEPENDS_ON in relations
        ):
            score += weight
            feature, basis = "dependency_relations", "DEPENDS_ON relation present"
        if kind is RepresentationKind.CAUSAL_GRAPH and ProblemRelationKind.CAUSES in relations:
            score += weight
            feature, basis = "causal_relations", "CAUSES relation present"
        if (
            kind in {RepresentationKind.EVENT_LOG, RepresentationKind.STATE_MACHINE}
            and ProblemRelationKind.TEMPORALLY_PRECEDES in relations
        ):
            score += weight
            feature, basis = "temporal_relations", "TEMPORALLY_PRECEDES relation present"
        if kind is RepresentationKind.SCENARIO_TREE and problem.unknowns:
            score += weight
            feature, basis = "unknowns", "one or more unresolved unknowns"
        if kind is RepresentationKind.TEXT_TABLE_FALLBACK:
            score = weight
            feature, basis = "fallback", "baseline typed fallback"
        score = min(score, 1.0)
        components = (
            (RepresentationScoreComponent(feature=feature, contribution=score, basis=basis),)
            if score
            else ()
        )
        return score, components

    def _resolve_actual_builder(
        self, actual: RepresentationKind, registry: tuple[RepresentationDefinition, ...]
    ) -> tuple[str, str]:
        """Look up the builder id/version that a fallback kind is ACTUALLY registered under.

        Falls back to the module-level `default_registry()` (which always
        carries `TEXT_TABLE_FALLBACK`) if the caller's own registry happens
        not to declare it, so a builder identity is always resolvable for the
        one kind `build()` can substitute in.
        """
        for definition in (*registry, *default_registry()):
            if definition.kind is actual:
                return definition.builder_id, definition.builder_version
        raise ValueError(f"no registered builder identity for fallback kind {actual}")

    def build(
        self,
        view: RepresentationView,
        problem: ProblemSpec,
        snapshot_version: int,
        registry: tuple[RepresentationDefinition, ...] | None = None,
    ) -> RepresentationArtifact:
        registry = registry or self.registry
        supported = view.kind in _BUILDERS
        available = view.builder_available and supported
        actual = view.kind if available else RepresentationKind.TEXT_TABLE_FALLBACK
        content = _BUILDERS[actual](problem)
        problem_hash = canonical_hash(problem)
        # F10 fix: when `available` is False, `content` above was produced by
        # the FALLBACK builder (`_BUILDERS[TEXT_TABLE_FALLBACK]`), never by
        # `view.builder_ref`/`view.builder_version` (the REQUESTED builder).
        # Before this fix, `builder_id`/`builder_version` below were always
        # set from the requested view regardless of `available`, silently
        # attributing fallback content to a builder that never ran --
        # `actual_kind`/`fallback_reason` were the only signals a caller could
        # use to even notice the substitution. Both the actual builder
        # identity (resolved from the registry for `actual`, not merely
        # assumed) and the originally requested identity are now always
        # recorded, distinctly.
        actual_builder_id, actual_builder_version = (
            (view.builder_ref, view.builder_version)
            if available
            else self._resolve_actual_builder(actual, registry)
        )
        preimage = {
            "requested_kind": view.kind,
            "actual_kind": actual,
            "builder_id": actual_builder_id,
            "builder_version": actual_builder_version,
            "requested_builder_id": view.builder_ref,
            "requested_builder_version": view.builder_version,
            "registry_version": view.registry_version,
            "selection_policy_version": view.selection_policy_version,
            "problem_spec_hash": problem_hash,
            "source_snapshot_version": snapshot_version,
            "content": content,
        }
        return RepresentationArtifact(
            requested_kind=view.kind,
            actual_kind=actual,
            builder_id=actual_builder_id,
            builder_version=actual_builder_version,
            requested_builder_id=view.builder_ref,
            requested_builder_version=view.builder_version,
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

    # ------------------------------------------------------------------
    # v2 bound API (Objectives 1-5, Phase 6/C07). New APIs emit bound v2
    # plans/artifacts only; the legacy `select`/`build` above remain the
    # decode-compatible v1 surface (still exercised by every pre-existing
    # test in this module) and are otherwise untouched except for the F10
    # attribution fix inside `build` itself, which needed no new wire shape.
    # ------------------------------------------------------------------

    def select_bound(
        self,
        problem: ProblemSpec,
        signature: TaskSignature,
        budget: BudgetPlan,
        source_snapshot_version: int,
        registry: tuple[RepresentationDefinition, ...] | None = None,
        policy: RepresentationSelectionPolicy | None = None,
        *,
        adjudication_record_ref: str | None = None,
    ) -> RepresentationPlanV2:
        """Deterministic, reproducible, hash-bound selection (Objective 2).

        Reproducible from a declared registry and exact inputs: `registry_hash`
        seals the exact candidate registry consulted (`default_registry_v2()`
        by default -- the F11-remediated, complete vocabulary), `input_hash`
        seals the exact `(problem, signature, budget)` triple scored, and
        `plan_hash` is a self-referential seal over every other field that
        `RunReducer.apply` independently re-derives and rejects on mismatch.
        `tie_band`/`tie_triggered` make the pre-existing 0.05 boundary
        semantics an explicit, declared decision on the plan itself, rather
        than something a caller can only infer by re-running the comparison.
        """
        registry = registry or default_registry_v2()
        policy = policy or self.policy
        scored = [
            (definition, *self._score_details(definition, problem)) for definition in registry
        ]
        ordered = sorted(scored, key=lambda item: (-item[1], item[0].priority, item[0].kind.value))
        compatible = [item for item in ordered if item[1] >= policy.minimum_compatibility]
        fallback_used = False
        if not compatible:
            fallback_used = True
            fallback_definition = next(
                (
                    item[0]
                    for item in ordered
                    if item[0].kind is RepresentationKind.TEXT_TABLE_FALLBACK
                ),
                None,
            )
            if fallback_definition is None:
                fallback_definition = next(
                    item
                    for item in default_registry_v2()
                    if item.kind is RepresentationKind.TEXT_TABLE_FALLBACK
                )
            score, components = self._score_details(fallback_definition, problem)
            compatible = [(fallback_definition, score, components)]
        limit = budget.search.max_representation_views
        tie_triggered = (
            len(compatible) > 1 and abs(compatible[0][1] - compatible[1][1]) <= policy.tie_band
        )
        selected = compatible[:1] if limit >= 1 else []
        if limit >= 2 and tie_triggered:
            selected.append(compatible[1])
            if selected[-1][0].kind is RepresentationKind.TEXT_TABLE_FALLBACK:
                fallback_used = True
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
                registry_version=REGISTRY_VERSION_V2,
                selection_policy_version=SELECTION_POLICY_VERSION_V2,
                limitations=definition.limitations,
            )
            for index, (definition, score, components) in enumerate(selected)
        )
        omitted: tuple[str, ...] = ()
        if limit == 1 and tie_triggered:
            omitted = (f"{compatible[1][0].kind}: omitted by one-view budget",)
        if limit < 1:
            omitted = (*omitted, "view budget is zero")
        candidate_scores = tuple(
            RepresentationCandidateScore(
                kind=definition.kind,
                compatibility_score=score,
                score_components=components,
                builder_available=definition.builder_available,
                cost=definition.cost_weight,
                expected_benefit=round(definition.expected_benefit_weight * score, 6),
            )
            for definition, score, components in ordered
        )
        if adjudication_record_ref is not None and not tie_triggered:
            raise ValueError(
                "adjudication_record_ref may only be set for a plan that is genuinely tied "
                "within its own declared tie_band"
            )
        registry_hash = canonical_hash(registry)
        problem_spec_hash = canonical_hash(problem)
        input_hash = canonical_hash({"problem": problem, "signature": signature, "budget": budget})
        selection_basis = (
            SELECTION_POLICY_VERSION_V2,
            REGISTRY_VERSION_V2,
            "deterministic compatibility scoring",
        )
        # `fields` exists solely to compute `plan_hash` from exactly the same
        # keys/values the constructor call below uses -- see `RunReducer.
        # apply`'s independent re-derivation of this same seal. It is kept
        # loosely typed (never passed to the constructor via `**`) so mypy
        # --strict can check the constructor call itself field-by-field.
        fields: dict[str, object] = {
            "registry_version": REGISTRY_VERSION_V2,
            "registry_hash": registry_hash,
            "selection_policy_version": SELECTION_POLICY_VERSION_V2,
            "problem_spec_hash": problem_spec_hash,
            "input_hash": input_hash,
            "source_snapshot_version": source_snapshot_version,
            "candidate_scores": candidate_scores,
            "views": views,
            "selection_basis": selection_basis,
            "omitted_reasons": omitted,
            "tie_band": policy.tie_band,
            "tie_triggered": tie_triggered,
            "fallback_used": fallback_used,
            "adjudication_record_ref": adjudication_record_ref,
        }
        plan_hash = canonical_hash(fields)
        return RepresentationPlanV2(
            registry_version=REGISTRY_VERSION_V2,
            registry_hash=registry_hash,
            selection_policy_version=SELECTION_POLICY_VERSION_V2,
            problem_spec_hash=problem_spec_hash,
            input_hash=input_hash,
            source_snapshot_version=source_snapshot_version,
            candidate_scores=candidate_scores,
            views=views,
            selection_basis=selection_basis,
            omitted_reasons=omitted,
            tie_band=policy.tie_band,
            tie_triggered=tie_triggered,
            fallback_used=fallback_used,
            adjudication_record_ref=adjudication_record_ref,
            plan_hash=plan_hash,
        )

    class AdjudicationOutcome(NamedTuple):
        plan: RepresentationPlanV2
        # Never a silent swallow (the gap this phase flags in the legacy,
        # untouched `apply_adjudication`): always states exactly why the
        # deterministic plan was kept, or `None` when adjudication genuinely
        # applied.
        diagnostic: str | None

    @staticmethod
    def apply_adjudication_v2(
        deterministic: RepresentationPlanV2,
        proposal: RepresentationAdjudicationOutput | None,
        *,
        allowed_kinds: frozenset[RepresentationKind],
        view_limit: int,
        adjudication_record_ref: str | None,
    ) -> "RepresentationSelector.AdjudicationOutcome":
        outcome = RepresentationSelector.AdjudicationOutcome
        if proposal is None:
            return outcome(deterministic, "NO_PROPOSAL: runtime refusal or unavailability")
        try:
            selected = tuple(RepresentationKind(item) for item in proposal.selected_kinds)
        except ValueError:
            return outcome(
                deterministic, "INVALID_KIND: proposal cited an unknown RepresentationKind"
            )
        if not selected:
            return outcome(deterministic, "EMPTY_SELECTION: proposal selected no kinds")
        if len(selected) > view_limit:
            return outcome(deterministic, "OVER_LIMIT: proposal exceeds the declared view budget")
        if len(selected) != len(set(selected)):
            return outcome(deterministic, "DUPLICATE_KIND: proposal repeats a kind")
        if any(kind not in allowed_kinds for kind in selected):
            return outcome(
                deterministic,
                "UNKNOWN_KIND: proposal cites a kind outside the disclosed candidates",
            )
        by_kind = {view.kind: view for view in deterministic.views}
        if any(kind not in by_kind for kind in selected):
            return outcome(
                deterministic,
                "UNRESOLVED_KIND: proposal cites a kind absent from the deterministic plan's views",
            )
        views = tuple(
            by_kind[kind].model_copy(update={"role": "PRIMARY" if index == 0 else "AUXILIARY"})
            for index, kind in enumerate(selected)
        )
        selection_basis = (*deterministic.selection_basis, "bounded model adjudication")
        fields: dict[str, object] = {
            "registry_version": deterministic.registry_version,
            "registry_hash": deterministic.registry_hash,
            "selection_policy_version": deterministic.selection_policy_version,
            "problem_spec_hash": deterministic.problem_spec_hash,
            "input_hash": deterministic.input_hash,
            "source_snapshot_version": deterministic.source_snapshot_version,
            "candidate_scores": deterministic.candidate_scores,
            "views": views,
            "selection_basis": selection_basis,
            "omitted_reasons": deterministic.omitted_reasons,
            "tie_band": deterministic.tie_band,
            "tie_triggered": deterministic.tie_triggered,
            "fallback_used": deterministic.fallback_used,
            "adjudication_record_ref": adjudication_record_ref,
        }
        plan_hash = canonical_hash(fields)
        return outcome(
            RepresentationPlanV2(
                registry_version=deterministic.registry_version,
                registry_hash=deterministic.registry_hash,
                selection_policy_version=deterministic.selection_policy_version,
                problem_spec_hash=deterministic.problem_spec_hash,
                input_hash=deterministic.input_hash,
                source_snapshot_version=deterministic.source_snapshot_version,
                candidate_scores=deterministic.candidate_scores,
                views=views,
                selection_basis=selection_basis,
                omitted_reasons=deterministic.omitted_reasons,
                tie_band=deterministic.tie_band,
                tie_triggered=deterministic.tie_triggered,
                fallback_used=deterministic.fallback_used,
                adjudication_record_ref=adjudication_record_ref,
                plan_hash=plan_hash,
            ),
            None,
        )

    async def select_with_adjudication_bound(
        self,
        problem: ProblemSpec,
        signature: TaskSignature,
        budget: BudgetPlan,
        source_snapshot_version: int,
        *,
        runtime: SemanticModelRuntime | None,
        run_id: object,
        registry: tuple[RepresentationDefinition, ...] | None = None,
        policy: RepresentationSelectionPolicy | None = None,
    ) -> "RepresentationSelector.AdjudicationOutcome":
        """v2 counterpart of `select_with_adjudication` (Objective 3).

        Semantic adjudication is charged through the C04 `SemanticModelRuntime`
        exactly as the legacy path does -- a real budget reservation is made
        and settled by `runtime.execute`, producing a real, persisted
        `SemanticModelCallRecord` -- and is only ever attempted when the
        deterministic plan is genuinely ambiguous within its own declared
        `tie_band`, never as a floating condition outside that one declared
        policy gate.
        """
        registry = registry or default_registry_v2()
        policy = policy or self.policy
        outcome = RepresentationSelector.AdjudicationOutcome
        deterministic = self.select_bound(
            problem, signature, budget, source_snapshot_version, registry, policy
        )
        candidates = tuple(view for view in deterministic.views if view.builder_available)
        if not policy.model_adjudication_enabled:
            return outcome(
                deterministic, "ADJUDICATION_DISABLED: policy.model_adjudication_enabled is False"
            )
        if not deterministic.tie_triggered:
            return outcome(
                deterministic, "NOT_AMBIGUOUS: deterministic plan is outside the tie band"
            )
        if runtime is None or len(candidates) < 2:
            return outcome(deterministic, "NO_RUNTIME_OR_CANDIDATES: adjudication is not possible")
        execution = await runtime.execute(
            run_id=run_id,
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
                "view_limit": budget.search.max_representation_views,
            },
        )
        proposal = (
            execution.proposal
            if isinstance(execution.proposal, RepresentationAdjudicationOutput)
            else None
        )
        if execution.record is None:
            return outcome(
                deterministic, f"RUNTIME_UNAVAILABLE: {execution.cause or 'no record produced'}"
            )
        return self.apply_adjudication_v2(
            deterministic,
            proposal,
            allowed_kinds=frozenset(view.kind for view in candidates),
            view_limit=budget.search.max_representation_views,
            adjudication_record_ref=execution.record.idempotency_key,
        )

    def build_bound(
        self,
        view: RepresentationView,
        problem: ProblemSpec,
        plan: RepresentationPlanV2,
        artifact_writer: Callable[[bytes], ArtifactRef],
        registry: tuple[RepresentationDefinition, ...] | None = None,
    ) -> RepresentationArtifactV2:
        """v2 counterpart of `build` (Objectives 1, 4, 5; carries the F10 fix).

        `artifact_writer` is the caller's real content-addressed artifact
        store (e.g. a thin adapter over `FrontierReasoningEngine.
        store_artifact`) -- this module stays decoupled from the engine, but
        the artifact's bytes are always genuinely persisted through it, never
        merely described inline as v1's `content: JsonValue` is. `content_hash`
        is computed from those exact bytes; `RunReducer.apply` independently
        re-reads them by their registered sha256 and re-derives this same
        hash before trusting the artifact at all (the decisive "reject a
        forged artifact hash" invariant).
        """
        registry = registry or default_registry_v2()
        supported = view.kind in _BUILDERS
        available = view.builder_available and supported
        actual = view.kind if available else RepresentationKind.TEXT_TABLE_FALLBACK
        content = _BUILDERS[actual](problem)
        content_bytes = canonical_json(content)
        content_hash = canonical_hash(content)
        stored_ref = artifact_writer(content_bytes)
        if stored_ref.sha256 != content_hash:
            raise ValueError(
                "artifact writer stored bytes whose sha256 does not match the computed "
                "content hash; refusing to bind a mismatched artifact"
            )
        # F10 fix (see `build` above for the full exploit-shape explanation):
        # the actual builder identity is always resolved from the registry
        # for `actual`, never assumed to equal the requested one.
        actual_builder_id, actual_builder_version = (
            (view.builder_ref, view.builder_version)
            if available
            else self._resolve_actual_builder(actual, registry)
        )
        determinism_hash = canonical_hash(
            {
                "registry_hash": plan.registry_hash,
                "actual_kind": actual,
                "problem_spec_hash": plan.problem_spec_hash,
                "actual_builder_id": actual_builder_id,
                "actual_builder_version": actual_builder_version,
                "content_hash": content_hash,
            }
        )
        return RepresentationArtifactV2(
            plan_hash=plan.plan_hash,
            registry_hash=plan.registry_hash,
            registry_version=plan.registry_version,
            selection_policy_version=plan.selection_policy_version,
            problem_spec_hash=plan.problem_spec_hash,
            source_snapshot_version=plan.source_snapshot_version,
            requested_kind=view.kind,
            actual_kind=actual,
            requested_builder_id=view.builder_ref,
            requested_builder_version=view.builder_version,
            actual_builder_id=actual_builder_id,
            actual_builder_version=actual_builder_version,
            physical_artifact_ref=stored_ref,
            content_hash=content_hash,
            determinism_hash=determinism_hash,
            fallback_reason=None if available else "requested builder unavailable or unsupported",
            limitations=view.limitations
            if available
            else (*view.limitations, f"requested {view.kind.value}; emitted typed fallback"),
        )


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
