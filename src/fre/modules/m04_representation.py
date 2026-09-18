"""M04 deterministic-first representation selection and structural builders."""

from collections.abc import Callable
from typing import NamedTuple

from fre.domain.budget import BudgetPlan
from fre.domain.common import ArtifactRef, JsonValue, bind_hash, canonical_hash, canonical_json
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
from fre.domain.representation_registry import (
    ORPHAN_KIND_REASONS,
    REGISTRY_VERSION_V2,
    SCORE_WEIGHTS,
    SELECTION_POLICY_VERSION_V2,
    RepresentationDefinition,
    RepresentationSelectionPolicy,
    default_registry,
    default_registry_v2,
    representation_determinism_hash,
    score_details,
)
from fre.domain.task import TaskSignature
from fre.prompts.schemas import RepresentationAdjudicationOutput
from fre.semantic_runtime import SemanticModelRuntime

# `ORPHAN_KIND_REASONS`, `REGISTRY_VERSION_V2`, `SELECTION_POLICY_VERSION_V2`,
# `RepresentationDefinition`, `RepresentationSelectionPolicy`,
# `default_registry`, `default_registry_v2` are all re-exported, unchanged in
# behaviour, from `fre.domain.representation_registry` -- see that module's
# docstring for why the pure registry/scoring logic lives there now (breaking
# the import cycle `fre.runtime.reducer` would otherwise hit trying to
# independently recompute the same values, per the C07 remediation root-cause
# fix). Every existing `from fre.modules.m04_representation import
# default_registry_v2` (etc.) elsewhere in this codebase keeps working
# unchanged.
__all__ = [
    "ORPHAN_KIND_REASONS",
    "REGISTRY_VERSION_V2",
    "SELECTION_POLICY_VERSION_V2",
    "RepresentationDefinition",
    "RepresentationSelectionPolicy",
    "RepresentationSelector",
    "default_registry",
    "default_registry_v2",
]

# Kept as a module-level alias for any external reader of the (formerly
# private) weights table; scoring itself always goes through
# `fre.domain.representation_registry.score_details`.
_SCORE_WEIGHTS = SCORE_WEIGHTS


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
        """Thin, backward-compatible wrapper.

        The actual scoring logic now lives in
        `fre.domain.representation_registry.score_details` (a pure function
        with no `fre.semantic_runtime`/`fre.engine` dependency), so
        `RunReducer.apply` can call the exact same implementation to
        independently recompute a claimed plan's `candidate_scores` (finding
        B, C07 remediation) rather than a second, hand-copied reimplementation
        living only here. This method is kept so any existing caller of
        `RepresentationSelector._score_details`/`._score` is unaffected.
        """
        return score_details(definition, problem)

    def _resolve_actual_builder(
        self, actual: RepresentationKind, registry: tuple[RepresentationDefinition, ...]
    ) -> tuple[str, str]:
        """Look up the builder id/version that a fallback kind is ACTUALLY registered under.

        Used only by the legacy v1 `build()` surface below. Falls back to the
        module-level `default_registry()` (which always carries
        `TEXT_TABLE_FALLBACK`) if the caller's own registry happens not to
        declare it, so a builder identity is always resolvable for the one
        kind `build()` can substitute in -- this is long-standing, tested,
        decode-compatible v1 behaviour
        (`test_registry_builder_availability_produces_typed_fallback`,
        `test_fallback_records_actual_builder_not_requested_builder`
        construct a bare, single-definition ad-hoc registry specifically to
        exercise it) and is left unchanged here.

        Finding J (C07 remediation) is about the BOUND v2 surface
        (`build_bound`), not this one: a v2 artifact's builder identity must
        never be silently borrowed from the unrelated, unversioned legacy
        registry just because the caller's own v2-style registry happens to
        omit `TEXT_TABLE_FALLBACK` -- that failure must be loud, not silently
        patched over from a different registry generation. See
        `_resolve_actual_builder_bound` below, which `build_bound` uses
        instead of this method.
        """
        for definition in (*registry, *default_registry()):
            if definition.kind is actual:
                return definition.builder_id, definition.builder_version
        raise ValueError(f"no registered builder identity for fallback kind {actual}")

    @staticmethod
    def _resolve_actual_builder_bound(
        actual: RepresentationKind, registry: tuple[RepresentationDefinition, ...]
    ) -> tuple[str, str]:
        """Bound (v2) counterpart of `_resolve_actual_builder` (finding J).

        Looks up `actual`'s builder identity ONLY within the registry the
        caller actually supplied (`default_registry_v2()` by default, whose
        own `default_registry_v2` construction now unconditionally raises
        (finding K) if it fails to account for the full `RepresentationKind`
        vocabulary, so it can never itself be missing `TEXT_TABLE_FALLBACK`).
        Never silently falls through to the legacy, unversioned
        `default_registry()` the way the v1-only `_resolve_actual_builder`
        does: a v2 artifact whose registry genuinely lacks a builder identity
        for the fallback kind raises a clear, immediate error instead of
        quietly borrowing an unrelated registry generation's identity for it.
        """
        for definition in registry:
            if definition.kind is actual:
                return definition.builder_id, definition.builder_version
        raise ValueError(
            f"bound (v2) registry does not declare a builder identity for the fallback kind "
            f"{actual}; a v2 registry must declare every kind it can fall back to explicitly, "
            "and this method never borrows an identity from the unrelated legacy v1 registry"
        )

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
        # Finding M (C07 remediation): rather than hand-maintaining a second
        # `fields: dict[str, object]` mirror of `RepresentationPlanV2`'s own
        # fields just to compute `plan_hash` (a copy that could silently drift
        # from the constructor call below, or from `RunReducer.apply`'s own
        # independent re-derivation, the moment a field was added to one but
        # not the others), the placeholder-then-`bind_hash` idiom already used
        # by `ContextPacket.packet_hash` is used here instead: construct the
        # real model once with a placeholder seal, hash `model_dump(mode=
        # "json", exclude={"plan_hash"})` of THAT SAME OBJECT via the single
        # shared `bind_hash` helper, then `model_copy` the real value in. This
        # is the exact preimage `RunReducer.apply` re-derives, computed by the
        # exact same function, not a hand-copied mirror of it.
        provisional = RepresentationPlanV2(
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
            plan_hash="0" * 64,
        )
        plan_hash = bind_hash(provisional, exclude={"plan_hash"})
        return provisional.model_copy(update={"plan_hash": plan_hash})

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
        # Finding M: same shared `bind_hash` idiom as `select_bound` -- no
        # second hand-maintained field-list mirror.
        provisional = deterministic.model_copy(
            update={
                "views": views,
                "selection_basis": selection_basis,
                "adjudication_record_ref": adjudication_record_ref,
                "plan_hash": "0" * 64,
            }
        )
        plan_hash = bind_hash(provisional, exclude={"plan_hash"})
        return outcome(provisional.model_copy(update={"plan_hash": plan_hash}), None)

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
        # for `actual`, never assumed to equal the requested one. Finding J:
        # the BOUND resolver never silently borrows the unrelated legacy v1
        # registry's identity -- see `_resolve_actual_builder_bound`.
        actual_builder_id, actual_builder_version = (
            (view.builder_ref, view.builder_version)
            if available
            else self._resolve_actual_builder_bound(actual, registry)
        )
        determinism_hash = representation_determinism_hash(
            registry_hash=plan.registry_hash,
            actual_kind=actual,
            problem_spec_hash=plan.problem_spec_hash,
            actual_builder_id=actual_builder_id,
            actual_builder_version=actual_builder_version,
            content_hash=content_hash,
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
