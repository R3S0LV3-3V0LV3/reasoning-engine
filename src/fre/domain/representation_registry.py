"""Pure M04 representation registry and scoring logic.

This module exists to break an import cycle exposed by the C07 remediation
root-cause fix: `RunReducer.apply` (in `fre.runtime.reducer`) must
independently recompute the real registry hash and the real candidate scores
to verify a claimed `RepresentationPlanSelectedV2`/`RepresentationArtifactCompiledV2`
rather than trusting the plan/artifact's own self-reported fields (see the
long comment on those branches in `reducer.py`). The natural home for that
logic, `fre.modules.m04_representation`, cannot be imported from
`fre.runtime.reducer`: `m04_representation` imports `fre.semantic_runtime`,
which imports `fre.engine`, which imports `fre.runtime.reducer` -- a direct
cycle.

Everything here is pure domain logic with no dependency on
`fre.semantic_runtime`/`fre.engine`/`fre.runtime.*`, so both
`fre.modules.m04_representation` (which re-exports every name below for
backward compatibility -- every existing import of e.g. `default_registry_v2`
from `fre.modules.m04_representation` keeps working unchanged) and
`fre.runtime.reducer` (which imports directly from here) can depend on it
without forming a cycle.
"""

from pydantic import Field

from fre.domain.common import FrozenModel, canonical_hash
from fre.domain.problem import ProblemRelationKind, ProblemSpec
from fre.domain.representation import (
    RepresentationKind,
    RepresentationScoreComponent,
)

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
    # kind" (`SCORE_WEIGHTS`, resolved in `score_details`) -- preserving the
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
SCORE_WEIGHTS: dict[RepresentationKind, float] = {
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
            score_weight=SCORE_WEIGHTS.get(kind, 0.0),
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
    # Finding K (C07 remediation): this was a bare `assert`, which `python -O`
    # strips entirely -- silently disabling the one check that guarantees
    # every `RepresentationKind` member is accounted for (either live or a
    # documented orphan) every time this function runs. A real, unconditional
    # exception cannot be compiled away.
    if frozenset(live_kinds) | frozenset(orphan_kinds) != frozenset(RepresentationKind):
        raise ValueError(
            "default_registry_v2 does not account for the full RepresentationKind vocabulary: "
            "every kind must be either a live, buildable candidate or a documented "
            "ORPHAN_KIND_REASONS entry"
        )
    definitions = [
        RepresentationDefinition(
            kind=kind,
            builder_id=f"m04.{kind.value.lower()}",
            priority=index,
            purpose=f"Structural {kind.value.lower()} view",
            limitations=("Projection only; does not solve or infer truth.",),
            score_weight=SCORE_WEIGHTS.get(kind, 0.0),
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


def score_details(
    definition: RepresentationDefinition, problem: ProblemSpec
) -> tuple[float, tuple[RepresentationScoreComponent, ...]]:
    """Score one candidate against `problem`.

    Which structural feature signals which kind remains code (domain
    logic that a plain declared number cannot express), but the
    magnitude each trigger is worth is read from
    `definition.score_weight` -- a declared, versioned registry value --
    rather than a literal buried in this function (the previously
    "hardcoded, not declared" gap). `default_registry`/`default_registry_v2`
    populate `score_weight` from the single shared `SCORE_WEIGHTS`
    table, so both registries score identically for every kind they both
    carry.

    This is the single scoring implementation used by
    `RepresentationSelector.select`/`select_bound` (via a thin wrapper kept
    for backward compatibility in `fre.modules.m04_representation`) AND by
    `RunReducer.apply`'s independent recomputation of a claimed plan's
    `candidate_scores` (finding B, C07 remediation) -- both call this exact
    function, so there is no second, independently-drifting reimplementation
    of "how a candidate is scored" anywhere in the codebase.
    """
    kind = definition.kind
    weight = (
        definition.score_weight
        if definition.score_weight is not None
        else SCORE_WEIGHTS.get(kind, 0.0)
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
    if kind is RepresentationKind.DEPENDENCY_DAG and ProblemRelationKind.DEPENDS_ON in relations:
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


def representation_determinism_hash(
    *,
    registry_hash: str,
    actual_kind: RepresentationKind,
    problem_spec_hash: str,
    actual_builder_id: str,
    actual_builder_version: str,
    content_hash: str,
) -> str:
    """The one preimage computation for `RepresentationArtifactV2.determinism_hash`.

    Finding M (C07 remediation): before this function existed,
    `RepresentationSelector.build_bound` and `RunReducer.apply`'s independent
    verification each hand-built their own identical-looking dict literal of
    these same six keys to feed `canonical_hash` -- two copies that could
    silently drift the moment a field was added to one but not the other.
    Both call sites now call this one function instead.
    """
    return canonical_hash(
        {
            "registry_hash": registry_hash,
            "actual_kind": actual_kind,
            "problem_spec_hash": problem_spec_hash,
            "actual_builder_id": actual_builder_id,
            "actual_builder_version": actual_builder_version,
            "content_hash": content_hash,
        }
    )
