"""Representation projection contracts."""

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, SerializerFunctionWrapHandler, model_serializer, model_validator

from fre.domain.common import ArtifactRef, FrozenModel, JsonValue, ObjectRef


class RepresentationKind(StrEnum):
    TYPED_CONSTRAINT_SET = "TYPED_CONSTRAINT_SET"
    DECISION_TABLE = "DECISION_TABLE"
    PARETO_OBJECTIVE_MATRIX = "PARETO_OBJECTIVE_MATRIX"
    DEPENDENCY_DAG = "DEPENDENCY_DAG"
    CAUSAL_GRAPH = "CAUSAL_GRAPH"
    STATE_MACHINE = "STATE_MACHINE"
    EVENT_LOG = "EVENT_LOG"
    KNOWLEDGE_GRAPH = "KNOWLEDGE_GRAPH"
    CAUSAL_DAG = "CAUSAL_DAG"
    CSP = "CSP"
    ILP = "ILP"
    SCENARIO_TREE = "SCENARIO_TREE"
    HYPERGRAPH = "HYPERGRAPH"
    PROPERTY_GRAPH = "PROPERTY_GRAPH"
    MORPHOLOGICAL_SPACE = "MORPHOLOGICAL_SPACE"
    TEXT_FALLBACK = "TEXT_FALLBACK"
    TEXT_TABLE_FALLBACK = "TEXT_TABLE_FALLBACK"


class RepresentationScoreComponent(FrozenModel):
    feature: str
    contribution: float
    basis: str


class RepresentationView(FrozenModel):
    id: str
    kind: RepresentationKind
    role: Literal["PRIMARY", "AUXILIARY"]
    compatibility_score: float
    score_components: tuple[RepresentationScoreComponent, ...] = ()
    purpose: str = "structural projection"
    expected_value: str
    builder_ref: str
    builder_available: bool = True
    source_object_refs: tuple[ObjectRef, ...] = ()
    builder_version: str = "1.0"
    registry_version: str = "wave3-m04-registry/1.0"
    selection_policy_version: str = "wave3-m04/1.0"
    limitations: tuple[str, ...] = ()

    @model_serializer(mode="wrap")
    def serialize_compatibly(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Keep the original 1.0 wire shape for available builders.

        ``builder_available`` was added after representation views had already
        been persisted.  Its default therefore must not appear in canonical
        state JSON, or validating an old snapshot would change its hash.  The
        non-default ``False`` value remains explicit and auditable.
        """
        payload: dict[str, Any] = handler(self)
        if self.builder_available:
            payload.pop("builder_available", None)
        return payload


class RepresentationPlan(FrozenModel):
    # Wave 3's original persisted 1.0 shape did not contain this binding.  Keep
    # accepting that shape so old events and snapshots remain replayable; all
    # newly selected plans supply the hash.
    problem_spec_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    views: tuple[RepresentationView, ...]
    selection_basis: tuple[str, ...] = ()
    omitted_reasons: tuple[str, ...] = ()

    @model_serializer(mode="plain")
    def serialize_compatibly(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "views": self.views,
            "selection_basis": self.selection_basis,
            "omitted_reasons": self.omitted_reasons,
        }
        if self.problem_spec_hash is not None:
            payload["problem_spec_hash"] = self.problem_spec_hash
        return payload


class RepresentationArtifact(FrozenModel):
    requested_kind: RepresentationKind
    actual_kind: RepresentationKind
    # F10 fix: prior to this phase these two fields were populated from the
    # REQUESTED view unconditionally, even when `actual_kind != requested_kind`
    # (i.e. a fallback ran) -- silently attributing fallback content to a
    # builder that never executed. They now always describe the builder that
    # actually produced `content`. `requested_builder_id`/`requested_builder_
    # version` below are new, optional (decode-safe for pre-fix snapshots,
    # which have no way to recover the true value) fields carrying what was
    # originally asked for, so the two identities are never conflated again.
    builder_id: str
    builder_version: str
    requested_builder_id: str | None = None
    requested_builder_version: str | None = None
    registry_version: str
    selection_policy_version: str
    problem_spec_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_snapshot_version: int = Field(ge=0)
    content: JsonValue
    projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_artifact_ref: ArtifactRef | None = None
    fallback_reason: str | None = None
    limitations: tuple[str, ...] = ()


class RepresentationCandidateScore(FrozenModel):
    """One declared candidate's fully-scored outcome, kept for reproducibility audits."""

    kind: RepresentationKind
    compatibility_score: float = Field(ge=0, le=1)
    score_components: tuple[RepresentationScoreComponent, ...] = ()
    builder_available: bool
    cost: float = Field(ge=0)
    expected_benefit: float = Field(ge=0)


class RepresentationPlanV2(FrozenModel):
    """Bound, reproducible v2 selection plan (Objective 2, Phase 6/C07).

    Unlike v1 `RepresentationPlan`, every field needed to reproduce this exact
    selection from the declared registry and exact inputs is present and
    hash-bound: `registry_hash` pins the exact candidate registry consulted,
    `input_hash` pins the exact (problem, signature, budget) triple scored,
    and `plan_hash` is a self-referential seal (computed over every other
    field, exactly like `ContextPacket.packet_hash`) that `RunReducer.apply`
    re-derives and rejects on mismatch. `tie_band`/`tie_triggered` preserve
    the pre-existing 0.05 boundary semantics explicitly rather than leaving
    them implicit in code that could silently drift across the three call
    sites that consult it (`RepresentationSelector.select`, its
    `omitted_reasons` branch, and `select_with_adjudication`'s ambiguity
    gate).
    """

    registry_version: str
    registry_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_policy_version: str
    problem_spec_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_snapshot_version: int = Field(ge=0)
    candidate_scores: tuple[RepresentationCandidateScore, ...]
    views: tuple[RepresentationView, ...]
    selection_basis: tuple[str, ...] = ()
    omitted_reasons: tuple[str, ...] = ()
    tie_band: float = Field(ge=0, le=1)
    tie_triggered: bool
    fallback_used: bool
    # Set only when a semantic adjudication genuinely ran inside the tie band;
    # binds to that call's own `idempotency_key`. `RunReducer.apply` rejects a
    # plan that sets this without a matching, already-applied
    # `SemanticModelCallRecord(V2)` in state -- see Objective 3.
    adjudication_record_ref: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # Self-referential seal over every other field, exactly like
    # `ContextPacket.packet_hash`: computed as `canonical_hash(self.model_dump(
    # exclude={"plan_hash"}))`. It cannot be checked by a `model_validator` on
    # this class itself (the seal necessarily excludes itself, so validating
    # it at construction time would require already knowing the field's own
    # final value -- exactly the chicken-and-egg problem `ContextPacket`
    # resolves the same way: construct with a placeholder, hash the rest,
    # `model_copy(update=...)` in the real value). `RunReducer.apply`
    # independently recomputes and enforces this seal before persistence,
    # mirroring `ContextCompiled`'s `expected_packet_hash` check.
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class RepresentationArtifactV2(FrozenModel):
    """Bound v2 artifact (Objectives 1 and 4, Phase 6/C07).

    Binds to: the exact bound plan (`plan_hash`) and registry
    (`registry_hash`/`registry_version`) that selected it, the exact
    `ProblemSpec`/run-state revision it was built against
    (`problem_spec_hash`/`source_snapshot_version`), REQUESTED and ACTUAL
    builder identity (the F10 fix, made structurally explicit -- both are
    required, not optional, on this bound type), and its actual stored bytes
    (`physical_artifact_ref`, `content_hash`). `content_hash` is the
    caller-asserted sha256 of the canonical content bytes; `RunReducer.apply`
    independently recomputes it from the bytes actually retrievable through
    `physical_artifact_ref.sha256` and rejects the event outright on any
    disagreement -- the computed hash is trusted, never the caller's claim.
    `determinism_hash` binds `content_hash` to the declared inputs that must
    deterministically reproduce it (`registry_hash`, `actual_kind`,
    `problem_spec_hash`, `actual_builder_id`, `actual_builder_version`),
    supporting the "selection/build is reproducible" invariant independent of
    the reducer's own bytes check.
    """

    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    registry_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    registry_version: str
    selection_policy_version: str
    problem_spec_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_snapshot_version: int = Field(ge=0)
    requested_kind: RepresentationKind
    actual_kind: RepresentationKind
    requested_builder_id: str = Field(min_length=1)
    requested_builder_version: str = Field(min_length=1)
    actual_builder_id: str = Field(min_length=1)
    actual_builder_version: str = Field(min_length=1)
    content_media_type: str = "application/json"
    physical_artifact_ref: ArtifactRef
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    determinism_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_result: str = "PROJECTION_ONLY_NO_SOLVE"
    fallback_reason: str | None = None
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def fallback_attribution_is_consistent(self) -> "RepresentationArtifactV2":
        fell_back = self.actual_kind != self.requested_kind
        if fell_back and self.fallback_reason is None:
            raise ValueError("a substituted representation kind requires a fallback_reason")
        if not fell_back and self.fallback_reason is not None:
            raise ValueError("fallback_reason is only valid when the actual kind was substituted")
        if fell_back and (
            self.actual_builder_id == self.requested_builder_id
            and self.actual_builder_version == self.requested_builder_version
        ):
            # This is exactly the F10 exploit shape: content was produced by a
            # different builder than the one requested, yet builder identity
            # was left unchanged -- attributing fallback content to a builder
            # that never ran it.
            raise ValueError(
                "requested and actual builder identity must differ when the representation "
                "kind was substituted by a fallback"
            )
        if not fell_back and (
            self.actual_builder_id != self.requested_builder_id
            or self.actual_builder_version != self.requested_builder_version
        ):
            raise ValueError(
                "requested and actual builder identity must match when no fallback occurred"
            )
        return self
