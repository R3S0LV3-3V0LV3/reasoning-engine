"""Task ingestion and classification contracts."""

from enum import StrEnum
from uuid import UUID

from pydantic import Field, model_validator

from fre.domain.common import ArtifactRef, FrozenModel, OutputContract, PermissionSet
from fre.domain.semantic import SourceAnchor


class TaskType(StrEnum):
    ANALYSIS = "ANALYSIS"
    DECISION = "DECISION"
    DESIGN = "DESIGN"
    DIAGNOSIS = "DIAGNOSIS"
    RESEARCH = "RESEARCH"


class Ordinal4(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SearchSpaceClass(StrEnum):
    CLOSED = "CLOSED"
    BOUNDED = "BOUNDED"
    OPEN = "OPEN"


class HorizonClass(StrEnum):
    IMMEDIATE = "IMMEDIATE"
    SHORT = "SHORT"
    LONG = "LONG"


class OutputForm(StrEnum):
    TEXT = "TEXT"
    STRUCTURED = "STRUCTURED"
    ARTIFACT = "ARTIFACT"


class TaskEnvelope(FrozenModel):
    task_id: UUID
    text: str
    attachments: tuple[ArtifactRef, ...] = ()
    explicit_constraints: tuple[str, ...] = ()
    requested_output: OutputContract
    user_metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    execution_permissions: PermissionSet


class TaskSignature(FrozenModel):
    task_type: TaskType
    consequence: Ordinal4
    irreversibility: Ordinal4
    ambiguity: Ordinal4
    search_space: SearchSpaceClass
    evidence_scarcity: Ordinal4
    horizon: HorizonClass
    output_form: OutputForm
    dimension_confidence: dict[str, float | None]
    evidence_refs: tuple[str, ...] = ()


class ClassificationDimensionResult(FrozenModel):
    """Complete per-axis classification provenance.

    Every one of the 8 classification axes (``task_type``, ``consequence``,
    ``irreversibility``, ``ambiguity``, ``evidence_scarcity``,
    ``search_space``, ``horizon``, ``output_form``) carries this full
    provenance record: a validated estimate (``estimated``), the value
    actually used downstream after floors/escalation (``effective``), a
    confidence/uncertainty treatment (``confidence``), a conservative bound
    (``conservative_upper``), a rationale, resolvable support
    (``source_anchors``) or an explicit deterministic-policy basis
    (``basis``/``policy_version``), and -- when ``effective`` differs from
    ``estimated`` -- the audited reason for that deviation
    (``override_basis``).

    C05 remediation (finding #5): not all 8 axes currently *influence*
    budget or routing, even though every one of them carries this
    provenance. `fre.modules.m02_budget.BudgetAllocator.allocate` only reads
    ``consequence``, ``irreversibility``, ``ambiguity``, ``evidence_scarcity``,
    and ``search_space`` when computing the reasoning tier. ``task_type``,
    ``horizon``, and ``output_form`` are recorded with the same full
    provenance discipline (for audit, future routing, and forward
    compatibility) but do not yet feed any budget or routing decision. Wiring
    a real, justified budget/routing effect for ``horizon``/``task_type`` is
    tracked as future work, not done in this pass, to avoid changing M02's
    tier semantics as a side effect of a classification-provenance fix.
    """

    estimated: str
    effective: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    conservative_upper: str | None = None
    source_anchors: tuple[SourceAnchor, ...] = ()
    basis: str
    rationale: str | None = None
    override_basis: str | None = None
    policy_version: str | None = None


class FloorOverrideRecord(FrozenModel):
    """Audit trail entry for a deterministic floor/escalation changing a dimension."""

    axis: str
    reason: str
    policy_version: str
    policy_hash: str
    previous_floor: str
    new_floor: str
    # C05 remediation (finding #13, documentation-only): at the only
    # construction site today (`m01_classifier.py`'s `record_override`,
    # called from `dimension()`/`categorical_dimension()`), `approving_rule`
    # is always set to exactly the same value as `reason`. It is
    # deliberately kept as a distinct field rather than collapsed into
    # `reason` (or derived from it) -- a future caller could set the two
    # independently (e.g. `reason` as a free-text audit note,
    # `approving_rule` as a stable machine-matchable rule identifier from a
    # fixed vocabulary) without a schema change. Note the field is *not*
    # cross-checked against `reason` (or anything else) by
    # `ClassificationRecord._floor_overrides_are_exhaustive` below.
    # `FloorOverrideRecord` is embedded in `ClassificationRecord.floor_overrides`,
    # part of persisted event payloads referenced by golden fixtures
    # (`tests/fixtures/golden/*.json`) and `m01_classifier.py`'s diagnostic
    # string (`f"{override.axis}:{override.reason}"`) -- removing or
    # collapsing this field would be a persisted-schema change requiring
    # fixture regeneration, and is explicitly out of scope for this cleanup
    # pass (tier-5, non-blocking).
    approving_rule: str


class ClassificationRecord(FrozenModel):
    policy_version: str
    policy_hash: str
    mode: str
    fallback_used: bool
    dimensions: dict[str, ClassificationDimensionResult]
    model_call_key: str | None = None
    diagnostics: tuple[str, ...] = ()
    floor_overrides: tuple[FloorOverrideRecord, ...] = ()

    @model_validator(mode="after")
    def _floor_overrides_are_exhaustive(self) -> "ClassificationRecord":
        deviated = {
            name
            for name, dimension in self.dimensions.items()
            if dimension.effective != dimension.estimated
        }
        audited = {record.axis for record in self.floor_overrides}
        if deviated != audited:
            raise ValueError(
                "every dimension whose effective value differs from its estimate must "
                "carry a matching, audited FloorOverrideRecord (unaudited floor change)"
            )
        # C05 remediation (finding #4): axis-name presence alone is not
        # sufficient -- a `FloorOverrideRecord` for the right axis but with a
        # fabricated `previous_floor`/`new_floor` pair (not matching the
        # dimension's own `estimated`/`effective` values) would previously
        # pass this validator untouched. Every audited record's before/after
        # values must match the dimension's own recorded values exactly.
        by_axis = {record.axis: record for record in self.floor_overrides}
        for name in deviated:
            dimension = self.dimensions[name]
            override = by_axis[name]
            if override.previous_floor != dimension.estimated:
                raise ValueError(
                    f"{name}: FloorOverrideRecord.previous_floor ({override.previous_floor}) "
                    f"does not match the dimension's estimated value ({dimension.estimated})"
                )
            if override.new_floor != dimension.effective:
                raise ValueError(
                    f"{name}: FloorOverrideRecord.new_floor ({override.new_floor}) does not "
                    f"match the dimension's effective value ({dimension.effective})"
                )
        return self


class ClassificationBlocked(ValueError):
    """M01's quality-gate failure: a material classification dimension failed
    a structural integrity check (missing rationale, no resolvable support,
    an ill-ordered conservative bound, or an anchor irrelevant to its axis).

    C05 remediation (finding #6): this is the specific, documented,
    catchable exception type `TaskClassifier.classify()`/`canonical_events()`
    raise for exactly this failure mode. Every internal quality-gate check in
    `m01_classifier.py` raises this type (never a bare `ValueError` or an
    unrelated exception), so a caller can reliably catch `ClassificationBlocked`
    specifically to distinguish "the proposal failed M01's quality gate" from
    other, unrelated failures, and decide how to recover (e.g. request a
    repaired proposal, fall back to a deterministic-only classification, or
    surface a diagnosable error to the run) instead of letting it crash the
    run as an unhandled exception. See
    `tests/unit/test_c05_remediation.py::test_classification_blocked_can_be_caught_and_handled_by_a_caller`
    for a worked example of a caller doing exactly this. A full event-based
    redesign (emitting an in-band "classification blocked" event the way
    M03's `ProblemBlockerRecorded` does for its own quality gate) remains
    future work -- out of scope for this pass, since it would change what a
    run's persisted event history looks like for every classification
    failure, not just fix this exception's catchability.
    """
