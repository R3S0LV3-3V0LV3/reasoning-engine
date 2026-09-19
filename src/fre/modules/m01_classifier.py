"""M01 deterministic-first task classifier."""

from datetime import datetime
from typing import cast

from pydantic import Field

from fre.domain.budget import BudgetProjection, DeploymentLimits, TierPolicy
from fre.domain.common import FrozenModel, ObjectRef, canonical_hash, canonical_json
from fre.domain.ledger import EpistemicStatus, LedgerNodeType
from fre.domain.semantic import SourceAnchor, SourceKind
from fre.domain.task import (
    ClassificationBlocked,
    ClassificationDimensionResult,
    ClassificationRecord,
    FloorOverrideRecord,
    HorizonClass,
    Ordinal4,
    OutputForm,
    SearchSpaceClass,
    TaskEnvelope,
    TaskSignature,
    TaskType,
)
from fre.modules.m02_budget import BudgetAllocator
from fre.modules.m09_ledger import make_node
from fre.modules.source_anchors import validate_anchor_relevance, validate_source_anchor
from fre.ports.clock import UUIDFactory
from fre.prompts.schemas import (
    ClassificationOutput,
    HorizonProposal,
    SearchSpaceProposal,
    TaskTypeProposal,
)
from fre.runtime.events import (
    BudgetAllocated,
    BudgetRevised,
    EventPayload,
    LedgerNodeAdded,
    TaskClassified,
    TaskPreliminarilyClassified,
)

ORDINAL = tuple(Ordinal4)

# Axes whose ordinal risk increases with index (LOW..CRITICAL == safest..riskiest),
# i.e. a "conservative" bound must sit at or above the point estimate.
_ASCENDING_AXES = frozenset({"consequence", "ambiguity", "evidence_scarcity"})


_OUTPUT_ALIASES = {
    OutputForm.TEXT: frozenset({"text", "plain_text", "prose", "markdown", "md", "report"}),
    OutputForm.STRUCTURED: frozenset(
        {"structured", "json", "yaml", "yml", "table", "tabular", "schema", "schema_bound"}
    ),
    OutputForm.ARTIFACT: frozenset(
        {"artifact", "file", "document", "media", "image", "audio", "video", "binary"}
    ),
}


def normalise_output_form(value: str) -> OutputForm:
    """Conservatively normalize common contracts without changing the original envelope."""
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    for output_form, aliases in _OUTPUT_ALIASES.items():
        if normalized in aliases:
            return output_form
    return OutputForm.ARTIFACT


def _field_anchor(envelope: TaskEnvelope, selector: str) -> SourceAnchor:
    return SourceAnchor(
        source_kind=SourceKind.TASK_FIELD,
        source_ref=ObjectRef(object_type="TaskEnvelope", object_id=str(envelope.task_id)),
        selector=selector,
    )


def _anchor_ref(anchor: SourceAnchor) -> str:
    return canonical_json(anchor).decode("utf-8")


class ClassificationPolicy(FrozenModel):
    version: str = "wave3-m01/1.1"
    confidence_threshold: float = Field(default=0.70, ge=0, le=1)
    low_confidence_floor: Ordinal4 = Ordinal4.HIGH
    fallback_ordinal: Ordinal4 = Ordinal4.CRITICAL
    fallback_search_space: SearchSpaceClass = SearchSpaceClass.OPEN
    fallback_horizon: HorizonClass = HorizonClass.LONG
    # C05 remediation (finding #8): independent low-confidence escalation
    # target for `horizon`, decoupled from `fallback_horizon` (used only when
    # there is no proposal at all). Mirrors how `search_space` already keeps
    # its two fallbacks independent: a hardcoded `SearchSpaceClass.OPEN` for
    # low-confidence escalation vs. `policy.fallback_search_space` for "no
    # proposal". Defaults to the same value as `fallback_horizon` so existing
    # callers/tests that never diverge the two see unchanged behavior.
    low_confidence_horizon_fallback: HorizonClass = HorizonClass.LONG

    @property
    def policy_hash(self) -> str:
        return canonical_hash(self)


def harder(left: Ordinal4, right: Ordinal4) -> Ordinal4:
    return max((left, right), key=ORDINAL.index)


def reversibility_to_irreversibility(value: Ordinal4) -> Ordinal4:
    return ORDINAL[len(ORDINAL) - 1 - ORDINAL.index(value)]


def _floor_consequence(envelope: TaskEnvelope) -> Ordinal4:
    return Ordinal4.HIGH if envelope.execution_permissions.allow_external_writes else Ordinal4.LOW


def _floor_irreversibility(envelope: TaskEnvelope) -> Ordinal4:
    permissions = envelope.execution_permissions
    if permissions.allow_external_writes:
        return Ordinal4.HIGH
    if permissions.allow_network:
        return Ordinal4.MEDIUM
    return Ordinal4.LOW


def _floor_evidence_scarcity(envelope: TaskEnvelope) -> Ordinal4:
    """Deterministic floor for `evidence_scarcity` (C05 remediation, finding #1).

    A task that supplies no resolvable context inputs at all -- no
    attachments, no explicit constraints -- and is not even permitted to
    fetch evidence externally (`allow_network=False`) cannot possibly ground
    its classification in given or fetchable evidence, regardless of how
    confidently a model proposal reports low scarcity. This mirrors
    `_floor_consequence`/`_floor_irreversibility`: a sound, envelope-derived
    signal that a self-reported low value must never fall below.
    """
    has_context_inputs = bool(envelope.attachments) or bool(envelope.explicit_constraints)
    if not has_context_inputs and not envelope.execution_permissions.allow_network:
        return Ordinal4.MEDIUM
    return Ordinal4.LOW


def _floor_ambiguity(envelope: TaskEnvelope) -> Ordinal4:
    """Deterministic floor for `ambiguity` (C05 remediation, finding #1).

    Execution permissions that span two or more distinct external systems
    (arbitrary declared capabilities, network access, and/or external writes,
    counted independently) put multiple unresolved external interactions in
    play that the envelope itself does not disambiguate -- a sound,
    deterministic lower bound on ambiguity that a self-reported low value must
    never fall below.
    """
    permissions = envelope.execution_permissions
    external_systems = len(permissions.capabilities)
    if permissions.allow_network:
        external_systems += 1
    if permissions.allow_external_writes:
        external_systems += 1
    if external_systems >= 2:
        return Ordinal4.MEDIUM
    return Ordinal4.LOW


def _explicit_ordinals(envelope: TaskEnvelope) -> dict[str, Ordinal4]:
    metadata = envelope.user_metadata
    explicit: dict[str, Ordinal4] = {}
    for key in ("consequence", "irreversibility", "ambiguity", "evidence_scarcity"):
        value = metadata.get(key)
        if isinstance(value, str) and value in Ordinal4:
            explicit[key] = Ordinal4(value)
    return explicit


def _require_model_rationale_and_support(
    name: str, rationale: str | None, anchors: tuple[SourceAnchor, ...]
) -> None:
    """Guard shared by `dimension()`/`categorical_dimension()` (C05 remediation, finding #12).

    A material, MODEL-basis classification dimension must carry both a
    rationale and at least one resolvable supporting anchor. The two call
    sites' *gating* condition (when this guard applies at all) genuinely
    differs -- `dimension()` additionally requires `not is_explicit and
    proposed is not None`, `categorical_dimension()` requires only
    `basis == "MODEL"` -- but the guard body itself (these two checks, with
    this exact error message text) was byte-identical duplication. Each call
    site still evaluates its own gating condition before calling this helper.
    """
    if not rationale or not rationale.strip():
        raise ClassificationBlocked(
            f"{name}: material classification dimension is missing a rationale"
        )
    if not anchors:
        raise ClassificationBlocked(
            f"{name}: material classification dimension has no resolvable support"
        )


def _validate_ordinal_bound(
    axis: str, estimate: Ordinal4, upper: Ordinal4, *, ascending: bool
) -> None:
    """Reject an ill-ordered conservative bound, honoring each axis's risk orientation.

    For an ascending-risk axis (higher ordinal == riskier: consequence,
    ambiguity, evidence_scarcity, and irreversibility once derived) the
    conservative bound must sit at or above the estimate. `reversibility`
    itself is descending-risk (higher ordinal == *safer*, i.e. more
    reversible), so its conservative bound must sit at or *below* the
    estimate -- this is the orientation check that must run before
    `reversibility_to_irreversibility` inverts it, per the F05 defect.
    """
    est_idx, up_idx = ORDINAL.index(estimate), ORDINAL.index(upper)
    if ascending and up_idx < est_idx:
        raise ClassificationBlocked(
            f"{axis}: conservative bound ({upper}) is less conservative than the estimate "
            f"({estimate})"
        )
    if not ascending and up_idx > est_idx:
        raise ClassificationBlocked(
            f"{axis}: conservative bound orientation is inverted -- reversibility's "
            f"conservative bound ({upper}) must assume equal or LOWER reversibility than the "
            f"estimate ({estimate}), never higher, because lower reversibility is the riskier "
            "(more irreversible) direction"
        )


class TaskClassifier:
    def __init__(self, policy: ClassificationPolicy | None = None) -> None:
        self.policy = policy or ClassificationPolicy()

    def bootstrap_signature(
        self, envelope: TaskEnvelope, policy: ClassificationPolicy | None = None
    ) -> TaskSignature:
        """Deterministic-only preliminary signature.

        Every axis takes its floor or explicit value -- never the pessimistic
        model-absent (`fallback_ordinal`) fallback `classify()` uses when a
        proposal never materializes. This keeps the bootstrap tier a true
        lower bound on the final, authoritative tier, so M02's later
        `BudgetAllocator.revise()` (which forbids lowering the committed
        tier) can never be violated by this handoff. It is used ONLY to size
        the M02 bootstrap budget and is never itself persisted as a material
        classification -- see `TaskPreliminarilyClassified`.
        """
        policy = policy or self.policy
        explicit = _explicit_ordinals(envelope)
        consequence = explicit.get("consequence", _floor_consequence(envelope))
        irreversibility = explicit.get("irreversibility", _floor_irreversibility(envelope))
        ambiguity = explicit.get("ambiguity", Ordinal4.LOW)
        evidence_scarcity = explicit.get("evidence_scarcity", Ordinal4.LOW)
        output_form = normalise_output_form(envelope.requested_output.form)
        return TaskSignature(
            task_type=TaskType.ANALYSIS,
            consequence=consequence,
            irreversibility=irreversibility,
            ambiguity=ambiguity,
            search_space=SearchSpaceClass.CLOSED,
            evidence_scarcity=evidence_scarcity,
            horizon=HorizonClass.IMMEDIATE,
            output_form=output_form,
            dimension_confidence={},
            evidence_refs=(),
        )

    def classify(
        self,
        envelope: TaskEnvelope,
        proposal: ClassificationOutput | None,
        policy: ClassificationPolicy | None = None,
        *,
        model_call_key: str | None = None,
    ) -> tuple[TaskSignature, ClassificationRecord]:
        policy = policy or self.policy
        available_artifacts = frozenset(attachment.sha256 for attachment in envelope.attachments)
        explicit = _explicit_ordinals(envelope)
        floor_consequence = _floor_consequence(envelope)
        floor_irreversibility = _floor_irreversibility(envelope)
        floor_ambiguity = _floor_ambiguity(envelope)
        floor_evidence_scarcity = _floor_evidence_scarcity(envelope)
        dimensions: dict[str, ClassificationDimensionResult] = {}
        floor_overrides: list[FloorOverrideRecord] = []

        def record_override(name: str, estimate: object, effective: object, reason: str) -> None:
            floor_overrides.append(
                FloorOverrideRecord(
                    axis=name,
                    reason=reason,
                    policy_version=policy.version,
                    policy_hash=policy.policy_hash,
                    previous_floor=str(getattr(estimate, "value", estimate)),
                    new_floor=str(getattr(effective, "value", effective)),
                    approving_rule=reason,
                )
            )

        def dimension(
            name: str,
            proposed: Ordinal4 | None,
            confidence: float | None,
            upper: Ordinal4 | None,
            rationale: str | None,
            floor: Ordinal4 = Ordinal4.LOW,
            proposed_anchors: tuple[SourceAnchor, ...] = (),
            deterministic_anchors: tuple[SourceAnchor, ...] = (),
        ) -> Ordinal4:
            for anchor in proposed_anchors:
                validate_source_anchor(anchor, envelope, available_artifacts)
                validate_anchor_relevance(name, anchor)
            is_explicit = name in explicit
            estimate = explicit.get(name, proposed or policy.fallback_ordinal)
            basis = "EXPLICIT" if is_explicit else "MODEL" if proposal else "POLICY_FALLBACK"
            if basis == "MODEL" and not is_explicit and proposed is not None:
                _require_model_rationale_and_support(name, rationale, proposed_anchors)
                if upper is not None:
                    _validate_ordinal_bound(
                        name,
                        proposed,
                        upper,
                        ascending=name in _ASCENDING_AXES or name == "irreversibility",
                    )
            effective_confidence = None if is_explicit else confidence
            effective_upper = None if is_explicit else upper
            # C05 remediation (finding #15): a single ordered pipeline of
            # three escalation stages (permission floor -> low-confidence
            # escalation -> no-proposal fallback), each computing its own
            # "did this stage actually change the running value" flag exactly
            # once, right where it computes the new value -- instead of the
            # prior shape, where the low-confidence and fallback stages each
            # tracked their own boolean but the floor stage did not, forcing
            # a separate, later re-comparison (`after_floor != estimate`) to
            # recover the same information. Each stage's `applied` flag is
            # reused directly to build `reasons`, with no re-derivation.
            # Preserves the exact fixed stage order and the exact `"+"`-joined
            # `override_basis` string for every input combination -- this
            # value is persisted in event payloads and golden fixtures.
            after_floor = harder(estimate, floor)
            floor_applied = after_floor != estimate

            low_confidence_applies = (
                not is_explicit
                and confidence is not None
                and confidence < policy.confidence_threshold
            )
            after_low_confidence = (
                harder(after_floor, harder(upper or estimate, policy.low_confidence_floor))
                if low_confidence_applies
                else after_floor
            )
            low_confidence_applied = after_low_confidence != after_floor

            fallback_applies = proposal is None and name not in explicit
            after_fallback = (
                harder(after_low_confidence, policy.fallback_ordinal)
                if fallback_applies
                else after_low_confidence
            )
            fallback_applied = after_fallback != after_low_confidence

            effective = after_fallback
            stages = (
                (floor_applied, "permission_floor"),
                (low_confidence_applied, "low_confidence_escalation"),
                (fallback_applied, "no_proposal_fallback"),
            )
            reasons = [label for applied, label in stages if applied]
            override_basis = "+".join(reasons) if reasons else None
            dimensions[name] = ClassificationDimensionResult(
                estimated=estimate.value,
                effective=effective.value,
                confidence=effective_confidence,
                conservative_upper=effective_upper.value if effective_upper else None,
                source_anchors=deterministic_anchors + proposed_anchors,
                basis=basis,
                rationale=rationale if not is_explicit else None,
                override_basis=override_basis,
                policy_version=policy.version,
            )
            if override_basis is not None:
                record_override(name, estimate, effective, override_basis)
            return effective

        def categorical_dimension(
            name: str,
            proposal_obj: TaskTypeProposal | SearchSpaceProposal | HorizonProposal | None,
            default: TaskType | SearchSpaceClass | HorizonClass,
            *,
            escalate_to: TaskType | SearchSpaceClass | HorizonClass | None = None,
        ) -> TaskType | SearchSpaceClass | HorizonClass:
            anchors = proposal_obj.anchors if proposal_obj else ()
            for anchor in anchors:
                validate_source_anchor(anchor, envelope, available_artifacts)
                validate_anchor_relevance(name, anchor)
            estimate = proposal_obj.estimate if proposal_obj else default
            confidence = proposal_obj.confidence if proposal_obj else None
            rationale = proposal_obj.rationale if proposal_obj else None
            basis = "MODEL" if proposal_obj else "POLICY_FALLBACK"
            if basis == "MODEL":
                _require_model_rationale_and_support(name, rationale, anchors)
            effective = estimate
            override_basis = None
            if (
                escalate_to is not None
                and confidence is not None
                and confidence < policy.confidence_threshold
                and effective != escalate_to
            ):
                effective = escalate_to
                override_basis = "low_confidence_escalation"
            dimensions[name] = ClassificationDimensionResult(
                estimated=estimate.value,
                effective=effective.value,
                confidence=confidence,
                conservative_upper=None,
                source_anchors=anchors,
                basis=basis,
                rationale=rationale,
                override_basis=override_basis,
                policy_version=policy.version,
            )
            if override_basis is not None:
                record_override(name, estimate, effective, override_basis)
            return effective

        # C05 remediation (finding #10): one upfront per-axis unpack of the
        # proposal's dimension sub-object, instead of repeating
        # `proposal.<axis>.<field> if proposal else None` for every field
        # below. Same resulting values for every input combination -- purely
        # a mechanical de-duplication of the "no proposal at all" guard.
        consequence_proposal = proposal.consequence if proposal else None
        reversibility_proposal = proposal.reversibility if proposal else None
        ambiguity_proposal = proposal.ambiguity if proposal else None
        evidence_scarcity_proposal = proposal.evidence_scarcity if proposal else None

        irreversibility: Ordinal4 | None
        irreversible_upper: Ordinal4 | None
        if reversibility_proposal is not None:
            if "irreversibility" not in explicit:
                _validate_ordinal_bound(
                    "reversibility",
                    reversibility_proposal.estimate,
                    reversibility_proposal.conservative_upper,
                    ascending=False,
                )
            irreversibility = reversibility_to_irreversibility(reversibility_proposal.estimate)
            irreversible_upper = reversibility_to_irreversibility(
                reversibility_proposal.conservative_upper
            )
        else:
            irreversibility = irreversible_upper = None
        consequence = dimension(
            "consequence",
            consequence_proposal.estimate if consequence_proposal else None,
            consequence_proposal.confidence if consequence_proposal else None,
            consequence_proposal.conservative_upper if consequence_proposal else None,
            consequence_proposal.rationale if consequence_proposal else None,
            floor_consequence,
            consequence_proposal.anchors if consequence_proposal else (),
            (
                (_field_anchor(envelope, "/user_metadata/consequence"),)
                if "consequence" in explicit
                else (_field_anchor(envelope, "/execution_permissions/allow_external_writes"),)
                if envelope.execution_permissions.allow_external_writes
                else ()
            ),
        )
        irreversible = dimension(
            "irreversibility",
            irreversibility,
            reversibility_proposal.confidence if reversibility_proposal else None,
            irreversible_upper,
            reversibility_proposal.rationale if reversibility_proposal else None,
            floor_irreversibility,
            reversibility_proposal.anchors if reversibility_proposal else (),
            (
                (_field_anchor(envelope, "/user_metadata/irreversibility"),)
                if "irreversibility" in explicit
                else (_field_anchor(envelope, "/execution_permissions/allow_external_writes"),)
                if envelope.execution_permissions.allow_external_writes
                else (_field_anchor(envelope, "/execution_permissions/allow_network"),)
                if envelope.execution_permissions.allow_network
                else ()
            ),
        )
        ambiguity = dimension(
            "ambiguity",
            ambiguity_proposal.estimate if ambiguity_proposal else None,
            ambiguity_proposal.confidence if ambiguity_proposal else None,
            ambiguity_proposal.conservative_upper if ambiguity_proposal else None,
            ambiguity_proposal.rationale if ambiguity_proposal else None,
            floor_ambiguity,
            ambiguity_proposal.anchors if ambiguity_proposal else (),
            (
                (_field_anchor(envelope, "/user_metadata/ambiguity"),)
                if "ambiguity" in explicit
                else (_field_anchor(envelope, "/execution_permissions"),)
                if floor_ambiguity != Ordinal4.LOW
                else ()
            ),
        )
        scarcity = dimension(
            "evidence_scarcity",
            evidence_scarcity_proposal.estimate if evidence_scarcity_proposal else None,
            evidence_scarcity_proposal.confidence if evidence_scarcity_proposal else None,
            evidence_scarcity_proposal.conservative_upper if evidence_scarcity_proposal else None,
            evidence_scarcity_proposal.rationale if evidence_scarcity_proposal else None,
            floor_evidence_scarcity,
            evidence_scarcity_proposal.anchors if evidence_scarcity_proposal else (),
            (
                (_field_anchor(envelope, "/user_metadata/evidence_scarcity"),)
                if "evidence_scarcity" in explicit
                else (_field_anchor(envelope, "/execution_permissions/allow_network"),)
                if floor_evidence_scarcity != Ordinal4.LOW
                else ()
            ),
        )
        task_type = cast(
            TaskType,
            categorical_dimension(
                "task_type", proposal.task_type if proposal else None, TaskType.ANALYSIS
            ),
        )
        search = cast(
            SearchSpaceClass,
            categorical_dimension(
                "search_space",
                proposal.search_space if proposal else None,
                policy.fallback_search_space,
                escalate_to=SearchSpaceClass.OPEN,
            ),
        )
        horizon = cast(
            HorizonClass,
            categorical_dimension(
                "horizon",
                proposal.horizon if proposal else None,
                policy.fallback_horizon,
                escalate_to=policy.low_confidence_horizon_fallback,
            ),
        )
        output_form = normalise_output_form(envelope.requested_output.form)
        output_form_anchor = _field_anchor(envelope, "/requested_output")
        dimensions["output_form"] = ClassificationDimensionResult(
            estimated=output_form.value,
            effective=output_form.value,
            confidence=1.0,
            conservative_upper=None,
            source_anchors=(output_form_anchor,),
            basis="DETERMINISTIC",
            rationale="Derived deterministically from the task's requested_output.form contract.",
            override_basis=None,
            policy_version=policy.version,
        )
        deterministic_evidence = {
            _anchor_ref(_field_anchor(envelope, f"/explicit_constraints/{index}"))
            for index in range(len(envelope.explicit_constraints))
        }
        deterministic_evidence.add(_anchor_ref(output_form_anchor))
        for key in sorted(envelope.user_metadata):
            escaped = key.replace("~", "~0").replace("/", "~1")
            deterministic_evidence.add(
                _anchor_ref(_field_anchor(envelope, f"/user_metadata/{escaped}"))
            )
        permissions = envelope.execution_permissions
        if (
            permissions.capabilities
            or permissions.allow_network
            or permissions.allow_external_writes
            or permissions.allowed_paths
        ):
            deterministic_evidence.add(
                _anchor_ref(_field_anchor(envelope, "/execution_permissions"))
            )
        signature = TaskSignature(
            task_type=task_type,
            consequence=consequence,
            irreversibility=irreversible,
            ambiguity=ambiguity,
            search_space=search,
            evidence_scarcity=scarcity,
            horizon=horizon,
            output_form=output_form,
            dimension_confidence={name: item.confidence for name, item in dimensions.items()},
            evidence_refs=tuple(
                sorted(
                    deterministic_evidence
                    | {
                        _anchor_ref(anchor)
                        for d in dimensions.values()
                        for anchor in d.source_anchors
                    }
                )
            ),
        )
        diagnostics = tuple(f"{override.axis}:{override.reason}" for override in floor_overrides)
        record = ClassificationRecord(
            policy_version=policy.version,
            policy_hash=policy.policy_hash,
            mode="HYBRID" if proposal else "DETERMINISTIC_FALLBACK",
            fallback_used=proposal is None,
            dimensions=dimensions,
            model_call_key=model_call_key,
            diagnostics=diagnostics,
            floor_overrides=tuple(floor_overrides),
        )
        return signature, record

    def provenance_events(
        self,
        record: ClassificationRecord,
        *,
        created_at: datetime,
        uuids: UUIDFactory,
    ) -> tuple[EventPayload, ...]:
        """Emit M09 provenance for every material, model-derived classification decision.

        C05 remediation (finding #8, re-scoped): emits at most one
        `LedgerNodeAdded` per MODEL-basis dimension. There are 7 classifiable
        dimensions (`task_type`, `consequence`, `irreversibility`, `ambiguity`,
        `evidence_scarcity`, `search_space`, `horizon`) -- the 8th dimension,
        `output_form`, is provably never MODEL-basis (it is always derived
        deterministically from `envelope.requested_output.form`, see its
        `basis="DETERMINISTIC"` assignment in `classify()`), so the true
        maximum is 7 events (14 UUIDs: one node id + one action id per event),
        not 8/16. This one-event-per-MODEL-axis granularity is intentional,
        not accidental fan-out: the reducer's admission check
        (`RunReducer.apply`, `TaskClassified` branch, "C05 remediation finding
        #2") looks up provenance per axis by name, so each MODEL axis needs
        its own independently citable `LedgerNodeRef`. Consolidating these
        into one event per classification would require changing that
        per-axis lookup contract in lockstep -- a larger, out-of-scope change
        touching persisted event shapes and every golden fixture -- so this is
        deliberately left as-is here.
        """
        events: list[EventPayload] = []
        for name in sorted(record.dimensions):
            result = record.dimensions[name]
            if result.basis != "MODEL":
                continue
            node = make_node(
                node_id=uuids.new(),
                revision=1,
                node_type=LedgerNodeType.INFERENCE,
                content={"axis": name, **result.model_dump(mode="json")},
                status=EpistemicStatus.PROVISIONAL,
                created_at=created_at,
                action_id=uuids.new(),
                module_id="M01",
            )
            events.append(LedgerNodeAdded(node=node))
        return tuple(events)

    def canonical_events(
        self,
        envelope: TaskEnvelope,
        proposal: ClassificationOutput | None,
        *,
        tier_policy: TierPolicy,
        deployment: DeploymentLimits,
        created_at: datetime,
        uuids: UUIDFactory,
        policy: ClassificationPolicy | None = None,
        model_call_key: str | None = None,
        current_projection: BudgetProjection | None = None,
    ) -> tuple[EventPayload, ...]:
        """Build one logically atomic classification/provenance/budget event batch.

        Order: a deterministic-only preliminary signature seeds a bootstrap
        M02 budget (`TaskPreliminarilyClassified` + `BudgetAllocated`); M09
        provenance for every model-derived axis is persisted *before* the
        authoritative classification (`TaskClassified`) that claims those
        axes; the budget is then revised to the authoritative tier
        (`BudgetRevised`). All events are returned as a single tuple, so the
        caller (mirroring the C04/F09 all-or-nothing batch-append pattern)
        commits them together or not at all.

        C05 remediation (finding #2/#3): provenance is deliberately ordered
        *before* `TaskClassified`, and `TaskClassified` before `BudgetRevised`,
        so `RunReducer.apply` can enforce -- as a simple backward look at
        already-applied state, exactly like its `ModelCallRecordedV2`
        reservation check -- that every MODEL-basis dimension in
        `TaskClassified` has a matching M09 node already applied, and that
        `BudgetRevised` never lands without an applied `TaskClassified` behind
        it. Reordering this tuple is not cosmetic: it is load-bearing for
        those reducer-level admission checks.

        C05 remediation (finding #9): this method is designed for exactly one
        calling context -- a task's *initial* classification, on a run with no
        prior budget activity at all. The bootstrap-vs-final atomicity precheck
        below (the `allocator.revise(...)` call) is only accurate for that
        context, because it is computed against a projection freshly
        synthesized from the bootstrap plan with zero committed/reserved
        usage -- correct for a genesis run, meaningless for one with real
        prior activity. `current_projection` makes that precondition explicit
        and machine-checked rather than merely assumed: pass the run's real,
        current `BudgetProjection` (e.g. from `engine.inspect(run_id).budget`)
        when the caller has one, and this method raises immediately if it is
        not fresh (i.e. already carries an allocated plan), instead of
        silently proceeding with a precheck that would not reflect reality.
        Leaving it `None` (the default, for today's only caller -- always a
        freshly created run) skips this extra guard and keeps prior behavior
        unchanged. A future orchestrator (C09) driving re-classification of an
        already-active run must not call this method at all; it would need
        its own revision path built around the real projection, not this one.

        EU-36 (w3-cleanup) cross-reference: `fre.composition.Wave3Engine.
        classify_task` is exactly that C09 orchestrator, and it deliberately
        does NOT call this method -- see its own docstring for why the
        bootstrap-then-semantic-call-then-finalize ordering it needs cannot
        be expressed as one call to `canonical_events` (this method runs the
        semantic call synchronously with no interleaving point to persist a
        bootstrap budget first). The two implementations look similar but
        exist for structurally different calling contexts; this is
        intentional duplication, not drift.
        """
        if current_projection is not None and current_projection.plan is not None:
            raise ValueError(
                "canonical_events requires a fresh run with no prior budget activity; "
                "the supplied current_projection already carries an allocated plan"
            )
        policy = policy or self.policy
        allocator = BudgetAllocator()
        bootstrap_signature = self.bootstrap_signature(envelope, policy)
        bootstrap_plan, bootstrap_hash = allocator.allocate(
            bootstrap_signature, tier_policy, deployment
        )
        final_signature, record = self.classify(
            envelope, proposal, policy, model_call_key=model_call_key
        )
        final_plan, final_hash = allocator.allocate(final_signature, tier_policy, deployment)
        # C05 remediation (finding #9, investigated): this `revise()` call is
        # intentionally validate-only -- its returned `BudgetProjection` is
        # discarded; only its invariant checks (monotone-tier,
        # validation-floor, committed-plus-reserved-usage) matter here, to
        # prove the bootstrap -> final handoff never violates M02's revision
        # invariants before any event is emitted. This is acceptable as-is:
        # both `BudgetAllocator.allocate` and `.revise` (see `m02_budget.py`)
        # are pure, cheap, in-memory arithmetic over already-constructed
        # `TaskSignature`/`BudgetPlan`/`BudgetProjection` objects -- floor
        # lookups, `max()`/comparisons, and a `canonical_hash` call for the
        # policy hash -- with no I/O and nothing that scales with anything
        # other than the small, fixed set of budget dimensions. There is no
        # meaningfully cheaper way to run `revise()`'s checks than calling
        # `revise()` itself: its invariant checks are woven through the same
        # few lines that would otherwise need duplicating into a standalone
        # function, which would itself be a second place for the two to drift
        # out of sync. Extracting a standalone check was considered and
        # rejected for this reason.
        allocator.revise(
            BudgetProjection(plan=bootstrap_plan, policy_hash=bootstrap_hash),
            final_plan,
            final_hash,
        )
        events: list[EventPayload] = [
            TaskPreliminarilyClassified(signature=bootstrap_signature),
            BudgetAllocated(
                plan=bootstrap_plan,
                policy_version=bootstrap_plan.policy_version,
                policy_hash=bootstrap_hash,
            ),
        ]
        events.extend(self.provenance_events(record, created_at=created_at, uuids=uuids))
        events.append(TaskClassified(signature=final_signature, record=record))
        events.append(
            BudgetRevised(
                plan=final_plan, policy_version=final_plan.policy_version, policy_hash=final_hash
            )
        )
        return tuple(events)
