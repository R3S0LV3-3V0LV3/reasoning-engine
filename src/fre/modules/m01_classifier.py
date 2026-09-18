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
from fre.modules.source_anchors import validate_source_anchor
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


def _explicit_ordinals(envelope: TaskEnvelope) -> dict[str, Ordinal4]:
    metadata = envelope.user_metadata
    explicit: dict[str, Ordinal4] = {}
    for key in ("consequence", "irreversibility", "ambiguity", "evidence_scarcity"):
        value = metadata.get(key)
        if isinstance(value, str) and value in Ordinal4:
            explicit[key] = Ordinal4(value)
    return explicit


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
            is_explicit = name in explicit
            estimate = explicit.get(name, proposed or policy.fallback_ordinal)
            basis = "EXPLICIT" if is_explicit else "MODEL" if proposal else "POLICY_FALLBACK"
            if basis == "MODEL" and not is_explicit and proposed is not None:
                if not rationale or not rationale.strip():
                    raise ClassificationBlocked(
                        f"{name}: material classification dimension is missing a rationale"
                    )
                if not proposed_anchors:
                    raise ClassificationBlocked(
                        f"{name}: material classification dimension has no resolvable support"
                    )
                if upper is not None:
                    _validate_ordinal_bound(
                        name,
                        proposed,
                        upper,
                        ascending=name in _ASCENDING_AXES or name == "irreversibility",
                    )
            after_floor = harder(estimate, floor)
            effective = after_floor
            effective_confidence = None if is_explicit else confidence
            effective_upper = None if is_explicit else upper
            low_confidence_applied = False
            if (
                not is_explicit
                and confidence is not None
                and confidence < policy.confidence_threshold
            ):
                escalated = harder(
                    after_floor, harder(upper or estimate, policy.low_confidence_floor)
                )
                low_confidence_applied = escalated != after_floor
                effective = escalated
            fallback_applied = False
            if proposal is None and name not in explicit:
                escalated = harder(effective, policy.fallback_ordinal)
                fallback_applied = escalated != effective
                effective = escalated
            reasons = []
            if after_floor != estimate:
                reasons.append("permission_floor")
            if low_confidence_applied:
                reasons.append("low_confidence_escalation")
            if fallback_applied:
                reasons.append("no_proposal_fallback")
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
            estimate = proposal_obj.estimate if proposal_obj else default
            confidence = proposal_obj.confidence if proposal_obj else None
            rationale = proposal_obj.rationale if proposal_obj else None
            basis = "MODEL" if proposal_obj else "POLICY_FALLBACK"
            if basis == "MODEL":
                if not rationale or not rationale.strip():
                    raise ClassificationBlocked(
                        f"{name}: material classification dimension is missing a rationale"
                    )
                if not anchors:
                    raise ClassificationBlocked(
                        f"{name}: material classification dimension has no resolvable support"
                    )
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

        irreversibility: Ordinal4 | None
        irreversible_upper: Ordinal4 | None
        if proposal:
            if "irreversibility" not in explicit:
                _validate_ordinal_bound(
                    "reversibility",
                    proposal.reversibility.estimate,
                    proposal.reversibility.conservative_upper,
                    ascending=False,
                )
            irreversibility = reversibility_to_irreversibility(proposal.reversibility.estimate)
            irreversible_upper = reversibility_to_irreversibility(
                proposal.reversibility.conservative_upper
            )
        else:
            irreversibility = irreversible_upper = None
        consequence = dimension(
            "consequence",
            proposal.consequence.estimate if proposal else None,
            proposal.consequence.confidence if proposal else None,
            proposal.consequence.conservative_upper if proposal else None,
            proposal.consequence.rationale if proposal else None,
            floor_consequence,
            proposal.consequence.anchors if proposal else (),
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
            proposal.reversibility.confidence if proposal else None,
            irreversible_upper,
            proposal.reversibility.rationale if proposal else None,
            floor_irreversibility,
            proposal.reversibility.anchors if proposal else (),
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
            proposal.ambiguity.estimate if proposal else None,
            proposal.ambiguity.confidence if proposal else None,
            proposal.ambiguity.conservative_upper if proposal else None,
            proposal.ambiguity.rationale if proposal else None,
            proposed_anchors=proposal.ambiguity.anchors if proposal else (),
            deterministic_anchors=(
                (_field_anchor(envelope, "/user_metadata/ambiguity"),)
                if "ambiguity" in explicit
                else ()
            ),
        )
        scarcity = dimension(
            "evidence_scarcity",
            proposal.evidence_scarcity.estimate if proposal else None,
            proposal.evidence_scarcity.confidence if proposal else None,
            proposal.evidence_scarcity.conservative_upper if proposal else None,
            proposal.evidence_scarcity.rationale if proposal else None,
            proposed_anchors=proposal.evidence_scarcity.anchors if proposal else (),
            deterministic_anchors=(
                (_field_anchor(envelope, "/user_metadata/evidence_scarcity"),)
                if "evidence_scarcity" in explicit
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
                escalate_to=policy.fallback_horizon,
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

    def _provenance_events(
        self,
        record: ClassificationRecord,
        *,
        created_at: datetime,
        uuids: UUIDFactory,
    ) -> tuple[EventPayload, ...]:
        """Emit M09 provenance for every material, model-derived classification decision."""
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
    ) -> tuple[EventPayload, ...]:
        """Build one logically atomic classification/provenance/budget event batch.

        Order: a deterministic-only preliminary signature seeds a bootstrap
        M02 budget (`TaskPreliminarilyClassified` + `BudgetAllocated`); the
        authoritative classification is then persisted separately
        (`TaskClassified`) together with M09 provenance for every
        model-derived axis, and the budget is revised to the authoritative
        tier (`BudgetRevised`). All events are returned as a single tuple, so
        the caller (mirroring the C04/F09 all-or-nothing batch-append
        pattern) commits them together or not at all.
        """
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
        # Validate-only: proves the bootstrap -> final handoff never violates
        # M02's monotone-tier/validation-floor revision invariants before any
        # event is emitted.
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
            TaskClassified(signature=final_signature, record=record),
        ]
        events.extend(self._provenance_events(record, created_at=created_at, uuids=uuids))
        events.append(
            BudgetRevised(
                plan=final_plan, policy_version=final_plan.policy_version, policy_hash=final_hash
            )
        )
        return tuple(events)
