"""M01 deterministic-first task classifier."""

from pydantic import Field

from fre.domain.common import FrozenModel, ObjectRef, canonical_hash, canonical_json
from fre.domain.semantic import SourceAnchor, SourceKind
from fre.domain.task import (
    ClassificationDimensionResult,
    ClassificationRecord,
    HorizonClass,
    Ordinal4,
    OutputForm,
    SearchSpaceClass,
    TaskEnvelope,
    TaskSignature,
    TaskType,
)
from fre.modules.source_anchors import validate_source_anchor
from fre.prompts.schemas import ClassificationOutput

ORDINAL = tuple(Ordinal4)


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
    version: str = "wave3-m01/1.0"
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


class TaskClassifier:
    def __init__(self, policy: ClassificationPolicy | None = None) -> None:
        self.policy = policy or ClassificationPolicy()

    def classify(
        self,
        envelope: TaskEnvelope,
        proposal: ClassificationOutput | None,
        policy: ClassificationPolicy | None = None,
        *,
        model_call_key: str | None = None,
    ) -> tuple[TaskSignature, ClassificationRecord]:
        policy = policy or self.policy
        metadata = envelope.user_metadata
        available_artifacts = frozenset(attachment.sha256 for attachment in envelope.attachments)
        explicit: dict[str, Ordinal4] = {}
        for key in ("consequence", "irreversibility", "ambiguity", "evidence_scarcity"):
            value = metadata.get(key)
            if isinstance(value, str) and value in Ordinal4:
                explicit[key] = Ordinal4(value)
        floor_consequence = (
            Ordinal4.HIGH if envelope.execution_permissions.allow_external_writes else Ordinal4.LOW
        )
        floor_irreversibility = (
            Ordinal4.HIGH
            if envelope.execution_permissions.allow_external_writes
            else Ordinal4.MEDIUM
            if envelope.execution_permissions.allow_network
            else Ordinal4.LOW
        )
        dimensions: dict[str, ClassificationDimensionResult] = {}

        def dimension(
            name: str,
            proposed: Ordinal4 | None,
            confidence: float | None,
            upper: Ordinal4 | None,
            floor: Ordinal4 = Ordinal4.LOW,
            proposed_anchors: tuple[SourceAnchor, ...] = (),
            deterministic_anchors: tuple[SourceAnchor, ...] = (),
        ) -> Ordinal4:
            for anchor in proposed_anchors:
                validate_source_anchor(anchor, envelope, available_artifacts)
            is_explicit = name in explicit
            estimate = explicit.get(name, proposed or policy.fallback_ordinal)
            effective = harder(estimate, floor)
            effective_confidence = None if is_explicit else confidence
            effective_upper = None if is_explicit else upper
            if (
                not is_explicit
                and confidence is not None
                and confidence < policy.confidence_threshold
            ):
                effective = harder(
                    effective, harder(upper or estimate, policy.low_confidence_floor)
                )
            if proposal is None and name not in explicit:
                effective = harder(effective, policy.fallback_ordinal)
            dimensions[name] = ClassificationDimensionResult(
                estimated=estimate,
                effective=effective,
                confidence=effective_confidence,
                conservative_upper=effective_upper,
                source_anchors=deterministic_anchors + proposed_anchors,
                basis="EXPLICIT"
                if name in explicit
                else "MODEL"
                if proposal
                else "POLICY_FALLBACK",
            )
            return effective

        irreversibility: Ordinal4 | None
        irreversible_upper: Ordinal4 | None
        if proposal:
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
            proposed_anchors=proposal.evidence_scarcity.anchors if proposal else (),
            deterministic_anchors=(
                (_field_anchor(envelope, "/user_metadata/evidence_scarcity"),)
                if "evidence_scarcity" in explicit
                else ()
            ),
        )
        search = proposal.search_space if proposal else policy.fallback_search_space
        if proposal and proposal.search_space_confidence < policy.confidence_threshold:
            search = SearchSpaceClass.OPEN
        task_type = proposal.task_type if proposal else TaskType.ANALYSIS
        horizon = proposal.horizon if proposal else policy.fallback_horizon
        output_form = normalise_output_form(envelope.requested_output.form)
        deterministic_evidence = {
            _anchor_ref(_field_anchor(envelope, f"/explicit_constraints/{index}"))
            for index in range(len(envelope.explicit_constraints))
        }
        deterministic_evidence.add(_anchor_ref(_field_anchor(envelope, "/requested_output")))
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
        return signature, ClassificationRecord(
            policy_version=policy.version,
            policy_hash=policy.policy_hash,
            mode="HYBRID" if proposal else "DETERMINISTIC_FALLBACK",
            fallback_used=proposal is None,
            dimensions=dimensions,
            model_call_key=model_call_key,
        )
