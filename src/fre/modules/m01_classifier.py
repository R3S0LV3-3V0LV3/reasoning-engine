"""M01 deterministic-first task classifier."""

from pydantic import Field

from fre.domain.common import FrozenModel, canonical_hash
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
from fre.prompts.schemas import ClassificationOutput

ORDINAL = tuple(Ordinal4)


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
    def classify(
        self,
        envelope: TaskEnvelope,
        proposal: ClassificationOutput | None,
        policy: ClassificationPolicy | None = None,
        *,
        model_call_key: str | None = None,
    ) -> tuple[TaskSignature, ClassificationRecord]:
        policy = policy or ClassificationPolicy()
        metadata = envelope.user_metadata
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
        ) -> Ordinal4:
            estimate = explicit.get(name, proposed or policy.fallback_ordinal)
            effective = harder(estimate, floor)
            if confidence is not None and confidence < policy.confidence_threshold:
                effective = harder(
                    effective, harder(upper or estimate, policy.low_confidence_floor)
                )
            if proposal is None and name not in explicit:
                effective = harder(effective, policy.fallback_ordinal)
            dimensions[name] = ClassificationDimensionResult(
                estimated=estimate,
                effective=effective,
                confidence=confidence,
                conservative_upper=upper,
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
        )
        irreversible = dimension(
            "irreversibility",
            irreversibility,
            proposal.reversibility.confidence if proposal else None,
            irreversible_upper,
            floor_irreversibility,
        )
        ambiguity = dimension(
            "ambiguity",
            proposal.ambiguity.estimate if proposal else None,
            proposal.ambiguity.confidence if proposal else None,
            proposal.ambiguity.conservative_upper if proposal else None,
        )
        scarcity = dimension(
            "evidence_scarcity",
            proposal.evidence_scarcity.estimate if proposal else None,
            proposal.evidence_scarcity.confidence if proposal else None,
            proposal.evidence_scarcity.conservative_upper if proposal else None,
        )
        search = proposal.search_space if proposal else policy.fallback_search_space
        if proposal and proposal.search_space_confidence < policy.confidence_threshold:
            search = SearchSpaceClass.OPEN
        task_type = proposal.task_type if proposal else TaskType.ANALYSIS
        horizon = proposal.horizon if proposal else policy.fallback_horizon
        output_form = OutputForm(envelope.requested_output.form.upper())
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
                sorted({str(anchor) for d in dimensions.values() for anchor in d.source_anchors})
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
