"""M03 provenance-rich problem formaliser."""

from datetime import datetime
from typing import Literal, cast

from fre.domain.ledger import (
    EpistemicStatus,
    LedgerEdge,
    LedgerNodeRef,
    LedgerNodeType,
    LedgerRelation,
)
from fre.domain.problem import (
    AcceptanceCriterion,
    AssumptionSpec,
    ConstraintSpec,
    DecisionVariable,
    FixedParameter,
    ObjectiveSpec,
    ObservableSpec,
    ProblemRelation,
    ProblemRelationKind,
    ProblemSpec,
    UnknownSpec,
    VerificationStatus,
)
from fre.domain.semantic import EpistemicItemProvenance, EpistemicOriginLabel
from fre.domain.task import TaskEnvelope
from fre.modules.m09_ledger import make_node
from fre.modules.source_anchors import validate_source_anchor
from fre.ports.clock import UUIDFactory
from fre.prompts.schemas import ProblemFormalisationOutput, ProblemItemProposal
from fre.runtime.events import EventPayload, LedgerEdgeAdded, LedgerNodeAdded


class InvalidProblemSpec(ValueError):
    pass


class ProblemFormaliser:
    def ledger_events(
        self,
        proposal: ProblemFormalisationOutput,
        *,
        created_at: datetime,
        uuids: UUIDFactory,
    ) -> tuple[EventPayload, ...]:
        """Map material semantic items onto the frozen M09 vocabulary."""
        node_types = {
            EpistemicOriginLabel.EXPLICIT_INPUT: LedgerNodeType.FACT,
            EpistemicOriginLabel.SUPPORTED_INFERENCE: LedgerNodeType.INFERENCE,
            EpistemicOriginLabel.WORKING_ASSUMPTION: LedgerNodeType.ASSUMPTION,
            EpistemicOriginLabel.UNRESOLVED: LedgerNodeType.UNKNOWN,
            EpistemicOriginLabel.CONTRADICTED: LedgerNodeType.CONTESTED,
        }
        statuses = {
            EpistemicOriginLabel.EXPLICIT_INPUT: EpistemicStatus.SUPPORTED,
            EpistemicOriginLabel.SUPPORTED_INFERENCE: EpistemicStatus.PROVISIONAL,
            EpistemicOriginLabel.WORKING_ASSUMPTION: EpistemicStatus.PROVISIONAL,
            EpistemicOriginLabel.UNRESOLVED: EpistemicStatus.UNRESOLVED,
            EpistemicOriginLabel.CONTRADICTED: EpistemicStatus.CONTESTED,
        }
        events: list[EventPayload] = []
        refs: dict[str, LedgerNodeRef] = {}
        for item in proposal.items:
            if item.kind == "RELATION":
                continue
            node_id = uuids.new()
            node = make_node(
                node_id=node_id,
                revision=1,
                node_type=node_types[item.origin],
                content=item.model_dump(mode="json"),
                status=statuses[item.origin],
                created_at=created_at,
                action_id=uuids.new(),
                module_id="M03",
            )
            refs[item.id] = node.ref
            events.append(LedgerNodeAdded(node=node))
        for item in proposal.items:
            if item.kind != "RELATION" or item.attributes.get("relation_kind") != "CONTRADICTS":
                continue
            source = refs.get(str(item.attributes.get("source_id")))
            target = refs.get(str(item.attributes.get("target_id")))
            if source is None or target is None:
                raise InvalidProblemSpec("contradiction relation references an unknown item")
            events.append(
                LedgerEdgeAdded(
                    edge=LedgerEdge(
                        edge_id=uuids.new(),
                        source=source,
                        target=target,
                        relation=LedgerRelation.CONTRADICTS,
                    )
                )
            )
        return tuple(events)

    def formalise(
        self,
        envelope: TaskEnvelope,
        proposal: ProblemFormalisationOutput | None,
        *,
        available_artifacts: frozenset[str] = frozenset(),
    ) -> ProblemSpec:
        items = proposal.items if proposal else ()
        variables: list[DecisionVariable] = []
        fixed: list[FixedParameter] = []
        objectives: list[ObjectiveSpec] = []
        constraints: list[ConstraintSpec] = []
        unknowns: list[UnknownSpec] = []
        observables: list[ObservableSpec] = []
        assumptions: list[AssumptionSpec] = []
        criteria: list[AcceptanceCriterion] = []
        relations: list[ProblemRelation] = []
        ids: set[str] = set()
        for item in items:
            if item.id in ids:
                raise InvalidProblemSpec("problem item IDs must be unique")
            ids.add(item.id)
            provenance = self._provenance(item, envelope, available_artifacts)
            attrs = item.attributes
            if item.kind == "OBJECTIVE":
                direction = str(attrs.get("direction", "UNRESOLVED"))
                if direction not in {
                    "MIN",
                    "MAX",
                    "TARGET",
                    "LEXICOGRAPHIC",
                    "QUALITATIVE",
                    "UNRESOLVED",
                }:
                    direction = "UNRESOLVED"
                objectives.append(
                    ObjectiveSpec(
                        id=item.id,
                        name=item.description,
                        direction=cast(
                            Literal[
                                "MIN", "MAX", "TARGET", "LEXICOGRAPHIC", "QUALITATIVE", "UNRESOLVED"
                            ],
                            direction,
                        ),
                        unit=(unit if isinstance((unit := attrs.get("unit")), str) else None),
                        description=item.description,
                        provenance=provenance,
                    )
                )
            elif item.kind == "CONSTRAINT":
                kind = str(attrs.get("constraint_kind", "HARD"))
                mode = str(attrs.get("verification_mode", "UNAVAILABLE"))
                status = VerificationStatus(str(attrs.get("verification_status", "UNKNOWN")))
                if kind not in {"HARD", "SOFT"} or mode not in {
                    "DETERMINISTIC",
                    "MODEL",
                    "HUMAN",
                    "UNAVAILABLE",
                }:
                    raise InvalidProblemSpec("invalid constraint semantics")
                constraints.append(
                    ConstraintSpec(
                        id=item.id,
                        description=item.description,
                        kind=cast(Literal["HARD", "SOFT"], kind),
                        verification_mode=cast(
                            Literal["DETERMINISTIC", "MODEL", "HUMAN", "UNAVAILABLE"], mode
                        ),
                        verification_status=status,
                        source_refs=item.supporting_refs,
                        provenance=provenance,
                    )
                )
            elif item.kind == "DECISION_VARIABLE":
                variables.append(
                    DecisionVariable(
                        id=item.id,
                        name=item.description,
                        domain=attrs.get("domain"),
                        provenance=provenance,
                    )
                )
            elif item.kind == "FIXED_PARAMETER":
                if "value" not in attrs:
                    raise InvalidProblemSpec("fixed parameter requires value")
                fixed.append(
                    FixedParameter(
                        id=item.id,
                        name=item.description,
                        value=attrs["value"],
                        provenance=provenance,
                    )
                )
            elif item.kind == "UNKNOWN":
                unknowns.append(
                    UnknownSpec(
                        id=item.id,
                        description=item.description,
                        domain=attrs.get("domain"),
                        resolvable=(
                            resolvable
                            if isinstance((resolvable := attrs.get("resolvable")), bool)
                            else None
                        ),
                        provenance=provenance,
                    )
                )
            elif item.kind == "OBSERVABLE":
                observables.append(
                    ObservableSpec(
                        id=item.id,
                        description=item.description,
                        unit=(unit if isinstance((unit := attrs.get("unit")), str) else None),
                        provenance=provenance,
                    )
                )
            elif item.kind == "ASSUMPTION":
                assumptions.append(
                    AssumptionSpec(
                        id=item.id,
                        statement=item.description,
                        why_needed=item.basis or "required for formalisation",
                        provenance=provenance,
                    )
                )
            elif item.kind == "ACCEPTANCE_CRITERION":
                mode = str(attrs.get("verification_mode", "UNAVAILABLE"))
                criteria.append(
                    AcceptanceCriterion(
                        id=item.id,
                        predicate_description=item.description,
                        verification_mode=mode,
                        required=bool(attrs.get("required", True)),
                        provenance=provenance,
                    )
                )
            elif item.kind == "RELATION":
                relations.append(
                    ProblemRelation(
                        source_id=str(attrs["source_id"]),
                        target_id=str(attrs["target_id"]),
                        kind=ProblemRelationKind(str(attrs["relation_kind"])),
                    )
                )
        # Honest fallback preserves only mechanically explicit constraints.
        if proposal is None:
            for index, statement in enumerate(envelope.explicit_constraints):
                constraints.append(
                    ConstraintSpec(
                        id=f"explicit-{index}",
                        description=statement,
                        kind="HARD",
                        verification_mode="UNAVAILABLE",
                        verification_status=VerificationStatus.UNKNOWN,
                        source_refs=(
                            f"TaskEnvelope:{envelope.task_id}:/explicit_constraints/{index}",
                        ),
                    )
                )
        overlap = {item.id for item in variables} & {item.id for item in fixed}
        if overlap:
            raise InvalidProblemSpec("item cannot be both decision variable and fixed parameter")
        return ProblemSpec(
            decision_variables=tuple(variables),
            fixed_parameters=tuple(fixed),
            objectives=tuple(objectives),
            constraints=tuple(constraints),
            assumption_items=tuple(assumptions),
            unknowns=tuple(unknowns),
            observables=tuple(observables),
            acceptance_criteria=tuple(criteria),
            relations=tuple(relations),
            output_contract=envelope.requested_output,
        )

    @staticmethod
    def _provenance(
        item: ProblemItemProposal, envelope: TaskEnvelope, available_artifacts: frozenset[str]
    ) -> EpistemicItemProvenance:
        for anchor in item.anchors:
            validate_source_anchor(anchor, envelope, available_artifacts)
        return EpistemicItemProvenance(
            origin=item.origin,
            anchors=item.anchors,
            supporting_refs=item.supporting_refs,
            basis=item.basis,
            policy_basis=item.policy_basis,
        )
