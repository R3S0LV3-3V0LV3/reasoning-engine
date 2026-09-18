"""M03 provenance-rich problem formaliser."""

from datetime import datetime
from typing import Literal, cast
from uuid import UUID

from fre.domain.common import ObjectRef
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
    ContradictionDiagnostic,
    DecisionVariable,
    FixedParameter,
    ObjectiveSpec,
    ObservableSpec,
    ProblemBlocker,
    ProblemRelation,
    ProblemRelationKind,
    ProblemSpec,
    UnknownSpec,
    VerificationStatus,
)
from fre.domain.semantic import (
    EpistemicItemProvenance,
    EpistemicOriginLabel,
    SourceAnchor,
    SourceKind,
    SupportProblemItemRef,
)
from fre.domain.task import TaskEnvelope
from fre.modules.m09_ledger import make_node
from fre.modules.source_anchors import validate_source_anchor, validate_support_graph
from fre.ports.clock import UUIDFactory
from fre.prompts.schemas import ProblemFormalisationOutput, ProblemItemProposal
from fre.runtime.events import (
    EventPayload,
    LedgerEdgeAdded,
    LedgerNodeAdded,
    ProblemBlockerRecorded,
    ProblemContradictionRecorded,
    ProblemFormalised,
)

# C06 remediation (F07): versioned materiality rules. A material unresolved
# item must produce a concrete blocker (and keep the run non-terminal until
# resolved); a non-material one keeps its retained provenance and explicit
# M09 basis but never fabricates a blocker. `attributes["material"]` is
# authoritative when the proposal states it explicitly; otherwise
# `attributes["decision_relevance"]` is compared against this threshold; and
# absent *both* signals the conservative default is `True` -- an unresolved
# item's materiality must never be silently downgraded to "not material" by
# mere omission.
MATERIALITY_POLICY_VERSION = "1.0"
MATERIALITY_DECISION_RELEVANCE_THRESHOLD = 0.5


class InvalidProblemSpec(ValueError):
    pass


def _topologically_ordered_items(
    items: tuple[ProblemItemProposal, ...],
) -> tuple[ProblemItemProposal, ...]:
    """Order `items` (already known acyclic) so a `ProblemItemRef` target's
    node always precedes the node of the item that cites it."""
    by_id = {item.id: item for item in items}
    visited: set[str] = set()
    ordered: list[ProblemItemProposal] = []

    def visit(item_id: str) -> None:
        if item_id in visited or item_id not in by_id:
            return
        visited.add(item_id)
        for ref in by_id[item_id].support:
            if isinstance(ref, SupportProblemItemRef):
                visit(ref.item_id)
        ordered.append(by_id[item_id])

    for item in items:
        visit(item.id)
    return tuple(ordered)


def _is_material(item: ProblemItemProposal) -> bool:
    material = item.attributes.get("material")
    if isinstance(material, bool):
        return material
    relevance = item.attributes.get("decision_relevance")
    if isinstance(relevance, int | float) and not isinstance(relevance, bool):
        return float(relevance) >= MATERIALITY_DECISION_RELEVANCE_THRESHOLD
    return True


class ProblemFormaliser:
    def _ledger_events(
        self,
        proposal: ProblemFormalisationOutput,
        *,
        created_at: datetime,
        uuids: UUIDFactory,
    ) -> tuple[tuple[EventPayload, ...], dict[str, LedgerNodeRef]]:
        """Map material semantic items onto the frozen M09 vocabulary.

        Private (C06 remediation, F06): this used to be a public
        `ledger_events()` method that any caller could invoke directly,
        emitting `LedgerNodeAdded`/`LedgerEdgeAdded`/blocker/contradiction
        events for a proposal whose anchors and support references were
        *never validated* -- `_validate_items` checks item/relation shape
        only, never `validate_source_anchor` or `validate_support_graph`.
        It is now reachable only from `canonical_events`, which always calls
        `formalise` (the sole place anchor/support resolution happens) on
        the very same `proposal` first; a caller can no longer skip
        provenance validation by going around `formalise` to this method
        directly, because there is no way to call it directly any more.
        Returns the ledger/blocker/contradiction events together with a map
        of `ACCEPTANCE_CRITERION` item id -> the `LedgerNodeRef` of the
        blocker recorded against it (if any), so `canonical_events` can
        thread that identity back into the immutable `ProblemSpec` it
        already built (`AcceptanceCriterion.blocker_ref`, C06 / F04).
        """
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
        blocker_refs: dict[str, LedgerNodeRef] = {}
        self._validate_items(proposal.items)
        contested_ids: set[str] = set()
        for item in proposal.items:
            if (
                item.kind == "RELATION"
                and item.attributes.get("relation_kind") == "CONTRADICTS"
                and item.attributes.get("material", True) is True
            ):
                contested_ids.add(str(item.attributes["source_id"]))
                contested_ids.add(str(item.attributes["target_id"]))
        # C06 remediation: persist ledger nodes before events that resolve
        # them, within this one atomic batch. A same-proposal `support`
        # reference (`ProblemItemRef`) may point at an item declared *later*
        # in `proposal.items` -- `validate_support_graph` only forbids a
        # cycle, never a specific declaration order -- so nodes are emitted
        # in dependency order (a target's node always precedes the node that
        # cites it), not raw proposal order. `_reject_support_cycles`
        # (already run by `validate_support_graph` inside `formalise`, which
        # `canonical_events` always calls first) guarantees this terminates.
        for item in _topologically_ordered_items(
            tuple(item for item in proposal.items if item.kind != "RELATION")
        ):
            node_id = uuids.new()
            node = make_node(
                node_id=node_id,
                revision=1,
                node_type=node_types[item.origin],
                content=item.model_dump(mode="json"),
                status=(
                    EpistemicStatus.CONTESTED if item.id in contested_ids else statuses[item.origin]
                ),
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
            material = item.attributes.get("material", True)
            if not isinstance(material, bool):
                raise InvalidProblemSpec("contradiction material flag must be boolean")
            events.append(
                ProblemContradictionRecorded(
                    diagnostic=ContradictionDiagnostic(
                        left_ref=source,
                        right_ref=target,
                        material=material,
                        message=item.description,
                    )
                )
            )
            if material:
                events.append(
                    ProblemBlockerRecorded(
                        blocker=ProblemBlocker(
                            blocker_id=f"contradiction:{item.id}",
                            description=item.description,
                            ledger_ref=source,
                            resolvable=True,
                        )
                    )
                )
        for item in proposal.items:
            if item.kind != "ACCEPTANCE_CRITERION":
                continue
            mode = str(item.attributes.get("verification_mode", "UNAVAILABLE"))
            required = item.attributes.get("required", True)
            if required is True and mode == "UNAVAILABLE":
                events.append(
                    ProblemBlockerRecorded(
                        blocker=ProblemBlocker(
                            blocker_id=f"acceptance:{item.id}",
                            description=(
                                "Required acceptance criterion has no verification route: "
                                f"{item.description}"
                            ),
                            ledger_ref=refs[item.id],
                            resolvable=False,
                        )
                    )
                )
                blocker_refs[item.id] = refs[item.id]
        # C06 remediation (F07): a material UNRESOLVED UNKNOWN item must
        # produce a concrete blocker -- unresolved state is never silently
        # downgraded just because nothing downstream happened to ask about
        # it. A non-material one keeps its ledger node (retained provenance,
        # explicit M09 basis) but never fabricates a blocker.
        for item in proposal.items:
            if item.kind != "UNKNOWN" or item.origin is not EpistemicOriginLabel.UNRESOLVED:
                continue
            if not _is_material(item):
                continue
            events.append(
                ProblemBlockerRecorded(
                    blocker=ProblemBlocker(
                        blocker_id=f"unknown:{item.id}",
                        description=(
                            f"Material unresolved item has no established value: {item.description}"
                        ),
                        ledger_ref=refs[item.id],
                        resolvable=bool(item.attributes.get("resolvable", False)),
                    )
                )
            )
        return tuple(events), blocker_refs

    def canonical_events(
        self,
        envelope: TaskEnvelope,
        proposal: ProblemFormalisationOutput | None,
        *,
        created_at: datetime,
        uuids: UUIDFactory,
        available_artifacts: frozenset[str] = frozenset(),
        trusted_verifier_results: dict[str, VerificationStatus] | None = None,
        known_ledger_refs: frozenset[tuple[UUID, int]] = frozenset(),
    ) -> tuple[EventPayload, ...]:
        """Build one logically atomic ProblemSpec/M09/M13 event batch.

        This is the *only* public path that can ever produce M03 ledger
        events (C06 / F06): it always calls `formalise` -- the sole place
        anchors and support references are resolved -- on the same
        `proposal` before the now-private `_ledger_events` ever runs, and
        those two calls share one immutable, already-validated
        `ProblemFormalisationOutput`. There is no remaining entry point that
        can construct M03 `LedgerNodeAdded`/`LedgerEdgeAdded`/blocker/
        contradiction events from an unvalidated proposal.
        """
        problem = self.formalise(
            envelope,
            proposal,
            available_artifacts=available_artifacts,
            trusted_verifier_results=trusted_verifier_results,
            known_ledger_refs=known_ledger_refs,
        )
        if proposal is None:
            return (ProblemFormalised(problem=problem),)
        ledger_events, blocker_refs = self._ledger_events(
            proposal, created_at=created_at, uuids=uuids
        )
        if blocker_refs:
            problem = problem.model_copy(
                update={
                    "acceptance_criteria": tuple(
                        criterion.model_copy(update={"blocker_ref": blocker_refs[criterion.id]})
                        if criterion.id in blocker_refs
                        else criterion
                        for criterion in problem.acceptance_criteria
                    )
                }
            )
        return (ProblemFormalised(problem=problem), *ledger_events)

    def formalise(
        self,
        envelope: TaskEnvelope,
        proposal: ProblemFormalisationOutput | None,
        *,
        available_artifacts: frozenset[str] = frozenset(),
        trusted_verifier_results: dict[str, VerificationStatus] | None = None,
        known_ledger_refs: frozenset[tuple[UUID, int]] = frozenset(),
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
        self._validate_items(items)
        # C06 remediation (F03, F06): build the complete proposal-item index
        # and resolve every support reference *before* any item is admitted
        # into the ProblemSpec or the ledger -- never per-item, in
        # declaration order, which would let a forward reference or a cycle
        # slip through undetected.
        validate_support_graph(
            items,
            envelope=envelope,
            available_artifacts=available_artifacts,
            known_ledger_refs=known_ledger_refs,
        )
        for item in items:
            if item.id in ids:
                raise InvalidProblemSpec("problem item IDs must be unique")
            ids.add(item.id)
            provenance = self._provenance(item, envelope, available_artifacts)
            attrs = item.attributes
            if item.kind == "OBJECTIVE":
                evaluator_ref = attrs.get("evaluator_ref")
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
                        priority=cast(int | None, attrs.get("priority")),
                        evaluator_ref=evaluator_ref if isinstance(evaluator_ref, str) else None,
                        description=item.description,
                        provenance=provenance,
                    )
                )
            elif item.kind == "CONSTRAINT":
                kind = str(attrs.get("constraint_kind", "HARD"))
                mode = str(attrs.get("verification_mode", "UNAVAILABLE"))
                proposed_status = VerificationStatus(
                    str(attrs.get("verification_status", "UNKNOWN"))
                )
                if kind not in {"HARD", "SOFT"} or mode not in {
                    "DETERMINISTIC",
                    "MODEL",
                    "HUMAN",
                    "UNAVAILABLE",
                }:
                    raise InvalidProblemSpec("invalid constraint semantics")
                verifier_ref = attrs.get("verifier_ref")
                if verifier_ref is not None and (
                    not isinstance(verifier_ref, str) or not verifier_ref.strip()
                ):
                    raise InvalidProblemSpec("verifier_ref must be a non-empty string")
                status = VerificationStatus.UNKNOWN
                if trusted_verifier_results and item.id in trusted_verifier_results:
                    if mode != "DETERMINISTIC" or verifier_ref is None:
                        raise InvalidProblemSpec(
                            "trusted results require a named deterministic verifier"
                        )
                    status = VerificationStatus(trusted_verifier_results[item.id])
                # Parse the value strictly, but a proposal is evidence to validate,
                # never itself a verifier result (including a proposed ERROR).
                _ = proposed_status
                constraints.append(
                    ConstraintSpec(
                        id=item.id,
                        description=item.description,
                        kind=cast(Literal["HARD", "SOFT"], kind),
                        verification_mode=cast(
                            Literal["DETERMINISTIC", "MODEL", "HUMAN", "UNAVAILABLE"], mode
                        ),
                        verifier_ref=verifier_ref,
                        verification_status=status,
                        source_refs=item.supporting_refs,
                        support=item.support,
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
                candidate_actions = attrs.get("candidate_actions")
                rationale = attrs.get("rationale")
                decision_relevance = attrs.get("decision_relevance")
                unknowns.append(
                    UnknownSpec(
                        id=item.id,
                        description=item.description,
                        domain=attrs.get("domain"),
                        rationale=rationale if isinstance(rationale, str) else None,
                        impact=attrs.get("impact"),
                        decision_relevance=(
                            float(decision_relevance)
                            if isinstance(decision_relevance, int | float)
                            and not isinstance(decision_relevance, bool)
                            else None
                        ),
                        resolvable=(
                            resolvable
                            if isinstance((resolvable := attrs.get("resolvable")), bool)
                            else None
                        ),
                        candidate_actions=(
                            tuple(str(action) for action in candidate_actions)
                            if isinstance(candidate_actions, list | tuple)
                            else ()
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
                decision_relevance = attrs.get("decision_relevance")
                scope = attrs.get("scope")
                assumptions.append(
                    AssumptionSpec(
                        id=item.id,
                        statement=item.description,
                        why_needed=item.basis or "required for formalisation",
                        decision_relevance=(
                            float(decision_relevance)
                            if isinstance(decision_relevance, int | float)
                            and not isinstance(decision_relevance, bool)
                            else None
                        ),
                        scope=scope if isinstance(scope, str) else None,
                        provenance=provenance,
                    )
                )
            elif item.kind == "ACCEPTANCE_CRITERION":
                mode = str(attrs.get("verification_mode", "UNAVAILABLE"))
                if mode not in {"DETERMINISTIC", "MODEL", "HUMAN", "UNAVAILABLE"}:
                    raise InvalidProblemSpec("invalid acceptance-criterion verification mode")
                required = attrs.get("required", True)
                if not isinstance(required, bool):
                    raise InvalidProblemSpec("acceptance-criterion required flag must be boolean")
                criteria.append(
                    AcceptanceCriterion(
                        id=item.id,
                        predicate_description=item.description,
                        verification_mode=mode,
                        required=required,
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
            else:
                raise InvalidProblemSpec(f"unsupported problem item kind: {item.kind}")
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
                        provenance=EpistemicItemProvenance(
                            origin=EpistemicOriginLabel.EXPLICIT_INPUT,
                            anchors=(
                                SourceAnchor(
                                    source_kind=SourceKind.TASK_FIELD,
                                    source_ref=ObjectRef(
                                        object_type="TaskEnvelope",
                                        object_id=str(envelope.task_id),
                                    ),
                                    selector=f"/explicit_constraints/{index}",
                                ),
                            ),
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
            # Legacy free-text mirror of `assumption_items` (C06 / F04: this
            # was declared but never populated).
            assumptions=tuple(item.statement for item in assumptions),
            assumption_items=tuple(assumptions),
            unknowns=tuple(unknowns),
            observables=tuple(observables),
            acceptance_criteria=tuple(criteria),
            relations=tuple(relations),
            output_contract=envelope.requested_output,
        )

    @staticmethod
    def _validate_items(items: tuple[ProblemItemProposal, ...]) -> dict[str, ProblemItemProposal]:
        supported = {
            "OBJECTIVE",
            "CONSTRAINT",
            "DECISION_VARIABLE",
            "FIXED_PARAMETER",
            "UNKNOWN",
            "OBSERVABLE",
            "ASSUMPTION",
            "ACCEPTANCE_CRITERION",
            "RELATION",
        }
        canonical: dict[str, ProblemItemProposal] = {}
        relations: set[tuple[str, str, ProblemRelationKind]] = set()
        lexicographic_priorities: list[int] = []
        for item in items:
            if item.kind not in supported:
                raise InvalidProblemSpec(f"unsupported problem item kind: {item.kind}")
            if item.id in canonical:
                raise InvalidProblemSpec("problem item IDs must be unique")
            canonical[item.id] = item
            if item.kind == "OBJECTIVE":
                priority = item.attributes.get("priority")
                if priority is not None and (
                    isinstance(priority, bool) or not isinstance(priority, int) or priority < 1
                ):
                    raise InvalidProblemSpec("objective priority must be a positive integer")
                if item.attributes.get("direction") == "LEXICOGRAPHIC":
                    if priority is None:
                        raise InvalidProblemSpec(
                            "lexicographic objectives require a complete ordering"
                        )
                    assert isinstance(priority, int) and not isinstance(priority, bool)
                    lexicographic_priorities.append(priority)
                evaluator = item.attributes.get("evaluator_ref")
                if evaluator is not None and (
                    not isinstance(evaluator, str) or not evaluator.strip()
                ):
                    raise InvalidProblemSpec("objective evaluator_ref must be a non-empty string")
        endpoint_ids = {item.id for item in items if item.kind != "RELATION"}
        for item in items:
            if item.kind != "RELATION":
                continue
            attrs = item.attributes
            source = attrs.get("source_id")
            target = attrs.get("target_id")
            if not isinstance(source, str) or not isinstance(target, str):
                raise InvalidProblemSpec("relation endpoints must be item IDs")
            if source not in endpoint_ids or target not in endpoint_ids:
                raise InvalidProblemSpec("relation references an unknown item")
            try:
                kind = ProblemRelationKind(str(attrs.get("relation_kind")))
            except ValueError as error:
                raise InvalidProblemSpec("unsupported problem relation kind") from error
            if source == target:
                raise InvalidProblemSpec("self-relations are invalid")
            key = (source, target, kind)
            if key in relations:
                raise InvalidProblemSpec("duplicate problem relation")
            relations.add(key)
        if len(lexicographic_priorities) != len(set(lexicographic_priorities)):
            raise InvalidProblemSpec("lexicographic objective priorities must be unique")
        if lexicographic_priorities and set(lexicographic_priorities) != set(
            range(1, len(lexicographic_priorities) + 1)
        ):
            raise InvalidProblemSpec("lexicographic objectives require a complete ordering")
        # C06 remediation (F07): an item claiming origin CONTRADICTED makes an
        # epistemic claim that only a genuine CONTRADICTS relation can
        # substantiate. Without one, "contradicted" is just an unverified
        # label -- reject rather than silently accept it as authoritative.
        contradicted_ids = {
            item.id
            for item in items
            if item.kind != "RELATION" and item.origin is EpistemicOriginLabel.CONTRADICTED
        }
        if contradicted_ids:
            related_ids: set[str] = set()
            for item in items:
                if (
                    item.kind == "RELATION"
                    and item.attributes.get("relation_kind") == "CONTRADICTS"
                ):
                    related_ids.add(str(item.attributes.get("source_id")))
                    related_ids.add(str(item.attributes.get("target_id")))
            missing = sorted(contradicted_ids - related_ids)
            if missing:
                raise InvalidProblemSpec(
                    f"item(s) {missing} claim CONTRADICTED origin without a valid "
                    "CONTRADICTS relation"
                )
        return canonical

    @staticmethod
    def _provenance(
        item: ProblemItemProposal, envelope: TaskEnvelope, available_artifacts: frozenset[str]
    ) -> EpistemicItemProvenance:
        for anchor in item.anchors:
            validate_source_anchor(anchor, envelope, available_artifacts)
        return EpistemicItemProvenance(
            origin=item.origin,
            anchors=item.anchors,
            support=item.support,
            basis=item.basis,
            policy_basis=item.policy_basis,
        )
