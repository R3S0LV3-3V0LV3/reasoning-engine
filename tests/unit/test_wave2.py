from uuid import UUID

import pytest

from fre.domain.budget import (
    BudgetExceeded,
    BudgetProjection,
    BudgetReservation,
    DeploymentLimits,
    DeploymentPolicyConflict,
    InvalidReservationSettlement,
    ReasoningTier,
    ReservationConflict,
    ResourceVector,
)
from fre.domain.common import ConfidenceAssessment, canonical_hash, canonical_json
from fre.domain.context import (
    CompilerProfile,
    ContextDeltaBaseMismatch,
    ContextOverflow,
    RejectedItem,
)
from fre.domain.ledger import (
    ContradictionResolution,
    ContradictionState,
    EpistemicStatus,
    InvalidLedgerRelation,
    LedgerCycleError,
    LedgerEdge,
    LedgerNode,
    LedgerNodeRef,
    LedgerNodeType,
    LedgerProjection,
    LedgerRelation,
    RevisionHashMismatch,
)
from fre.domain.stop import (
    AcceptanceStatus,
    CostEstimateInterval,
    MarginalValueEstimate,
    MissingRequiredContext,
    StopDisposition,
    StopInputs,
    StopPolicy,
    ValidationStatus,
)
from fre.domain.task import (
    HorizonClass,
    Ordinal4,
    OutputForm,
    SearchSpaceClass,
    TaskSignature,
    TaskType,
)
from fre.modules.m02_budget import BudgetAllocator, default_tier_policy
from fre.modules.m09_ledger import EpistemicLedger, make_node
from fre.modules.m12_context import ContextCompiler, apply_delta, generate_delta
from fre.modules.m13_stop import StopController
from fre.runtime.budget_meter import BudgetMeter


def uid(value: int) -> UUID:
    return UUID(int=value)


def node(
    value: int,
    *,
    revision: int = 1,
    status: EpistemicStatus = EpistemicStatus.SUPPORTED,
    predecessor: str | None = None,
    score: float | None = None,
) -> LedgerNode:
    return make_node(
        node_id=uid(value),
        revision=revision,
        node_type=LedgerNodeType.FACT,
        content={"value": value, "revision": revision},
        status=status,
        created_at="2026-01-01T00:00:00Z",
        action_id=uid(100 + value + revision),
        module_id="M09",
        predecessor_revision_hash=predecessor,
        confidence=ConfidenceAssessment(score=score, method="RULE", explanation="fixture")
        if score is not None
        else None,
    )


def edge(
    value: int,
    source: LedgerNode,
    target: LedgerNode,
    relation: LedgerRelation = LedgerRelation.SUPPORTS,
) -> LedgerEdge:
    return LedgerEdge(
        edge_id=uid(200 + value), source=source.ref, target=target.ref, relation=relation
    )


def signature(level: Ordinal4, search: SearchSpaceClass = SearchSpaceClass.CLOSED) -> TaskSignature:
    return TaskSignature(
        task_type=TaskType.ANALYSIS,
        consequence=level,
        irreversibility=level,
        ambiguity=level,
        search_space=search,
        evidence_scarcity=level,
        horizon=HorizonClass.IMMEDIATE,
        output_form=OutputForm.STRUCTURED,
        dimension_confidence={},
    )


def allocated(level: Ordinal4 = Ordinal4.LOW) -> BudgetProjection:
    allocator = BudgetAllocator()
    plan, policy_hash = allocator.allocate(
        signature(level), default_tier_policy(), DeploymentLimits()
    )
    return BudgetProjection(plan=plan, policy_hash=policy_hash)


def test_ledger_graph_revisions_envelopes_and_hash_chain() -> None:
    service = EpistemicLedger()
    first, dependent = node(1, score=0.2), node(2, score=0.9)
    projection = service.append_node(LedgerProjection(), first)
    projection = service.append_node(projection, dependent)
    projection = service.append_edge(projection, edge(1, first, dependent))
    assert service.ancestors(projection, dependent.ref) == (first.ref,)
    assert service.descendants(projection, first.ref) == (dependent.ref,)
    assert dependent.confidence and dependent.confidence.score == 0.9

    successor = node(1, revision=2, predecessor=first.revision_hash, score=0.4)
    revised = service.revise_node(projection, successor)
    assert service.effective_status(revised, dependent.ref) is EpistemicStatus.STALE
    assert revised.stale_envelopes[0].affected_node_ref == dependent.ref
    assert (
        dependent.confidence
        == next(item for item in revised.nodes if item.ref == dependent.ref).confidence
    )
    assert service.verify_revision_chain(revised, first.node_id)
    assert service.projection_hash(revised) == service.projection_hash(revised)

    tampered = successor.model_copy(update={"content": {"tampered": True}})
    with pytest.raises(RevisionHashMismatch):
        service.verify_revision_chain(
            revised.model_copy(update={"nodes": (*projection.nodes, tampered)}), first.node_id
        )


def test_relation_specific_cycles_and_contradiction_resolution() -> None:
    service = EpistemicLedger()
    left, right = node(1), node(2, status=EpistemicStatus.CONTESTED)
    projection = service.append_node(service.append_node(LedgerProjection(), left), right)
    projection = service.append_edge(projection, edge(1, left, right))
    with pytest.raises(LedgerCycleError):
        service.append_edge(projection, edge(2, right, left))

    base = service.append_node(service.append_node(LedgerProjection(), left), right)
    conflict = edge(3, left, right, LedgerRelation.CONTRADICTS)
    reverse = edge(4, right, left, LedgerRelation.CONTRADICTS)
    base = service.append_edge(service.append_edge(base, conflict), reverse)
    assert len(service.contradictions(base, ContradictionState.OPEN)) == 2
    with pytest.raises(InvalidLedgerRelation):
        service.append_edge(base, edge(5, left, right, LedgerRelation.RESOLVES))
    resolution = ContradictionResolution(
        contradiction_edge_ids=(conflict.edge_id,),
        resolver_ref=left.ref,
        affected_refs=(right.ref,),
        status_transitions=((right.ref, EpistemicStatus.REFUTED),),
        outcome="left supported",
        action_id=uid(50),
        module_id="M09",
    )
    resolved = service.resolve_contradiction(
        base, resolution, edge(5, left, right, LedgerRelation.RESOLVES)
    )
    assert service.contradictions(resolved, ContradictionState.RESOLVED) == (conflict,)
    assert service.effective_status(resolved, right.ref) is EpistemicStatus.REFUTED


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        (Ordinal4.LOW, "T0"),
        (Ordinal4.MEDIUM, "T1"),
        (Ordinal4.HIGH, "T3"),
        (Ordinal4.CRITICAL, "T5"),
    ],
)
def test_allocation_tier_floors_and_policy_provenance(level: Ordinal4, expected: str) -> None:
    allocator = BudgetAllocator()
    policy = default_tier_policy()
    plan, digest = allocator.allocate(signature(level), policy, DeploymentLimits())
    assert plan.tier == expected
    assert digest == allocator.policy_hash(policy, DeploymentLimits())
    if level is Ordinal4.CRITICAL:
        with pytest.raises(DeploymentPolicyConflict):
            allocator.allocate(
                signature(level), policy, DeploymentLimits(maximum_tier=ReasoningTier.T4)
            )


def test_budget_meter_hard_ceiling_reservations_settlement_release_and_burn_rate() -> None:
    meter = BudgetMeter()
    projection = allocated()
    assert meter.remaining(projection).resources.iterations == 1
    reserved = meter.reserve(
        projection,
        BudgetReservation(
            reservation_id="r1", action_id="a1", resources=ResourceVector(iterations=1)
        ),
    )
    with pytest.raises(ReservationConflict):
        meter.reserve(
            reserved,
            BudgetReservation(
                reservation_id="r2", action_id="a2", resources=ResourceVector(iterations=1)
            ),
        )
    with pytest.raises(InvalidReservationSettlement):
        meter.settle(reserved, "r1", ResourceVector(iterations=2))
    released = meter.release(reserved, "r1")
    assert meter.remaining(released).resources.iterations == 1
    consumed = meter.consume(released, ResourceVector(iterations=1))
    assert meter.remaining(consumed).resources.iterations == 0
    with pytest.raises(BudgetExceeded):
        meter.consume(consumed, ResourceVector(iterations=1))
    burn = meter.burn_rate(consumed)
    assert burn.resources["iterations"].committed == 1
    assert meter.remaining(consumed).resources.iterations == 0


def test_context_profiles_hash_markdown_overflow_and_delta() -> None:
    service = EpistemicLedger()
    projection = service.append_node(LedgerProjection(), node(1))
    remaining = BudgetMeter().remaining(allocated())
    compiler = ContextCompiler()
    outputs = [
        compiler.compile(
            run_id=uid(1),
            snapshot_version=2,
            ledger=projection,
            budget_remaining=remaining,
            profile=profile,
            rejected_items=(RejectedItem(ref="c1", reason="fails constraint"),),
        )
        for profile in CompilerProfile
    ]
    assert len({item.packet.packet_hash for item in outputs}) == 3
    assert outputs[0].markdown == compiler.render_markdown(outputs[0].packet)
    assert outputs[0].packet.packet_hash == canonical_hash(
        outputs[0].packet.model_dump(exclude={"packet_hash"})
    )
    delta = generate_delta(outputs[0].packet, outputs[1].packet)
    reconstructed = apply_delta(outputs[0].packet, delta)
    assert reconstructed == outputs[1].packet
    assert canonical_json(reconstructed) == outputs[1].canonical_bytes
    with pytest.raises(ContextDeltaBaseMismatch):
        apply_delta(outputs[2].packet, delta)
    with pytest.raises(ContextOverflow):
        compiler.compile(
            run_id=uid(1),
            snapshot_version=2,
            ledger=projection,
            budget_remaining=remaining,
            profile=CompilerProfile.HANDOFF,
            size_target=10,
        )


def stop_inputs(**changes: object) -> StopInputs:
    values: dict[str, object] = dict(
        budget=BudgetMeter().remaining(allocated()),
        acceptance=AcceptanceStatus.PENDING,
        validation=ValidationStatus.NOT_APPLICABLE,
    )
    values.update(changes)
    return StopInputs.model_validate(values)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"cancelled": True}, StopDisposition.CANCELLED),
        ({"invariant_failure": True}, StopDisposition.FAILED_INVARIANT),
        ({"blocker_required": True}, StopDisposition.BLOCKED),
        ({"mandatory_action": True}, StopDisposition.CONTINUE),
        ({"acceptance": AcceptanceStatus.SATISFIED}, StopDisposition.COMPLETE),
    ],
)
def test_stop_precedence_dispositions(
    changes: dict[str, object], expected: StopDisposition
) -> None:
    if changes.get("blocker_required"):
        changes["epistemic_trigger_refs"] = (LedgerNodeRef(node_id=uid(1), revision=1),)
    decision = StopController().evaluate(stop_inputs(**changes), StopPolicy(version="1.0"))
    assert decision.disposition is expected
    assert (decision.context_request is not None) is (expected is not StopDisposition.CONTINUE)


def test_bounded_marginal_value_and_provenance() -> None:
    confidence = ConfidenceAssessment(score=0.8, method="RULE", explanation="fixture")
    valuable = MarginalValueEstimate(
        lower=3, point=4, upper=5, confidence=confidence, method_version="v1"
    )
    cost = CostEstimateInterval(lower=1, point=1, upper=2, method_version="c1")
    controller = StopController()
    assert (
        controller.evaluate(
            stop_inputs(executable_work=True, value=valuable, cost=cost), StopPolicy(version="1")
        ).disposition
        is StopDisposition.CONTINUE
    )
    not_valuable = valuable.model_copy(update={"lower": 0.0, "point": 0.5, "upper": 0.9})
    assert (
        controller.evaluate(
            stop_inputs(executable_work=True, value=not_valuable, cost=cost),
            StopPolicy(version="1"),
        ).disposition
        is StopDisposition.COMPLETE
    )
    overlap = valuable.model_copy(update={"lower": 1.0, "point": 1.5, "upper": 2.5})
    assert (
        controller.evaluate(
            stop_inputs(executable_work=True, value=overlap, cost=cost), StopPolicy(version="1")
        ).disposition
        is StopDisposition.CONTINUE
    )
    with pytest.raises(MissingRequiredContext):
        controller.evaluate(stop_inputs(blocker_required=True), StopPolicy(version="1"))


def test_adversarial_optional_proposer_terminates_at_hard_ceiling() -> None:
    meter = BudgetMeter()
    projection = allocated()
    controller = StopController()
    decisions = []
    while True:
        remaining = meter.remaining(projection)
        decision = controller.evaluate(
            StopInputs(
                budget=remaining,
                acceptance=AcceptanceStatus.PENDING,
                validation=ValidationStatus.NOT_APPLICABLE,
                required_work=True,
                executable_work=remaining.resources.iterations > 0,
            ),
            StopPolicy(version="termination/1.0"),
        )
        decisions.append(decision.disposition)
        if decision.disposition is not StopDisposition.CONTINUE:
            break
        projection = meter.consume(projection, ResourceVector(iterations=1))
    assert decisions == [StopDisposition.CONTINUE, StopDisposition.PARTIAL_BUDGET]


@pytest.mark.parametrize(
    ("consequence", "irreversibility", "ambiguity", "search", "evidence", "expected"),
    [
        (Ordinal4.LOW, Ordinal4.LOW, Ordinal4.LOW, SearchSpaceClass.CLOSED, Ordinal4.LOW, "T0"),
        (Ordinal4.MEDIUM, Ordinal4.LOW, Ordinal4.LOW, SearchSpaceClass.CLOSED, Ordinal4.LOW, "T1"),
        (Ordinal4.LOW, Ordinal4.LOW, Ordinal4.HIGH, SearchSpaceClass.CLOSED, Ordinal4.LOW, "T2"),
        (Ordinal4.HIGH, Ordinal4.LOW, Ordinal4.LOW, SearchSpaceClass.CLOSED, Ordinal4.LOW, "T3"),
        (
            Ordinal4.CRITICAL,
            Ordinal4.LOW,
            Ordinal4.LOW,
            SearchSpaceClass.CLOSED,
            Ordinal4.LOW,
            "T4",
        ),
        (
            Ordinal4.CRITICAL,
            Ordinal4.CRITICAL,
            Ordinal4.LOW,
            SearchSpaceClass.CLOSED,
            Ordinal4.LOW,
            "T5",
        ),
    ],
)
def test_each_tier_fixture(
    consequence: Ordinal4,
    irreversibility: Ordinal4,
    ambiguity: Ordinal4,
    search: SearchSpaceClass,
    evidence: Ordinal4,
    expected: str,
) -> None:
    task = TaskSignature(
        task_type=TaskType.ANALYSIS,
        consequence=consequence,
        irreversibility=irreversibility,
        ambiguity=ambiguity,
        search_space=search,
        evidence_scarcity=evidence,
        horizon=HorizonClass.IMMEDIATE,
        output_form=OutputForm.TEXT,
        dimension_confidence={},
    )
    plan, _ = BudgetAllocator().allocate(task, default_tier_policy(), DeploymentLimits())
    assert plan.tier == expected
