"""Decisive coverage for the M01 complete-classification/provenance contract (F05).

Every one of the 8 classification axes must carry a validated value, a
confidence/uncertainty treatment, a rationale, and resolvable support (or an
explicit deterministic-policy basis) -- regardless of whether that axis
currently feeds a budget/routing decision (see the C05 remediation note on
`ClassificationDimensionResult` for exactly which axes do today). These
tests pin that provenance invariant across all 8 axes, the preliminary/final
signature handoff into M02, and the M09 provenance batch.
"""

from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from fre.adapters.testing import FakeUUIDFactory
from fre.domain.budget import DeploymentLimits, InvalidBudgetRevision
from fre.domain.common import ObjectRef, OutputContract, PermissionSet
from fre.domain.semantic import SourceAnchor, SourceKind
from fre.domain.task import (
    ClassificationBlocked,
    ClassificationRecord,
    FloorOverrideRecord,
    HorizonClass,
    Ordinal4,
    SearchSpaceClass,
    TaskEnvelope,
    TaskType,
)
from fre.engine import FrontierReasoningEngine
from fre.modules.m01_classifier import TaskClassifier
from fre.modules.m02_budget import BudgetAllocator, default_tier_policy
from fre.prompts.schemas import (
    ClassificationDimensionProposal,
    ClassificationOutput,
    HorizonProposal,
    SearchSpaceProposal,
    TaskTypeProposal,
)
from fre.runtime.events import (
    LedgerNodeAdded,
    TaskClassified,
    TaskPreliminarilyClassified,
)

TASK_ID = UUID(int=4200)


def envelope(**updates: object) -> TaskEnvelope:
    value = TaskEnvelope(
        task_id=TASK_ID,
        text="Decide whether to proceed.",
        explicit_constraints=("budget <= 100",),
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(),
    )
    return value.model_copy(update=updates)


def anchor(selector: str = "/explicit_constraints/0") -> SourceAnchor:
    return SourceAnchor(
        source_kind=SourceKind.TASK_FIELD,
        source_ref=ObjectRef(object_type="TaskEnvelope", object_id=str(TASK_ID)),
        selector=selector,
    )


def ordinal_dimension(
    estimate: Ordinal4,
    upper: Ordinal4,
    *,
    rationale: str = "modeled",
    anchors: tuple[SourceAnchor, ...] = (anchor(),),
) -> ClassificationDimensionProposal:
    return ClassificationDimensionProposal(
        estimate=estimate,
        confidence=0.9,
        conservative_upper=upper,
        anchors=anchors,
        rationale=rationale,
    )


def full_proposal(
    *,
    consequence: ClassificationDimensionProposal | None = None,
    reversibility: ClassificationDimensionProposal | None = None,
    ambiguity: ClassificationDimensionProposal | None = None,
    evidence_scarcity: ClassificationDimensionProposal | None = None,
    search_space: SearchSpaceProposal | None = None,
    horizon: HorizonProposal | None = None,
    task_type: TaskTypeProposal | None = None,
) -> ClassificationOutput:
    return ClassificationOutput(
        task_type=task_type
        or TaskTypeProposal(
            estimate=TaskType.ANALYSIS, confidence=0.9, anchors=(anchor(),), rationale="modeled"
        ),
        consequence=consequence or ordinal_dimension(Ordinal4.LOW, Ordinal4.MEDIUM),
        # Reversibility is descending-risk: conservative bound <= estimate.
        reversibility=reversibility or ordinal_dimension(Ordinal4.MEDIUM, Ordinal4.LOW),
        ambiguity=ambiguity or ordinal_dimension(Ordinal4.LOW, Ordinal4.MEDIUM),
        evidence_scarcity=evidence_scarcity or ordinal_dimension(Ordinal4.LOW, Ordinal4.MEDIUM),
        search_space=search_space
        or SearchSpaceProposal(
            estimate=SearchSpaceClass.BOUNDED,
            confidence=0.9,
            anchors=(anchor(),),
            rationale="modeled",
        ),
        horizon=horizon
        or HorizonProposal(
            estimate=HorizonClass.SHORT, confidence=0.9, anchors=(anchor(),), rationale="modeled"
        ),
    )


@pytest.mark.unit
def test_all_eight_axes_carry_full_provenance() -> None:
    _signature, record = TaskClassifier().classify(envelope(), full_proposal())
    expected_axes = {
        "task_type",
        "consequence",
        "irreversibility",
        "ambiguity",
        "evidence_scarcity",
        "search_space",
        "horizon",
        "output_form",
    }
    assert set(record.dimensions) == expected_axes
    for name, result in record.dimensions.items():
        assert result.estimated
        assert result.effective
        assert result.rationale, name
        assert result.basis
        assert result.policy_version


@pytest.mark.unit
def test_missing_rationale_on_material_dimension_is_blocked() -> None:
    bad = ordinal_dimension(Ordinal4.LOW, Ordinal4.MEDIUM, rationale="   ")
    with pytest.raises(ClassificationBlocked):
        TaskClassifier().classify(envelope(), full_proposal(consequence=bad))


@pytest.mark.unit
def test_missing_support_on_material_dimension_is_blocked() -> None:
    bad = ordinal_dimension(Ordinal4.LOW, Ordinal4.MEDIUM, anchors=())
    with pytest.raises(ClassificationBlocked):
        TaskClassifier().classify(envelope(), full_proposal(ambiguity=bad))


@pytest.mark.unit
@pytest.mark.parametrize("axis", ["consequence", "ambiguity", "evidence_scarcity"])
def test_ill_ordered_bounds_are_rejected_on_every_ascending_axis(axis: str) -> None:
    bad = ordinal_dimension(Ordinal4.HIGH, Ordinal4.LOW)  # upper below estimate: ill-ordered
    proposal = (
        full_proposal(consequence=bad)
        if axis == "consequence"
        else full_proposal(ambiguity=bad)
        if axis == "ambiguity"
        else full_proposal(evidence_scarcity=bad)
    )
    with pytest.raises(ClassificationBlocked):
        TaskClassifier().classify(envelope(), proposal)


@pytest.mark.unit
def test_reversibility_orientation_and_derived_irreversibility_bound() -> None:
    # Correct orientation: conservative bound is *less* reversible (LOW) than
    # the estimate (HIGH) -- must be accepted and produce a well-formed
    # derived irreversibility bound.
    good = ordinal_dimension(Ordinal4.HIGH, Ordinal4.LOW)
    signature, record = TaskClassifier().classify(envelope(), full_proposal(reversibility=good))
    irreversibility = record.dimensions["irreversibility"]
    # reversibility HIGH -> irreversibility MEDIUM; conservative reversibility
    # LOW -> irreversibility CRITICAL.
    assert irreversibility.estimated == Ordinal4.MEDIUM
    assert irreversibility.conservative_upper == Ordinal4.CRITICAL
    assert signature.irreversibility in {Ordinal4.LOW, Ordinal4.HIGH, Ordinal4.MEDIUM}

    # Inverted (wrong) orientation: conservative bound is *more* reversible
    # than the estimate -- this is the exact defect being fixed and must be
    # rejected before any inversion happens.
    inverted = ordinal_dimension(Ordinal4.LOW, Ordinal4.HIGH)
    with pytest.raises(ClassificationBlocked):
        TaskClassifier().classify(envelope(), full_proposal(reversibility=inverted))


@pytest.mark.unit
def test_low_horizon_confidence_escalates_to_conservative_default() -> None:
    uncertain_horizon = HorizonProposal(
        estimate=HorizonClass.IMMEDIATE, confidence=0.1, anchors=(anchor(),), rationale="unsure"
    )
    signature, record = TaskClassifier().classify(
        envelope(), full_proposal(horizon=uncertain_horizon)
    )
    assert signature.horizon is HorizonClass.LONG
    horizon_result = record.dimensions["horizon"]
    assert horizon_result.estimated == HorizonClass.IMMEDIATE
    assert horizon_result.effective == HorizonClass.LONG
    assert horizon_result.override_basis == "low_confidence_escalation"
    assert any(
        override.axis == "horizon" and override.reason == "low_confidence_escalation"
        for override in record.floor_overrides
    )


@pytest.mark.unit
def test_ambiguity_floor_overrides_a_confident_low_self_report() -> None:
    """C05 remediation (finding #1): ambiguity must not be an unconditional
    no-op floor. Two or more distinct external systems in play (here: network
    access plus external writes) deterministically floor ambiguity at MEDIUM,
    regardless of how confidently a proposal reports LOW."""
    confident_low = ClassificationDimensionProposal(
        estimate=Ordinal4.LOW,
        confidence=0.99,
        conservative_upper=Ordinal4.LOW,
        anchors=(anchor(),),
        rationale="modeled",
    )
    signature, record = TaskClassifier().classify(
        envelope(
            execution_permissions=PermissionSet(allow_network=True, allow_external_writes=True)
        ),
        full_proposal(ambiguity=confident_low),
    )
    ambiguity = record.dimensions["ambiguity"]
    assert ambiguity.estimated == Ordinal4.LOW
    assert ambiguity.effective == Ordinal4.MEDIUM
    assert ambiguity.override_basis == "permission_floor"
    assert signature.ambiguity is Ordinal4.MEDIUM


@pytest.mark.unit
def test_evidence_scarcity_floor_overrides_a_confident_low_self_report() -> None:
    """C05 remediation (finding #1): evidence_scarcity must not be an
    unconditional no-op floor. A task with no attachments, no explicit
    constraints, and no permission to fetch evidence externally cannot
    possibly ground a LOW-scarcity claim -- that must be floored to MEDIUM
    regardless of self-reported confidence."""
    # No explicit_constraints in this envelope, so the default `/explicit_
    # constraints/0` anchor cannot resolve; anchor into `user_metadata`
    # instead (relevant to every axis, and harmless here since "context" is
    # not one of the reserved explicit-ordinal keys).
    context_anchor = anchor(selector="/user_metadata/context")
    confident_low = ClassificationDimensionProposal(
        estimate=Ordinal4.LOW,
        confidence=0.99,
        conservative_upper=Ordinal4.LOW,
        anchors=(context_anchor,),
        rationale="modeled",
    )
    signature, record = TaskClassifier().classify(
        envelope(
            explicit_constraints=(),
            user_metadata={"context": "note"},
            execution_permissions=PermissionSet(),
        ),
        full_proposal(
            consequence=ordinal_dimension(Ordinal4.LOW, Ordinal4.MEDIUM, anchors=(context_anchor,)),
            reversibility=ordinal_dimension(
                Ordinal4.MEDIUM, Ordinal4.LOW, anchors=(context_anchor,)
            ),
            ambiguity=ordinal_dimension(Ordinal4.LOW, Ordinal4.MEDIUM, anchors=(context_anchor,)),
            evidence_scarcity=confident_low,
            search_space=SearchSpaceProposal(
                estimate=SearchSpaceClass.BOUNDED,
                confidence=0.9,
                anchors=(context_anchor,),
                rationale="modeled",
            ),
            horizon=HorizonProposal(
                estimate=HorizonClass.SHORT,
                confidence=0.9,
                anchors=(context_anchor,),
                rationale="modeled",
            ),
            task_type=TaskTypeProposal(
                estimate=TaskType.ANALYSIS,
                confidence=0.9,
                anchors=(context_anchor,),
                rationale="modeled",
            ),
        ),
    )
    scarcity = record.dimensions["evidence_scarcity"]
    assert scarcity.estimated == Ordinal4.LOW
    assert scarcity.effective == Ordinal4.MEDIUM
    assert scarcity.override_basis == "permission_floor"
    assert signature.evidence_scarcity is Ordinal4.MEDIUM


@pytest.mark.unit
def test_ambiguity_and_evidence_scarcity_floors_are_no_ops_when_signals_are_absent() -> None:
    """Sanity check: the new floors only engage on their deterministic
    trigger, not unconditionally -- the default envelope (single permission,
    explicit constraints present) stays at LOW."""
    signature, _record = TaskClassifier().classify(envelope(), full_proposal())
    assert signature.ambiguity is Ordinal4.LOW
    assert signature.evidence_scarcity is Ordinal4.LOW


@pytest.mark.unit
def test_irrelevant_anchor_is_rejected_for_unconnected_axis() -> None:
    """C05 remediation (finding #7): an anchor whose path is entirely
    unrelated to the axis it is claimed to support must be rejected, not
    merely checked for existence. `/requested_output` (the task's output
    contract) cannot plausibly substantiate a `consequence` claim."""
    from fre.modules.source_anchors import IrrelevantSourceAnchor

    unrelated_anchor = SourceAnchor(
        source_kind=SourceKind.TASK_FIELD,
        source_ref=ObjectRef(object_type="TaskEnvelope", object_id=str(TASK_ID)),
        selector="/requested_output",
    )
    bad = ordinal_dimension(Ordinal4.LOW, Ordinal4.MEDIUM, anchors=(unrelated_anchor,))
    with pytest.raises(IrrelevantSourceAnchor):
        TaskClassifier().classify(envelope(), full_proposal(consequence=bad))


@pytest.mark.unit
def test_horizon_low_confidence_fallback_is_independent_of_no_proposal_fallback() -> None:
    """C05 remediation (finding #8): `low_confidence_horizon_fallback` must be
    settable independently of `fallback_horizon` (the "no proposal at all"
    target), the way `search_space` already keeps its two fallbacks apart."""
    from fre.modules.m01_classifier import ClassificationPolicy

    policy = ClassificationPolicy(
        fallback_horizon=HorizonClass.LONG,
        low_confidence_horizon_fallback=HorizonClass.SHORT,
    )
    # No proposal at all -> uses fallback_horizon.
    no_proposal_signature, _ = TaskClassifier(policy).classify(envelope(), None)
    assert no_proposal_signature.horizon is HorizonClass.LONG

    # Low-confidence proposal -> uses the independent low-confidence target,
    # which now diverges from fallback_horizon.
    uncertain_horizon = HorizonProposal(
        estimate=HorizonClass.IMMEDIATE, confidence=0.1, anchors=(anchor(),), rationale="unsure"
    )
    low_confidence_signature, record = TaskClassifier(policy).classify(
        envelope(), full_proposal(horizon=uncertain_horizon)
    )
    assert low_confidence_signature.horizon is HorizonClass.SHORT
    assert record.dimensions["horizon"].effective == HorizonClass.SHORT


@pytest.mark.unit
def test_floor_overrides_are_exhaustively_audited() -> None:
    """Every dimension whose effective value diverges from its estimate must be audited."""
    signature, record = TaskClassifier().classify(
        envelope(execution_permissions=PermissionSet(allow_external_writes=True)),
        full_proposal(consequence=ordinal_dimension(Ordinal4.LOW, Ordinal4.MEDIUM)),
    )
    consequence = record.dimensions["consequence"]
    assert consequence.estimated == Ordinal4.LOW
    assert consequence.effective == Ordinal4.HIGH
    assert consequence.override_basis == "permission_floor"
    assert any(override.axis == "consequence" for override in record.floor_overrides)
    assert signature.consequence is Ordinal4.HIGH


@pytest.mark.unit
def test_unaudited_floor_change_is_rejected_by_the_record_invariant() -> None:
    _signature, record = TaskClassifier().classify(
        envelope(execution_permissions=PermissionSet(allow_external_writes=True)),
        full_proposal(),
    )
    tampered = record.model_dump(mode="json")
    # Strip the audit trail while a real deviation still exists: this must be
    # structurally impossible to reconstruct as a valid ClassificationRecord.
    tampered["floor_overrides"] = ()
    assert any(
        dimension["effective"] != dimension["estimated"]
        for dimension in tampered["dimensions"].values()
    )
    with pytest.raises(ValidationError):
        ClassificationRecord.model_validate(tampered)


@pytest.mark.unit
def test_floor_override_record_shape_is_complete() -> None:
    record = FloorOverrideRecord(
        axis="consequence",
        reason="permission_floor",
        policy_version="wave3-m01/1.1",
        policy_hash="0" * 64,
        previous_floor="LOW",
        new_floor="HIGH",
        approving_rule="permission_floor",
    )
    assert record.axis and record.previous_floor != record.new_floor


@pytest.mark.unit
def test_preliminary_signature_is_distinct_from_and_never_more_permissive_than_final() -> None:
    classifier = TaskClassifier()
    task = envelope(execution_permissions=PermissionSet(allow_external_writes=True))
    preliminary = classifier.bootstrap_signature(task)
    final, _record = classifier.classify(task, full_proposal())
    assert preliminary != final
    allocator = BudgetAllocator()
    tier_policy = default_tier_policy()
    deployment = DeploymentLimits()
    preliminary_plan, _ = allocator.allocate(preliminary, tier_policy, deployment)
    final_plan, _ = allocator.allocate(final, tier_policy, deployment)
    from fre.modules.m02_budget import TIER_ORDER

    assert TIER_ORDER[final_plan.tier] >= TIER_ORDER[preliminary_plan.tier]


@pytest.mark.unit
def test_bootstrap_to_final_revision_cannot_fall_below_committed_plus_reserved() -> None:
    """Extends the existing InvalidBudgetRevision guard to the M01 bootstrap handoff."""
    from fre.domain.budget import BudgetProjection, BudgetReservation, ResourceVector

    classifier = TaskClassifier()
    task = envelope()
    bootstrap = classifier.bootstrap_signature(task)
    allocator = BudgetAllocator()
    tier_policy = default_tier_policy()
    deployment = DeploymentLimits()
    bootstrap_plan, bootstrap_hash = allocator.allocate(bootstrap, tier_policy, deployment)
    projection = BudgetProjection(
        plan=bootstrap_plan,
        policy_hash=bootstrap_hash,
        committed=ResourceVector(llm_calls=bootstrap_plan.limits.max_llm_calls),
        reservations=(
            BudgetReservation(
                reservation_id="r1", action_id="a1", resources=ResourceVector(llm_calls=1)
            ),
        ),
    )
    # A "final" plan identical to bootstrap cannot satisfy committed + reserved usage.
    with pytest.raises(InvalidBudgetRevision):
        allocator.revise(projection, bootstrap_plan, bootstrap_hash)


@pytest.mark.integration
def test_canonical_events_batch_is_atomic_and_distinguishes_preliminary_from_final(
    engine: FrontierReasoningEngine,
) -> None:
    handle = engine.create_run({"m01-classification-contract": True})
    task = envelope()
    classifier = TaskClassifier()
    payloads = classifier.canonical_events(
        task,
        full_proposal(),
        tier_policy=default_tier_policy(),
        deployment=DeploymentLimits(),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        uuids=FakeUUIDFactory(UUID(int=index) for index in range(5000, 5100)),
    )
    kinds = [type(payload).__name__ for payload in payloads]
    assert kinds[0] == "TaskPreliminarilyClassified"
    assert kinds[1] == "BudgetAllocated"
    assert kinds[-1] == "BudgetRevised"
    assert any(kind == "LedgerNodeAdded" for kind in kinds)
    # C05 remediation (finding #2/#3): M09 provenance for every MODEL-basis
    # axis must land *before* `TaskClassified`, which must itself land before
    # `BudgetRevised` -- this ordering is what lets `RunReducer.apply` enforce
    # both admission checks as simple backward looks at already-applied state.
    classified_index = kinds.index("TaskClassified")
    assert all(kind != "LedgerNodeAdded" for kind in kinds[classified_index + 1 :]), (
        "all provenance must precede TaskClassified"
    )
    assert classified_index < len(kinds) - 1, "TaskClassified must precede BudgetRevised"

    preliminary_payload = next(p for p in payloads if isinstance(p, TaskPreliminarilyClassified))
    final_payload = next(p for p in payloads if isinstance(p, TaskClassified))
    assert preliminary_payload.signature != final_payload.signature

    events = tuple(
        engine.make_event(handle.run_id, payload, module_id="M01") for payload in payloads
    )

    # Corrupt the LedgerNodeAdded event so the batch fails mid-way: nothing in
    # the batch (including TaskClassified and the BudgetRevised event) may
    # commit -- the same all-or-nothing guarantee C04 established for M09/F09.
    ledger_index = next(
        index for index, payload in enumerate(payloads) if isinstance(payload, LedgerNodeAdded)
    )
    ledger_payload = payloads[ledger_index]
    assert isinstance(ledger_payload, LedgerNodeAdded)
    bad_node = ledger_payload.node.model_copy(update={"revision_hash": "f" * 64})
    invalid_events = list(events)
    invalid_events[ledger_index] = engine.make_event(
        handle.run_id, LedgerNodeAdded(node=bad_node), module_id="M01"
    )
    with pytest.raises(ValueError, match="revision hash"):
        engine.append(handle.run_id, handle.version, tuple(invalid_events))
    assert engine.inspect(handle.run_id).version == handle.version
    assert engine.inspect(handle.run_id).task_signature is None

    # The clean batch commits atomically and replays deterministically.
    engine.append(handle.run_id, handle.version, events)
    state = engine.inspect(handle.run_id)
    assert state.preliminary_task_signature == preliminary_payload.signature
    assert state.task_signature == final_payload.signature
    assert state.budget.plan is not None
    snapshot_hash = engine.snapshot(handle.run_id)
    assert engine.replay(handle.run_id).state_hash == snapshot_hash
    assert engine.replay_from_snapshot(handle.run_id).state_hash == snapshot_hash


@pytest.mark.unit
@pytest.mark.parametrize(
    "proposal_factory",
    [
        lambda: full_proposal(
            consequence=ordinal_dimension(Ordinal4.LOW, Ordinal4.LOW),
            reversibility=ordinal_dimension(Ordinal4.CRITICAL, Ordinal4.CRITICAL),
            ambiguity=ordinal_dimension(Ordinal4.LOW, Ordinal4.LOW),
            evidence_scarcity=ordinal_dimension(Ordinal4.LOW, Ordinal4.LOW),
            search_space=SearchSpaceProposal(
                estimate=SearchSpaceClass.CLOSED,
                confidence=0.95,
                anchors=(anchor(),),
                rationale="closed",
            ),
            horizon=HorizonProposal(
                estimate=HorizonClass.IMMEDIATE,
                confidence=0.95,
                anchors=(anchor(),),
                rationale="short",
            ),
        ),  # low complexity, no search, short horizon
        lambda: full_proposal(
            consequence=ordinal_dimension(Ordinal4.CRITICAL, Ordinal4.CRITICAL),
            reversibility=ordinal_dimension(Ordinal4.LOW, Ordinal4.LOW),
            ambiguity=ordinal_dimension(Ordinal4.CRITICAL, Ordinal4.CRITICAL),
            evidence_scarcity=ordinal_dimension(Ordinal4.CRITICAL, Ordinal4.CRITICAL),
            search_space=SearchSpaceProposal(
                estimate=SearchSpaceClass.OPEN,
                confidence=0.95,
                anchors=(anchor(),),
                rationale="open",
            ),
            horizon=HorizonProposal(
                estimate=HorizonClass.LONG, confidence=0.95, anchors=(anchor(),), rationale="long"
            ),
        ),  # high complexity, open search, long horizon
        lambda: full_proposal(
            horizon=HorizonProposal(
                estimate=HorizonClass.SHORT, confidence=0.2, anchors=(anchor(),), rationale="unsure"
            )
        ),  # uncertain horizon -> escalation
    ],
)
def test_golden_classification_cases_round_trip(proposal_factory) -> None:  # type: ignore[no-untyped-def]
    signature, record = TaskClassifier().classify(envelope(), proposal_factory())
    allocator = BudgetAllocator()
    plan, _hash = allocator.allocate(signature, default_tier_policy(), DeploymentLimits())
    assert plan.tier is not None
    assert record.dimensions  # every case fully populated


@pytest.mark.unit
def test_golden_floor_override_case() -> None:
    signature, record = TaskClassifier().classify(
        envelope(execution_permissions=PermissionSet(allow_network=True)),
        full_proposal(reversibility=ordinal_dimension(Ordinal4.CRITICAL, Ordinal4.LOW)),
    )
    assert record.floor_overrides == () or all(
        override.policy_hash == record.policy_hash for override in record.floor_overrides
    )
    assert signature.irreversibility is not None
