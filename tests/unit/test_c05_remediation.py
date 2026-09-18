"""Decisive tests for the C05 independent-review remediation round (PR #20).

Each test below targets one numbered finding from the review that produced
these fixes (see the PR body for the full list). Findings that are
documentation/judgment-call only (#5, #6, #11) are exercised minimally here
to prove the chosen scope, not re-litigated beyond that.
"""

from datetime import UTC, datetime
from uuid import UUID

import pytest

from fre.adapters.testing import FakeUUIDFactory
from fre.domain.budget import BudgetProjection, DeploymentLimits
from fre.domain.common import FrozenModel, ObjectRef, OutputContract, PermissionSet
from fre.domain.ledger import EpistemicStatus, LedgerNodeType
from fre.domain.semantic import SourceAnchor, SourceKind
from fre.domain.task import (
    ClassificationBlocked,
    ClassificationDimensionResult,
    ClassificationRecord,
    FloorOverrideRecord,
    Ordinal4,
    TaskEnvelope,
)
from fre.engine import FrontierReasoningEngine
from fre.modules.m01_classifier import TaskClassifier
from fre.modules.m02_budget import default_tier_policy
from fre.modules.m09_ledger import make_node
from fre.runtime.events import (
    BudgetAllocated,
    BudgetRevised,
    LedgerNodeAdded,
    StoredEvent,
    TaskClassified,
)
from fre.runtime.reducer import RunReducer

TASK_ID = UUID(int=9100)


def envelope(**updates: object) -> TaskEnvelope:
    value = TaskEnvelope(
        task_id=TASK_ID,
        text="Decide.",
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(),
    )
    return value.model_copy(update=updates)


def anchor() -> SourceAnchor:
    return SourceAnchor(
        source_kind=SourceKind.TASK_FIELD,
        source_ref=ObjectRef(object_type="TaskEnvelope", object_id=str(TASK_ID)),
        selector="/explicit_constraints/0",
    )


def _dimension(basis: str = "MODEL") -> ClassificationDimensionResult:
    return ClassificationDimensionResult(
        estimated="LOW",
        effective="LOW",
        confidence=0.9 if basis == "MODEL" else None,
        conservative_upper="MEDIUM" if basis == "MODEL" else None,
        source_anchors=(anchor(),) if basis == "MODEL" else (),
        basis=basis,
        rationale="modeled" if basis == "MODEL" else None,
        override_basis=None,
        policy_version="wave3-m01/1.1",
    )


def _record(dimensions: dict[str, ClassificationDimensionResult]) -> ClassificationRecord:
    return ClassificationRecord(
        policy_version="wave3-m01/1.1",
        policy_hash="0" * 64,
        mode="HYBRID",
        fallback_used=False,
        dimensions=dimensions,
    )


def _stored(
    engine: FrontierReasoningEngine, run_id: UUID, payload: FrozenModel, sequence: int
) -> StoredEvent:
    event = engine.make_event(run_id, payload, module_id="test")
    return StoredEvent.model_validate(
        {**event.model_dump(), "created_at": event.created_at, "sequence": sequence}, strict=True
    )


# --- Finding #2 ---------------------------------------------------------


def test_task_classified_model_dimension_without_provenance_is_rejected_by_reducer_directly(
    engine: FrontierReasoningEngine,
) -> None:
    """A forged `TaskClassified` with a MODEL-basis dimension and NO
    `LedgerNodeAdded` provenance anywhere must be rejected by `RunReducer.apply`
    itself -- called directly, bypassing `TaskClassifier`/`engine.append`
    entirely. This must fail on the pre-fix reducer (which never checked
    provenance for `TaskClassified` at all)."""
    run_id = engine.uuids.new()
    reducer = RunReducer()
    state = reducer.initial(run_id)
    record = _record({"consequence": _dimension("MODEL")})
    signature = TaskClassifier().bootstrap_signature(envelope())
    forged = _stored(engine, run_id, TaskClassified(signature=signature, record=record), 1)
    with pytest.raises(ValueError, match="M09 LedgerNodeAdded provenance"):
        reducer.apply(state, forged)


def test_task_classified_model_dimension_with_matching_provenance_is_accepted(
    engine: FrontierReasoningEngine,
) -> None:
    """Positive control for the above: a genuine, matching provenance node
    applied first allows the `TaskClassified` through."""
    run_id = engine.uuids.new()
    reducer = RunReducer()
    state = reducer.initial(run_id)
    result = _dimension("MODEL")
    node = make_node(
        node_id=engine.uuids.new(),
        revision=1,
        node_type=LedgerNodeType.INFERENCE,
        content={"axis": "consequence", **result.model_dump(mode="json")},
        status=EpistemicStatus.PROVISIONAL,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        action_id=engine.uuids.new(),
        module_id="M01",
    )
    provenance = _stored(engine, run_id, LedgerNodeAdded(node=node), 1)
    state = reducer.apply(state, provenance)
    record = _record({"consequence": result})
    signature = TaskClassifier().bootstrap_signature(envelope())
    classified = _stored(engine, run_id, TaskClassified(signature=signature, record=record), 2)
    state = reducer.apply(state, classified)
    assert state.task_signature == signature


def test_task_classified_deterministic_dimension_needs_no_provenance(
    engine: FrontierReasoningEngine,
) -> None:
    """Non-MODEL bases (EXPLICIT/POLICY_FALLBACK/DETERMINISTIC) never made a
    model-derived claim in the first place, so they are exempt from the
    provenance requirement."""
    run_id = engine.uuids.new()
    reducer = RunReducer()
    state = reducer.initial(run_id)
    record = _record({"consequence": _dimension("POLICY_FALLBACK")})
    signature = TaskClassifier().bootstrap_signature(envelope())
    event = _stored(engine, run_id, TaskClassified(signature=signature, record=record), 1)
    state = reducer.apply(state, event)
    assert state.task_signature == signature


# --- Finding #3 ----------------------------------------------------------


def test_budget_revised_without_task_classified_is_rejected_by_reducer_directly(
    engine: FrontierReasoningEngine,
) -> None:
    """A `BudgetRevised` submitted alone -- no accompanying `TaskClassified`
    in the batch, and none previously persisted -- must be rejected."""
    run_id = engine.uuids.new()
    reducer = RunReducer()
    state = reducer.initial(run_id)
    signature = TaskClassifier().bootstrap_signature(envelope())
    from fre.modules.m02_budget import BudgetAllocator

    plan, policy_hash = BudgetAllocator().allocate(
        signature, default_tier_policy(), DeploymentLimits()
    )
    allocated = _stored(
        engine,
        run_id,
        BudgetAllocated(plan=plan, policy_version=plan.policy_version, policy_hash=policy_hash),
        1,
    )
    state = reducer.apply(state, allocated)
    revised = _stored(
        engine,
        run_id,
        BudgetRevised(plan=plan, policy_version=plan.policy_version, policy_hash=policy_hash),
        2,
    )
    with pytest.raises(ValueError, match="budget revision requires an authoritative"):
        reducer.apply(state, revised)


# --- Finding #4 ------------------------------------------------------------


def test_floor_override_record_with_wrong_previous_or_new_floor_is_rejected() -> None:
    """`_floor_overrides_are_exhaustive` must check content, not just axis-name
    presence: a `FloorOverrideRecord` for the right axis but wrong
    previous_floor/new_floor must now be rejected."""
    dimension = ClassificationDimensionResult(
        estimated="LOW",
        effective="HIGH",
        basis="MODEL",
        rationale="modeled",
        override_basis="permission_floor",
        policy_version="wave3-m01/1.1",
    )
    bad_override = FloorOverrideRecord(
        axis="consequence",
        reason="permission_floor",
        policy_version="wave3-m01/1.1",
        policy_hash="0" * 64,
        previous_floor="MEDIUM",  # wrong: dimension.estimated is LOW
        new_floor="HIGH",
        approving_rule="permission_floor",
    )
    with pytest.raises(ValueError):
        ClassificationRecord(
            policy_version="wave3-m01/1.1",
            policy_hash="0" * 64,
            mode="HYBRID",
            fallback_used=False,
            dimensions={"consequence": dimension},
            floor_overrides=(bad_override,),
        )


# --- Finding #6 --------------------------------------------------------


def test_classification_blocked_can_be_caught_and_handled_by_a_caller() -> None:
    """C05 remediation (finding #6): `ClassificationBlocked` is a specific,
    documented, catchable exception type -- a caller wrapping `classify()`
    can catch it and recover gracefully rather than crashing the run."""
    from fre.domain.task import HorizonClass, SearchSpaceClass, TaskType
    from fre.prompts.schemas import (
        ClassificationDimensionProposal,
        ClassificationOutput,
        HorizonProposal,
        SearchSpaceProposal,
        TaskTypeProposal,
    )

    bad_dimension = ClassificationDimensionProposal(
        estimate=Ordinal4.LOW,
        confidence=0.9,
        conservative_upper=Ordinal4.MEDIUM,
        anchors=(anchor(),),
        rationale="   ",  # blank rationale -> ClassificationBlocked
    )
    proposal = ClassificationOutput(
        task_type=TaskTypeProposal(
            estimate=TaskType.ANALYSIS, confidence=0.9, anchors=(anchor(),), rationale="modeled"
        ),
        consequence=bad_dimension,
        reversibility=ClassificationDimensionProposal(
            estimate=Ordinal4.MEDIUM,
            confidence=0.9,
            conservative_upper=Ordinal4.LOW,
            anchors=(anchor(),),
            rationale="modeled",
        ),
        ambiguity=ClassificationDimensionProposal(
            estimate=Ordinal4.LOW,
            confidence=0.9,
            conservative_upper=Ordinal4.MEDIUM,
            anchors=(anchor(),),
            rationale="modeled",
        ),
        evidence_scarcity=ClassificationDimensionProposal(
            estimate=Ordinal4.LOW,
            confidence=0.9,
            conservative_upper=Ordinal4.MEDIUM,
            anchors=(anchor(),),
            rationale="modeled",
        ),
        search_space=SearchSpaceProposal(
            estimate=SearchSpaceClass.BOUNDED,
            confidence=0.9,
            anchors=(anchor(),),
            rationale="modeled",
        ),
        horizon=HorizonProposal(
            estimate=HorizonClass.SHORT, confidence=0.9, anchors=(anchor(),), rationale="modeled"
        ),
    )

    def caller_with_fallback() -> str:
        try:
            TaskClassifier().classify(envelope(explicit_constraints=("x",)), proposal)
        except ClassificationBlocked:
            return "handled-gracefully"
        return "unreachable"

    assert caller_with_fallback() == "handled-gracefully"


# --- Finding #9 ----------------------------------------------------------


def test_canonical_events_rejects_a_non_fresh_prior_projection() -> None:
    """C05 remediation (finding #9): `canonical_events` is only correct for a
    task's initial classification on a run with no prior budget activity.
    Passing a `current_projection` that already carries an allocated plan
    (a non-fresh run) must be rejected with an explicit precondition error,
    rather than silently precomputing the atomicity check against a
    synthetic, zero-usage projection that has nothing to do with real state.
    """
    classifier = TaskClassifier()
    task = envelope(explicit_constraints=("x",))
    tier_policy = default_tier_policy()
    deployment = DeploymentLimits()
    signature = classifier.bootstrap_signature(task)
    from fre.modules.m02_budget import BudgetAllocator

    plan, policy_hash = BudgetAllocator().allocate(signature, tier_policy, deployment)
    non_fresh = BudgetProjection(plan=plan, policy_hash=policy_hash)
    with pytest.raises(ValueError, match="fresh"):
        classifier.canonical_events(
            task,
            None,
            tier_policy=tier_policy,
            deployment=deployment,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            uuids=FakeUUIDFactory(UUID(int=index) for index in range(9000, 9010)),
            current_projection=non_fresh,
        )
