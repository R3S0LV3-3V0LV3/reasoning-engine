"""Decisive Wave 3 semantic-front-end unit tests."""

import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from fre.adapters.testing import FakeUUIDFactory
from fre.domain.budget import DeploymentLimits
from fre.domain.common import ObjectRef, OutputContract, PermissionSet, canonical_hash
from fre.domain.ledger import EpistemicStatus, LedgerRelation
from fre.domain.problem import ProblemRelation, ProblemRelationKind, VerificationStatus
from fre.domain.representation import RepresentationKind
from fre.domain.semantic import (
    EpistemicItemProvenance,
    EpistemicOriginLabel,
    SourceAnchor,
    SourceKind,
)
from fre.domain.task import Ordinal4, TaskEnvelope
from fre.modules.m01_classifier import (
    TaskClassifier,
    reversibility_to_irreversibility,
)
from fre.modules.m02_budget import BudgetAllocator, default_tier_policy
from fre.modules.m03_formaliser import InvalidProblemSpec, ProblemFormaliser
from fre.modules.m04_representation import RepresentationSelector
from fre.modules.source_anchors import InvalidSourceAnchor, validate_source_anchor
from fre.prompts import default_output_schema_registry, default_prompt_registry
from fre.prompts.registry import PromptDefinition, PromptVersionConflict
from fre.prompts.schemas import (
    ClassificationOutput,
    ProblemFormalisationOutput,
    RepresentationAdjudicationOutput,
)
from fre.runtime.events import (
    LedgerEdgeAdded,
    LedgerNodeAdded,
    ProblemBlockerRecorded,
    ProblemContradictionRecorded,
    ProblemFormalised,
)


def envelope(*, external_write: bool = False) -> TaskEnvelope:
    return TaskEnvelope(
        task_id=UUID(int=1),
        text="Minimize cost subject to cost <= 10.",
        explicit_constraints=("cost <= 10",),
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(allow_external_writes=external_write),
    )


def _anchor() -> dict[str, object]:
    return {
        "source_kind": "TASK_FIELD",
        "source_ref": {
            "object_type": "TaskEnvelope",
            "object_id": "00000000-0000-0000-0000-000000000001",
        },
        "selector": "/explicit_constraints/0",
    }


def classification(confidence: float = 0.9) -> ClassificationOutput:
    dimension = {
        "estimate": "LOW",
        "confidence": confidence,
        "conservative_upper": "MEDIUM",
        "anchors": [_anchor()],
        "rationale": "fixture",
    }
    # `reversibility` is descending-risk (higher ordinal == more reversible, i.e.
    # safer), so its conservative bound must sit at or below the estimate --
    # the opposite orientation from every other ordinal axis.
    reversibility_dimension = {
        "estimate": "MEDIUM",
        "confidence": confidence,
        "conservative_upper": "LOW",
        "anchors": [_anchor()],
        "rationale": "fixture",
    }
    categorical = {"confidence": confidence, "anchors": [_anchor()], "rationale": "fixture"}
    return ClassificationOutput.model_validate_json(
        json.dumps(
            {
                "task_type": {"estimate": "DECISION", **categorical},
                "consequence": dimension,
                "reversibility": reversibility_dimension,
                "ambiguity": dimension,
                "evidence_scarcity": dimension,
                "search_space": {"estimate": "CLOSED", **categorical},
                "horizon": {"estimate": "SHORT", **categorical},
            }
        )
    )


@pytest.mark.unit
def test_prompt_and_schema_registries_are_versioned_and_integrity_checked() -> None:
    prompts = default_prompt_registry()
    first = prompts.get("m01.classify", "1.0")
    assert prompts.render("m01.classify", "1.0", {"b": 2, "a": 1}) == prompts.render(
        "m01.classify", "1.0", {"a": 1, "b": 2}
    )
    prompts.register(first)
    conflicting = PromptDefinition.create(
        **{**first.model_dump(exclude={"template_hash"}), "template": "materially different"}
    )
    with pytest.raises(PromptVersionConflict):
        prompts.register(conflicting)
    schemas = default_output_schema_registry()
    definition, model = schemas.get("m01.classification-output", "1.0")
    assert definition.schema_hash == canonical_hash(model.model_json_schema())
    with pytest.raises(ValidationError):
        schemas.validate("m01.classification-output", "1.0", {"task_type": "ANALYSIS"})


@pytest.mark.unit
def test_source_anchor_and_explicit_origin_are_mechanical() -> None:
    task = envelope()
    anchor = SourceAnchor(
        source_kind=SourceKind.TASK_TEXT,
        source_ref=ObjectRef(object_type="TaskEnvelope", object_id=str(task.task_id)),
        selector="/text",
        char_start=0,
        char_end=8,
        excerpt_hash=hashlib.sha256(b"Minimize").hexdigest(),
    )
    validate_source_anchor(anchor, task)
    with pytest.raises(InvalidSourceAnchor):
        validate_source_anchor(anchor.model_copy(update={"char_end": 999}), task)
    with pytest.raises(ValidationError):
        EpistemicItemProvenance(origin=EpistemicOriginLabel.EXPLICIT_INPUT)


@pytest.mark.unit
def test_m01_risk_floor_low_confidence_and_fallback_feed_frozen_m02() -> None:
    classifier = TaskClassifier()
    high_confidence, _ = classifier.classify(envelope(), classification())
    low_confidence, _ = classifier.classify(envelope(), classification(0.2))
    risk, _ = classifier.classify(envelope(external_write=True), classification())
    fallback, record = classifier.classify(envelope(), None)
    assert low_confidence.ambiguity is Ordinal4.HIGH
    assert risk.consequence in {Ordinal4.HIGH, Ordinal4.CRITICAL}
    assert risk.irreversibility in {Ordinal4.HIGH, Ordinal4.CRITICAL}
    assert fallback.consequence is Ordinal4.CRITICAL
    # `output_form` is always deterministic (confidence 1.0); every other axis
    # is genuinely unmodeled in full-fallback mode and so carries no confidence.
    assert record.fallback_used and all(
        value is None
        for name, value in fallback.dimension_confidence.items()
        if name != "output_form"
    )
    assert fallback.dimension_confidence["output_form"] == 1.0
    allocator = BudgetAllocator()
    policy = default_tier_policy()
    tiers = [
        allocator.allocate(item, policy, DeploymentLimits())[0].tier
        for item in (high_confidence, low_confidence, risk, fallback)
    ]
    assert tiers[1] >= tiers[0] and tiers[2] >= tiers[0] and tiers[3] >= tiers[0]
    assert reversibility_to_irreversibility(Ordinal4.LOW) is Ordinal4.CRITICAL


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["MODEL", "HUMAN", "UNAVAILABLE"])
def test_m03_preserves_hard_unknown_and_epistemic_labels(mode: str) -> None:
    task = envelope()
    anchor = {
        "source_kind": "TASK_FIELD",
        "source_ref": {
            "object_type": "TaskEnvelope",
            "object_id": str(task.task_id),
            "revision": None,
        },
        "selector": "/explicit_constraints",
    }
    proposal = ProblemFormalisationOutput.model_validate_json(
        json.dumps(
            {
                "items": (
                    {
                        "id": "c1",
                        "kind": "CONSTRAINT",
                        "description": "cost <= 10",
                        "origin": "EXPLICIT_INPUT",
                        "anchors": [anchor],
                        "attributes": {
                            "constraint_kind": "HARD",
                            "verification_mode": mode,
                            "verification_status": "UNKNOWN",
                        },
                    },
                    {
                        "id": "u1",
                        "kind": "UNKNOWN",
                        "description": "future demand",
                        "origin": EpistemicOriginLabel.UNRESOLVED,
                        "attributes": {"resolvable": False},
                    },
                )
            }
        )
    )
    problem = ProblemFormaliser().formalise(task, proposal)
    assert problem.constraints[0].kind == "HARD"
    assert problem.constraints[0].verification_status is VerificationStatus.UNKNOWN
    assert problem.unknowns[0].provenance is not None
    assert problem.unknowns[0].provenance.origin is EpistemicOriginLabel.UNRESOLVED
    sparse = ProblemFormaliser().formalise(task, None)
    assert sparse.constraints[0].kind == "HARD" and not sparse.objectives


@pytest.mark.unit
def test_m03_merges_explicit_hard_constraint_omitted_by_proposal() -> None:
    """W3 final-gate remediation: a model proposal that silently drops one of
    the user's `explicit_constraints` (while faithfully representing another)
    must not produce a `ProblemSpec` missing it. Before the fix, `formalise`
    trusted `proposal.items` exclusively whenever a proposal was present, so
    the second explicit constraint below vanished with no error and no
    ledger trace -- this test fails on the pre-fix code and passes after."""
    task = TaskEnvelope(
        task_id=UUID(int=1),
        text="Minimize cost subject to cost <= 10 and weight <= 5kg.",
        explicit_constraints=("cost <= 10", "weight <= 5kg"),
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(allow_external_writes=False),
    )
    anchor = {
        "source_kind": "TASK_FIELD",
        "source_ref": {
            "object_type": "TaskEnvelope",
            "object_id": str(task.task_id),
            "revision": None,
        },
        "selector": "/explicit_constraints/0",
    }
    # The proposal covers "cost <= 10" verbatim but never mentions the
    # "weight <= 5kg" constraint anywhere -- a realistic omission, not a
    # paraphrase.
    proposal = ProblemFormalisationOutput.model_validate_json(
        json.dumps(
            {
                "items": (
                    {
                        "id": "c1",
                        "kind": "CONSTRAINT",
                        "description": "cost <= 10",
                        "origin": "EXPLICIT_INPUT",
                        "anchors": [anchor],
                        "attributes": {
                            "constraint_kind": "HARD",
                            "verification_mode": "UNAVAILABLE",
                            "verification_status": "UNKNOWN",
                        },
                    },
                )
            }
        )
    )
    problem = ProblemFormaliser().formalise(task, proposal)
    hard_descriptions = {c.description for c in problem.constraints if c.kind == "HARD"}
    assert "cost <= 10" in hard_descriptions
    assert "weight <= 5kg" in hard_descriptions
    # No duplicate was synthesised for the constraint the proposal already
    # represented.
    assert sum(1 for c in problem.constraints if c.description == "cost <= 10") == 1
    synthesised = next(c for c in problem.constraints if c.description == "weight <= 5kg")
    assert synthesised.kind == "HARD"
    assert synthesised.verification_status is VerificationStatus.UNKNOWN
    assert synthesised.provenance is not None
    assert synthesised.provenance.origin is EpistemicOriginLabel.EXPLICIT_INPUT
    assert len(synthesised.provenance.anchors) == 1
    synthesised_anchor = synthesised.provenance.anchors[0]
    assert synthesised_anchor.source_kind is SourceKind.TASK_FIELD
    assert isinstance(synthesised_anchor.source_ref, ObjectRef)
    assert synthesised_anchor.source_ref.object_id == str(task.task_id)
    assert synthesised_anchor.selector == "/explicit_constraints/1"
    assert synthesised.id not in {"c1"}


@pytest.mark.unit
def test_m03_downgraded_explicit_constraint_is_restored_as_hard() -> None:
    """A proposal that restates the user's constraint verbatim but downgrades
    it to SOFT must not let the HARD requirement quietly disappear: the
    explicit statement is not "represented" by a HARD constraint, so it is
    merged back in as its own synthesised HARD item alongside the proposal's
    (retained) SOFT one."""
    task = envelope()
    anchor = {
        "source_kind": "TASK_FIELD",
        "source_ref": {
            "object_type": "TaskEnvelope",
            "object_id": str(task.task_id),
            "revision": None,
        },
        "selector": "/explicit_constraints/0",
    }
    proposal = ProblemFormalisationOutput.model_validate_json(
        json.dumps(
            {
                "items": (
                    {
                        "id": "c1",
                        "kind": "CONSTRAINT",
                        "description": "cost <= 10",
                        "origin": "EXPLICIT_INPUT",
                        "anchors": [anchor],
                        "attributes": {
                            "constraint_kind": "SOFT",
                            "verification_mode": "UNAVAILABLE",
                            "verification_status": "UNKNOWN",
                        },
                    },
                )
            }
        )
    )
    problem = ProblemFormaliser().formalise(task, proposal)
    kinds = {c.kind for c in problem.constraints if c.description == "cost <= 10"}
    assert kinds == {"HARD", "SOFT"}


@pytest.mark.unit
def test_m04_selection_budget_projection_hash_and_fallback() -> None:
    task = envelope()
    problem = ProblemFormaliser().formalise(task, None)
    signature, _ = TaskClassifier().classify(task, None)
    budget_plan, _ = BudgetAllocator().allocate(
        signature, default_tier_policy(), DeploymentLimits()
    )
    selector = RepresentationSelector()
    plan = selector.select(problem, signature, budget_plan)
    assert len(plan.views) <= budget_plan.search.max_representation_views
    artifact = selector.build(plan.views[0], problem, 4)
    assert artifact.projection_hash == selector.build(plan.views[0], problem, 4).projection_hash
    unavailable = plan.views[0].model_copy(update={"builder_available": False})
    fallback = selector.build(unavailable, problem, 4)
    assert fallback.actual_kind is RepresentationKind.TEXT_TABLE_FALLBACK
    assert fallback.fallback_reason
    assert not selector.is_stale(artifact, problem)
    invalid = RepresentationAdjudicationOutput(
        selected_kinds=("ILP",), explanation="attempted invention"
    )
    assert (
        selector.apply_adjudication(
            plan,
            invalid,
            allowed_kinds=frozenset(view.kind for view in plan.views),
            view_limit=budget_plan.search.max_representation_views,
        )
        == plan
    )


@pytest.mark.unit
def test_m04_close_scores_retain_two_or_record_budget_omission() -> None:
    task = envelope()
    problem = (
        ProblemFormaliser()
        .formalise(task, None)
        .model_copy(
            update={
                "relations": (
                    ProblemRelation(
                        source_id="a", target_id="b", kind=ProblemRelationKind.DEPENDS_ON
                    ),
                    ProblemRelation(source_id="a", target_id="b", kind=ProblemRelationKind.CAUSES),
                )
            }
        )
    )
    signature, _ = TaskClassifier().classify(task, None)
    budget, _ = BudgetAllocator().allocate(signature, default_tier_policy(), DeploymentLimits())
    selector = RepresentationSelector()
    two = selector.select(problem, signature, budget)
    assert len(two.views) == 2
    one_budget = budget.model_copy(
        update={"search": budget.search.model_copy(update={"max_representation_views": 1})}
    )
    one = selector.select(problem, signature, one_budget)
    assert len(one.views) == 1 and one.omitted_reasons


@pytest.mark.unit
def test_m03_contradiction_maps_to_frozen_m09_events() -> None:
    proposal = ProblemFormalisationOutput.model_validate_json(
        json.dumps(
            {
                "items": (
                    {
                        "id": "left",
                        "kind": "UNKNOWN",
                        "description": "value is A",
                        "origin": "CONTRADICTED",
                        "attributes": {},
                    },
                    {
                        "id": "right",
                        "kind": "UNKNOWN",
                        "description": "value is B",
                        "origin": "CONTRADICTED",
                        "attributes": {},
                    },
                    {
                        "id": "conflict",
                        "kind": "RELATION",
                        "description": "conflict",
                        "origin": "CONTRADICTED",
                        "attributes": {
                            "source_id": "left",
                            "target_id": "right",
                            "relation_kind": "CONTRADICTS",
                        },
                    },
                )
            }
        )
    )
    # C06 remediation (F06): `ledger_events()` is no longer a public entry
    # point. The only way to obtain M03 ledger events is `canonical_events`,
    # which always resolves anchors/support via `formalise` first and
    # prepends the resulting `ProblemFormalised` event.
    events = ProblemFormaliser().canonical_events(
        envelope(),
        proposal,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        uuids=FakeUUIDFactory(UUID(int=index) for index in range(1, 10)),
    )
    assert len(events) == 6
    assert isinstance(events[0], ProblemFormalised)
    edge = events[3]
    assert isinstance(edge, LedgerEdgeAdded)
    assert edge.edge.relation is LedgerRelation.CONTRADICTS
    assert isinstance(events[4], ProblemContradictionRecorded)
    assert isinstance(events[5], ProblemBlockerRecorded)
    left_node = events[1]
    assert isinstance(left_node, LedgerNodeAdded)
    assert left_node.node.epistemic_status is EpistemicStatus.CONTESTED
    assert events[5].blocker.ledger_ref == left_node.node.ref


@pytest.mark.unit
def test_m03_material_contradiction_contests_supported_endpoints() -> None:
    left_anchor = SourceAnchor(
        source_kind=SourceKind.TASK_FIELD,
        source_ref=ObjectRef(
            object_type="TaskEnvelope", object_id="00000000-0000-0000-0000-000000000001"
        ),
        selector="/explicit_constraints/0",
    )
    proposal = ProblemFormalisationOutput.model_validate(
        {
            "items": (
                {
                    "id": "left",
                    "kind": "UNKNOWN",
                    "description": "value is A",
                    "origin": EpistemicOriginLabel.EXPLICIT_INPUT,
                    "anchors": (left_anchor,),
                },
                {
                    "id": "right",
                    "kind": "UNKNOWN",
                    "description": "value is B",
                    "origin": EpistemicOriginLabel.SUPPORTED_INFERENCE,
                    "basis": "derived from the explicit constraint",
                    "support": ({"item_id": "left"},),
                },
                {
                    "id": "conflict",
                    "kind": "RELATION",
                    "description": "material conflict",
                    "origin": EpistemicOriginLabel.SUPPORTED_INFERENCE,
                    "basis": "both endpoints observed",
                    "support": ({"item_id": "left"},),
                    "attributes": {
                        "source_id": "left",
                        "target_id": "right",
                        "relation_kind": "CONTRADICTS",
                        "material": True,
                    },
                },
            )
        }
    )
    # C06 remediation (F06): route through `canonical_events`, the only
    # remaining public path to M03 ledger events -- see the note on the
    # previous call site.
    events = ProblemFormaliser().canonical_events(
        envelope(),
        proposal,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        uuids=FakeUUIDFactory(UUID(int=index) for index in range(40, 50)),
    )
    nodes = [event.node for event in events if isinstance(event, LedgerNodeAdded)]
    assert [node.epistemic_status for node in nodes] == [
        EpistemicStatus.CONTESTED,
        EpistemicStatus.CONTESTED,
    ]


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["DETERMINISTIC", "MODEL", "HUMAN", "UNAVAILABLE"])
@pytest.mark.parametrize("status", list(VerificationStatus))
def test_m03_model_constraint_statuses_begin_unknown(mode: str, status: VerificationStatus) -> None:
    proposal = ProblemFormalisationOutput.model_validate(
        {
            "items": (
                {
                    "id": "c1",
                    "kind": "CONSTRAINT",
                    "description": "safe",
                    "origin": EpistemicOriginLabel.UNRESOLVED,
                    "attributes": {
                        "constraint_kind": "HARD",
                        "verification_mode": mode,
                        "verification_status": status.value,
                    },
                },
            )
        }
    )
    constraint = ProblemFormaliser().formalise(envelope(), proposal).constraints[0]
    assert constraint.kind == "HARD"
    assert constraint.verification_status is VerificationStatus.UNKNOWN


@pytest.mark.unit
def test_m03_only_trusted_named_deterministic_verifier_establishes_status() -> None:
    proposal = ProblemFormalisationOutput.model_validate(
        {
            "items": (
                {
                    "id": "c1",
                    "kind": "CONSTRAINT",
                    "description": "safe",
                    "origin": EpistemicOriginLabel.UNRESOLVED,
                    "attributes": {
                        "constraint_kind": "HARD",
                        "verification_mode": "DETERMINISTIC",
                        "verifier_ref": "checker:safety-v1",
                        "verification_status": "FAIL",
                    },
                },
            )
        }
    )
    formaliser = ProblemFormaliser()
    for status in VerificationStatus:
        constraint = formaliser.formalise(
            envelope(), proposal, trusted_verifier_results={"c1": status}
        ).constraints[0]
        assert constraint.verification_status is status
    untrusted_mode = proposal.model_copy(
        update={
            "items": (
                proposal.items[0].model_copy(
                    update={
                        "attributes": {
                            **proposal.items[0].attributes,
                            "verification_mode": "MODEL",
                        }
                    }
                ),
            )
        }
    )
    with pytest.raises(InvalidProblemSpec, match="named deterministic"):
        formaliser.formalise(
            envelope(), untrusted_mode, trusted_verifier_results={"c1": VerificationStatus.PASS}
        )


@pytest.mark.unit
def test_m03_relations_are_order_independent_and_strict() -> None:
    relation = {
        "id": "r1",
        "kind": "RELATION",
        "description": "depends",
        "origin": EpistemicOriginLabel.UNRESOLVED,
        "attributes": {
            "source_id": "later",
            "target_id": "first",
            "relation_kind": "DEPENDS_ON",
        },
    }
    nodes = (
        relation,
        {
            "id": "first",
            "kind": "UNKNOWN",
            "description": "first",
            "origin": EpistemicOriginLabel.UNRESOLVED,
        },
        {
            "id": "later",
            "kind": "UNKNOWN",
            "description": "later",
            "origin": EpistemicOriginLabel.UNRESOLVED,
        },
    )
    proposal = ProblemFormalisationOutput.model_validate({"items": nodes})
    assert ProblemFormaliser().formalise(envelope(), proposal).relations[0].source_id == "later"
    missing = {
        "id": "r1",
        "kind": "RELATION",
        "description": "depends",
        "origin": EpistemicOriginLabel.UNRESOLVED,
        "attributes": {
            "source_id": "later",
            "target_id": "missing",
            "relation_kind": "DEPENDS_ON",
        },
    }
    self_relation = {
        "id": "r1",
        "kind": "RELATION",
        "description": "depends",
        "origin": EpistemicOriginLabel.UNRESOLVED,
        "attributes": {
            "source_id": "later",
            "target_id": "later",
            "relation_kind": "DEPENDS_ON",
        },
    }
    for bad in (
        (missing, *nodes[1:]),
        (self_relation, *nodes[1:]),
        (*nodes, {**relation, "id": "r2"}),
    ):
        with pytest.raises(InvalidProblemSpec):
            ProblemFormaliser().formalise(
                envelope(), ProblemFormalisationOutput.model_validate({"items": bad})
            )
    unsupported = ProblemFormalisationOutput.model_validate(
        {
            "items": (
                {
                    "id": "x",
                    "kind": "MAGIC",
                    "description": "x",
                    "origin": EpistemicOriginLabel.UNRESOLVED,
                },
            )
        }
    )
    with pytest.raises(InvalidProblemSpec, match="unsupported"):
        ProblemFormaliser().formalise(envelope(), unsupported)


@pytest.mark.unit
def test_m03_fallback_provenance_objectives_and_unavailable_acceptance_blocker() -> None:
    task = envelope()
    fallback = ProblemFormaliser().formalise(task, None)
    provenance = fallback.constraints[0].provenance
    assert provenance is not None
    assert provenance.origin is EpistemicOriginLabel.EXPLICIT_INPUT
    assert provenance.anchors[0].selector == "/explicit_constraints/0"
    validate_source_anchor(provenance.anchors[0], task)

    proposal = ProblemFormalisationOutput.model_validate(
        {
            "items": (
                {
                    "id": "objective",
                    "kind": "OBJECTIVE",
                    "description": "minimise cost",
                    "origin": EpistemicOriginLabel.UNRESOLVED,
                    "attributes": {
                        "direction": "LEXICOGRAPHIC",
                        "priority": 1,
                        "evaluator_ref": "evaluator:cost-v1",
                    },
                },
                {
                    "id": "criterion",
                    "kind": "ACCEPTANCE_CRITERION",
                    "description": "must be reviewed",
                    "origin": EpistemicOriginLabel.UNRESOLVED,
                    "attributes": {"verification_mode": "UNAVAILABLE", "required": True},
                },
            )
        }
    )
    # C06 remediation (F06): route through `canonical_events`, the only
    # remaining public path to M03 ledger events -- see the note on earlier
    # call sites in this file.
    events = ProblemFormaliser().canonical_events(
        task,
        proposal,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        uuids=FakeUUIDFactory(UUID(int=index) for index in range(20, 40)),
    )
    formalised = next(item for item in events if isinstance(item, ProblemFormalised))
    problem = formalised.problem
    assert problem.objectives[0].priority == 1
    assert problem.objectives[0].evaluator_ref == "evaluator:cost-v1"
    blocker = next(item for item in events if isinstance(item, ProblemBlockerRecorded))
    assert not blocker.blocker.resolvable
    nodes = (item.node for item in events if isinstance(item, LedgerNodeAdded))
    criterion_node = next(
        node
        for node in nodes
        if isinstance(node.content, dict) and node.content["id"] == "criterion"
    )
    assert blocker.blocker.ledger_ref == criterion_node.ref
    # C06 remediation (F04): `AcceptanceCriterion.blocker_ref` used to be
    # always null -- the link only lived in the separate
    # `ProblemBlockerRecorded` event. `canonical_events` now threads the same
    # ledger identity back into the immutable `ProblemSpec` it returns.
    criterion_spec = next(item for item in problem.acceptance_criteria if item.id == "criterion")
    assert criterion_spec.blocker_ref == criterion_node.ref

    duplicate_priorities = proposal.model_copy(
        update={"items": (proposal.items[0], proposal.items[0].model_copy(update={"id": "o2"}))}
    )
    with pytest.raises(InvalidProblemSpec, match="priorities"):
        ProblemFormaliser().formalise(task, duplicate_priorities)
    incomplete_priorities = proposal.model_copy(
        update={
            "items": (
                proposal.items[0],
                proposal.items[0].model_copy(
                    update={"id": "o2", "attributes": {"direction": "LEXICOGRAPHIC"}}
                ),
            )
        }
    )
    with pytest.raises(InvalidProblemSpec, match="complete ordering"):
        ProblemFormaliser().formalise(task, incomplete_priorities)
    unrelated_priority = proposal.model_copy(
        update={
            "items": (
                proposal.items[0].model_copy(update={"attributes": {"direction": "LEXICOGRAPHIC"}}),
                proposal.items[0].model_copy(
                    update={"id": "o2", "attributes": {"direction": "MIN", "priority": 1}}
                ),
            )
        }
    )
    with pytest.raises(InvalidProblemSpec, match="complete ordering"):
        ProblemFormaliser().formalise(task, unrelated_priority)
    invalid_criterion = proposal.model_copy(
        update={
            "items": (
                proposal.items[1].model_copy(
                    update={"attributes": {"verification_mode": "MAGIC", "required": True}}
                ),
            )
        }
    )
    with pytest.raises(InvalidProblemSpec, match="verification mode"):
        ProblemFormaliser().formalise(task, invalid_criterion)
