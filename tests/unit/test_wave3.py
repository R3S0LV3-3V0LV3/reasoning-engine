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
from fre.domain.ledger import LedgerRelation
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
from fre.modules.m03_formaliser import ProblemFormaliser
from fre.modules.m04_representation import RepresentationSelector
from fre.modules.source_anchors import InvalidSourceAnchor, validate_source_anchor
from fre.prompts import default_output_schema_registry, default_prompt_registry
from fre.prompts.registry import PromptDefinition, PromptVersionConflict
from fre.prompts.schemas import (
    ClassificationOutput,
    ProblemFormalisationOutput,
    RepresentationAdjudicationOutput,
)
from fre.runtime.events import LedgerEdgeAdded


def envelope(*, external_write: bool = False) -> TaskEnvelope:
    return TaskEnvelope(
        task_id=UUID(int=1),
        text="Minimize cost subject to cost <= 10.",
        explicit_constraints=("cost <= 10",),
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(allow_external_writes=external_write),
    )


def classification(confidence: float = 0.9) -> ClassificationOutput:
    dimension = {
        "estimate": "LOW",
        "confidence": confidence,
        "conservative_upper": "MEDIUM",
        "anchors": (),
        "rationale": "fixture",
    }
    return ClassificationOutput.model_validate_json(
        json.dumps(
            {
                "task_type": "DECISION",
                "consequence": dimension,
                "reversibility": dimension,
                "ambiguity": dimension,
                "evidence_scarcity": dimension,
                "search_space": "CLOSED",
                "search_space_confidence": confidence,
                "horizon": "SHORT",
                "horizon_confidence": confidence,
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
    assert record.fallback_used and all(
        value is None for value in fallback.dimension_confidence.values()
    )
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
                        "origin": "UNRESOLVED",
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
    fallback = selector.build(plan.views[0], problem, 4, builder_available=False)
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
    events = ProblemFormaliser().ledger_events(
        proposal,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        uuids=FakeUUIDFactory(UUID(int=index) for index in range(1, 10)),
    )
    assert len(events) == 3
    edge = events[-1]
    assert isinstance(edge, LedgerEdgeAdded)
    assert edge.edge.relation is LedgerRelation.CONTRADICTS
