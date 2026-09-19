"""Regression coverage for Wave 3 source-anchor and M01 evidence defects."""

import hashlib
from uuid import UUID

import pytest

from fre.domain.common import ArtifactRef, ObjectRef, OutputContract, PermissionSet
from fre.domain.semantic import SourceAnchor, SourceKind
from fre.domain.task import (
    HorizonClass,
    Ordinal4,
    OutputForm,
    SearchSpaceClass,
    TaskEnvelope,
    TaskType,
)
from fre.modules.m01_classifier import TaskClassifier
from fre.modules.source_anchors import InvalidSourceAnchor, validate_source_anchor
from fre.prompts.schemas import (
    ClassificationDimensionProposal,
    ClassificationOutput,
    HorizonProposal,
    SearchSpaceProposal,
    TaskTypeProposal,
)


def task(**updates: object) -> TaskEnvelope:
    value = TaskEnvelope(
        task_id=UUID(int=42),
        text="analyse carefully",
        explicit_constraints=("first", "second"),
        requested_output=OutputContract(form="Markdown", requirements=("concise",)),
        user_metadata={"consequence": "HIGH", "a/b~c": "quoted"},
        execution_permissions=PermissionSet(allow_network=True),
    )
    return value.model_copy(update=updates)


def anchor(selector: str, **updates: object) -> SourceAnchor:
    value = SourceAnchor(
        source_kind=SourceKind.TASK_FIELD,
        source_ref=ObjectRef(object_type="TaskEnvelope", object_id=str(UUID(int=42))),
        selector=selector,
    )
    return value.model_copy(update=updates)


def proposal(source_anchor: SourceAnchor) -> ClassificationOutput:
    dimension = ClassificationDimensionProposal(
        estimate=Ordinal4.LOW,
        confidence=0.9,
        conservative_upper=Ordinal4.MEDIUM,
        anchors=(source_anchor,),
        rationale="anchored",
    )
    # `reversibility` is descending-risk: its conservative bound must sit at
    # or below the estimate (see `_validate_ordinal_bound` in m01_classifier).
    reversibility_dimension = ClassificationDimensionProposal(
        estimate=Ordinal4.MEDIUM,
        confidence=0.9,
        conservative_upper=Ordinal4.LOW,
        anchors=(source_anchor,),
        rationale="anchored",
    )
    return ClassificationOutput(
        task_type=TaskTypeProposal(
            estimate=TaskType.ANALYSIS,
            confidence=0.9,
            anchors=(source_anchor,),
            rationale="anchored",
        ),
        consequence=dimension,
        reversibility=reversibility_dimension,
        ambiguity=dimension,
        evidence_scarcity=dimension,
        search_space=SearchSpaceProposal(
            estimate=SearchSpaceClass.CLOSED,
            confidence=0.9,
            anchors=(source_anchor,),
            rationale="anchored",
        ),
        horizon=HorizonProposal(
            estimate=HorizonClass.SHORT,
            confidence=0.9,
            anchors=(source_anchor,),
            rationale="anchored",
        ),
    )


@pytest.mark.unit
def test_complete_json_pointer_resolution_and_escaping() -> None:
    envelope = task()
    validate_source_anchor(anchor("/explicit_constraints/1"), envelope)
    validate_source_anchor(
        anchor(
            "/user_metadata/a~1b~0c",
            char_start=0,
            char_end=6,
            excerpt_hash=hashlib.sha256(b"quoted").hexdigest(),
        ),
        envelope,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "selector",
    [
        "/explicit_constraints/-1",
        "/explicit_constraints/-",
        "/explicit_constraints/+1",
        "/explicit_constraints/01",
        "/explicit_constraints/2",
        "/explicit_constraints/0/more",
        "/user_metadata/missing",
        "/user_metadata/a~2b",
        "user_metadata/consequence",
    ],
)
def test_invalid_pointer_indices_keys_escaping_and_scalar_traversal(selector: str) -> None:
    with pytest.raises(InvalidSourceAnchor):
        validate_source_anchor(anchor(selector), task())


@pytest.mark.unit
def test_object_identity_text_spans_hashes_and_non_text_are_strict() -> None:
    envelope = task()
    text = SourceAnchor(
        source_kind=SourceKind.TASK_TEXT,
        source_ref=ObjectRef(object_type="TaskEnvelope", object_id=str(envelope.task_id)),
        selector="/text",
        char_start=0,
        char_end=7,
        excerpt_hash=hashlib.sha256(b"analyse").hexdigest(),
    )
    validate_source_anchor(text, envelope)
    invalid = (
        text.model_copy(
            update={"source_ref": ObjectRef(object_type="task", object_id=str(envelope.task_id))}
        ),
        text.model_copy(update={"char_end": 99}),
        text.model_copy(update={"excerpt_hash": "0" * 64}),
        anchor("/requested_output", char_start=0, char_end=1),
    )
    for value in invalid:
        with pytest.raises(InvalidSourceAnchor):
            validate_source_anchor(value, envelope)


@pytest.mark.unit
def test_m01_retains_valid_anchors_orders_evidence_and_preserves_explicit_values() -> None:
    envelope = task()
    claimed = anchor("/explicit_constraints/0")
    signature, record = TaskClassifier().classify(envelope, proposal(claimed))
    consequence = record.dimensions["consequence"]
    assert consequence.estimated == Ordinal4.HIGH
    assert consequence.effective == Ordinal4.HIGH
    assert consequence.confidence is None
    assert consequence.basis == "EXPLICIT"
    assert consequence.source_anchors == (anchor("/user_metadata/consequence"), claimed)
    assert record.dimensions["ambiguity"].source_anchors == (claimed,)
    assert signature.evidence_refs == tuple(sorted(signature.evidence_refs))
    assert len(signature.evidence_refs) == len(set(signature.evidence_refs))
    assert any("explicit_constraints/1" in ref for ref in signature.evidence_refs)
    assert any("execution_permissions" in ref for ref in signature.evidence_refs)


@pytest.mark.unit
def test_m01_rejects_any_invalid_model_claimed_anchor() -> None:
    with pytest.raises(InvalidSourceAnchor):
        TaskClassifier().classify(task(), proposal(anchor("/explicit_constraints/99")))


@pytest.mark.unit
def test_m01_accepts_model_anchor_for_envelope_attachment() -> None:
    attachment = ArtifactRef(artifact_id=UUID(int=7), sha256="a" * 64)
    envelope = task(attachments=(attachment,))
    claimed = SourceAnchor(
        source_kind=SourceKind.ARTIFACT,
        source_ref=attachment,
        selector="",
    )

    signature, record = TaskClassifier().classify(envelope, proposal(claimed))

    assert record.dimensions["ambiguity"].source_anchors == (claimed,)
    assert any(attachment.sha256 in ref for ref in signature.evidence_refs)


@pytest.mark.unit
def test_m01_rejects_model_anchor_for_unattached_artifact() -> None:
    claimed = SourceAnchor(
        source_kind=SourceKind.ARTIFACT,
        source_ref=ArtifactRef(artifact_id=UUID(int=7), sha256="a" * 64),
        selector="",
    )

    with pytest.raises(InvalidSourceAnchor):
        TaskClassifier().classify(task(), proposal(claimed))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("text", OutputForm.TEXT),
        ("Markdown", OutputForm.TEXT),
        ("report", OutputForm.TEXT),
        ("JSON", OutputForm.STRUCTURED),
        ("YAML", OutputForm.STRUCTURED),
        ("schema-bound", OutputForm.STRUCTURED),
        ("table", OutputForm.STRUCTURED),
        ("file", OutputForm.ARTIFACT),
        ("image", OutputForm.ARTIFACT),
        ("something-new", OutputForm.ARTIFACT),
    ],
)
def test_output_forms_are_normalized_without_mutating_contract(
    requested: str, expected: OutputForm
) -> None:
    envelope = task(requested_output=OutputContract(form=requested))
    signature, _ = TaskClassifier().classify(envelope, None)
    assert signature.output_form is expected
    assert envelope.requested_output.form == requested
