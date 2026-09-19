"""Decisive C06 remediation tests: M03 resolved support, UNKNOWN preservation,
and materiality (defects F03, F04, F06, F07).

Each test below maps directly onto one bullet in the C06 validation strategy
in FRE_WAVE3_C01_C10_EXECUTION_COMPLETION_AND_VALIDATION_REGISTER.md.
"""

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from fre.adapters.testing import FakeUUIDFactory
from fre.composition import compose_wave3
from fre.domain.common import ArtifactRef as ArtifactDataRef
from fre.domain.common import JsonValue, ObjectRef, OutputContract, PermissionSet
from fre.domain.ledger import EpistemicStatus, LedgerNodeType
from fre.domain.semantic import (
    EpistemicItemProvenance,
    EpistemicOriginLabel,
    SemanticCallUsage,
    SourceAnchor,
    SourceKind,
    StructuredModelResult,
    StructuredModelStatus,
    SupportArtifactRef,
    SupportLedgerNodeRef,
    SupportProblemItemRef,
)
from fre.domain.task import TaskEnvelope
from fre.engine import FrontierReasoningEngine
from fre.modules.m03_formaliser import (
    InvalidProblemSpec,
    ProblemFormaliser,
    _coerce_optional_float,
)
from fre.modules.source_anchors import (
    DanglingLedgerSupportReference,
    IncompatibleSupportReferenceKind,
    InvalidSourceAnchor,
    SelfSupportReference,
    UnauthorizedArtifactReference,
)
from fre.prompts.schemas import ProblemFormalisationOutput
from fre.runtime.events import (
    ArtifactRegistered,
    LedgerNodeAdded,
    ProblemBlockerRecorded,
    StoredEvent,
    event_wire_identity,
)
from fre.runtime.reducer import RunReducer


def envelope(*, explicit_constraints: tuple[str, ...] = ("cost <= 10",)) -> TaskEnvelope:
    return TaskEnvelope(
        task_id=UUID(int=1),
        text="Minimize cost subject to cost <= 10.",
        explicit_constraints=explicit_constraints,
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(),
    )


def task_anchor(selector: str = "/explicit_constraints/0") -> SourceAnchor:
    return SourceAnchor(
        source_kind=SourceKind.TASK_FIELD,
        source_ref=ObjectRef(object_type="TaskEnvelope", object_id=str(UUID(int=1))),
        selector=selector,
    )


# ---------------------------------------------------------------------------
# F03: SUPPORTED_INFERENCE must resolve to real admissible evidence, never a
# fictitious opaque string.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_supported_inference_rejects_fictitious_opaque_string() -> None:
    """The F03 headline case: a non-empty but wholly fictitious identifier in
    the legacy, decode-only `supporting_refs` field must never satisfy
    SUPPORTED_INFERENCE -- only a resolved, typed `support` tuple can."""
    with pytest.raises(ValidationError, match="resolved SupportRef"):
        EpistemicItemProvenance(
            origin=EpistemicOriginLabel.SUPPORTED_INFERENCE,
            supporting_refs=("totally-fictitious-id-nobody-checked",),
            basis="claimed basis",
        )


@pytest.mark.unit
def test_supported_inference_never_promoted_to_fact_by_string_presence() -> None:
    """Even with a fully resolved `support`, a SUPPORTED_INFERENCE item must
    stay INFERENCE/PROVISIONAL in the ledger -- never silently promoted to
    FACT/SUPPORTED merely because *some* support string/ref is present."""
    proposal = ProblemFormalisationOutput.model_validate(
        {
            "items": (
                {
                    "id": "anchor-item",
                    "kind": "UNKNOWN",
                    "description": "value is A",
                    "origin": EpistemicOriginLabel.EXPLICIT_INPUT,
                    "anchors": (task_anchor(),),
                },
                {
                    "id": "inference-item",
                    "kind": "UNKNOWN",
                    "description": "inferred value",
                    "origin": EpistemicOriginLabel.SUPPORTED_INFERENCE,
                    "basis": "derived from anchor-item",
                    "support": ({"item_id": "anchor-item"},),
                },
            )
        }
    )
    events = ProblemFormaliser().canonical_events(
        envelope(),
        proposal,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        uuids=FakeUUIDFactory(UUID(int=index) for index in range(1, 10)),
    )
    nodes = {
        node.content["id"]: node
        for node in (e.node for e in events if isinstance(e, LedgerNodeAdded))
        if isinstance(node.content, dict)
    }
    inference_node = nodes["inference-item"]
    assert inference_node.node_type is LedgerNodeType.INFERENCE
    assert inference_node.epistemic_status is EpistemicStatus.PROVISIONAL


# ---------------------------------------------------------------------------
# Reject self-support, cross-kind confusion, unauthorized artifacts, dangling
# prior-ledger references.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_self_support_is_rejected() -> None:
    proposal = ProblemFormalisationOutput.model_validate(
        {
            "items": (
                {
                    "id": "x",
                    "kind": "UNKNOWN",
                    "description": "self-supporting",
                    "origin": EpistemicOriginLabel.SUPPORTED_INFERENCE,
                    "basis": "circular",
                    "support": ({"item_id": "x"},),
                },
            )
        }
    )
    with pytest.raises(SelfSupportReference):
        ProblemFormaliser().formalise(envelope(), proposal)


@pytest.mark.unit
def test_support_referencing_a_relation_item_is_rejected() -> None:
    """A `RELATION` pseudo-item describes an edge between items, never
    evidence in its own right -- citing one as support is a cross-kind
    confusion that must be rejected."""
    proposal = ProblemFormalisationOutput.model_validate(
        {
            "items": (
                {
                    "id": "a",
                    "kind": "UNKNOWN",
                    "description": "a",
                    "origin": EpistemicOriginLabel.UNRESOLVED,
                },
                {
                    "id": "b",
                    "kind": "UNKNOWN",
                    "description": "b",
                    "origin": EpistemicOriginLabel.UNRESOLVED,
                },
                {
                    "id": "rel",
                    "kind": "RELATION",
                    "description": "a depends on b",
                    "origin": EpistemicOriginLabel.UNRESOLVED,
                    "attributes": {
                        "source_id": "a",
                        "target_id": "b",
                        "relation_kind": "DEPENDS_ON",
                    },
                },
                {
                    "id": "citing",
                    "kind": "UNKNOWN",
                    "description": "cites a relation as evidence",
                    "origin": EpistemicOriginLabel.SUPPORTED_INFERENCE,
                    "basis": "bogus",
                    "support": ({"item_id": "rel"},),
                },
            )
        }
    )
    with pytest.raises(IncompatibleSupportReferenceKind):
        ProblemFormaliser().formalise(envelope(), proposal)


@pytest.mark.unit
def test_unauthorized_artifact_reference_is_rejected() -> None:
    proposal = ProblemFormalisationOutput.model_validate(
        {
            "items": (
                {
                    "id": "x",
                    "kind": "UNKNOWN",
                    "description": "artifact-supported",
                    "origin": EpistemicOriginLabel.SUPPORTED_INFERENCE,
                    "basis": "from an attachment",
                    "support": (SupportArtifactRef(artifact_id=UUID(int=99), sha256="a" * 64),),
                },
            )
        }
    )
    with pytest.raises(UnauthorizedArtifactReference):
        ProblemFormaliser().formalise(envelope(), proposal, available_artifacts=frozenset())


@pytest.mark.unit
def test_dangling_prior_ledger_reference_is_rejected() -> None:
    proposal = ProblemFormalisationOutput.model_validate(
        {
            "items": (
                {
                    "id": "x",
                    "kind": "UNKNOWN",
                    "description": "ledger-supported",
                    "origin": EpistemicOriginLabel.SUPPORTED_INFERENCE,
                    "basis": "from a prior ledger node",
                    "support": (SupportLedgerNodeRef(node_id=UUID(int=5), revision=1),),
                },
            )
        }
    )
    with pytest.raises(DanglingLedgerSupportReference):
        ProblemFormaliser().formalise(envelope(), proposal, known_ledger_refs=frozenset())


@pytest.mark.unit
def test_exact_text_anchor_with_invalid_offsets_is_rejected() -> None:
    proposal = ProblemFormalisationOutput.model_validate(
        {
            "items": (
                {
                    "id": "x",
                    "kind": "UNKNOWN",
                    "description": "bad span",
                    "origin": EpistemicOriginLabel.SUPPORTED_INFERENCE,
                    "basis": "quoting the task text",
                    "support": (
                        SourceAnchor(
                            source_kind=SourceKind.TASK_TEXT,
                            source_ref=ObjectRef(
                                object_type="TaskEnvelope", object_id=str(UUID(int=1))
                            ),
                            selector="/text",
                            char_start=0,
                            char_end=9999,
                        ),
                    ),
                },
            )
        }
    )
    with pytest.raises(InvalidSourceAnchor):
        ProblemFormaliser().formalise(envelope(), proposal)


# ---------------------------------------------------------------------------
# Resolve valid anchor, artifact, prior-ledger, and same-proposal refs.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_all_four_support_reference_kinds_resolve_when_valid() -> None:
    artifact_bytes = b"attachment contents"
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    prior_ledger_node_id = UUID(int=42)
    proposal = ProblemFormalisationOutput.model_validate(
        {
            "items": (
                {
                    "id": "same-proposal-target",
                    "kind": "UNKNOWN",
                    "description": "another item in this proposal",
                    "origin": EpistemicOriginLabel.UNRESOLVED,
                },
                {
                    "id": "multi-supported",
                    "kind": "UNKNOWN",
                    "description": "supported by all four reference kinds",
                    "origin": EpistemicOriginLabel.SUPPORTED_INFERENCE,
                    "basis": "cross-checked against every admissible evidence kind",
                    "support": (
                        SourceAnchor(
                            source_kind=SourceKind.TASK_FIELD,
                            source_ref=ObjectRef(
                                object_type="TaskEnvelope", object_id=str(UUID(int=1))
                            ),
                            selector="/explicit_constraints/0",
                        ),
                        SupportArtifactRef(artifact_id=UUID(int=7), sha256=artifact_sha256),
                        SupportLedgerNodeRef(node_id=prior_ledger_node_id, revision=1),
                        SupportProblemItemRef(item_id="same-proposal-target"),
                    ),
                },
            )
        }
    )
    problem = ProblemFormaliser().formalise(
        envelope(),
        proposal,
        available_artifacts=frozenset({artifact_sha256}),
        known_ledger_refs=frozenset({(prior_ledger_node_id, 1)}),
    )
    supported = next(item for item in problem.unknowns if item.id == "multi-supported")
    assert supported.provenance is not None
    assert len(supported.provenance.support) == 4


@pytest.mark.unit
def test_ledger_nodes_are_emitted_in_dependency_order_for_a_three_level_support_chain() -> None:
    """EU-18 (C06 cleanup): `_topologically_ordered_items` must still emit a
    target's ledger node strictly before the node of any item -- however
    many `SupportProblemItemRef` hops away -- that (transitively) cites it,
    regardless of declaration order in the proposal. Declared here in
    reverse dependency order (`leaf` first, `root` last) so a naive
    declaration-order emission would fail this assertion."""
    proposal = ProblemFormalisationOutput.model_validate(
        {
            "items": (
                {
                    "id": "leaf",
                    "kind": "UNKNOWN",
                    "description": "cites mid",
                    "origin": EpistemicOriginLabel.SUPPORTED_INFERENCE,
                    "basis": "derived from mid",
                    "support": ({"item_id": "mid"},),
                },
                {
                    "id": "mid",
                    "kind": "UNKNOWN",
                    "description": "cites root",
                    "origin": EpistemicOriginLabel.SUPPORTED_INFERENCE,
                    "basis": "derived from root",
                    "support": ({"item_id": "root"},),
                },
                {
                    "id": "root",
                    "kind": "UNKNOWN",
                    "description": "no support",
                    "origin": EpistemicOriginLabel.UNRESOLVED,
                },
            )
        }
    )
    # `explicit_constraints=()`: this test is about dependency-order
    # emission for support-chained items, not explicit-constraint synthesis
    # (W3 final-gate fix #2 now also emits a ledger node for any synthesised
    # explicit constraint, which would otherwise appear in `emitted_order`
    # and break this test's exact-sequence assertion below).
    events = ProblemFormaliser().canonical_events(
        envelope(explicit_constraints=()),
        proposal,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        uuids=FakeUUIDFactory(UUID(int=index) for index in range(1, 20)),
    )
    emitted_order = [
        node.content["id"]
        for node in (e.node for e in events if isinstance(e, LedgerNodeAdded))
        if isinstance(node.content, dict)
    ]
    assert emitted_order == ["root", "mid", "leaf"]


# ---------------------------------------------------------------------------
# EU-23 (C06 cleanup, re-scoped item #23): `SupportLedgerNodeRef` resolution
# through the REAL `composition.py` coordinator -- not a hand-built
# `known_ledger_refs` (which is all `test_all_four_support_reference_kinds_
# resolve_when_valid` above exercises).
# ---------------------------------------------------------------------------


class _QueueModel:
    """A structured-model port whose responses are drained one call at a
    time (deliberately duplicated here, not imported, from
    `tests/integration/test_wave3_gate.py`'s identically-named helper --
    see `tests/golden/_common.py`'s docstring for why a cross-test-module
    import is avoided in this repository: with no `tests/__init__.py`,
    mypy resolves the same file under two different module identities
    depending on how it is collected/imported, and refuses to type-check
    the result)."""

    def __init__(self, responses: list[StructuredModelResult]) -> None:
        self.responses = iter(responses)

    async def generate(self, request: object) -> StructuredModelResult:
        return next(self.responses)


def _classification_response() -> StructuredModelResult:
    anchor: JsonValue = {
        "source_kind": "TASK_TEXT",
        "source_ref": {"object_type": "TaskEnvelope", "object_id": str(UUID(int=900))},
        "selector": "/text",
        "char_start": 0,
        "char_end": len("Choose a safe option."),
    }
    valid: dict[str, JsonValue] = {
        "task_type": {
            "estimate": "DECISION",
            "confidence": 0.9,
            "anchors": [anchor],
            "rationale": "fixture",
        },
        "consequence": {
            "estimate": "LOW",
            "confidence": 0.9,
            "conservative_upper": "MEDIUM",
            "anchors": [anchor],
            "rationale": "fixture",
        },
        "reversibility": {
            "estimate": "HIGH",
            "confidence": 0.9,
            "conservative_upper": "HIGH",
            "anchors": [anchor],
            "rationale": "fixture",
        },
        "ambiguity": {
            "estimate": "MEDIUM",
            "confidence": 0.9,
            "conservative_upper": "MEDIUM",
            "anchors": [anchor],
            "rationale": "fixture",
        },
        "evidence_scarcity": {
            "estimate": "MEDIUM",
            "confidence": 0.9,
            "conservative_upper": "MEDIUM",
            "anchors": [anchor],
            "rationale": "fixture",
        },
        "search_space": {
            "estimate": "BOUNDED",
            "confidence": 0.9,
            "anchors": [anchor],
            "rationale": "fixture",
        },
        "horizon": {
            "estimate": "SHORT",
            "confidence": 0.9,
            "anchors": [anchor],
            "rationale": "fixture",
        },
    }
    return StructuredModelResult(
        status=StructuredModelStatus.SUCCESS,
        adapter_id="fake",
        model_id="fixture",
        raw_response=json.dumps(valid).encode(),
        decoded=valid,
        usage=SemanticCallUsage(input_tokens=100, output_tokens=50),
    )


def _formalisation_response(items: list[dict[str, JsonValue]]) -> StructuredModelResult:
    payload: JsonValue = {"items": [dict(item) for item in items]}
    return StructuredModelResult(
        status=StructuredModelStatus.SUCCESS,
        adapter_id="fake",
        model_id="fixture",
        raw_response=json.dumps(payload).encode(),
        decoded=payload,
        usage=SemanticCallUsage(input_tokens=100, output_tokens=50),
    )


@pytest.mark.unit
def test_support_ledger_node_ref_resolves_through_the_real_coordinator(
    engine: FrontierReasoningEngine,
) -> None:
    """A proposal whose `support` cites a real prior `SupportLedgerNodeRef`
    must resolve when driven through the actual `composition.py`
    coordinator: (1) `classify_task_semantic` (M01) is driven through the
    real coordinator with a MODEL-derived classification, persisting a real
    M01 ledger node; (2) that node's real `(node_id, revision)` is captured
    from a fresh `engine.inspect`; (3) a second M03 proposal whose `support`
    cites that exact `(node_id, revision)` as a `SupportLedgerNodeRef` is
    driven through `formalise_problem` (also via the real coordinator, which
    computes `known_ledger_refs` from its own fresh post-await
    `engine.inspect`, never a hand-built set); (4) the reference must resolve
    successfully end-to-end, with no `SelfSupportReference`/dangling-ref
    rejection -- proving the wiring `composition.py`'s docstrings claim is
    actually exercised end-to-end, not just at the `ProblemFormaliser.
    formalise()` unit level (as in `test_all_four_support_reference_kinds_
    resolve_when_valid` above, which hand-builds `known_ledger_refs`)."""
    model = _QueueModel([_classification_response()])
    wave3 = compose_wave3(engine, model)
    handle = wave3.create_run()
    task = TaskEnvelope(
        task_id=UUID(int=900),
        text="Choose a safe option.",
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(),
    )

    asyncio.run(wave3.classify_task_semantic(handle.run_id, task, allow_model=True))
    state = engine.inspect(handle.run_id)
    prior_node = next(node for node in state.ledger.nodes if node.producing_module == "M01")

    model.responses = iter(
        [
            _formalisation_response(
                [
                    {
                        "id": "cites-prior-m01-ledger-node",
                        "kind": "UNKNOWN",
                        "description": "supported by a real prior M01 ledger node",
                        "origin": "SUPPORTED_INFERENCE",
                        "basis": (
                            "cross-checked against the classifier's own model-derived inference"
                        ),
                        "support": [
                            {
                                "node_id": str(prior_node.node_id),
                                "revision": prior_node.revision,
                            }
                        ],
                    }
                ]
            )
        ]
    )
    problem = asyncio.run(wave3.formalise_problem(handle.run_id, task, allow_model=True))
    resolved = next(item for item in problem.unknowns if item.id == "cites-prior-m01-ledger-node")
    assert resolved.provenance is not None
    assert len(resolved.provenance.support) == 1


@pytest.mark.unit
def test_formalise_problem_registers_envelope_attachments_before_appending(
    engine: FrontierReasoningEngine,
) -> None:
    """W3 final-gate fix #4: `formalise_problem` built `available_artifacts`
    from `envelope.attachments` purely as a LOCAL `frozenset` for M03's own
    `ProblemFormaliser.formalise()` anchor/support check, but never emitted
    a real `ArtifactRegistered` event for any of them. The reducer's OWN
    independent verification of an M03 `LedgerNodeAdded`'s anchors
    (`_validate_m03_ledger_node_provenance` in `runtime/reducer.py`) checks
    against `state.artifacts`, which is populated ONLY by applied
    `ArtifactRegistered` events -- so a proposal item anchored to a real
    envelope attachment passed `formalise()` locally and then was rejected
    by the reducer at `engine.append`, with no way to ever succeed. This
    test fails on the pre-fix code (the coordinator call below raises) and
    passes after (the coordinator registers every not-yet-known attachment
    in the same atomic batch as the formalisation events)."""
    attachment = ArtifactDataRef(artifact_id=UUID(int=42), sha256="a" * 64)
    task = TaskEnvelope(
        task_id=UUID(int=901),
        text="Use the attached document.",
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(),
        attachments=(attachment,),
    )
    model = _QueueModel(
        [
            _formalisation_response(
                [
                    {
                        "id": "anchored-to-attachment",
                        "kind": "UNKNOWN",
                        "description": "derived from the attached document",
                        "origin": "EXPLICIT_INPUT",
                        "anchors": [
                            {
                                "source_kind": "ARTIFACT",
                                "source_ref": {
                                    "artifact_id": str(attachment.artifact_id),
                                    "sha256": attachment.sha256,
                                },
                                "selector": "",
                            }
                        ],
                    }
                ]
            )
        ]
    )
    wave3 = compose_wave3(engine, model)
    handle = wave3.create_run()
    # Bootstrap the budget a real semantic call needs to reserve against
    # (deterministic-only: this model's queue holds only the formalisation
    # response above, so `allow_model=False` here must never draw from it).
    asyncio.run(wave3.classify_task_semantic(handle.run_id, task, allow_model=False))

    problem = asyncio.run(wave3.formalise_problem(handle.run_id, task, allow_model=True))

    resolved = next(item for item in problem.unknowns if item.id == "anchored-to-attachment")
    assert resolved.provenance is not None
    state = engine.inspect(handle.run_id)
    assert attachment.sha256 in state.artifacts


# ---------------------------------------------------------------------------
# F07: contradicted-without-relation must be rejected; material vs.
# non-material items produce blockers only when material.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_contradicted_origin_without_a_contradicts_relation_is_rejected() -> None:
    proposal = ProblemFormalisationOutput.model_validate(
        {
            "items": (
                {
                    "id": "orphan",
                    "kind": "UNKNOWN",
                    "description": "claims to be contradicted",
                    "origin": EpistemicOriginLabel.CONTRADICTED,
                },
            )
        }
    )
    with pytest.raises(InvalidProblemSpec, match="CONTRADICTED"):
        ProblemFormaliser().formalise(envelope(), proposal)


@pytest.mark.unit
def test_material_unresolved_unknown_produces_blocker_non_material_does_not() -> None:
    proposal = ProblemFormalisationOutput.model_validate(
        {
            "items": (
                {
                    "id": "material-gap",
                    "kind": "UNKNOWN",
                    "description": "must be known before proceeding",
                    "origin": EpistemicOriginLabel.UNRESOLVED,
                    "attributes": {"material": True},
                },
                {
                    "id": "cosmetic-gap",
                    "kind": "UNKNOWN",
                    "description": "does not affect the decision",
                    "origin": EpistemicOriginLabel.UNRESOLVED,
                    "attributes": {"material": False},
                    # Finding D (C06 remediation, round 2): a self-reported
                    # LOW-materiality claim is honoured only when corroborated
                    # by at least one independently-resolvable support
                    # reference -- without one it is always treated as
                    # material regardless of the claim (see
                    # `test_self_reported_low_materiality_without_support_is_still_material`
                    # below for the corroboration-free case).
                    "support": ({"item_id": "material-gap"},),
                },
            )
        }
    )
    events = ProblemFormaliser().canonical_events(
        envelope(),
        proposal,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        uuids=FakeUUIDFactory(UUID(int=index) for index in range(1, 20)),
    )
    blockers = [event for event in events if isinstance(event, ProblemBlockerRecorded)]
    assert len(blockers) == 1
    assert blockers[0].blocker.blocker_id == "unknown:material-gap"
    # The non-material item still keeps its ledger node (retained provenance):
    nodes = {
        node.content["id"]: node
        for node in (e.node for e in events if isinstance(e, LedgerNodeAdded))
        if isinstance(node.content, dict)
    }
    assert "cosmetic-gap" in nodes


@pytest.mark.unit
def test_material_defaults_true_when_no_explicit_signal_is_given() -> None:
    """F07: materiality must never be silently downgraded by mere omission --
    an unresolved item with no explicit `material`/`decision_relevance`
    signal is conservatively treated as material."""
    proposal = ProblemFormalisationOutput.model_validate(
        {
            "items": (
                {
                    "id": "unspecified",
                    "kind": "UNKNOWN",
                    "description": "no materiality signal given",
                    "origin": EpistemicOriginLabel.UNRESOLVED,
                },
            )
        }
    )
    events = ProblemFormaliser().canonical_events(
        envelope(),
        proposal,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        uuids=FakeUUIDFactory(UUID(int=index) for index in range(1, 10)),
    )
    assert any(isinstance(event, ProblemBlockerRecorded) for event in events)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.75, 0.75),
        (2, 2.0),
        (0, 0.0),
        (True, None),
        (False, None),
        ("0.75", None),
        (None, None),
        ({"nested": 1}, None),
    ],
)
def test_coerce_optional_float_isolated(value: object, expected: float | None) -> None:
    """EU-19 (C06 cleanup): `_coerce_optional_float` -- isolated from its
    three call sites -- must coerce `int`/`float` to `float`, reject `bool`
    (an `int` subclass in Python) despite that, and treat every other type
    (`str`, `None`, other) as "no numeric signal"."""
    assert _coerce_optional_float(value) == expected


# ---------------------------------------------------------------------------
# F04: UNKNOWN and assumption metadata round-trips losslessly through the
# full formalisation -> events -> reducer -> SQLite -> snapshot -> replay ->
# M09 content pipeline.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unknown_and_assumption_fields_round_trip_through_engine(
    engine: FrontierReasoningEngine,
) -> None:
    run = engine.create_run({"c06": "round-trip"})
    proposal = ProblemFormalisationOutput.model_validate(
        {
            "items": (
                {
                    "id": "u1",
                    "kind": "UNKNOWN",
                    "description": "future demand",
                    "origin": EpistemicOriginLabel.UNRESOLVED,
                    "attributes": {
                        "domain": "numeric",
                        "rationale": "no forecast is available yet",
                        "impact": {"revenue": "high"},
                        "decision_relevance": 0.75,
                        "resolvable": True,
                        "candidate_actions": ["run a market survey", "wait one quarter"],
                        "material": True,
                    },
                },
                {
                    "id": "a1",
                    "kind": "ASSUMPTION",
                    "description": "demand is roughly stable",
                    "origin": EpistemicOriginLabel.WORKING_ASSUMPTION,
                    "basis": "no contrary signal observed",
                    "policy_basis": "default planning policy",
                    "attributes": {"decision_relevance": 0.2, "scope": "/forecast/next_quarter"},
                },
            )
        }
    )
    events = ProblemFormaliser().canonical_events(
        envelope(), proposal, created_at=engine.clock.now(), uuids=engine.uuids
    )
    uncommitted = tuple(engine.make_event(run.run_id, event, module_id="M03") for event in events)
    engine.append(run.run_id, run.version, uncommitted)

    for state in (engine.inspect(run.run_id), engine.replay(run.run_id)):
        problem_spec = state.problem_spec
        assert problem_spec is not None
        unknown = next(item for item in problem_spec.unknowns if item.id == "u1")
        assert unknown.domain == "numeric"
        assert unknown.rationale == "no forecast is available yet"
        assert unknown.impact == {"revenue": "high"}
        assert unknown.decision_relevance == 0.75
        assert unknown.resolvable is True
        assert unknown.candidate_actions == ("run a market survey", "wait one quarter")

        assumption = next(item for item in problem_spec.assumption_items if item.id == "a1")
        assert assumption.decision_relevance == 0.2
        assert assumption.scope == "/forecast/next_quarter"
        assert "demand is roughly stable" in problem_spec.assumptions

    # Explicit snapshot/replay-from-snapshot round trip.
    engine.snapshot(run.run_id)
    replayed = engine.replay_from_snapshot(run.run_id)
    replayed_spec = replayed.problem_spec
    assert replayed_spec is not None
    unknown = next(item for item in replayed_spec.unknowns if item.id == "u1")
    assert unknown.candidate_actions == ("run a market survey", "wait one quarter")
    assert unknown.impact == {"revenue": "high"}

    # M09 content: the ledger node's own content dict must also carry every
    # accepted field (not just the ProblemSpec projection).
    unknown_node_content = next(
        node.content
        for node in replayed.ledger.nodes
        if node.producing_module == "M03"
        and isinstance(node.content, dict)
        and node.content.get("id") == "u1"
    )
    assert isinstance(unknown_node_content, dict)
    attributes = unknown_node_content["attributes"]
    assert isinstance(attributes, dict)
    assert attributes["rationale"] == "no forecast is available yet"
    assert attributes["candidate_actions"] == [
        "run a market survey",
        "wait one quarter",
    ]


# ---------------------------------------------------------------------------
# F06: no public API bypasses provenance/ledger validation; a direct
# `RunReducer.apply` caller cannot smuggle an unvalidated M03 ledger node.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ledger_events_is_no_longer_a_public_method() -> None:
    assert not hasattr(ProblemFormaliser, "ledger_events")


@pytest.mark.unit
def test_reducer_apply_rejects_forged_m03_ledger_node_with_unauthorized_artifact() -> None:
    """Bypass `ProblemFormaliser.canonical_events` entirely: hand-build a
    self-consistent `LedgerNodeAdded` claiming `producing_module="M03"` whose
    embedded `support` cites an artifact nobody ever registered, and call
    `RunReducer.apply` directly. The reducer itself must reject it -- this is
    the C04/C05 lesson applied to C06: the invariant must not depend on
    every caller going through `ProblemFormaliser`."""
    from fre.domain.common import canonical_hash as _hash
    from fre.modules.m09_ledger import make_node
    from fre.runtime.events import RunCreated, StoredEvent, event_wire_identity

    run_id = UUID(int=1)
    reducer = RunReducer()
    state = reducer.initial(run_id)

    def _stored(payload: object, sequence: int) -> StoredEvent:
        event_type, schema_version = event_wire_identity(payload)  # type: ignore[arg-type]
        return StoredEvent(
            event_id=UUID(int=sequence + 1000),
            run_id=run_id,
            event_type=event_type,
            action_id=UUID(int=sequence + 2000),
            module_id="attacker",
            schema_version=schema_version,
            module_version="1.0",
            input_hash=_hash(None),
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            sequence=sequence,
            payload=payload.model_dump(mode="json"),  # type: ignore[attr-defined]
        )

    state = reducer.apply(state, _stored(RunCreated(config_hash="c" * 64), 1))

    forged_node = make_node(
        node_id=UUID(int=3000),
        revision=1,
        node_type=LedgerNodeType.INFERENCE,
        content={
            "id": "forged",
            "kind": "UNKNOWN",
            "description": "forged support",
            "origin": "SUPPORTED_INFERENCE",
            "anchors": [],
            "supporting_refs": [],
            "support": [{"artifact_id": str(UUID(int=9)), "sha256": "f" * 64}],
            "basis": "forged",
            "policy_basis": None,
            "attributes": {},
        },
        status=EpistemicStatus.PROVISIONAL,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        action_id=UUID(int=4000),
        module_id="M03",
    )
    with pytest.raises(ValueError, match="artifact that is not registered"):
        reducer.apply(state, _stored(LedgerNodeAdded(node=forged_node), 2))


@pytest.mark.unit
def test_reducer_apply_accepts_m03_ledger_node_once_artifact_is_registered() -> None:
    """The same forged-shape node, but with a genuinely registered artifact,
    must be accepted -- proving the reducer check resolves real evidence
    rather than merely rejecting everything."""
    from fre.domain.common import canonical_hash as _hash
    from fre.modules.m09_ledger import make_node
    from fre.runtime.events import RunCreated, StoredEvent, event_wire_identity

    run_id = UUID(int=1)
    reducer = RunReducer()
    state = reducer.initial(run_id)

    def _stored(payload: object, sequence: int) -> StoredEvent:
        event_type, schema_version = event_wire_identity(payload)  # type: ignore[arg-type]
        return StoredEvent(
            event_id=UUID(int=sequence + 1000),
            run_id=run_id,
            event_type=event_type,
            action_id=UUID(int=sequence + 2000),
            module_id="test",
            schema_version=schema_version,
            module_version="1.0",
            input_hash=_hash(None),
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            sequence=sequence,
            payload=payload.model_dump(mode="json"),  # type: ignore[attr-defined]
        )

    state = reducer.apply(state, _stored(RunCreated(config_hash="c" * 64), 1))
    state = reducer.apply(
        state,
        _stored(
            ArtifactRegistered(
                artifact=ArtifactDataRef(artifact_id=UUID(int=9), sha256="f" * 64),
                media_type="text/plain",
                byte_size=3,
            ),
            2,
        ),
    )
    node = make_node(
        node_id=UUID(int=3000),
        revision=1,
        node_type=LedgerNodeType.INFERENCE,
        content={
            "id": "ok",
            "kind": "UNKNOWN",
            "description": "genuine support",
            "origin": "SUPPORTED_INFERENCE",
            "anchors": [],
            "supporting_refs": [],
            "support": [{"artifact_id": str(UUID(int=9)), "sha256": "f" * 64}],
            "basis": "genuine",
            "policy_basis": None,
            "attributes": {},
        },
        status=EpistemicStatus.PROVISIONAL,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        action_id=UUID(int=4000),
        module_id="M03",
    )
    state = reducer.apply(state, _stored(LedgerNodeAdded(node=node), 3))
    admitted_content = state.ledger.nodes[0].content
    assert isinstance(admitted_content, dict)
    assert admitted_content["id"] == "ok"


# ---------------------------------------------------------------------------
# C06 remediation, round 2: independent-review findings A-G.
# ---------------------------------------------------------------------------


def _stored_event(payload: object, run_id: UUID, sequence: int) -> StoredEvent:
    """Shared helper for the hand-built `RunReducer.apply` forgery tests
    below (mirrors the local `_stored` closures already used above)."""
    event_type, schema_version = event_wire_identity(payload)  # type: ignore[arg-type]
    return StoredEvent(
        event_id=UUID(int=sequence + 1000),
        run_id=run_id,
        event_type=event_type,
        action_id=UUID(int=sequence + 2000),
        module_id="test",
        schema_version=schema_version,
        module_version="1.0",
        input_hash=hashlib.sha256(b"noop").hexdigest(),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        sequence=sequence,
        payload=payload.model_dump(mode="json"),  # type: ignore[attr-defined]
    )


@pytest.mark.unit
def test_reducer_rejects_cross_batch_same_proposal_item_reference() -> None:
    """Finding A (P0, empirically demonstrated exploitable): a hand-built
    `LedgerNodeAdded` citing an item-id from an unrelated, already-committed
    EARLIER M03 batch/proposal must be REJECTED by the reducer -- exactly as
    the same content is, and always was, rejected by
    `ProblemFormaliser.formalise()` (whose `index` is scoped to the current
    proposal only). Before this fix, `known_m03_ids` was built from the
    ENTIRE ledger's M03 history with no batch scoping at all, so this exact
    forgery was silently ACCEPTED."""
    from fre.modules.m09_ledger import make_node
    from fre.runtime.events import RunCreated

    run_id = UUID(int=1)
    reducer = RunReducer()
    state = reducer.initial(run_id)
    state = reducer.apply(state, _stored_event(RunCreated(config_hash="c" * 64), run_id, 1))

    # An unrelated, already-committed earlier batch contributes "shared-id".
    earlier_batch_node = make_node(
        node_id=UUID(int=3000),
        revision=1,
        node_type=LedgerNodeType.UNKNOWN,
        content={
            "batch_id": "batch-1",
            "id": "shared-id",
            "kind": "UNKNOWN",
            "description": "an unrelated, already-committed item",
            "origin": "UNRESOLVED",
            "anchors": [],
            "supporting_refs": [],
            "support": [],
            "basis": None,
            "policy_basis": None,
            "attributes": {},
        },
        status=EpistemicStatus.UNRESOLVED,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        action_id=UUID(int=4000),
        module_id="M03",
    )
    state = reducer.apply(state, _stored_event(LedgerNodeAdded(node=earlier_batch_node), run_id, 2))

    # A DIFFERENT batch/proposal (its own batch_id) forges a citation of
    # "shared-id" as if it were a same-proposal support reference.
    forged_node = make_node(
        node_id=UUID(int=3001),
        revision=1,
        node_type=LedgerNodeType.INFERENCE,
        content={
            "batch_id": "batch-2",
            "id": "citing",
            "kind": "UNKNOWN",
            "description": "cites an unrelated batch's item as support",
            "origin": "SUPPORTED_INFERENCE",
            "anchors": [],
            "supporting_refs": [],
            "support": [{"item_id": "shared-id"}],
            "basis": "forged cross-batch citation",
            "policy_basis": None,
            "attributes": {},
        },
        status=EpistemicStatus.PROVISIONAL,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        action_id=UUID(int=4001),
        module_id="M03",
    )
    with pytest.raises(ValueError, match="not been admitted to the ledger yet"):
        reducer.apply(state, _stored_event(LedgerNodeAdded(node=forged_node), run_id, 3))

    # Same cross-proposal citation, through the real entry point: always
    # rejected, both before and after this fix.
    proposal = ProblemFormalisationOutput.model_validate(
        {
            "items": (
                {
                    "id": "citing",
                    "kind": "UNKNOWN",
                    "description": "cites an item from a different proposal",
                    "origin": EpistemicOriginLabel.SUPPORTED_INFERENCE,
                    "basis": "forged cross-batch citation",
                    "support": ({"item_id": "shared-id"},),
                },
            )
        }
    )
    with pytest.raises(InvalidSourceAnchor):
        ProblemFormaliser().formalise(envelope(), proposal)


@pytest.mark.unit
def test_reducer_rejects_explicit_input_node_with_unregistered_artifact_anchor() -> None:
    """Finding B (P0, empirically demonstrated exploitable):
    `EpistemicItemProvenance.anchors` (mandatory for EXPLICIT_INPUT origin)
    was never re-checked by the reducer at all -- only `content["support"]`
    was inspected. Forge an EXPLICIT_INPUT node whose sole anchor is an
    ARTIFACT-kind anchor pointing at an artifact nobody ever registered."""
    from fre.modules.m09_ledger import make_node
    from fre.runtime.events import RunCreated

    run_id = UUID(int=1)
    reducer = RunReducer()
    state = reducer.initial(run_id)
    state = reducer.apply(state, _stored_event(RunCreated(config_hash="c" * 64), run_id, 1))

    forged_node = make_node(
        node_id=UUID(int=3000),
        revision=1,
        node_type=LedgerNodeType.FACT,
        content={
            "batch_id": "batch-1",
            "id": "forged-fact",
            "kind": "UNKNOWN",
            "description": "claims explicit input from an unregistered artifact",
            "origin": "EXPLICIT_INPUT",
            "anchors": [
                {
                    "source_kind": "ARTIFACT",
                    "source_ref": {"artifact_id": str(UUID(int=9)), "sha256": "e" * 64},
                    "selector": "",
                }
            ],
            "supporting_refs": [],
            "support": [],
            "basis": None,
            "policy_basis": None,
            "attributes": {},
        },
        status=EpistemicStatus.SUPPORTED,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        action_id=UUID(int=4000),
        module_id="M03",
    )
    with pytest.raises(ValueError, match="artifact that is not registered"):
        reducer.apply(state, _stored_event(LedgerNodeAdded(node=forged_node), run_id, 2))


@pytest.mark.unit
def test_reducer_rejects_explicit_input_node_with_no_anchors_at_all() -> None:
    """Finding B, second half: EXPLICIT_INPUT with an empty `anchors` list is
    just as forged as one with an unregistered artifact -- both must be
    rejected at the reducer, not only by `EpistemicItemProvenance`'s own
    pydantic validator (which a hand-built dict content bypasses entirely)."""
    from fre.modules.m09_ledger import make_node
    from fre.runtime.events import RunCreated

    run_id = UUID(int=1)
    reducer = RunReducer()
    state = reducer.initial(run_id)
    state = reducer.apply(state, _stored_event(RunCreated(config_hash="c" * 64), run_id, 1))

    forged_node = make_node(
        node_id=UUID(int=3000),
        revision=1,
        node_type=LedgerNodeType.FACT,
        content={
            "batch_id": "batch-1",
            "id": "forged-fact",
            "kind": "UNKNOWN",
            "description": "claims explicit input with no anchor at all",
            "origin": "EXPLICIT_INPUT",
            "anchors": [],
            "supporting_refs": [],
            "support": [],
            "basis": None,
            "policy_basis": None,
            "attributes": {},
        },
        status=EpistemicStatus.SUPPORTED,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        action_id=UUID(int=4000),
        module_id="M03",
    )
    with pytest.raises(ValueError, match="EXPLICIT_INPUT origin without any anchor"):
        reducer.apply(state, _stored_event(LedgerNodeAdded(node=forged_node), run_id, 2))


@pytest.mark.unit
def test_reducer_rejects_support_citing_a_relation_item() -> None:
    """Finding C: a minimal, kind-based relevance safeguard, extended to the
    reducer for the first time -- a `SupportProblemItemRef` naming a
    `RELATION`-kind target (relations describe edges between items, never
    evidence in their own right) is rejected here too, not only in
    `formalise()`. This cannot and does not verify semantic relevance of the
    target's actual content -- only that its structural kind is even
    evidentially eligible."""
    from fre.modules.m09_ledger import make_node
    from fre.runtime.events import RunCreated

    run_id = UUID(int=1)
    reducer = RunReducer()
    state = reducer.initial(run_id)
    state = reducer.apply(state, _stored_event(RunCreated(config_hash="c" * 64), run_id, 1))

    relation_node = make_node(
        node_id=UUID(int=3000),
        revision=1,
        node_type=LedgerNodeType.UNKNOWN,
        content={
            "batch_id": "batch-1",
            "id": "rel",
            "kind": "RELATION",
            "description": "a depends on b",
            "origin": "UNRESOLVED",
            "anchors": [],
            "supporting_refs": [],
            "support": [],
            "basis": None,
            "policy_basis": None,
            "attributes": {"source_id": "a", "target_id": "b", "relation_kind": "DEPENDS_ON"},
        },
        status=EpistemicStatus.UNRESOLVED,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        action_id=UUID(int=4000),
        module_id="M03",
    )
    state = reducer.apply(state, _stored_event(LedgerNodeAdded(node=relation_node), run_id, 2))

    forged_node = make_node(
        node_id=UUID(int=3001),
        revision=1,
        node_type=LedgerNodeType.INFERENCE,
        content={
            "batch_id": "batch-1",
            "id": "citing",
            "kind": "UNKNOWN",
            "description": "cites a relation pseudo-item as evidence",
            "origin": "SUPPORTED_INFERENCE",
            "anchors": [],
            "supporting_refs": [],
            "support": [{"item_id": "rel"}],
            "basis": "forged",
            "policy_basis": None,
            "attributes": {},
        },
        status=EpistemicStatus.PROVISIONAL,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        action_id=UUID(int=4001),
        module_id="M03",
    )
    with pytest.raises(IncompatibleSupportReferenceKind):
        reducer.apply(state, _stored_event(LedgerNodeAdded(node=forged_node), run_id, 3))


@pytest.mark.unit
def test_self_reported_low_materiality_without_support_is_still_material() -> None:
    """Finding D: a self-reported LOW-materiality claim (`decision_relevance`
    below threshold, or `material=False`) with NO independently-resolvable
    support must still produce a blocker -- the self-report alone, with
    nothing to corroborate it, is never trusted (there is no independent,
    deterministic materiality signal available for M03 items the way M01 has
    an execution-permission-derived floor)."""
    proposal = ProblemFormalisationOutput.model_validate(
        {
            "items": (
                {
                    "id": "unsubstantiated-low",
                    "kind": "UNKNOWN",
                    "description": "claims low relevance with nothing to back it",
                    "origin": EpistemicOriginLabel.UNRESOLVED,
                    "attributes": {"decision_relevance": 0.1},
                },
                {
                    "id": "unsubstantiated-false",
                    "kind": "UNKNOWN",
                    "description": "claims non-materiality with nothing to back it",
                    "origin": EpistemicOriginLabel.UNRESOLVED,
                    "attributes": {"material": False},
                },
            )
        }
    )
    events = ProblemFormaliser().canonical_events(
        envelope(),
        proposal,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        uuids=FakeUUIDFactory(UUID(int=index) for index in range(1, 20)),
    )
    blocker_ids = {
        event.blocker.blocker_id for event in events if isinstance(event, ProblemBlockerRecorded)
    }
    assert blocker_ids == {"unknown:unsubstantiated-low", "unknown:unsubstantiated-false"}


@pytest.mark.unit
def test_reducer_rejects_malformed_text_span_inside_a_support_anchor() -> None:
    """Finding E: a `SourceAnchor`-kind `support` entry with an out-of-order
    text span (`char_end <= char_start`) is caught by the reducer purely by
    parsing it back into the typed `SourceAnchor` model, whose own
    `valid_range` validator enforces this structurally -- without needing
    the originating `TaskEnvelope` at all. `TASK_FIELD`/`TASK_TEXT`
    resolution against that envelope remains `formalise()`-only."""
    from fre.modules.m09_ledger import make_node
    from fre.runtime.events import RunCreated

    run_id = UUID(int=1)
    reducer = RunReducer()
    state = reducer.initial(run_id)
    state = reducer.apply(state, _stored_event(RunCreated(config_hash="c" * 64), run_id, 1))

    forged_node = make_node(
        node_id=UUID(int=3000),
        revision=1,
        node_type=LedgerNodeType.INFERENCE,
        content={
            "batch_id": "batch-1",
            "id": "bad-span",
            "kind": "UNKNOWN",
            "description": "cites a malformed text span as support",
            "origin": "SUPPORTED_INFERENCE",
            "anchors": [],
            "supporting_refs": [],
            "support": [
                {
                    "source_kind": "TASK_TEXT",
                    "source_ref": {
                        "object_type": "TaskEnvelope",
                        "object_id": str(UUID(int=1)),
                    },
                    "selector": "/text",
                    "char_start": 9999,
                    "char_end": 3,
                }
            ],
            "basis": "forged",
            "policy_basis": None,
            "attributes": {},
        },
        status=EpistemicStatus.PROVISIONAL,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        action_id=UUID(int=4000),
        module_id="M03",
    )
    with pytest.raises(ValueError, match="malformed support reference"):
        reducer.apply(state, _stored_event(LedgerNodeAdded(node=forged_node), run_id, 2))


@pytest.mark.unit
def test_reducer_rejects_cross_batch_duplicate_item_id() -> None:
    """Finding F: two independent batches each formalising a distinct item
    under the SAME `id` string, with different content, is never silently
    accepted as an unrelated collision. Only a `LedgerNodeRevised` event
    represents a legitimate revision of an existing item's content (a
    deliberate design choice, documented on
    `_validate_m03_ledger_node_provenance`); a second `LedgerNodeAdded`
    reusing an id already claimed by a different batch is always rejected."""
    from fre.modules.m09_ledger import make_node
    from fre.runtime.events import RunCreated

    run_id = UUID(int=1)
    reducer = RunReducer()
    state = reducer.initial(run_id)
    state = reducer.apply(state, _stored_event(RunCreated(config_hash="c" * 64), run_id, 1))

    first_node = make_node(
        node_id=UUID(int=3000),
        revision=1,
        node_type=LedgerNodeType.UNKNOWN,
        content={
            "batch_id": "batch-1",
            "id": "dup-id",
            "kind": "UNKNOWN",
            "description": "the first, genuine item under this id",
            "origin": "UNRESOLVED",
            "anchors": [],
            "supporting_refs": [],
            "support": [],
            "basis": None,
            "policy_basis": None,
            "attributes": {},
        },
        status=EpistemicStatus.UNRESOLVED,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        action_id=UUID(int=4000),
        module_id="M03",
    )
    state = reducer.apply(state, _stored_event(LedgerNodeAdded(node=first_node), run_id, 2))

    colliding_node = make_node(
        node_id=UUID(int=3001),
        revision=1,
        node_type=LedgerNodeType.UNKNOWN,
        content={
            "batch_id": "batch-2",
            "id": "dup-id",
            "kind": "UNKNOWN",
            "description": "a completely different item that reuses the same id",
            "origin": "UNRESOLVED",
            "anchors": [],
            "supporting_refs": [],
            "support": [],
            "basis": None,
            "policy_basis": None,
            "attributes": {},
        },
        status=EpistemicStatus.UNRESOLVED,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        action_id=UUID(int=4001),
        module_id="M03",
    )
    with pytest.raises(ValueError, match="collides with an item already admitted"):
        reducer.apply(state, _stored_event(LedgerNodeAdded(node=colliding_node), run_id, 3))


@pytest.mark.unit
def test_reducer_and_formalise_share_identical_support_ref_error_taxonomy() -> None:
    """Finding G: the reducer's re-validation now parses `support` back into
    the same typed `SupportRef` models and resolves them through the same
    envelope-independent helper `formalise()`'s own validation uses
    (`resolve_support_ref_at_reduction` / `_validate_single_support_ref`
    both delegate to `_check_artifact_support_ref`), rather than maintaining
    a second, independently-drifting duck-typed reimplementation. Proven
    here by both call sites raising the exact SAME exception class for the
    exact same malformed reference shape."""
    from fre.modules.m09_ledger import make_node
    from fre.runtime.events import RunCreated

    proposal = ProblemFormalisationOutput.model_validate(
        {
            "items": (
                {
                    "id": "x",
                    "kind": "UNKNOWN",
                    "description": "artifact-supported",
                    "origin": EpistemicOriginLabel.SUPPORTED_INFERENCE,
                    "basis": "from an attachment",
                    "support": (SupportArtifactRef(artifact_id=UUID(int=99), sha256="a" * 64),),
                },
            )
        }
    )
    with pytest.raises(UnauthorizedArtifactReference):
        ProblemFormaliser().formalise(envelope(), proposal, available_artifacts=frozenset())

    run_id = UUID(int=1)
    reducer = RunReducer()
    state = reducer.initial(run_id)
    state = reducer.apply(state, _stored_event(RunCreated(config_hash="c" * 64), run_id, 1))
    forged_node = make_node(
        node_id=UUID(int=3000),
        revision=1,
        node_type=LedgerNodeType.INFERENCE,
        content={
            "batch_id": "batch-1",
            "id": "x",
            "kind": "UNKNOWN",
            "description": "artifact-supported",
            "origin": "SUPPORTED_INFERENCE",
            "anchors": [],
            "supporting_refs": [],
            "support": [{"artifact_id": str(UUID(int=99)), "sha256": "a" * 64}],
            "basis": "from an attachment",
            "policy_basis": None,
            "attributes": {},
        },
        status=EpistemicStatus.PROVISIONAL,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        action_id=UUID(int=4000),
        module_id="M03",
    )
    with pytest.raises(UnauthorizedArtifactReference):
        reducer.apply(state, _stored_event(LedgerNodeAdded(node=forged_node), run_id, 2))
