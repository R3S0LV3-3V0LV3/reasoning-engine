"""Decisive C06 remediation tests: M03 resolved support, UNKNOWN preservation,
and materiality (defects F03, F04, F06, F07).

Each test below maps directly onto one bullet in the C06 validation strategy
in FRE_WAVE3_C01_C10_EXECUTION_COMPLETION_AND_VALIDATION_REGISTER.md.
"""

import hashlib
from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from fre.adapters.testing import FakeUUIDFactory
from fre.domain.common import ArtifactRef as ArtifactDataRef
from fre.domain.common import ObjectRef, OutputContract, PermissionSet
from fre.domain.ledger import EpistemicStatus, LedgerNodeType
from fre.domain.semantic import (
    EpistemicItemProvenance,
    EpistemicOriginLabel,
    SourceAnchor,
    SourceKind,
    SupportArtifactRef,
    SupportLedgerNodeRef,
    SupportProblemItemRef,
)
from fre.domain.task import TaskEnvelope
from fre.engine import FrontierReasoningEngine
from fre.modules.m03_formaliser import InvalidProblemSpec, ProblemFormaliser
from fre.modules.source_anchors import (
    DanglingLedgerSupportReference,
    IncompatibleSupportReferenceKind,
    InvalidSourceAnchor,
    SelfSupportReference,
    UnauthorizedArtifactReference,
)
from fre.prompts.schemas import ProblemFormalisationOutput
from fre.runtime.events import ArtifactRegistered, LedgerNodeAdded, ProblemBlockerRecorded
from fre.runtime.reducer import RunReducer


def envelope() -> TaskEnvelope:
    return TaskEnvelope(
        task_id=UUID(int=1),
        text="Minimize cost subject to cost <= 10.",
        explicit_constraints=("cost <= 10",),
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
