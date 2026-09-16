from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from fre.domain.budget import BudgetProjection, DeploymentLimits
from fre.domain.common import FrozenModel, canonical_json
from fre.domain.context import CompilerProfile, ContextDeltaPacket
from fre.domain.ledger import (
    EpistemicStatus,
    InvalidStatusTransition,
    LedgerNode,
    LedgerNodeType,
    LedgerProjection,
)
from fre.domain.task import TaskSignature
from fre.modules.m02_budget import BudgetAllocator, default_tier_policy
from fre.modules.m09_ledger import EpistemicLedger, make_node
from fre.modules.m12_context import ContextCompiler, apply_delta
from fre.runtime.events import UncommittedEvent


def low_signature() -> TaskSignature:
    from fre.domain.task import (
        HorizonClass,
        Ordinal4,
        OutputForm,
        SearchSpaceClass,
        TaskSignature,
        TaskType,
    )

    return TaskSignature(
        task_type=TaskType.ANALYSIS,
        consequence=Ordinal4.LOW,
        irreversibility=Ordinal4.LOW,
        ambiguity=Ordinal4.LOW,
        search_space=SearchSpaceClass.CLOSED,
        evidence_scarcity=Ordinal4.LOW,
        horizon=HorizonClass.IMMEDIATE,
        output_form=OutputForm.TEXT,
        dimension_confidence={},
    )


class NestedFixture(FrozenModel):
    payload: dict[str, list[dict[str, int]]]


def test_persistent_values_are_recursively_immutable() -> None:
    fixture = NestedFixture(payload={"items": [{"value": 1}]})
    with pytest.raises(TypeError, match="immutable"):
        fixture.payload["new"] = []
    with pytest.raises(TypeError, match="immutable"):
        fixture.payload["items"].append({"value": 2})
    with pytest.raises(TypeError, match="immutable"):
        fixture.payload["items"][0]["value"] = 3
    assert canonical_json(fixture) == b'{"payload":{"items":[{"value":1}]}}'


def test_reducer_outputs_remain_recursively_immutable() -> None:
    from datetime import UTC, datetime

    from fre.domain.common import canonical_hash
    from fre.runtime.events import RunCreated, StoredEvent, TestValueSet
    from fre.runtime.reducer import RunReducer

    run_id = UUID(int=80)
    payloads = (
        RunCreated(config_hash="a" * 64),
        TestValueSet(key="nested", value={"items": [1]}),
    )
    events = tuple(
        StoredEvent(
            event_id=UUID(int=81 + index),
            run_id=run_id,
            sequence=index + 1,
            event_type=type(payload).__name__,
            action_id=UUID(int=91 + index),
            module_id="runtime",
            module_version="1.0",
            input_hash=canonical_hash(payload),
            created_at=datetime(2026, 1, 1, 0, 0, index, tzinfo=UTC),
            payload=payload.model_dump(mode="json"),
        )
        for index, payload in enumerate(payloads)
    )
    state = RunReducer().reduce(run_id, events)
    nested = state.values["nested"]
    assert isinstance(nested, dict)
    with pytest.raises(TypeError, match="immutable"):
        nested["new"] = 2


def test_canonical_json_rejects_non_string_mapping_keys_and_non_finite_numbers() -> None:
    with pytest.raises(TypeError, match="keys must be strings"):
        canonical_json({1: "ambiguous"})
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            canonical_json(value)


def test_event_registry_rejects_unknown_schema_and_empty_provenance() -> None:
    base = {
        "event_id": UUID(int=1),
        "run_id": UUID(int=2),
        "event_type": "RunCreated",
        "action_id": UUID(int=3),
        "module_id": "runtime",
        "module_version": "1.0",
        "input_hash": "0" * 64,
        "created_at": "2026-01-01T00:00:00Z",
        "payload": {"config_hash": "a" * 64},
    }
    event = UncommittedEvent.model_validate({**base, "schema_version": "2.0"}, strict=False)
    with pytest.raises(ValueError, match="type/schema"):
        event.validated_payload()
    with pytest.raises(ValidationError):
        UncommittedEvent.model_validate({**base, "module_id": ""}, strict=False)


def test_invalid_status_transition_requires_successor_revision() -> None:
    service = EpistemicLedger()
    node = make_node(
        node_id=UUID(int=1),
        revision=1,
        node_type=LedgerNodeType.FACT,
        content={"claim": "refuted"},
        status=EpistemicStatus.REFUTED,
        created_at="2026-01-01T00:00:00Z",
        action_id=UUID(int=2),
        module_id="M09",
    )
    projection = service.append_node(LedgerProjection(), node)
    with pytest.raises(InvalidStatusTransition):
        service.mark_status(projection, node.ref, EpistemicStatus.SUPPORTED)


def test_policy_hash_is_stable_and_projection_is_historic_event_data() -> None:
    allocator = BudgetAllocator()
    policy = default_tier_policy()
    plan, first_hash = allocator.allocate(low_signature(), policy, DeploymentLimits())
    _, second_hash = allocator.allocate(low_signature(), policy, DeploymentLimits())
    projection = BudgetProjection(plan=plan, policy_hash=first_hash)
    changed_policy = policy.model_copy(update={"policy_version": "future/9.0"})
    assert first_hash == second_hash
    assert projection.plan == plan
    assert projection.policy_hash != allocator.policy_hash(changed_policy, DeploymentLimits())


def test_tampered_delta_hash_and_target_packet_are_rejected() -> None:
    # The exact round-trip is covered by the context suite; this test hardens integrity failures.
    invalid = ContextDeltaPacket(
        base_packet_hash="a" * 64,
        target_packet_hash="b" * 64,
        compiler_version="1.0",
        compression_policy_version="1.0",
        profile=CompilerProfile.HANDOFF,
        operations=(),
        delta_hash="c" * 64,
    )
    from fre.modules.m12_context import ContextCompiler
    from fre.runtime.budget_meter import BudgetMeter

    allocator = BudgetAllocator()
    plan, policy_hash = allocator.allocate(
        low_signature(), default_tier_policy(), DeploymentLimits()
    )
    base = (
        ContextCompiler()
        .compile(
            run_id=UUID(int=1),
            snapshot_version=1,
            ledger=LedgerProjection(),
            budget_remaining=BudgetMeter().remaining(
                BudgetProjection(plan=plan, policy_hash=policy_hash)
            ),
            profile=CompilerProfile.HANDOFF,
        )
        .packet
    )
    with pytest.raises(ValueError, match="delta hash"):
        apply_delta(base, invalid.model_copy(update={"base_packet_hash": base.packet_hash}))


def test_pep561_marker_exists_in_source_package() -> None:
    assert Path("src/fre/py.typed").is_file()


def test_dependency_confidence_envelope_tracks_depth_ranges_and_direct_fraction() -> None:
    from fre.domain.common import ConfidenceAssessment
    from fre.domain.ledger import LedgerEdge, LedgerRelation

    service = EpistemicLedger()

    def make(value: int, score: float | None = None) -> LedgerNode:
        return make_node(
            node_id=UUID(int=value),
            revision=1,
            node_type=LedgerNodeType.FACT,
            content={"value": value},
            status=EpistemicStatus.SUPPORTED,
            created_at="2026-01-01T00:00:00Z",
            action_id=UUID(int=100 + value),
            module_id="M09",
            confidence=ConfidenceAssessment(score=score, method="RULE", explanation="fixture")
            if score is not None
            else None,
        )

    root, direct, transitive = make(1, 0.3), make(2), make(3)
    projection = LedgerProjection()
    for item in (root, direct, transitive):
        projection = service.append_node(projection, item)
    projection = service.append_edge(
        projection,
        LedgerEdge(
            edge_id=UUID(int=10),
            source=root.ref,
            target=direct.ref,
            relation=LedgerRelation.SUPPORTS,
            independence_group="source-a",
        ),
    )
    projection = service.append_edge(
        projection,
        LedgerEdge(
            edge_id=UUID(int=11),
            source=direct.ref,
            target=transitive.ref,
            relation=LedgerRelation.SUPPORTS,
        ),
    )
    successor = make_node(
        node_id=root.node_id,
        revision=2,
        node_type=root.node_type,
        content={"value": "revised"},
        status=EpistemicStatus.SUPPORTED,
        created_at="2026-01-01T00:00:01Z",
        action_id=UUID(int=200),
        module_id="M09",
        predecessor_revision_hash=root.revision_hash,
        confidence=ConfidenceAssessment(score=0.5, method="RULE", explanation="fixture"),
    )
    revised = service.revise_node(projection, successor)
    envelopes = {item.affected_node_ref: item for item in revised.stale_envelopes}
    assert envelopes[direct.ref].maximum_dependency_depth == 1
    assert envelopes[direct.ref].direct_dependency_fraction == 1.0
    assert envelopes[direct.ref].previous_confidence_range == (0.3, 0.3)
    assert envelopes[direct.ref].current_confidence_range == (0.5, 0.5)
    assert envelopes[direct.ref].confidence_delta_interval == pytest.approx((0.2, 0.2))
    assert envelopes[direct.ref].independence_groups == ("source-a",)
    assert envelopes[transitive.ref].maximum_dependency_depth == 2
    assert envelopes[transitive.ref].direct_dependencies_changed == 0
    assert envelopes[transitive.ref].previous_confidence_range == (0.3, 0.3)
    assert next(item for item in revised.nodes if item.ref == transitive.ref).confidence is None


def test_wave2_metrics_are_complete_projection_derived_and_non_mutating() -> None:
    from fre.domain.budget import BudgetReservation, ResourceVector
    from fre.domain.context import CompilerProfile, ContextCompilationRecord
    from fre.projections.metrics import project_metrics
    from fre.runtime.budget_meter import BudgetMeter

    allocator = BudgetAllocator()
    plan, policy_hash = allocator.allocate(
        low_signature(), default_tier_policy(), DeploymentLimits()
    )
    budget = BudgetProjection(plan=plan, policy_hash=policy_hash)
    budget = BudgetMeter().reserve(
        budget,
        BudgetReservation(
            reservation_id="metric-r", action_id="metric-a", resources=ResourceVector(llm_calls=1)
        ),
    )
    remaining_before = BudgetMeter().remaining(budget)
    context = (
        ContextCompiler()
        .compile(
            run_id=UUID(int=1),
            snapshot_version=1,
            ledger=LedgerProjection(),
            budget_remaining=remaining_before,
            profile=CompilerProfile.STANDARD,
        )
        .packet
    )
    metrics = project_metrics(
        LedgerProjection(),
        budget=budget,
        budget_remaining=remaining_before,
        context=context,
        compilation=ContextCompilationRecord(
            packet_hash=context.packet_hash,
            profile=context.profile,
            compiler_version=context.compiler_version,
            compression_policy_version=context.compression_policy_version,
            applied_rule_ids=context.applied_rule_ids,
            renderer_version="1.0",
            canonical_byte_size=1,
            json_artifact_sha256="d" * 64,
        ),
    )
    assert metrics.budget_allocated == plan.limits.resources()
    assert metrics.budget_committed == ResourceVector()
    assert metrics.budget_reserved == ResourceVector(llm_calls=1)
    assert metrics.budget_policy_version == plan.policy_version
    assert metrics.budget_policy_hash == policy_hash
    assert metrics.latest_context_size is not None
    assert metrics.latest_context_artifact_hash == "d" * 64
    assert metrics.latest_compression_policy_version == context.compression_policy_version
    assert BudgetMeter().remaining(budget) == remaining_before


def test_burn_rate_is_deterministic_advisory_and_iteration_based() -> None:
    from fre.domain.budget import ResourceVector
    from fre.runtime.budget_meter import BudgetMeter

    allocator = BudgetAllocator()
    plan, policy_hash = allocator.allocate(
        low_signature(), default_tier_policy(), DeploymentLimits()
    )
    projection = BudgetProjection(plan=plan, policy_hash=policy_hash)
    projection = BudgetMeter().consume(projection, ResourceVector(iterations=1, input_tokens=10))
    before = BudgetMeter().remaining(projection)
    first = BudgetMeter().burn_rate(projection, window_size=3)
    second = BudgetMeter().burn_rate(projection, window_size=3)
    assert first == second
    assert first.source_event_range == (1, 1)
    assert first.resources["input_tokens"].consumption_per_iteration == 10.0
    assert first.resources["input_tokens"].availability.value == "INSUFFICIENT_DATA"
    assert BudgetMeter().remaining(projection) == before


def test_compression_deduplicates_and_retains_successor_linkage() -> None:
    from fre.domain.context import RejectedItem
    from fre.runtime.budget_meter import BudgetMeter

    service = EpistemicLedger()
    original = make_node(
        node_id=UUID(int=30),
        revision=1,
        node_type=LedgerNodeType.FACT,
        content={"claim": "old"},
        status=EpistemicStatus.SUPPORTED,
        created_at="2026-01-01T00:00:00Z",
        action_id=UUID(int=31),
        module_id="M09",
    )
    projection = service.append_node(LedgerProjection(), original)
    successor = make_node(
        node_id=original.node_id,
        revision=2,
        node_type=LedgerNodeType.FACT,
        content={"claim": "current"},
        status=EpistemicStatus.SUPPORTED,
        created_at="2026-01-01T00:00:01Z",
        action_id=UUID(int=32),
        module_id="M09",
        predecessor_revision_hash=original.revision_hash,
    )
    projection = service.revise_node(projection, successor)
    allocator = BudgetAllocator()
    plan, policy_hash = allocator.allocate(
        low_signature(), default_tier_policy(), DeploymentLimits()
    )
    result = ContextCompiler().compile(
        run_id=UUID(int=1),
        snapshot_version=2,
        ledger=projection,
        budget_remaining=BudgetMeter().remaining(
            BudgetProjection(plan=plan, policy_hash=policy_hash)
        ),
        profile=CompilerProfile.STANDARD,
        hard_constraints=({"id": "h1"}, {"id": "h1"}),
        rejected_items=(
            RejectedItem(ref="candidate", reason="invalid"),
            RejectedItem(ref="candidate", reason="invalid"),
        ),
        unresolved_blockers=("permission", "permission"),
    )
    assert len(result.packet.ledger_items) == 1
    assert result.packet.ledger_items[0].predecessor_ref == original.ref
    assert len(result.packet.hard_constraints) == 1
    assert len(result.packet.rejected_items) == 1
    assert result.packet.unresolved_blockers == ("permission",)
    assert result.packet.applied_rule_ids == ("C01", "C02", "C04", "C05")


def test_stop_precedence_requires_stability_and_complete_beats_exhaustion() -> None:
    from fre.domain.budget import BudgetRemaining, ResourceVector
    from fre.domain.stop import (
        AcceptanceStatus,
        StopDisposition,
        StopInputs,
        StopPolicy,
        ValidationStatus,
    )
    from fre.modules.m13_stop import StopController

    exhausted = BudgetRemaining(
        resources=ResourceVector(), active_concurrent_actions=0, projection_hash="e" * 64
    )
    controller = StopController()
    complete = controller.evaluate(
        StopInputs(
            budget=exhausted,
            acceptance=AcceptanceStatus.SATISFIED,
            validation=ValidationStatus.COMPLETE,
            stable=True,
        ),
        StopPolicy(version="1.0"),
    )
    assert complete.disposition is StopDisposition.COMPLETE
    unstable = controller.evaluate(
        StopInputs(
            budget=exhausted,
            acceptance=AcceptanceStatus.SATISFIED,
            validation=ValidationStatus.COMPLETE,
            stable=False,
        ),
        StopPolicy(version="1.0"),
    )
    assert unstable.disposition is StopDisposition.BLOCKED
