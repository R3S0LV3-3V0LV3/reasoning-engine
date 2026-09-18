"""Direct regression lock on the event-wire registry (EVENT_PAYLOADS/EVENT_WIRE_IDENTITIES).

Wave 3 (defect F02, stop-decision freshness) introduced an explicit wire-identity
layer so V2 payload classes serialize under their V1-compatible event name at a
bumped schema version (e.g. ``ModelCallRecordedV2`` -> ``ModelCallRecorded@2.0``).
These tests decode each registered (event_type, schema_version) pair end to end
through ``UncommittedEvent.validated_payload`` and lock in the deliberate removal
of ``ModelCallRecordedV2`` from the ``@1.0`` wire-identity set.
"""

import asyncio
from collections.abc import Sequence
from uuid import UUID

import pytest

from fre.domain.budget import DeploymentLimits
from fre.domain.common import JsonValue, OutputContract, PermissionSet
from fre.domain.semantic import (
    SemanticModelCallRecord,
    SemanticModelCallRecordV2,
    StructuredModelRequest,
    StructuredModelResult,
    StructuredModelStatus,
)
from fre.domain.stop import (
    AcceptanceStatus,
    StopDecisionRecord,
    StopInputs,
    StopPolicy,
    ValidationStatus,
)
from fre.domain.task import (
    HorizonClass,
    Ordinal4,
    OutputForm,
    SearchSpaceClass,
    TaskEnvelope,
    TaskSignature,
    TaskType,
)
from fre.engine import FrontierReasoningEngine
from fre.modules.m01_classifier import TaskClassifier
from fre.modules.m02_budget import BudgetAllocator, default_tier_policy
from fre.modules.m13_stop import StopController
from fre.prompts import default_output_schema_registry, default_prompt_registry
from fre.runtime.budget_meter import BudgetMeter
from fre.runtime.events import (
    EVENT_PAYLOADS,
    EVENT_WIRE_IDENTITIES,
    BudgetAllocated,
    ModelCallFailed,
    ModelCallFailedV2,
    ModelCallRecorded,
    ModelCallRecordedV2,
    RunCreated,
    StopDecisionRecorded,
    StopDecisionRecordedV2,
)
from fre.semantic_runtime import SemanticModelRuntime, SemanticRuntimePolicy

VALID: dict[str, JsonValue] = {
    "task_type": {"estimate": "DECISION", "confidence": 0.9, "anchors": [], "rationale": "test"},
    "consequence": {
        "estimate": "LOW",
        "confidence": 0.9,
        "conservative_upper": "MEDIUM",
        "anchors": [],
        "rationale": "test",
    },
    "reversibility": {
        "estimate": "LOW",
        "confidence": 0.9,
        "conservative_upper": "MEDIUM",
        "anchors": [],
        "rationale": "test",
    },
    "ambiguity": {
        "estimate": "LOW",
        "confidence": 0.9,
        "conservative_upper": "MEDIUM",
        "anchors": [],
        "rationale": "test",
    },
    "evidence_scarcity": {
        "estimate": "LOW",
        "confidence": 0.9,
        "conservative_upper": "MEDIUM",
        "anchors": [],
        "rationale": "test",
    },
    "search_space": {"estimate": "BOUNDED", "confidence": 0.9, "anchors": [], "rationale": "test"},
    "horizon": {"estimate": "SHORT", "confidence": 0.9, "anchors": [], "rationale": "test"},
}


class CapturingModel:
    def __init__(self, responses: Sequence[StructuredModelResult]) -> None:
        self.responses = iter(responses)
        self.requests: list[StructuredModelRequest] = []

    async def generate(self, request: StructuredModelRequest) -> StructuredModelResult:
        self.requests.append(request)
        return next(self.responses)


def _result() -> StructuredModelResult:
    return StructuredModelResult(
        status=StructuredModelStatus.SUCCESS,
        adapter_id="test",
        model_id="test",
        raw_response=b"",
        decoded=VALID,
    )


def _signature() -> TaskSignature:
    return TaskSignature(
        task_type=TaskType.ANALYSIS,
        consequence=Ordinal4.MEDIUM,
        irreversibility=Ordinal4.MEDIUM,
        ambiguity=Ordinal4.MEDIUM,
        search_space=SearchSpaceClass.BOUNDED,
        evidence_scarcity=Ordinal4.MEDIUM,
        horizon=HorizonClass.SHORT,
        output_form=OutputForm.STRUCTURED,
        dimension_confidence={},
    )


def _setup(engine: FrontierReasoningEngine) -> tuple[UUID, JsonValue]:
    handle = engine.create_run({"registry": True})
    task = TaskEnvelope(
        task_id=UUID(int=999),
        text="Choose.",
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(),
    )
    signature, _ = TaskClassifier().classify(task, None)
    plan, policy_hash = BudgetAllocator().allocate(
        signature, default_tier_policy(), DeploymentLimits()
    )
    payload = BudgetAllocated(
        plan=plan, policy_version=plan.policy_version, policy_hash=policy_hash
    )
    engine.append(
        handle.run_id, handle.version, (engine.make_event(handle.run_id, payload, module_id="M02"),)
    )
    return handle.run_id, task.model_dump(mode="json")


def _model_call_records(
    engine: FrontierReasoningEngine,
) -> tuple[SemanticModelCallRecordV2, SemanticModelCallRecord]:
    """Return a real, schema-valid V2 record and its downgraded V1 projection."""
    run_id, value = _setup(engine)
    model = CapturingModel([_result()])
    runtime = SemanticModelRuntime(
        model, engine, default_prompt_registry(), default_output_schema_registry()
    )
    execution = asyncio.run(
        runtime.execute(
            run_id=run_id,
            module_id="M01",
            module_version="1.0",
            operation="classify",
            prompt_id="m01.classify",
            prompt_version="1.0",
            canonical_input=value,
            policy=SemanticRuntimePolicy(maximum_repair_attempts=0),
        )
    )
    assert isinstance(execution.record, SemanticModelCallRecordV2)
    v2_record = execution.record
    v1_record = SemanticModelCallRecord.model_validate(
        v2_record.model_dump(
            exclude={"reservation_id", "reported_usage", "charged_usage", "charge_basis"}
        ),
        strict=True,
    )
    return v2_record, v1_record


def _stop_decision_record(engine: FrontierReasoningEngine) -> StopDecisionRecord:
    run = engine.create_run({"registry": "stop"})
    plan, policy_hash = BudgetAllocator().allocate(
        _signature(), default_tier_policy(), DeploymentLimits()
    )
    engine.append(
        run.run_id,
        run.version,
        (
            engine.make_event(
                run.run_id,
                BudgetAllocated(
                    plan=plan, policy_version=plan.policy_version, policy_hash=policy_hash
                ),
                module_id="M02",
            ),
        ),
    )
    state = engine.inspect(run.run_id)
    decision = StopController().evaluate(
        StopInputs(
            budget=BudgetMeter().remaining(state.budget),
            acceptance=AcceptanceStatus.SATISFIED,
            validation=ValidationStatus.COMPLETE,
        ),
        StopPolicy(version="stop/1.0"),
    )
    return StopDecisionRecord(
        decision=decision,
        evaluated_state_version=state.version,
        evaluated_state_hash=state.state_hash,
        budget_projection_hash=decision.budget_projection_hash,
    )


@pytest.mark.unit
def test_model_call_recorded_decodes_at_both_versions(engine: FrontierReasoningEngine) -> None:
    v2_record, v1_record = _model_call_records(engine)

    v1_event = engine.make_event(UUID(int=1), ModelCallRecorded(record=v1_record), module_id="test")
    assert v1_event.schema_version == "1.0"
    decoded_v1 = v1_event.validated_payload()
    assert isinstance(decoded_v1, ModelCallRecorded)
    assert decoded_v1.record == v1_record

    v2_event = engine.make_event(
        UUID(int=1), ModelCallRecordedV2(record=v2_record), module_id="test"
    )
    assert v2_event.event_type == "ModelCallRecorded"
    assert v2_event.schema_version == "2.0"
    decoded_v2 = v2_event.validated_payload()
    assert isinstance(decoded_v2, ModelCallRecordedV2)
    assert decoded_v2.record == v2_record


@pytest.mark.unit
def test_model_call_failed_decodes_at_both_versions(engine: FrontierReasoningEngine) -> None:
    v2_record, v1_record = _model_call_records(engine)

    v1_event = engine.make_event(
        UUID(int=1), ModelCallFailed(record=v1_record, reason="fixture"), module_id="test"
    )
    assert v1_event.schema_version == "1.0"
    decoded_v1 = v1_event.validated_payload()
    assert isinstance(decoded_v1, ModelCallFailed)
    assert decoded_v1.record == v1_record

    v2_event = engine.make_event(
        UUID(int=1), ModelCallFailedV2(record=v2_record, reason="fixture"), module_id="test"
    )
    assert v2_event.event_type == "ModelCallFailed"
    assert v2_event.schema_version == "2.0"
    decoded_v2 = v2_event.validated_payload()
    assert isinstance(decoded_v2, ModelCallFailedV2)
    assert decoded_v2.record == v2_record


@pytest.mark.unit
def test_stop_decision_recorded_decodes_at_both_versions(engine: FrontierReasoningEngine) -> None:
    record = _stop_decision_record(engine)

    v1_event = engine.make_event(
        UUID(int=1), StopDecisionRecorded(decision=record.decision), module_id="test"
    )
    assert v1_event.schema_version == "1.0"
    decoded_v1 = v1_event.validated_payload()
    assert isinstance(decoded_v1, StopDecisionRecorded)
    assert decoded_v1.decision == record.decision

    v2_event = engine.make_event(
        UUID(int=1), StopDecisionRecordedV2(record=record), module_id="test"
    )
    assert v2_event.event_type == "StopDecisionRecorded"
    assert v2_event.schema_version == "2.0"
    decoded_v2 = v2_event.validated_payload()
    assert isinstance(decoded_v2, StopDecisionRecordedV2)
    assert decoded_v2.record == record


@pytest.mark.unit
def test_run_created_decodes_at_v1_and_v2_is_unregistered(
    engine: FrontierReasoningEngine,
) -> None:
    """RunCreated has no V2 wire schema; the RunReducer's own "2.0" version number
    is a reducer/snapshot-compatibility marker, unrelated to event schema
    versioning. Confirm the baseline decodes, and that a hypothetical
    ``RunCreated@2.0`` (which does not exist in EVENT_PAYLOADS) is correctly
    rejected -- doubling as the "unsupported version" regression case."""
    v1_event = engine.make_event(UUID(int=1), RunCreated(config_hash="a" * 64), module_id="test")
    assert v1_event.schema_version == "1.0"
    decoded = v1_event.validated_payload()
    assert isinstance(decoded, RunCreated)
    assert decoded.config_hash == "a" * 64

    unregistered = v1_event.model_copy(update={"schema_version": "99.0"})
    with pytest.raises(ValueError, match="unregistered event type/schema"):
        unregistered.validated_payload()

    hypothetical_v2 = v1_event.model_copy(update={"schema_version": "2.0"})
    with pytest.raises(ValueError, match="unregistered event type/schema"):
        hypothetical_v2.validated_payload()
    assert ("RunCreated", "2.0") not in EVENT_PAYLOADS


@pytest.mark.unit
def test_model_call_recorded_v2_is_not_in_the_v1_wire_identity_set() -> None:
    """Locks in the deliberate Wave 3 removal: ModelCallRecordedV2 must serialize
    under schema_version "2.0", never fall back to being treated as an "@1.0"
    identity. EVENT_WIRE_IDENTITIES is the single source of truth consulted by
    ``event_wire_identity`` for every V2 payload class."""
    v1_only_types = {
        payload_type
        for (_name, version), payload_type in EVENT_PAYLOADS.items()
        if version == "1.0"
    }
    assert ModelCallRecordedV2 not in v1_only_types
    assert ModelCallFailedV2 not in v1_only_types
    assert StopDecisionRecordedV2 not in v1_only_types
    assert ModelCallRecordedV2 in EVENT_WIRE_IDENTITIES
    assert EVENT_WIRE_IDENTITIES[ModelCallRecordedV2] == ("ModelCallRecorded", "2.0")
