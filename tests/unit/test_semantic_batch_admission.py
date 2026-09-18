"""Decisive tests for the F09 semantic batch-admission validator.

`FrontierReasoningEngine.append` proves, for every `ModelCallRecordedV2`/
`ModelCallFailedV2` in a proposed batch, that it is paired -- within that same
batch -- with exactly one `BudgetReservationSettled` (never a
`BudgetReservationReleased`) event for its own `reservation_id`, whose
`actual_usage` matches the record's `charged_usage`. See
`fre.runtime.reducer.validate_semantic_reservation_admission`.

These tests forge events directly through the low-level `engine.append`/
`engine.make_event` API -- bypassing `SemanticModelRuntime` entirely -- to
prove the reducer/engine boundary itself rejects a caller (honest or hostile)
that cannot produce matching budget evidence, rather than trusting the shape
of the record alone.
"""

import asyncio
from collections.abc import Sequence
from uuid import UUID

import pytest

from fre.domain.budget import BudgetReservation, DeploymentLimits, ResourceVector
from fre.domain.common import ArtifactRef, JsonValue, OutputContract, PermissionSet
from fre.domain.semantic import (
    SemanticAccountingCondition,
    SemanticModelCallRecordV2,
    StructuredModelRequest,
    StructuredModelResult,
    StructuredModelStatus,
)
from fre.domain.task import TaskEnvelope
from fre.engine import FrontierReasoningEngine
from fre.modules.m01_classifier import TaskClassifier
from fre.modules.m02_budget import BudgetAllocator, default_tier_policy
from fre.prompts import default_output_schema_registry, default_prompt_registry
from fre.runtime.events import (
    BudgetAllocated,
    BudgetReservationReleased,
    BudgetReservationSettled,
    BudgetReserved,
    ModelCallFailedV2,
    ModelCallRecordedV2,
)
from fre.semantic_runtime import SemanticModelRuntime, SemanticRuntimePolicy

VALID: dict[str, JsonValue] = {
    "task_type": "DECISION",
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
    "search_space": "BOUNDED",
    "search_space_confidence": 0.9,
    "horizon": "SHORT",
    "horizon_confidence": 0.9,
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


def _setup(engine: FrontierReasoningEngine) -> tuple[UUID, JsonValue]:
    handle = engine.create_run({"admission": True})
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


def _genuine_record(
    engine: FrontierReasoningEngine, run_id: UUID, value: JsonValue
) -> SemanticModelCallRecordV2:
    """Execute one real semantic call end to end and return its committed V2 record."""
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
    return execution.record


@pytest.mark.unit
def test_forged_success_without_any_reservation_is_rejected(
    engine: FrontierReasoningEngine,
) -> None:
    run_id, value = _setup(engine)
    genuine = _genuine_record(engine, run_id, value)
    before = engine.inspect(run_id)

    forged = genuine.model_copy(
        update={
            "call_id": engine.uuids.new(),
            "idempotency_key": "f" * 64,
            "reservation_id": "never-reserved",
        }
    )
    with pytest.raises(ValueError, match="reservation settlement evidence is missing"):
        engine.append(
            run_id,
            before.version,
            (engine.make_event(run_id, ModelCallRecordedV2(record=forged), module_id="attacker"),),
        )
    assert engine.inspect(run_id) == before


@pytest.mark.unit
def test_settlement_linked_to_another_calls_reservation_is_rejected(
    engine: FrontierReasoningEngine,
) -> None:
    run_id, value = _setup(engine)
    genuine = _genuine_record(engine, run_id, value)
    before = engine.inspect(run_id)

    other_reservation = BudgetReservation(
        reservation_id="other-call-reservation",
        action_id="other-action",
        resources=genuine.charged_usage,
    )
    forged = genuine.model_copy(
        update={
            "call_id": engine.uuids.new(),
            "idempotency_key": "e" * 64,
            "reservation_id": "hijacked-reservation",
        }
    )
    batch = (
        engine.make_event(
            run_id, BudgetReserved(reservation=other_reservation), module_id="attacker"
        ),
        engine.make_event(
            run_id,
            BudgetReservationSettled(
                reservation_id=other_reservation.reservation_id, actual_usage=genuine.charged_usage
            ),
            module_id="attacker",
        ),
        engine.make_event(run_id, ModelCallRecordedV2(record=forged), module_id="attacker"),
    )
    with pytest.raises(ValueError, match="reservation settlement evidence is missing"):
        engine.append(run_id, before.version, batch)
    assert engine.inspect(run_id) == before


@pytest.mark.unit
def test_model_call_recorded_v2_paired_with_release_instead_of_settlement_is_rejected(
    engine: FrontierReasoningEngine,
) -> None:
    """The reviewer-flagged gap: a ModelCallRecordedV2 whose OWN reservation_id
    was released (not settled) in the same batch must be rejected -- even
    though the reservation_id matches exactly and existed beforehand."""
    run_id, value = _setup(engine)
    genuine = _genuine_record(engine, run_id, value)
    before = engine.inspect(run_id)

    reservation = BudgetReservation(
        reservation_id="r-released-not-settled",
        action_id="a-release",
        resources=genuine.charged_usage,
    )
    forged = genuine.model_copy(
        update={
            "call_id": engine.uuids.new(),
            "idempotency_key": "d" * 64,
            "reservation_id": reservation.reservation_id,
        }
    )
    batch = (
        engine.make_event(run_id, BudgetReserved(reservation=reservation), module_id="attacker"),
        engine.make_event(
            run_id,
            BudgetReservationReleased(reservation_id=reservation.reservation_id),
            module_id="attacker",
        ),
        engine.make_event(run_id, ModelCallRecordedV2(record=forged), module_id="attacker"),
    )
    with pytest.raises(ValueError, match="not a release"):
        engine.append(run_id, before.version, batch)
    assert engine.inspect(run_id) == before


@pytest.mark.unit
def test_charged_usage_mismatched_against_settlement_is_rejected(
    engine: FrontierReasoningEngine,
) -> None:
    run_id, value = _setup(engine)
    genuine = _genuine_record(engine, run_id, value)
    before = engine.inspect(run_id)

    reservation = BudgetReservation(
        reservation_id="r-mismatch", action_id="a-mismatch", resources=ResourceVector(llm_calls=1)
    )
    forged = genuine.model_copy(
        update={
            "call_id": engine.uuids.new(),
            "idempotency_key": "c" * 64,
            "reservation_id": reservation.reservation_id,
        }
    )
    batch = (
        engine.make_event(run_id, BudgetReserved(reservation=reservation), module_id="attacker"),
        engine.make_event(
            run_id,
            BudgetReservationSettled(
                reservation_id=reservation.reservation_id,
                actual_usage=ResourceVector(llm_calls=0),
            ),
            module_id="attacker",
        ),
        engine.make_event(run_id, ModelCallRecordedV2(record=forged), module_id="attacker"),
    )
    with pytest.raises(ValueError, match="charged usage does not match"):
        engine.append(run_id, before.version, batch)
    assert engine.inspect(run_id) == before


@pytest.mark.unit
def test_missing_registered_artifact_is_rejected(engine: FrontierReasoningEngine) -> None:
    run_id, value = _setup(engine)
    genuine = _genuine_record(engine, run_id, value)
    before = engine.inspect(run_id)

    reservation = BudgetReservation(
        reservation_id="r-artifact", action_id="a-artifact", resources=genuine.charged_usage
    )
    unregistered_ref = ArtifactRef(artifact_id=engine.uuids.new(), sha256="a" * 64)
    forged = genuine.model_copy(
        update={
            "call_id": engine.uuids.new(),
            "idempotency_key": "b" * 64,
            "reservation_id": reservation.reservation_id,
            "raw_artifact": unregistered_ref,
        }
    )
    batch = (
        engine.make_event(run_id, BudgetReserved(reservation=reservation), module_id="attacker"),
        engine.make_event(
            run_id,
            BudgetReservationSettled(
                reservation_id=reservation.reservation_id, actual_usage=genuine.charged_usage
            ),
            module_id="attacker",
        ),
        engine.make_event(run_id, ModelCallRecordedV2(record=forged), module_id="attacker"),
    )
    with pytest.raises(ValueError, match="artifact is not registered"):
        engine.append(run_id, before.version, batch)
    assert engine.inspect(run_id) == before


@pytest.mark.unit
def test_failed_record_cannot_present_success_status_without_recognised_override(
    engine: FrontierReasoningEngine,
) -> None:
    run_id, value = _setup(engine)
    genuine = _genuine_record(engine, run_id, value)
    before = engine.inspect(run_id)

    reservation = BudgetReservation(
        reservation_id="r-fake-failure", action_id="a-fake-failure", resources=genuine.charged_usage
    )
    forged = genuine.model_copy(
        update={
            "call_id": engine.uuids.new(),
            "idempotency_key": "9" * 64,
            "reservation_id": reservation.reservation_id,
            "status": StructuredModelStatus.SUCCESS,
            "accounting_condition": None,
        }
    )
    batch = (
        engine.make_event(run_id, BudgetReserved(reservation=reservation), module_id="attacker"),
        engine.make_event(
            run_id,
            BudgetReservationSettled(
                reservation_id=reservation.reservation_id, actual_usage=genuine.charged_usage
            ),
            module_id="attacker",
        ),
        engine.make_event(
            run_id,
            ModelCallFailedV2(record=forged, reason="attacker claims success"),
            module_id="attacker",
        ),
    )
    with pytest.raises(ValueError, match="recognised override"):
        engine.append(run_id, before.version, batch)
    assert engine.inspect(run_id) == before


@pytest.mark.unit
def test_recorded_success_may_not_carry_an_accounting_condition(
    engine: FrontierReasoningEngine,
) -> None:
    """A `SemanticModelCallRecordV2` may self-consistently combine SUCCESS status
    with a RESERVATION_CAP_ON_PROVIDER_OVERAGE charge basis and an accounting
    condition (the record's own validator allows that shape). The business
    rule that no such record may ever be wrapped in a *Recorded* (as opposed
    to *Failed*) event is enforced only at the reducer/batch-admission layer,
    which this test targets directly -- `SemanticModelRuntime` itself never
    constructs this combination (see `test_overreported_usage_...` in
    `test_semantic_runtime.py`, which always emits `ModelCallFailedV2` for it)."""
    from fre.domain.semantic import SemanticCallCharge, SemanticCallUsage, SemanticChargeBasis

    run_id, value = _setup(engine)
    genuine = _genuine_record(engine, run_id, value)
    before = engine.inspect(run_id)

    reservation = BudgetReservation(
        reservation_id="r-fake-condition",
        action_id="a-fake-condition",
        resources=genuine.charged_usage,
    )
    forged = genuine.model_copy(
        update={
            "call_id": engine.uuids.new(),
            "idempotency_key": "8" * 64,
            "reservation_id": reservation.reservation_id,
            "status": StructuredModelStatus.SUCCESS,
            "reported_usage": SemanticCallUsage(
                input_tokens=genuine.charged_usage.input_tokens + 1,
                output_tokens=genuine.charged_usage.output_tokens,
            ),
            "usage": SemanticCallUsage(
                input_tokens=genuine.charged_usage.input_tokens + 1,
                output_tokens=genuine.charged_usage.output_tokens,
            ),
            "accounting_condition": SemanticAccountingCondition.PROVIDER_USAGE_EXCEEDED_RESERVATION,
            "charge_basis": SemanticChargeBasis.RESERVATION_CAP_ON_PROVIDER_OVERAGE,
            "policy_charge": SemanticCallCharge(
                llm_calls=genuine.charged_usage.llm_calls,
                input_tokens=genuine.charged_usage.input_tokens,
                output_tokens=genuine.charged_usage.output_tokens,
                basis=SemanticChargeBasis.RESERVATION_CAP_ON_PROVIDER_OVERAGE.value,
            ),
        }
    )
    batch = (
        engine.make_event(run_id, BudgetReserved(reservation=reservation), module_id="attacker"),
        engine.make_event(
            run_id,
            BudgetReservationSettled(
                reservation_id=reservation.reservation_id, actual_usage=genuine.charged_usage
            ),
            module_id="attacker",
        ),
        engine.make_event(run_id, ModelCallRecordedV2(record=forged), module_id="attacker"),
    )
    with pytest.raises(ValueError, match="may not carry an accounting condition"):
        engine.append(run_id, before.version, batch)
    assert engine.inspect(run_id) == before


@pytest.mark.unit
def test_recover_outstanding_reservations_only_releases_never_touches_model_calls(
    engine: FrontierReasoningEngine,
) -> None:
    """2.2.3: the one legitimate direct-appender path must stay confined to
    `BudgetReservationReleased` -- it must never emit a ModelCall* event."""
    from fre.semantic_runtime import SemanticModelRuntime as Runtime

    run_id, value = _setup(engine)

    class HangingModel:
        async def generate(self, request: StructuredModelRequest) -> StructuredModelResult:
            raise asyncio.CancelledError()

    class LeakyModel:
        """Simulates a crash after the reservation but before `_release` runs,
        by reserving directly and never invoking the runtime's own cleanup."""

    runtime = Runtime(
        HangingModel(), engine, default_prompt_registry(), default_output_schema_registry()
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            runtime.execute(
                run_id=run_id,
                module_id="M01",
                module_version="1.0",
                operation="classify",
                prompt_id="m01.classify",
                prompt_version="1.0",
                canonical_input=value,
            )
        )
    # The runtime's own exception path already released this reservation, so
    # force an orphaned one to exist directly to exercise recovery.
    state = engine.inspect(run_id)
    orphan = BudgetReservation(
        reservation_id="orphan", action_id="orphan-action", resources=ResourceVector(llm_calls=1)
    )
    engine.append(
        run_id, state.version, (engine.make_event(run_id, BudgetReserved(reservation=orphan)),)
    )
    before_recovery = engine.inspect(run_id)
    recovered = runtime.recover_outstanding_reservations(run_id)
    assert recovered == ("orphan",)
    after = engine.inspect(run_id)
    assert after.budget.reservations == ()
    assert after.model_calls == before_recovery.model_calls
    new_event_types = {
        event.event_type
        for event in engine.store.load(run_id)[len(engine.store.load(run_id)) - 1 :]
    }
    assert new_event_types == {"BudgetReservationReleased"}
