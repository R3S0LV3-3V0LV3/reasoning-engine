"""Decisive tests for the C04 independent-review remediation round.

Each test below targets one numbered finding from the review that produced
these fixes (see the PR body / WAVE3_DEFERRED_CLEANUP_REGISTER.md for the
full list). Findings that are documentation-only (#6, #10, #11) are not
re-tested here beyond what already exists; #4 is subsumed by #1's content
check (see the comment in `RunReducer.apply`) and is exercised by the same
test.
"""

import asyncio
import contextlib
import time
from collections.abc import Sequence
from uuid import UUID

import pytest
from pydantic import BaseModel, Field

from fre.domain.budget import BudgetReservation, DeploymentLimits, ResourceVector
from fre.domain.common import ArtifactRef, JsonValue, OutputContract, PermissionSet, canonical_json
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
from fre.prompts.schemas import OutputSchemaError, enforce_schema_limits
from fre.runtime.events import (
    BudgetAllocated,
    BudgetReservationSettled,
    BudgetReserved,
    ModelCallFailedV2,
    ModelCallRecordedV2,
)
from fre.runtime.reducer import RunReducer, validate_semantic_reservation_admission
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


def _result() -> StructuredModelResult:
    return StructuredModelResult(
        status=StructuredModelStatus.SUCCESS,
        adapter_id="test",
        model_id="test",
        raw_response=b"",
        decoded=VALID,
    )


class CapturingModel:
    def __init__(self, responses: Sequence[StructuredModelResult]) -> None:
        self.responses = iter(responses)

    async def generate(self, request: StructuredModelRequest) -> StructuredModelResult:
        return next(self.responses)


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


# ---------------------------------------------------------------------------
# Finding #1 (+ #4): content-provenance / schema-hash re-derivation.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_forged_success_with_schema_invalid_artifact_bytes_is_rejected(
    engine: FrontierReasoningEngine,
) -> None:
    """A forged SUCCESS record whose `proposal_artifact` bytes do not actually
    validate against the schema it claims (by id/version/hash) must be
    rejected -- even though every structural/accounting check it presents is
    internally self-consistent. This is the decisive test for finding #1: it
    fails against the pre-remediation head (cfdde8877) because the old code
    only checked that the artifact's sha256 was *registered*, never that its
    *content* actually decoded against the claimed schema.
    """
    run_id, value = _setup(engine)
    genuine = _genuine_record(engine, run_id, value)
    before = engine.inspect(run_id)

    # Register an artifact whose bytes do not conform to ANY registered output
    # schema (missing every required ClassificationOutput field).
    bogus_bytes = canonical_json({"not": "a-valid-classification-output"})
    bogus_descriptor = engine.store_artifact(bogus_bytes, media_type="application/json")
    bogus_ref = ArtifactRef(artifact_id=bogus_descriptor.id, sha256=bogus_descriptor.sha256)

    reservation = BudgetReservation(
        reservation_id="r-forged-content",
        action_id="a-forged-content",
        resources=genuine.charged_usage,
    )
    forged = genuine.model_copy(
        update={
            "call_id": engine.uuids.new(),
            "idempotency_key": "1" * 64,
            "reservation_id": reservation.reservation_id,
            "proposal_artifact": bogus_ref,
            # output_schema_id/version/hash all still point at the legitimate,
            # registered ClassificationOutput schema -- only the artifact
            # bytes are forged.
        }
    )
    from fre.runtime.events import ArtifactRegistered

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
            ArtifactRegistered(
                artifact=bogus_ref, media_type="application/json", byte_size=len(bogus_bytes)
            ),
            module_id="attacker",
        ),
        engine.make_event(run_id, ModelCallRecordedV2(record=forged), module_id="attacker"),
    )
    with pytest.raises(ValueError, match="does not validate against its claimed output schema"):
        engine.append(run_id, before.version, batch)
    assert engine.inspect(run_id) == before


@pytest.mark.unit
def test_forged_success_with_mismatched_schema_hash_is_rejected(
    engine: FrontierReasoningEngine,
) -> None:
    """Finding #4: a record claiming a schema hash that does not match the
    registry's current hash for that schema id/version is rejected -- proven
    structurally at admission time, not merely at the one `_invoke` call
    site."""
    run_id, value = _setup(engine)
    genuine = _genuine_record(engine, run_id, value)
    before = engine.inspect(run_id)

    reservation = BudgetReservation(
        reservation_id="r-bad-hash", action_id="a-bad-hash", resources=genuine.charged_usage
    )
    forged = genuine.model_copy(
        update={
            "call_id": engine.uuids.new(),
            "idempotency_key": "2" * 64,
            "reservation_id": reservation.reservation_id,
            "output_schema_hash": "0" * 64,
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
    with pytest.raises(ValueError, match="schema hash does not match"):
        engine.append(run_id, before.version, batch)
    assert engine.inspect(run_id) == before


# ---------------------------------------------------------------------------
# Finding #2: the reducer-embedded duplicate of the F09 admission check.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reducer_apply_rejects_forged_record_without_going_through_engine_append() -> None:
    """Bypass `FrontierReasoningEngine.append` (and therefore
    `validate_semantic_reservation_admission`) entirely: call `RunReducer.apply`
    directly with a self-consistent but forged `ModelCallFailedV2` whose
    reservation was never settled anywhere. The reducer itself must reject it.
    """
    from datetime import UTC, datetime

    from fre.domain.common import FrozenModel, canonical_hash
    from fre.domain.semantic import (
        SemanticCallCharge,
        SemanticCallUsage,
        SemanticChargeBasis,
    )
    from fre.runtime.events import RunCreated, StoredEvent, event_wire_identity

    run_id = UUID(int=1)
    reducer = RunReducer()
    state = reducer.initial(run_id)

    def _stored(payload: FrozenModel, sequence: int) -> StoredEvent:
        event_type, schema_version = event_wire_identity(payload)
        return StoredEvent(
            event_id=UUID(int=sequence + 1000),
            run_id=run_id,
            event_type=event_type,
            action_id=UUID(int=sequence + 2000),
            module_id="attacker",
            schema_version=schema_version,
            module_version="1.0",
            input_hash=canonical_hash(None),
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            sequence=sequence,
            payload=payload.model_dump(mode="json"),
        )

    state = reducer.apply(state, _stored(RunCreated(config_hash="c" * 64), 1))

    # Use `ModelCallFailedV2` (any status, no artifact requirement) so the
    # forgery isolates the reservation-settlement check specifically, rather
    # than tripping an earlier, unrelated "artifacts required" check that only
    # applies to *Recorded* success events.
    forged_failed = SemanticModelCallRecordV2(
        call_id=UUID(int=3000),
        idempotency_key="f" * 64,
        module_id="M01",
        operation="classify",
        module_version="1.0",
        policy_version="policy/1.0",
        policy_hash="a" * 64,
        prompt_id="m01.classify",
        prompt_version="1.0",
        template_hash="b" * 64,
        output_schema_id="m01.classification-output",
        output_schema_version="1.0",
        output_schema_hash="c" * 64,
        canonical_input_hash="d" * 64,
        model_role="classifier",
        adapter_id="test",
        model_id="test",
        status=StructuredModelStatus.PERMANENT_FAILURE,
        raw_artifact=None,
        proposal_artifact=None,
        policy_charge=SemanticCallCharge(
            llm_calls=1,
            input_tokens=0,
            output_tokens=0,
            basis=SemanticChargeBasis.CONSERVATIVE_RESERVED_CAPACITY.value,
        ),
        reservation_id="never-settled",
        reported_usage=SemanticCallUsage(),
        charged_usage=ResourceVector(llm_calls=1),
        charge_basis=SemanticChargeBasis.CONSERVATIVE_RESERVED_CAPACITY,
    )
    with pytest.raises(ValueError, match="reservation settlement evidence is missing"):
        reducer.apply(
            state,
            _stored(ModelCallFailedV2(record=forged_failed, reason="forged"), 2),
        )


# ---------------------------------------------------------------------------
# Finding #3: schema nesting-depth memoization against shared-$defs blow-up.
# ---------------------------------------------------------------------------


def _build_shared_submodel_schema(depth: int = 20, branch: int = 5) -> type[BaseModel]:
    """A chain of `depth` levels where EVERY level has `branch` sibling fields
    that are all the exact same child submodel type (a "diamond" DAG, not
    just top-level fan-out). Pydantic flattens the repeated type into a single
    `$defs` entry referenced by many `$ref`s; without memoization keyed on the
    `$ref` target, `_schema_nesting_depth` independently re-expands that
    shared subtree from every one of the `branch` sibling references at every
    level, which is exponential (`branch**depth`) in the number of levels --
    the exact resource-exhaustion shape finding #3 closes. Empirically (see
    the C04 remediation notes), `depth=20, branch=5` takes several hundred
    milliseconds pre-fix and is effectively instant post-fix.
    """
    current: type[BaseModel] = type("Leaf", (BaseModel,), {"__annotations__": {"value": int}})
    for level in range(depth):
        annotations = {chr(97 + i): current for i in range(branch)}
        current = type(f"Level{level}", (BaseModel,), {"__annotations__": annotations})
    return current


@pytest.mark.unit
def test_schema_nesting_depth_memoizes_shared_defs_and_stays_fast() -> None:
    model = _build_shared_submodel_schema()
    start = time.monotonic()
    # The chain intentionally exceeds MAX_SCHEMA_NESTING_DEPTH; rejecting it is
    # the correct outcome. What this test actually proves is the *speed* of
    # reaching that verdict, not the verdict itself.
    with contextlib.suppress(OutputSchemaError):
        enforce_schema_limits(model)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, f"enforce_schema_limits took {elapsed:.3f}s -- exponential blow-up?"


# ---------------------------------------------------------------------------
# Finding #5: order-independent two-pass batch admission.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_admission_accepts_settlement_after_its_record_in_batch_tuple(
    engine: FrontierReasoningEngine,
) -> None:
    run_id, value = _setup(engine)
    genuine = _genuine_record(engine, run_id, value)
    before = engine.inspect(run_id)

    reservation = BudgetReservation(
        reservation_id="r-reordered", action_id="a-reordered", resources=genuine.charged_usage
    )
    reused_record = genuine.model_copy(
        update={
            "call_id": engine.uuids.new(),
            "idempotency_key": "3" * 64,
            "reservation_id": reservation.reservation_id,
        }
    )
    from fre.runtime.events import StoredEvent

    reserved_event = engine.make_event(run_id, BudgetReserved(reservation=reservation))
    record_event = engine.make_event(run_id, ModelCallRecordedV2(record=reused_record))
    settled_event = engine.make_event(
        run_id,
        BudgetReservationSettled(
            reservation_id=reservation.reservation_id, actual_usage=genuine.charged_usage
        ),
    )
    stored = tuple(
        StoredEvent.model_validate(
            {**event.model_dump(), "created_at": event.created_at, "sequence": before.version + i},
            strict=True,
        )
        for i, event in enumerate((reserved_event, record_event, settled_event), 1)
    )
    # Must NOT raise: the settlement is present in the batch, merely ordered
    # after its record.
    validate_semantic_reservation_admission(stored)


# ---------------------------------------------------------------------------
# Finding #7: optimistic-concurrency fast path never bypasses admission.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stale_expected_version_fast_path_never_persists_unvalidated_batch(
    engine: FrontierReasoningEngine,
) -> None:
    run_id, value = _setup(engine)
    genuine = _genuine_record(engine, run_id, value)
    before = engine.inspect(run_id)

    forged = genuine.model_copy(
        update={
            "call_id": engine.uuids.new(),
            "idempotency_key": "4" * 64,
            "reservation_id": "never-settled-race",
        }
    )
    stale_version = before.version - 1  # deliberately wrong: triggers the fast path
    from fre.adapters.storage_sqlite import ConcurrentAppendError

    with pytest.raises(ConcurrentAppendError):
        engine.append(
            run_id,
            stale_version,
            (engine.make_event(run_id, ModelCallRecordedV2(record=forged), module_id="attacker"),),
        )
    # Nothing was persisted: state is byte-for-byte unchanged.
    assert engine.inspect(run_id) == before


# ---------------------------------------------------------------------------
# Finding #8: ModelCallFailedV2 accounting-condition/status pairing.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_provider_overage_condition_rejected_for_incompatible_status(
    engine: FrontierReasoningEngine,
) -> None:
    from fre.domain.semantic import SemanticCallCharge, SemanticCallUsage, SemanticChargeBasis

    run_id, value = _setup(engine)
    genuine = _genuine_record(engine, run_id, value)
    before = engine.inspect(run_id)

    reservation = BudgetReservation(
        reservation_id="r-bad-pairing", action_id="a-bad-pairing", resources=genuine.charged_usage
    )
    forged = genuine.model_copy(
        update={
            "call_id": engine.uuids.new(),
            "idempotency_key": "5" * 64,
            "reservation_id": reservation.reservation_id,
            "status": StructuredModelStatus.UNAVAILABLE,
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
        engine.make_event(
            run_id,
            ModelCallFailedV2(record=forged, reason="attacker-forged pairing"),
            module_id="attacker",
        ),
    )
    with pytest.raises(ValueError, match="only valid when the semantic model call"):
        engine.append(run_id, before.version, batch)
    assert engine.inspect(run_id) == before


# ---------------------------------------------------------------------------
# Finding #9: regex `pattern` denylist (ReDoS defence).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_schema_with_arbitrary_regex_pattern_field_is_rejected() -> None:
    class Malicious(BaseModel):
        model_config = {"extra": "forbid"}
        value: str = Field(pattern=r"^(a+)+$")  # classic catastrophic-backtracking shape

    with pytest.raises(OutputSchemaError, match="pattern"):
        enforce_schema_limits(Malicious)


@pytest.mark.unit
def test_existing_registered_schemas_keep_their_allowlisted_hex_hash_pattern() -> None:
    # Confirms the fix did not collaterally break the two schemas that
    # legitimately carry the fixed, reviewed hex-digest pattern.
    registry = default_output_schema_registry()
    registry.get("m01.classification-output", "1.0")
    registry.get("m03.problem-formalisation-output", "1.0")
