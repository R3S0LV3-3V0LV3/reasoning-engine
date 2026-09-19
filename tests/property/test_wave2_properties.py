from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

import pytest
from hypothesis import given
from hypothesis import strategies as st

from fre.adapters.artifacts_local import LocalArtifactStore
from fre.adapters.storage_sqlite import SQLiteStore
from fre.adapters.testing import FakeClock, FakeUUIDFactory
from fre.domain.budget import (
    BudgetExceeded,
    BudgetProjection,
    BudgetReservation,
    DeploymentLimits,
    ResourceVector,
)
from fre.domain.context import CompilerProfile
from fre.domain.ledger import EpistemicStatus, LedgerNodeType, LedgerProjection
from fre.domain.stop import AcceptanceStatus, StopInputs, StopPolicy, ValidationStatus
from fre.domain.task import (
    HorizonClass,
    Ordinal4,
    OutputForm,
    SearchSpaceClass,
    TaskSignature,
    TaskType,
)
from fre.engine import FrontierReasoningEngine
from fre.modules.m02_budget import TIER_ORDER, BudgetAllocator, default_tier_policy
from fre.modules.m09_ledger import EpistemicLedger, make_node
from fre.modules.m12_context import ContextCompiler
from fre.modules.m13_stop import StopController
from fre.runtime.budget_meter import BudgetMeter
from fre.runtime.events import BudgetAllocated
from fre.runtime.events import TestValueSet as ValueSet
from fre.runtime.wave2 import Wave2Runtime

ORDINALS = tuple(Ordinal4)
SEARCHES = tuple(SearchSpaceClass)


def signature(index: int, search_index: int) -> TaskSignature:
    level = ORDINALS[index]
    return TaskSignature(
        task_type=TaskType.ANALYSIS,
        consequence=level,
        irreversibility=level,
        ambiguity=level,
        search_space=SEARCHES[search_index],
        evidence_scarcity=level,
        horizon=HorizonClass.IMMEDIATE,
        output_form=OutputForm.TEXT,
        dimension_confidence={},
    )


@given(st.integers(0, 3), st.integers(0, 3), st.integers(0, 2), st.integers(0, 2))
def test_allocation_is_monotone(a: int, b: int, sa: int, sb: int) -> None:
    low, high = min(a, b), max(a, b)
    slow, shigh = min(sa, sb), max(sa, sb)
    allocator = BudgetAllocator()
    policy = default_tier_policy()
    first, _ = allocator.allocate(signature(low, slow), policy, DeploymentLimits())
    second, _ = allocator.allocate(signature(high, shigh), policy, DeploymentLimits())
    assert TIER_ORDER[second.tier] >= TIER_ORDER[first.tier]


@given(st.integers(min_value=0, max_value=1))
def test_budget_consumption_never_exceeds_hard_ceiling(amount: int) -> None:
    allocator = BudgetAllocator()
    plan, digest = allocator.allocate(signature(0, 0), default_tier_policy(), DeploymentLimits())
    projection = BudgetProjection(plan=plan, policy_hash=digest)
    meter = BudgetMeter()
    consumed = meter.consume(projection, ResourceVector(iterations=amount))
    assert meter.remaining(consumed).resources.iterations == 1 - amount
    with pytest.raises(BudgetExceeded):
        meter.consume(consumed, ResourceVector(iterations=2))


@given(
    st.integers(min_value=1, max_value=4096),
    st.integers(min_value=1, max_value=2048),
)
def test_settled_invocation_reservation_never_becomes_available_again(
    input_tokens: int, output_tokens: int
) -> None:
    allocator = BudgetAllocator()
    plan, digest = allocator.allocate(signature(0, 0), default_tier_policy(), DeploymentLimits())
    meter = BudgetMeter()
    projection = BudgetProjection(plan=plan, policy_hash=digest)
    reservation = BudgetReservation(
        reservation_id="invoked",
        action_id="semantic-call",
        resources=ResourceVector(
            llm_calls=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )

    reserved = meter.reserve(projection, reservation)
    settled = meter.settle(reserved, reservation.reservation_id, reservation.resources)
    remaining = meter.remaining(settled).resources

    assert settled.reservations == ()
    assert settled.committed.llm_calls == 1
    assert settled.committed.input_tokens == input_tokens
    assert settled.committed.output_tokens == output_tokens
    assert remaining.llm_calls == plan.limits.max_llm_calls - 1
    assert remaining.input_tokens == plan.limits.max_input_tokens - input_tokens
    assert remaining.output_tokens == plan.limits.max_output_tokens - output_tokens


@given(st.integers(min_value=1, max_value=5), st.booleans())
def test_terminal_finalization_iff_equal_decision_is_rebound_after_unrelated_events(
    unrelated_event_count: int, recompute: bool
) -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        engine = FrontierReasoningEngine(
            SQLiteStore(root / "events.sqlite3"),
            LocalArtifactStore(root / "artifacts"),
            FakeClock(
                datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=index) for index in range(100)
            ),
            FakeUUIDFactory(UUID(int=index) for index in range(1, 500)),
        )
        run = engine.create_run({"property": "stop-freshness"})
        plan, policy_hash = BudgetAllocator().allocate(
            signature(0, 0), default_tier_policy(), DeploymentLimits()
        )
        engine.append(
            run.run_id,
            run.version,
            (
                engine.make_event(
                    run.run_id,
                    BudgetAllocated(
                        plan=plan,
                        policy_version=plan.policy_version,
                        policy_hash=policy_hash,
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
        runtime = Wave2Runtime(engine)
        runtime.record_decision(run.run_id, decision)
        for index in range(unrelated_event_count):
            current = engine.inspect(run.run_id)
            engine.append(
                run.run_id,
                current.version,
                (
                    engine.make_event(
                        run.run_id,
                        ValueSet(key=f"unrelated-{index}", value=index),
                    ),
                ),
            )

        if recompute:
            runtime.record_decision(run.run_id, decision)
            assert runtime.finalize(run.run_id, decision)
        else:
            with pytest.raises(ValueError, match="stale"):
                runtime.finalize(run.run_id, decision)


@given(st.integers(min_value=1, max_value=20))
def test_context_compilation_is_idempotent(value: int) -> None:
    allocator = BudgetAllocator()
    plan, digest = allocator.allocate(signature(0, 0), default_tier_policy(), DeploymentLimits())
    remaining = BudgetMeter().remaining(BudgetProjection(plan=plan, policy_hash=digest))
    node = make_node(
        node_id=UUID(int=value),
        revision=1,
        node_type=LedgerNodeType.FACT,
        content={"value": value},
        status=EpistemicStatus.SUPPORTED,
        created_at="2026-01-01T00:00:00Z",
        action_id=UUID(int=100 + value),
        module_id="M09",
    )
    ledger = EpistemicLedger().append_node(LedgerProjection(), node)
    compiler = ContextCompiler()
    first = compiler.compile(
        run_id=UUID(int=999),
        snapshot_version=2,
        ledger=ledger,
        budget_remaining=remaining,
        profile=CompilerProfile.STANDARD,
    )
    second = compiler.compile(
        run_id=UUID(int=999),
        snapshot_version=2,
        ledger=ledger,
        budget_remaining=remaining,
        profile=CompilerProfile.STANDARD,
    )
    assert first.canonical_bytes == second.canonical_bytes
    assert first.markdown == second.markdown


@given(st.permutations(("alpha", "beta", "gamma")))
def test_semantically_unordered_context_inputs_have_identical_bytes(
    blocker_order: list[str],
) -> None:
    allocator = BudgetAllocator()
    plan, digest = allocator.allocate(signature(0, 0), default_tier_policy(), DeploymentLimits())
    remaining = BudgetMeter().remaining(BudgetProjection(plan=plan, policy_hash=digest))
    compiler = ContextCompiler()
    baseline = compiler.compile(
        run_id=UUID(int=999),
        snapshot_version=2,
        ledger=LedgerProjection(),
        budget_remaining=remaining,
        profile=CompilerProfile.HANDOFF,
        unresolved_blockers=("alpha", "beta", "gamma"),
    )
    permuted = compiler.compile(
        run_id=UUID(int=999),
        snapshot_version=2,
        ledger=LedgerProjection(),
        budget_remaining=remaining,
        profile=CompilerProfile.HANDOFF,
        unresolved_blockers=tuple(blocker_order),
    )
    assert baseline.canonical_bytes == permuted.canonical_bytes
    assert baseline.packet.packet_hash == permuted.packet.packet_hash


def test_semantically_authoritative_sequences_remain_distinguishable() -> None:
    from fre.domain.common import canonical_hash
    from fre.domain.context import ContextCompressionPolicy

    first = ContextCompressionPolicy(ordered_rule_ids=("C01", "C02", "C05"))
    second = ContextCompressionPolicy(ordered_rule_ids=("C02", "C01", "C05"))
    assert canonical_hash(first) != canonical_hash(second)
