from uuid import UUID

import pytest
from hypothesis import given
from hypothesis import strategies as st

from fre.domain.budget import BudgetExceeded, BudgetProjection, DeploymentLimits, ResourceVector
from fre.domain.context import CompilerProfile
from fre.domain.ledger import EpistemicStatus, LedgerNodeType, LedgerProjection
from fre.domain.task import (
    HorizonClass,
    Ordinal4,
    OutputForm,
    SearchSpaceClass,
    TaskSignature,
    TaskType,
)
from fre.modules.m02_budget import TIER_ORDER, BudgetAllocator, default_tier_policy
from fre.modules.m09_ledger import EpistemicLedger, make_node
from fre.modules.m12_context import ContextCompiler
from fre.runtime.budget_meter import BudgetMeter

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
