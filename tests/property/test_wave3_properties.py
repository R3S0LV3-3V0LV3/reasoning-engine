"""Property proofs for Wave 3 monotonic semantic policies."""

from uuid import UUID

import pytest
from hypothesis import given
from hypothesis import strategies as st

from fre.domain.budget import DeploymentLimits
from fre.domain.common import OutputContract, PermissionSet
from fre.domain.task import Ordinal4, TaskEnvelope
from fre.modules.m01_classifier import TaskClassifier, reversibility_to_irreversibility
from fre.modules.m02_budget import TIER_ORDER, BudgetAllocator, default_tier_policy


@pytest.mark.property
@given(st.sampled_from(tuple(Ordinal4)))
def test_reversibility_transform_is_involution(value: Ordinal4) -> None:
    assert reversibility_to_irreversibility(reversibility_to_irreversibility(value)) is value


@pytest.mark.property
@given(
    st.sampled_from(("consequence", "irreversibility", "ambiguity", "evidence_scarcity")),
    st.integers(min_value=0, max_value=3),
    st.integers(min_value=0, max_value=3),
)
def test_m01_to_m02_axes_are_monotone(axis: str, left: int, right: int) -> None:
    low, high = sorted((left, right))
    task = TaskEnvelope(
        task_id=UUID(int=1),
        text="x",
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(),
        user_metadata={axis: tuple(Ordinal4)[low].value},
    )
    harder = task.model_copy(update={"user_metadata": {axis: tuple(Ordinal4)[high].value}})
    classifier = TaskClassifier()
    first, _ = classifier.classify(task, None)
    second, _ = classifier.classify(harder, None)
    allocator = BudgetAllocator()
    policy = default_tier_policy()
    one = allocator.allocate(first, policy, DeploymentLimits())[0].tier
    two = allocator.allocate(second, policy, DeploymentLimits())[0].tier
    assert TIER_ORDER[two] >= TIER_ORDER[one]
