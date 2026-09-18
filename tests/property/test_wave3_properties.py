"""Property proofs for Wave 3 monotonic semantic policies."""

from uuid import UUID

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel

from fre.domain.budget import DeploymentLimits
from fre.domain.common import JsonValue, OutputContract, PermissionSet, canonical_hash
from fre.domain.task import Ordinal4, TaskEnvelope
from fre.modules.m01_classifier import TaskClassifier, reversibility_to_irreversibility
from fre.modules.m02_budget import TIER_ORDER, BudgetAllocator, default_tier_policy
from fre.prompts.schemas import (
    MAX_SCHEMA_NESTING_DEPTH,
    ClassificationOutput,
    ProblemFormalisationOutput,
    RepresentationAdjudicationOutput,
    _schema_nesting_depth,
    canonical_schema_bytes,
    canonical_schema_hash,
)


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


@pytest.mark.property
@given(
    st.sampled_from(
        (ClassificationOutput, ProblemFormalisationOutput, RepresentationAdjudicationOutput)
    )
)
def test_canonical_schema_hash_is_stable_and_matches_canonical_hash(
    model: type[BaseModel],
) -> None:
    """F13: `canonical_schema_hash`/`canonical_schema_bytes` must be a
    deterministic, repeatable function of a model's JSON-Schema, identical to
    the pre-existing `canonical_hash(model.model_json_schema())` used
    throughout the registry -- computing it twice, or via either helper, must
    always agree."""
    first = canonical_schema_hash(model)
    second = canonical_schema_hash(model)
    assert first == second
    assert first == canonical_hash(model.model_json_schema())
    assert canonical_schema_bytes(model) == canonical_schema_bytes(model)


@pytest.mark.property
@given(st.integers(min_value=0, max_value=64))
def test_schema_nesting_depth_measures_a_finite_chain_exactly_up_to_the_cap(depth: int) -> None:
    """Fuzz the depth-measuring primitive itself (not just the fixed, hand-
    picked registered schemas) across a range of nesting depths, including
    depths well past the configured cap, to prove it always terminates with
    an exact count below the cap and a bounded, monotone "exceeded" signal
    above it -- it never raises, hangs, or silently under-counts."""
    value: JsonValue = "leaf"
    for _ in range(depth):
        value = {"nested": value}
    result = _schema_nesting_depth(value)
    if depth <= MAX_SCHEMA_NESTING_DEPTH:
        assert result == depth
    else:
        assert result > MAX_SCHEMA_NESTING_DEPTH
