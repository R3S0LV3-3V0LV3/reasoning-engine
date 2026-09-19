"""Property proofs for Wave 3 monotonic semantic policies."""

from uuid import UUID

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel

from fre.domain.budget import DeploymentLimits
from fre.domain.common import JsonValue, ObjectRef, OutputContract, PermissionSet, canonical_hash
from fre.domain.semantic import (
    EpistemicOriginLabel,
    SourceAnchor,
    SourceKind,
    SupportArtifactRef,
    SupportLedgerNodeRef,
    SupportProblemItemRef,
)
from fre.domain.task import Ordinal4, TaskEnvelope
from fre.modules.m01_classifier import TaskClassifier, reversibility_to_irreversibility
from fre.modules.m02_budget import TIER_ORDER, BudgetAllocator, default_tier_policy
from fre.modules.m03_formaliser import ProblemFormaliser
from fre.modules.source_anchors import SelfSupportReference, validate_support_graph
from fre.prompts.schemas import (
    MAX_SCHEMA_NESTING_DEPTH,
    ClassificationOutput,
    ProblemFormalisationOutput,
    ProblemItemProposal,
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


# ---------------------------------------------------------------------------
# C06 (F03/F06): reference-graph resolution, acyclicity, duplicate identity,
# and lossless serialization of the typed `SupportRef` union.
# ---------------------------------------------------------------------------


def _envelope() -> TaskEnvelope:
    return TaskEnvelope(
        task_id=UUID(int=1),
        text="fixture",
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(),
    )


def _has_cycle(edges: dict[str, tuple[str, ...]]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited or node not in edges:
            return False
        visiting.add(node)
        found = any(visit(target) for target in edges[node])
        visiting.discard(node)
        visited.add(node)
        return found

    return any(visit(node) for node in edges)


@pytest.mark.property
@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=4), st.integers(min_value=0, max_value=4)
        ).filter(lambda pair: pair[0] != pair[1]),
        min_size=0,
        max_size=8,
        unique=True,
    )
)
def test_support_graph_cycle_detection_matches_true_graph_cyclicity(
    edge_pairs: list[tuple[int, int]],
) -> None:
    """`validate_support_graph`'s cycle rejection must agree exactly with
    whether the same-proposal `ProblemItemRef` support graph is actually
    cyclic -- for arbitrary small graphs, not just the hand-picked cases."""
    node_ids = {f"n{index}" for pair in edge_pairs for index in pair} | {"n0"}
    edges: dict[str, tuple[str, ...]] = {
        node: tuple(f"n{target}" for source, target in edge_pairs if f"n{source}" == node)
        for node in node_ids
    }
    items = tuple(
        ProblemItemProposal(
            id=node_id,
            kind="UNKNOWN",
            description=node_id,
            origin=EpistemicOriginLabel.UNRESOLVED,
            support=tuple(SupportProblemItemRef(item_id=target) for target in edges[node_id]),
        )
        for node_id in sorted(node_ids)
    )
    cyclic = _has_cycle(edges)
    if cyclic:
        with pytest.raises(SelfSupportReference):
            validate_support_graph(items, envelope=_envelope())
    else:
        validate_support_graph(items, envelope=_envelope())


@pytest.mark.property
@given(
    st.one_of(
        st.builds(
            SourceAnchor,
            source_kind=st.just(SourceKind.TASK_FIELD),
            source_ref=st.just(ObjectRef(object_type="TaskEnvelope", object_id=str(UUID(int=1)))),
            selector=st.just("/explicit_constraints"),
        ),
        st.builds(SupportProblemItemRef, item_id=st.text(min_size=1, max_size=8)),
        st.builds(
            SupportArtifactRef,
            artifact_id=st.uuids(),
            sha256=st.just("a" * 64),
        ),
        st.builds(
            SupportLedgerNodeRef, node_id=st.uuids(), revision=st.integers(min_value=1, max_value=5)
        ),
    )
)
def test_support_ref_union_round_trips_losslessly(
    ref: SourceAnchor | SupportProblemItemRef | SupportArtifactRef | SupportLedgerNodeRef,
) -> None:
    """Every `SupportRef` variant must survive a JSON round-trip through the
    proposal schema unchanged -- no field silently dropped or coerced."""
    proposal = ProblemFormalisationOutput(
        items=(
            ProblemItemProposal(
                id="x",
                kind="UNKNOWN",
                description="x",
                origin=EpistemicOriginLabel.UNRESOLVED,
                support=(ref,),
            ),
        )
    )
    decoded = ProblemFormalisationOutput.model_validate_json(proposal.model_dump_json())
    assert decoded.items[0].support == (ref,)


@pytest.mark.property
@given(st.lists(st.sampled_from(("a", "b", "c")), min_size=2, max_size=6))
def test_duplicate_item_ids_are_always_rejected(ids: list[str]) -> None:
    """Regardless of how many items share an id (2 or more, from a small
    alphabet to force collisions), `formalise` must reject the proposal
    whenever any id repeats, and accept it when all ids happen to be
    distinct."""
    items = tuple(
        ProblemItemProposal(
            id=item_id,
            kind="UNKNOWN",
            description=f"item {index}",
            origin=EpistemicOriginLabel.UNRESOLVED,
        )
        for index, item_id in enumerate(ids)
    )
    proposal = ProblemFormalisationOutput(items=items)
    has_duplicates = len(set(ids)) != len(ids)
    if has_duplicates:
        with pytest.raises(Exception, match="unique"):
            ProblemFormaliser().formalise(_envelope(), proposal)
    else:
        ProblemFormaliser().formalise(_envelope(), proposal)
