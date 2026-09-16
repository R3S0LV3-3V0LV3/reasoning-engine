"""M09 deterministic epistemic ledger projection and command validation."""

from collections import deque
from datetime import UTC, datetime
from uuid import UUID

from fre.domain.common import canonical_hash
from fre.domain.ledger import (
    ContradictionResolution,
    ContradictionState,
    DanglingLedgerReference,
    DependencyConfidenceEnvelope,
    EpistemicStatus,
    GraphValidationResult,
    InvalidContradictionResolution,
    InvalidLedgerRelation,
    InvalidRevisionError,
    InvalidStatusTransition,
    LedgerCycleError,
    LedgerEdge,
    LedgerNode,
    LedgerNodeRef,
    LedgerNodeType,
    LedgerProjection,
    LedgerRelation,
    RevisionHashMismatch,
)

DEPENDENCY_RELATIONS = {
    LedgerRelation.SUPPORTS,
    LedgerRelation.DEPENDS_ON,
    LedgerRelation.DERIVED_FROM,
}


def revision_preimage(node: LedgerNode) -> dict[str, object]:
    return node.model_dump(mode="json", exclude={"revision_hash"})


def revision_hash(node: LedgerNode) -> str:
    return canonical_hash(revision_preimage(node))


def make_node(
    *,
    node_id: UUID,
    revision: int,
    node_type: LedgerNodeType,
    content: object,
    status: EpistemicStatus,
    created_at: str | datetime,
    action_id: UUID,
    module_id: str,
    predecessor_revision_hash: str | None = None,
    confidence: object = None,
) -> LedgerNode:
    normalized_created_at = (
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if isinstance(created_at, str)
        else created_at
    )
    if normalized_created_at.tzinfo is None or normalized_created_at.utcoffset() is None:
        raise ValueError("ledger timestamps must be timezone-aware")
    data = {
        "node_id": node_id,
        "revision": revision,
        "node_type": node_type,
        "content": content,
        "epistemic_status": status,
        "confidence": confidence,
        "created_at": normalized_created_at.astimezone(UTC),
        "created_by_action": action_id,
        "producing_module": module_id,
        "predecessor_revision_hash": predecessor_revision_hash,
        "revision_hash": "0" * 64,
    }
    provisional = LedgerNode.model_validate(data)
    return LedgerNode.model_validate({**data, "revision_hash": revision_hash(provisional)})


def _ref_key(ref: LedgerNodeRef) -> tuple[str, int]:
    return str(ref.node_id), ref.revision


def _edge_key(edge: LedgerEdge) -> tuple[str, str, int, str, int, str]:
    return (
        edge.relation,
        str(edge.source.node_id),
        edge.source.revision,
        str(edge.target.node_id),
        edge.target.revision,
        str(edge.edge_id),
    )


def _normalised(edge: LedgerEdge) -> tuple[LedgerNodeRef, LedgerNodeRef]:
    if edge.relation in {LedgerRelation.DEPENDS_ON, LedgerRelation.DERIVED_FROM}:
        return edge.target, edge.source
    return edge.source, edge.target


class EpistemicLedger:
    """Pure functional operations over immutable projections."""

    @staticmethod
    def projection_hash(projection: LedgerProjection) -> str:
        return canonical_hash(projection)

    @staticmethod
    def _node_map(projection: LedgerProjection) -> dict[tuple[str, int], LedgerNode]:
        return {_ref_key(node.ref): node for node in projection.nodes}

    def append_node(self, projection: LedgerProjection, node: LedgerNode) -> LedgerProjection:
        if revision_hash(node) != node.revision_hash:
            raise RevisionHashMismatch("node revision hash does not match its canonical preimage")
        nodes = self._node_map(projection)
        key = _ref_key(node.ref)
        if key in nodes:
            raise InvalidRevisionError("ledger revision already exists")
        if node.revision != 1:
            predecessor = nodes.get((str(node.node_id), node.revision - 1))
            if predecessor is None or node.predecessor_revision_hash != predecessor.revision_hash:
                raise RevisionHashMismatch("revision does not bind its immediate predecessor")
        return projection.model_copy(
            update={
                "nodes": tuple(sorted((*projection.nodes, node), key=lambda n: _ref_key(n.ref)))
            }
        )

    def append_edge(self, projection: LedgerProjection, edge: LedgerEdge) -> LedgerProjection:
        nodes = self._node_map(projection)
        if _ref_key(edge.source) not in nodes or _ref_key(edge.target) not in nodes:
            raise DanglingLedgerReference("edge references an unknown ledger revision")
        if edge.relation is LedgerRelation.RESOLVES:
            raise InvalidLedgerRelation(
                "RESOLVES is accepted only by atomic contradiction resolution"
            )
        if edge.relation is LedgerRelation.SUPERSEDES:
            source = nodes[_ref_key(edge.source)]
            target = nodes[_ref_key(edge.target)]
            if (
                source.node_id != target.node_id
                or source.revision != target.revision + 1
                or source.predecessor_revision_hash != target.revision_hash
            ):
                raise InvalidRevisionError(
                    "SUPERSEDES must bind consecutive revisions of one logical node"
                )
        if edge.relation in DEPENDENCY_RELATIONS and edge.source == edge.target:
            raise LedgerCycleError("direct dependency cycle")
        candidate = projection.model_copy(
            update={"edges": tuple(sorted((*projection.edges, edge), key=_edge_key))}
        )
        self.validate_graph(candidate, raise_on_error=True)
        return candidate

    def _adjacency(
        self, projection: LedgerProjection
    ) -> dict[LedgerNodeRef, tuple[LedgerNodeRef, ...]]:
        values: dict[LedgerNodeRef, set[LedgerNodeRef]] = {}
        for edge in projection.edges:
            if edge.relation in DEPENDENCY_RELATIONS:
                prerequisite, dependent = _normalised(edge)
                values.setdefault(prerequisite, set()).add(dependent)
        return {key: tuple(sorted(items, key=_ref_key)) for key, items in values.items()}

    def validate_graph(
        self, projection: LedgerProjection, *, raise_on_error: bool = False
    ) -> GraphValidationResult:
        nodes = self._node_map(projection)
        errors: list[str] = []
        for edge in projection.edges:
            if _ref_key(edge.source) not in nodes or _ref_key(edge.target) not in nodes:
                errors.append("dangling reference")
        adjacency = self._adjacency(projection)
        visiting: set[LedgerNodeRef] = set()
        visited: set[LedgerNodeRef] = set()

        def visit(ref: LedgerNodeRef) -> None:
            if ref in visiting:
                raise LedgerCycleError("dependency-family cycle")
            if ref in visited:
                return
            visiting.add(ref)
            for target in adjacency.get(ref, ()):
                visit(target)
            visiting.remove(ref)
            visited.add(ref)

        try:
            for ref in sorted(adjacency, key=_ref_key):
                visit(ref)
        except LedgerCycleError:
            errors.append("dependency-family cycle")
        if errors and raise_on_error:
            if "dependency-family cycle" in errors:
                raise LedgerCycleError(errors[0])
            raise DanglingLedgerReference(errors[0])
        return GraphValidationResult(valid=not errors, errors=tuple(errors))

    def descendants(
        self, projection: LedgerProjection, ref: LedgerNodeRef
    ) -> tuple[LedgerNodeRef, ...]:
        adjacency = self._adjacency(projection)
        found: set[LedgerNodeRef] = set()
        queue = deque(adjacency.get(ref, ()))
        while queue:
            item = queue.popleft()
            if item not in found:
                found.add(item)
                queue.extend(adjacency.get(item, ()))
        return tuple(sorted(found, key=_ref_key))

    def ancestors(
        self, projection: LedgerProjection, ref: LedgerNodeRef
    ) -> tuple[LedgerNodeRef, ...]:
        reverse: dict[LedgerNodeRef, set[LedgerNodeRef]] = {}
        for source, targets in self._adjacency(projection).items():
            for target in targets:
                reverse.setdefault(target, set()).add(source)
        found: set[LedgerNodeRef] = set()
        queue = deque(sorted(reverse.get(ref, ()), key=_ref_key))
        while queue:
            item = queue.popleft()
            if item not in found:
                found.add(item)
                queue.extend(sorted(reverse.get(item, ()), key=_ref_key))
        return tuple(sorted(found, key=_ref_key))

    def effective_status(self, projection: LedgerProjection, ref: LedgerNodeRef) -> EpistemicStatus:
        overlays = dict(projection.status_overlays)
        node = self._node_map(projection).get(_ref_key(ref))
        if node is None:
            raise DanglingLedgerReference("unknown ledger revision")
        return overlays.get(ref, node.epistemic_status)

    def mark_status(
        self, projection: LedgerProjection, ref: LedgerNodeRef, status: EpistemicStatus
    ) -> LedgerProjection:
        if _ref_key(ref) not in self._node_map(projection):
            raise DanglingLedgerReference("unknown ledger revision")
        current = self.effective_status(projection, ref)
        allowed = {
            EpistemicStatus.SUPPORTED: {
                EpistemicStatus.SUPPORTED,
                EpistemicStatus.CONTESTED,
                EpistemicStatus.REFUTED,
                EpistemicStatus.STALE,
            },
            EpistemicStatus.PROVISIONAL: set(EpistemicStatus),
            EpistemicStatus.CONTESTED: {
                EpistemicStatus.CONTESTED,
                EpistemicStatus.SUPPORTED,
                EpistemicStatus.REFUTED,
                EpistemicStatus.STALE,
            },
            EpistemicStatus.REFUTED: {
                EpistemicStatus.REFUTED,
                EpistemicStatus.STALE,
            },
            EpistemicStatus.UNRESOLVED: set(EpistemicStatus),
            EpistemicStatus.STALE: set(EpistemicStatus),
        }
        if status not in allowed[current]:
            raise InvalidStatusTransition(f"invalid status transition: {current} -> {status}")
        overlays = dict(projection.status_overlays)
        overlays[ref] = status
        ordered = tuple(sorted(overlays.items(), key=lambda item: _ref_key(item[0])))
        return projection.model_copy(update={"status_overlays": ordered})

    def revise_node(self, projection: LedgerProjection, successor: LedgerNode) -> LedgerProjection:
        if successor.revision <= 1:
            raise InvalidRevisionError("successor revision must exceed one")
        predecessor_ref = LedgerNodeRef(node_id=successor.node_id, revision=successor.revision - 1)
        predecessor = self._node_map(projection).get(_ref_key(predecessor_ref))
        if predecessor is None:
            raise InvalidRevisionError("immediate predecessor is missing")
        result = self.append_node(projection, successor)
        supersedes = LedgerEdge(
            edge_id=UUID(successor.revision_hash[:32]),
            source=successor.ref,
            target=predecessor.ref,
            relation=LedgerRelation.SUPERSEDES,
        )
        result = result.model_copy(
            update={"edges": tuple(sorted((*result.edges, supersedes), key=_edge_key))}
        )
        adjacency = self._adjacency(projection)
        distances: dict[LedgerNodeRef, int] = {}
        queue: deque[tuple[LedgerNodeRef, int]] = deque(
            (item, 1) for item in adjacency.get(predecessor.ref, ())
        )
        while queue:
            descendant, distance = queue.popleft()
            previous_distance = distances.get(descendant)
            if previous_distance is not None and previous_distance <= distance:
                continue
            distances[descendant] = distance
            queue.extend((item, distance + 1) for item in adjacency.get(descendant, ()))
        descendants = tuple(sorted(distances, key=_ref_key))
        envelopes: list[DependencyConfidenceEnvelope] = []
        for descendant in descendants:
            result = self.mark_status(result, descendant, EpistemicStatus.STALE)
            direct_prerequisites = {
                prerequisite
                for prerequisite, dependents in adjacency.items()
                if descendant in dependents
            }
            directly_changed = int(predecessor.ref in direct_prerequisites)
            previous_score = predecessor.confidence.score if predecessor.confidence else None
            current_score = successor.confidence.score if successor.confidence else None
            confidence_delta = (
                (current_score - previous_score, current_score - previous_score)
                if previous_score is not None and current_score is not None
                else None
            )
            independence_groups = tuple(
                sorted(
                    {
                        edge.independence_group
                        for edge in projection.edges
                        if edge.relation in DEPENDENCY_RELATIONS
                        and edge.independence_group is not None
                        and _normalised(edge) == (predecessor.ref, descendant)
                    }
                )
            )
            envelopes.append(
                DependencyConfidenceEnvelope(
                    affected_node_ref=descendant,
                    changed_dependency_refs=(predecessor.ref, successor.ref),
                    maximum_dependency_depth=distances[descendant],
                    previous_confidence_range=(previous_score, previous_score)
                    if previous_score is not None
                    else None,
                    current_confidence_range=(current_score, current_score)
                    if current_score is not None
                    else None,
                    confidence_delta_interval=confidence_delta,
                    direct_dependencies_changed=directly_changed,
                    direct_dependency_fraction=(
                        directly_changed / len(direct_prerequisites)
                        if direct_prerequisites
                        else 0.0
                    ),
                    independence_groups=independence_groups,
                    source_event_refs=(str(successor.created_by_action),),
                )
            )
        return result.model_copy(update={"stale_envelopes": (*result.stale_envelopes, *envelopes)})

    def contradictions(
        self, projection: LedgerProjection, state: ContradictionState | None = None
    ) -> tuple[LedgerEdge, ...]:
        edges = [edge for edge in projection.edges if edge.relation is LedgerRelation.CONTRADICTS]
        if state is ContradictionState.OPEN:
            edges = [
                edge
                for edge in edges
                if edge.edge_id not in projection.resolved_contradiction_edges
            ]
        elif state is ContradictionState.RESOLVED:
            edges = [
                edge for edge in edges if edge.edge_id in projection.resolved_contradiction_edges
            ]
        return tuple(sorted(edges, key=_edge_key))

    def resolve_contradiction(
        self, projection: LedgerProjection, resolution: ContradictionResolution, edge: LedgerEdge
    ) -> LedgerProjection:
        known = {item.edge_id for item in self.contradictions(projection, ContradictionState.OPEN)}
        if (
            not resolution.contradiction_edge_ids
            or not set(resolution.contradiction_edge_ids) <= known
        ):
            raise InvalidContradictionResolution(
                "resolution must identify open contradiction edges"
            )
        if not resolution.status_transitions or edge.relation is not LedgerRelation.RESOLVES:
            raise InvalidContradictionResolution(
                "resolution relation and status transitions are atomic"
            )
        if edge.source != resolution.resolver_ref:
            raise InvalidContradictionResolution(
                "resolver relation does not match resolution record"
            )
        result = projection.model_copy(
            update={"edges": tuple(sorted((*projection.edges, edge), key=_edge_key))}
        )
        for ref, status in resolution.status_transitions:
            result = self.mark_status(result, ref, status)
        return result.model_copy(
            update={
                "resolved_contradiction_edges": tuple(
                    sorted(
                        set(
                            (
                                *result.resolved_contradiction_edges,
                                *resolution.contradiction_edge_ids,
                            )
                        ),
                        key=str,
                    )
                )
            }
        )

    def unresolved_dependencies(
        self, projection: LedgerProjection, ref: LedgerNodeRef
    ) -> tuple[LedgerNodeRef, ...]:
        return tuple(
            item
            for item in self.ancestors(projection, ref)
            if self.effective_status(projection, item) is not EpistemicStatus.SUPPORTED
        )

    def query_by_type(
        self, projection: LedgerProjection, node_type: LedgerNodeType
    ) -> tuple[LedgerNode, ...]:
        return tuple(node for node in projection.nodes if node.node_type is node_type)

    def verify_revision_chain(
        self, projection: LedgerProjection, node_id: UUID, from_revision: int = 1
    ) -> bool:
        nodes = sorted(
            (node for node in projection.nodes if node.node_id == node_id),
            key=lambda node: node.revision,
        )
        previous: LedgerNode | None = None
        for node in nodes:
            if node.revision < from_revision:
                previous = node
                continue
            if revision_hash(node) != node.revision_hash:
                raise RevisionHashMismatch("revision preimage hash mismatch")
            if node.revision > 1 and (
                previous is None or node.predecessor_revision_hash != previous.revision_hash
            ):
                raise RevisionHashMismatch("predecessor hash mismatch")
            previous = node
        return True
