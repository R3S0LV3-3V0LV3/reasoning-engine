"""Shared, node-id-generic depth-first traversal primitive.

C06 remediation (EU-17/EU-18): `source_anchors._reject_support_cycles` and
`m09_ledger.EpistemicLedger.validate_graph` each implemented their own
depth-first cycle-detection DFS over a `dict[K, tuple[K, ...]]` adjacency
map, differing only in node-id type (`str` vs. `LedgerNodeRef`) and in what
happens when a cycle is found (immediate raise vs. append-to-errors-list via
a caught exception). `m03_formaliser._topologically_ordered_items` ran a
third, structurally identical DFS whose post-order node emission produces a
topological ordering. All three are expressed here as one generic traversal:
callers select their own on-cycle and (optional) on-finish behaviour via
callbacks, and remain free to raise, accumulate, or ignore as they always
did -- this module changes no externally observable behaviour anywhere it is
used.
"""

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import TypeVar

K = TypeVar("K", bound=Hashable)


def depth_first_traverse(
    edges: Mapping[K, Sequence[K]],
    *,
    order: Iterable[K],
    on_cycle: Callable[[K], None],
    on_finish: Callable[[K], None] | None = None,
) -> None:
    """Visit every node reachable from `order` via `edges`, depth-first.

    `on_cycle(node_id)` is invoked (and traversal into that node's subtree is
    skipped) whenever `node_id` is revisited while still on the current
    recursion stack -- callers that want to raise immediately (stopping the
    whole traversal) simply raise from within the callback; callers that
    want to accumulate and continue may swallow it. `on_finish(node_id)`, if
    given, is invoked in post-order (after all of `node_id`'s dependencies
    have been visited) for every node that is itself a key of `edges` --
    nodes referenced only as targets (dangling/unknown ids) are skipped
    without being marked finished, exactly as the original standalone
    implementations did.
    """
    visiting: set[K] = set()
    visited: set[K] = set()

    def visit(node_id: K) -> None:
        if node_id in visiting:
            on_cycle(node_id)
            return
        if node_id in visited or node_id not in edges:
            return
        visiting.add(node_id)
        for target_id in edges[node_id]:
            visit(target_id)
        visiting.discard(node_id)
        visited.add(node_id)
        if on_finish is not None:
            on_finish(node_id)

    for node_id in order:
        visit(node_id)
