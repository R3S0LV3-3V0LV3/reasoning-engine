# ADR-007: Wave 2 deterministic control spine

**Status:** Accepted

Wave 2 keeps the event stream authoritative and folds typed events into immutable in-memory
projections. No projection tables are introduced.

Ledger dependency traversal normalises `SUPPORTS` as source-to-target and `DEPENDS_ON` /
`DERIVED_FROM` as target-to-source (prerequisite-to-dependent). Only that relation family is
acyclic. Contradictions remain historical edges and close only through an atomic resolution record,
resolution edge, and status overlays. Revisions form per-node hash chains; whole-ledger canonical
hashes provide cross-node integrity.

Budget allocation is monotone and binds the resolved policy version/hash in events. The meter
subtracts committed use and active reservations from hard ceilings. Burn rate is advisory and
event-count based.

Context packets hash their canonical representation excluding `packet_hash`; JSON remains
authoritative and Markdown is deterministic presentation. Compression has fixed versioned rule
order and overflow is explicit. Deltas are content-addressed transfer optimisations only.

M13 applies fixed hard-terminal precedence before confidence-bounded soft value/cost comparison.
It requests terminal context; the minimal `Wave2Runtime` coordinates compilation, artifact storage,
association, and terminal status without introducing the later scheduler/orchestrator.
