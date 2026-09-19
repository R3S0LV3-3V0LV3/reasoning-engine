# ADR-008: Wave 3 semantic front end

**Status:** Accepted

Wave 3 introduces provider-neutral, budgeted structured-model calls as operational input only.
Raw responses are content-addressed independently; strict versioned output schemas create
untrusted proposals; M01/M03/M04 deterministic validators alone create canonical event payloads.
One schema-repair call is the maximum and consumes a separate reservation. Replay folds typed
events and never invokes a model.

M01 applies harder-is-higher axes, mandatory permission floors, conservative low-confidence
handling, and then delegates allocation to unchanged M02. M03 requires `SourceAnchor` for every
`EXPLICIT_INPUT`, keeps origin labels distinct from M09 status, preserves `HARD + UNKNOWN`, and
uses concrete ledger provenance for blockers. M04 builds immutable structural projections over
`ProblemSpec`; compatibility scores never become epistemic authority and no Wave 4 algorithm is
included.

The reducer remains version 1.0. Wave 3 state is omitted from historic snapshot hash payloads
when absent. Context packets incorporating semantic state use compiler version 2.0, while
historic packet behavior remains unchanged.

## Coordinator (C09) and acceptance evidence (C10)

`fre.composition.Wave3Engine` composes and sequences the M01 -> M02 -> M03 -> M04 -> M12 front
end above as `execute_front_end`: a single, RECOMMENDED, replay-safe, resumable entry point built
from the same module-level `canonical_events`/`select_bound`/`build_bound` surfaces described
above, with no additional reducer-level enforcement of its own (every ordering invariant is still
independently enforced by the reducer, not by this coordinator's calling convention). It is
RECOMMENDED, not structurally exclusive: the underlying modules remain independently callable
(every pre-coordinator test in this repository did exactly that), and `fre.composition`'s own
module docstring documents precisely which of the coordinator's own conveniences (resumability/
idempotency discipline) — as opposed to reducer-enforced invariants — are lost by bypassing it.

Every mandatory Wave 3 acceptance criterion traces to a real implementation symbol and a real,
machine-verified test in `docs/wave3-requirements-matrix.md` (checked by
`scripts/verify_requirements_matrix.py`), and twelve checked golden fixtures under `tests/golden/`
(A–L) exercise decisive coordinator scenarios end to end. `WAVE3_DEFERRED_CLEANUP_REGISTER.md`
tracks non-blocking Tier 4–7 findings from each phase's independent review, explicitly deferred
rather than resolved; their existence does not change any status in this ADR.
