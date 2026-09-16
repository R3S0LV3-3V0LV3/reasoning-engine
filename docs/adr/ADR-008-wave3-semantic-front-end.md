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
