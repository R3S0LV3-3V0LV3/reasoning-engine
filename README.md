# Frontier Reasoning Engine — Architecture Bundle v1

This bundle converts the supplied conceptual and module-level drafts into a build-authoritative architecture for Codex.

## Files

- `FRE_ARCHITECTURE_V1.md` — complete system, runtime, domain, module, persistence, testing, and build architecture.
- `FRE_MODULE_MANIFEST.yaml` — machine-readable module contracts, dependencies, tier defaults, and build waves.
- `CODEX_BUILD_HANDOFF.md` — ready-to-use implementation handoff beginning with Waves 0–1.

## Recommended use

Place all three files at the root of the new repository or under `docs/`, then give Codex `CODEX_BUILD_HANDOFF.md` as the execution instruction while keeping the architecture and manifest in context.

## Implemented waves

- Waves 0–1: strict canonical contracts, typed events, SQLite WAL persistence, snapshots,
  content-addressed artifacts, deterministic adapters, and replay.
- Wave 2: event-derived hash-linked epistemic ledger, monotone allocation and executable budget
  meter, deterministic context packets/deltas, confidence-bounded stopping, and a minimal terminal
  lifecycle coordinator.

- Wave 3: versioned prompt/output-schema governance, budgeted provider-neutral structured model
  calls, source-anchored M01/M03 semantics, deterministic M04 structural projections, replay-safe
  typed events, M12 compiler 2.0 semantic packets, and an authoritative `Wave3Engine` coordinator
  (`fre.composition`) that sequences M01 -> M02 -> M03 -> M04 -> M12 as one resumable, replay-safe
  front end.

Wave 4 and later search, feasibility, and decision modules remain intentionally unimplemented.

## Wave 3 acceptance evidence

`docs/wave3-requirements-matrix.md` traces every mandatory Wave 3 acceptance row to a real
implementation symbol, a real positive test, a real negative/adversarial test where one applies,
and replay/property/golden evidence where applicable — every cell is machine-checked against the
live test suite by `scripts/verify_requirements_matrix.py` (run in CI as part of the `unit` job),
which fails if any row cites a test that does not exist or does not collect. `tests/golden/`
holds twelve checked golden fixtures (A–L) that each drive a decisive Wave 3 scenario through the
real `Wave3Engine.execute_front_end` coordinator — never a hand-wired module call — and assert
its persisted output (event sequence, artifact hashes, projections, blockers/availability, budget
totals, final state hash) against checked-in evidence files under `tests/fixtures/golden/`.

This matrix and its golden fixtures report `PASS`-with-evidence for every mandatory row; they do
not claim the codebase has zero known issues. `WAVE3_DEFERRED_CLEANUP_REGISTER.md` at the
repository root tracks 28+ Tier 4–7 findings (code-quality/duplication observations and a small
number of documented, non-blocking residual limitations) raised by each phase's independent
review and explicitly deferred, by standing policy, to a single cleanup pass after C10 rather than
blocking any individual phase's merge. None of those findings are Tier 1–3, and none invalidate
any row in the requirements matrix.

## Frozen deterministic foundation

Waves 0–2 are the accepted deterministic substrate for subsequent development. Wave 3+ changes
must preserve canonical serialization, typed events, replay and snapshot compatibility, artifact
identity, epistemic revision semantics, budget accounting, context packet and stop semantics, and
dependency boundaries unless a deliberate versioned migration is approved.

Terminal context artifacts are content-addressed before their event batch commits. A failed batch
may therefore leave unreferenced physical objects, but those objects have no event authority, no
canonical reference, and cannot make a run terminal.

The original architecture-bundle README is preserved at `docs/README_BUNDLE_V1.md`.
