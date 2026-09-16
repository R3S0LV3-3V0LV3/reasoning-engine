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
  typed events, and M12 compiler 2.0 semantic packets.

Wave 4 and later search, feasibility, and decision modules remain intentionally unimplemented.

## Frozen deterministic foundation

Waves 0–2 are the accepted deterministic substrate for subsequent development. Wave 3+ changes
must preserve canonical serialization, typed events, replay and snapshot compatibility, artifact
identity, epistemic revision semantics, budget accounting, context packet and stop semantics, and
dependency boundaries unless a deliberate versioned migration is approved.

Terminal context artifacts are content-addressed before their event batch commits. A failed batch
may therefore leave unreferenced physical objects, but those objects have no event authority, no
canonical reference, and cannot make a run terminal.

The original architecture-bundle README is preserved at `docs/README_BUNDLE_V1.md`.
