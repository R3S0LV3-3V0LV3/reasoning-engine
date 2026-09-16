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

Wave 3 and later semantic/model-assisted modules remain intentionally unimplemented.
