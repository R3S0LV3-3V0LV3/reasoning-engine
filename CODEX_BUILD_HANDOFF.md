# Codex Build Handoff — Frontier Reasoning Engine v1

## Authority

Treat these files as the binding specification, in order:

1. `FRE_ARCHITECTURE_V1.md`
2. `FRE_MODULE_MANIFEST.yaml`
3. this handoff

Do not silently replace the specified epistemic, constraint, persistence, or stopping semantics with a simpler agent loop. Raise a documented ADR proposal before changing a binding contract.

## Build objective

Implement a local-first, provider-neutral Python package named `fre` that realises the thirteen-module Frontier Reasoning Engine as an evented control plane. The first working milestone is not an LLM demo. It is a deterministic, replayable foundation on which model-assisted modules can be added without changing state semantics.

## Non-negotiable rules

1. Modules never mutate shared run state directly. They return typed domain events.
2. The event log is authoritative; projections and snapshots are rebuildable.
3. `UNKNOWN` and `ERROR` are not `FAIL`.
4. A non-deterministically verifiable hard constraint remains hard.
5. Candidate and claim revisions are immutable.
6. A model response is untrusted until schema and module semantic validation pass.
7. Domain and module packages cannot import concrete provider/storage adapters.
8. No hidden scalar score may replace Pareto-first decision logic.
9. Hard budget ceilings must stop any run, including one whose modules continually propose more work.
10. Every prune, challenge, decision, and stop must have persisted provenance.

## Recommended implementation baseline

- Python package with `src/` layout.
- Pydantic models for persistent contracts.
- SQLite in WAL mode for events, snapshots, and projections.
- Content-addressed local artifact store using SHA-256.
- `asyncio` module/adapter interfaces; deterministic reducers remain synchronous and pure.
- `pytest` for tests, property-based tests where specified, static typing, and one canonical lint/format command.
- No external model/provider dependency is required before Wave 3; use fake adapters.

Do not overbuild a web service, GUI, distributed queue, or production authentication layer during the core waves.

## Execution method

Build one wave at a time. At the end of every wave:

1. run formatting, linting, typing, unit tests, property tests, and any wave integration tests;
2. update the relevant ADRs and README;
3. provide a changed-file list;
4. report the exact test commands and results;
5. report unresolved decisions and deviations;
6. stop before starting the next wave unless explicitly instructed to continue.

## First implementation task — Wave 0 and Wave 1 only

### Wave 0

Create the repository structure from Section 18 of the architecture. Add:

- `pyproject.toml`;
- `src/fre/` package and version;
- `tests/` hierarchy;
- config loading and schema-version helpers;
- formatting, linting, type-checking, and test commands;
- a minimal README linking the architecture;
- ADR directory and ADR-001 to ADR-003 stubs with accepted decisions.

### Wave 1

Implement:

1. common identifiers, timestamps, schema-version fields, object/artifact references;
2. all core Pydantic contracts from Architecture Section 6, using separate modules as specified;
3. domain-event envelope and a reducer protocol;
4. `RunCreated`, `RunStatusChanged`, `ArtifactRegistered`, and generic test events;
5. SQLite event store with optimistic expected-version append;
6. snapshot store with reducer version and state hash;
7. local content-addressed artifact store;
8. fake clock, fake structured-model port, and fake tool port;
9. run creation, event append, state reduction, snapshot, reload, and replay verification;
10. unit/integration tests proving replay equivalence and concurrent-version rejection.

### Wave 1 architectural constraints

- JSON payloads must be canonicalised before hashing.
- Stored events must include `run_id`, monotonically increasing sequence, `event_type`, `action_id`, `module_id`, schema version, module version, input hash, and timestamp.
- The event store must use transactions and reject an incorrect expected version.
- Snapshot verification must rebuild from events and compare the rebuilt hash.
- Artifacts must deduplicate by SHA-256.
- Raw artifacts and parsed domain objects remain distinct.
- Do not implement any real model API in Wave 1.
- Do not implement the full orchestrator in Wave 1; provide only enough engine/service code to exercise persistence and replay.

### Wave 1 acceptance gate

The wave passes only when a test can:

1. create a run;
2. append multiple events;
3. reduce them into state;
4. create a snapshot;
5. close and reopen storage;
6. reconstruct the state from events;
7. verify that the reconstructed hash equals the stored snapshot hash;
8. prove that a stale concurrent append is rejected;
9. prove that storing the same artifact twice creates one content object.

## Build sequence after Wave 1

Follow the wave order in the architecture and manifest:

```text
W2  M09 ledger + M02 budget + M12 context + M13 stop
W3  M01 classifier + M03 formaliser + M04 representation
W4  M05 generator + M06 constraints + M10 decision
W5  M07 synthesis + M08 falsifier
W6  M11 acquisition + capability routing
W7  full orchestrator + CLI + T5 branch reconciliation
W8  thin MCP/service adapter
```

Do not move M09 late merely because it is numbered ninth; it is part of the deterministic spine.

## Required test fixtures to establish early

Create reusable fixtures for:

- a minimal task envelope;
- a task signature at each tier boundary;
- a problem with one deterministic and one semantic hard constraint;
- candidates with complete, missing, and interval objective values;
- a ledger with support, contradiction, and revision relations;
- a fake model output registry keyed by idempotency key;
- a fake tool registry that can succeed, fail, or be unavailable;
- a deterministic clock and UUID factory for golden tests.

## Code review checklist

Before presenting a wave as complete, verify:

- Are domain objects immutable where required?
- Can every persistent object be serialised and versioned?
- Can state be rebuilt without invoking a model or tool again?
- Are external outputs stored and referenced rather than treated as ephemeral?
- Can one module spend budget without the meter recording it? It must not.
- Can an `UNKNOWN` constraint accidentally invalidate a candidate? It must not.
- Can an adapter leak into domain imports? It must not.
- Is every terminal run able to produce a context packet once M12 exists?
- Can an endless proposal loop exceed hard limits? It must not.

## Required response after the first build pass

Return:

```text
IMPLEMENTED
- files and capabilities

TESTS
- exact commands
- pass/fail counts

INVARIANTS VERIFIED
- replay
- optimistic concurrency
- artifact deduplication
- schema validation

DEVIATIONS
- none, or explicit items with rationale

OPEN DECISIONS
- only decisions that materially block Wave 2

NEXT WAVE
- precise W2 work plan, not implementation unless requested
```
