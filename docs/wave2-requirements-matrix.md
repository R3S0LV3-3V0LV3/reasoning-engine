# Wave 2 requirements matrix

This matrix traces the enhanced Wave 2 plan to implementation and decisive tests.  IDs map to
the numbered acceptance requirements in Section 26 of
`FRE_WAVE_2_BUILD_AUTHORITATIVE_PLAN_ENHANCED.txt`; ranges are expanded by the named tests.

| Requirement IDs | Owner | Public contract / event | Decisive tests | Status |
|---|---|---|---|---|
| W2-001–030, 109–118 | M09 | `LedgerProjection`; typed ledger events | `tests/unit/test_wave2.py`, `tests/property/test_wave2_properties.py` | PASS |
| W2-021–035, 119–122 | M02 | `TierPolicy`, `BudgetPlan`; `BudgetAllocated`, `BudgetRevised` | `tests/unit/test_wave2.py`, monotonicity property | PASS |
| W2-036–050, 123–126 | Budget Meter | `BudgetProjection`; consume/reserve/settle/release events | `tests/unit/test_wave2.py`, budget properties | PASS |
| W2-051–068, 127–135 | M12 | `ContextPacket`, compression/delta contracts; `ContextCompiled` | `tests/unit/test_wave2.py`, context properties | PASS |
| W2-069–089, 136–143 | M13 | `StopDecision`; `StopDecisionRecorded`, terminal request/association | `tests/unit/test_wave2.py`, termination property | PASS |
| W2-090–108, 144–149 | Runtime/replay | Wave 2 reducer state and `Wave2Runtime` | `tests/integration/test_wave2_gate.py`, differential replay | PASS |
| EVT-001–014 | Runtime | Explicit registry entries for every Wave 2 event | event/reducer and integration tests | PASS |
| ERR-001–017 | All | Typed ledger, budget, context, and stop errors | module unit tests | PASS |
| DET-001–010 | All | Canonical ordering/hashes; no clock/random/external calls in core | property and differential replay tests | PASS |
| REG-001–013 | Runtime | Wave 1 event/snapshot compatibility | existing Wave 1 suite plus compatibility test | PASS |
| CFG-001–010 | M02/M12/M13 | Versioned tier, compiler, compression, and stop policy | configuration and module tests | PASS |
| SCOPE-001–018 | Wave boundary | No M01/M03–M08/M10/M11, providers, scheduler, CLI, service | dependency test and diff review | PASS |

All projections are event-derived and rebuildable. No database projection table was added. The
event stream remains authoritative. “PASS” means the corresponding executable test is present and
the final quality gate must pass; this document is not a substitute for that gate.
