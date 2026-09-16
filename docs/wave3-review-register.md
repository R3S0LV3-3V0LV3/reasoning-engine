# Wave 3 review register

Only findings with decisive regression coverage are closed here.

| Finding | Resolution | Evidence | State |
|---|---|---|---|
| Runtime integrity defects 1–6 | Snapshot compatibility, revision-envelope markers, representation invalidation, model-call artifact provenance, contradiction-resolution reference validation, and stale-stop rejection were hardened without changing the reducer version. | `tests/fixtures/wave2_history.json`, `tests/integration/test_wave2_gate.py`, `tests/integration/test_wave3_gate.py`, and `tests/unit/test_wave2.py` | Fixed and regression-tested |
| Semantic runtime defects 7–11 and 30 | The provider-neutral port is asynchronous; repair executes its own identity-bound render; validated proposals are stored separately; reservations are committed before calls and deterministically released; repair refusal preserves the initial record; and operation ownership is enforced before spending. | `tests/unit/test_semantic_runtime.py` and `tests/integration/test_wave3_gate.py` | Fixed and regression-tested |
| Source-anchor and M01 defects 12–14 | Source anchors now resolve complete RFC 6901 pointers with strict identity, span, and hash validation; M01 validates and retains proposed evidence while giving explicit deterministic inputs precedence; evidence references are canonically ordered; and common output contracts normalize conservatively. | `tests/unit/test_source_anchors_m01.py`, `tests/property/test_wave3_properties.py`, and `tests/integration/test_wave3_gate.py` | Fixed and regression-tested |
