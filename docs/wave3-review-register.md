# Wave 3 review register

Only findings with decisive regression coverage are closed here.

| Finding | Resolution | Evidence | State |
|---|---|---|---|
| Runtime integrity defects 1–6 | Snapshot compatibility, revision-envelope markers, representation invalidation, model-call artifact provenance, contradiction-resolution reference validation, and stale-stop rejection were hardened without changing the reducer version. | `tests/fixtures/wave2_history.json`, `tests/integration/test_wave2_gate.py`, `tests/integration/test_wave3_gate.py`, and `tests/unit/test_wave2.py` | Fixed and regression-tested |
