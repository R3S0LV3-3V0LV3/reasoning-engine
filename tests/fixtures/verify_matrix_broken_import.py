"""Fixture module for `tests/unit/test_verify_requirements_matrix.py` only.

Deliberately fails to import under pytest (finding H's decisive test): a
scratch matrix citing both this file and the healthy
`tests/fixtures/verify_matrix_scenarios.py` must see the collection failure
attributed ONLY to rows citing THIS file -- never blanket-failing rows that
cite the unrelated, healthy file, the way the pre-fix batched
`--collect-only` invocation used to. See naming-pattern note in
`verify_matrix_scenarios.py`; the same applies here. Do not cite this file
from `docs/wave3-requirements-matrix.md`.
"""

raise ImportError("fixture: deliberately broken import for the checker's own decisive test")


def test_never_collected() -> None:  # pragma: no cover - import fails first
    assert True
