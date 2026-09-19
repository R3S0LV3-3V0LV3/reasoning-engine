"""Fixture module for `tests/unit/test_verify_requirements_matrix.py` only.

Deliberately named so pytest's default `testpaths = ["tests"]` recursive
discovery (which only auto-collects `test_*.py`) never picks this file up as
part of the real test suite -- it exists solely to be CITED BY NAME from a
scratch matrix in the decisive tests for `scripts/verify_requirements_matrix.py`
(findings A and H), giving those tests a real skipped test, a real failing
test, and a real passing test to point at. None of these are acceptance
evidence for any Wave 3 requirement; do not cite this file from
`docs/wave3-requirements-matrix.md`.
"""

import pytest


def test_fixture_passing_case() -> None:
    assert True


@pytest.mark.skip(reason="fixture: deliberately skipped for the checker's own decisive test")
def test_fixture_skipped_case() -> None:
    assert True


def test_fixture_failing_case() -> None:
    raise AssertionError("fixture: deliberately failing for the checker's own decisive test")
