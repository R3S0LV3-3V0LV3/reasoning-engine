"""Decisive tests for the C10 matrix referential-integrity checker (F14).

The checker (`scripts/verify_requirements_matrix.py`) exists to make the
requirements matrix's own claims falsifiable: if a row cites a test that
does not exist, or a function that is not defined in the file it names, the
checker must fail and name the row -- never silently pass a matrix whose
"evidence" is fabricated or stale.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.verify_requirements_matrix as vrm
from scripts.verify_requirements_matrix import (
    check_execution,
    check_matrix_matches_generator,
    check_references,
    main,
    parse_matrix,
)

VALID_ROW = (
    "| W3-999 | fixture requirement | `fre.example.Symbol` | "
    "`tests/unit/test_domain.py::test_domain_models_are_immutable` | N/A | N/A | `deadbeef` (C10) |"
)
DANGLING_FUNCTION_ROW = (
    "| W3-998 | fixture requirement | `fre.example.Symbol` | "
    "`tests/unit/test_domain.py::test_this_function_does_not_exist` | N/A | N/A "
    "| `deadbeef` (C10) |"
)
DANGLING_FILE_ROW = (
    "| W3-997 | fixture requirement | `fre.example.Symbol` | "
    "`tests/unit/test_this_file_does_not_exist.py::test_anything` | N/A | N/A | `deadbeef` (C10) |"
)
HEADER = (
    "| ID | REQUIREMENT | IMPLEMENTATION SYMBOL/FILE | POSITIVE TEST | "
    "NEGATIVE/ADVERSARIAL TEST | REPLAY/PROPERTY/GOLDEN EVIDENCE | EVIDENCE SHA |\n"
    "|---|---|---|---|---|---|---|\n"
)


def _write_matrix(tmp_path: Path, row: str) -> Path:
    path = tmp_path / "matrix.md"
    path.write_text("# Fixture matrix\n\n" + HEADER + row + "\n")
    return path


@pytest.mark.unit
def test_valid_matrix_reference_resolves_and_collects(tmp_path: Path) -> None:
    path = _write_matrix(tmp_path, VALID_ROW)
    references = parse_matrix(path)
    assert len(references) == 1
    assert references[0].row_id == "W3-999"
    findings = check_references(references)
    assert findings == []


@pytest.mark.unit
def test_dangling_function_reference_is_a_finding(tmp_path: Path) -> None:
    path = _write_matrix(tmp_path, DANGLING_FUNCTION_ROW)
    references = parse_matrix(path)
    findings = check_references(references)
    assert len(findings) == 1
    assert findings[0].row_id == "W3-998"
    assert "not defined" in findings[0].reason


@pytest.mark.unit
def test_dangling_file_reference_is_a_finding(tmp_path: Path) -> None:
    path = _write_matrix(tmp_path, DANGLING_FILE_ROW)
    references = parse_matrix(path)
    findings = check_references(references)
    assert len(findings) == 1
    assert findings[0].row_id == "W3-997"
    assert "does not exist" in findings[0].reason


@pytest.mark.unit
def test_main_exits_nonzero_for_a_matrix_with_a_dangling_reference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The decisive end-to-end case: point the checker's CLI entry point at
    a matrix with a dangling test reference and confirm it exits non-zero
    and names the offending row -- this is the exact scenario C10's own
    validation strategy requires ("give it a matrix with a dangling test
    reference and confirm it fails")."""
    path = _write_matrix(tmp_path, DANGLING_FUNCTION_ROW)
    exit_code = main(["verify_requirements_matrix.py", str(path)])
    assert exit_code != 0
    captured = capsys.readouterr()
    assert "W3-998" in captured.err


@pytest.mark.unit
def test_main_exits_zero_for_the_real_checked_in_matrix() -> None:
    """The real matrix this PR ships must itself pass the checker -- proving
    the checker is wired against genuine evidence, not just its own fixtures."""
    exit_code = main(["verify_requirements_matrix.py"])
    assert exit_code == 0


@pytest.mark.unit
def test_na_negative_column_is_not_a_finding(tmp_path: Path) -> None:
    """A row honestly declaring `N/A` for the negative/adversarial column
    (no adversarial test applies) must not be treated as a dangling
    reference -- `N/A` is a valid, declared choice, not a missing citation."""
    path = _write_matrix(tmp_path, VALID_ROW)
    references = parse_matrix(path)
    assert all(ref.file_path != "N/A" for ref in references)


# --- Finding A: a cited test must actually PASS, not merely exist/collect ---
#
# Before this PR's fix, `verify_requirements_matrix.py` only checked that a
# cited test file/function existed and that the FILE collected under pytest
# -- it never ran the cited test itself. A row citing an
# `@pytest.mark.skip`'d test, or a row citing an actively-failing test, both
# passed the pre-fix checker cleanly (confirmed manually against the pre-fix
# revision: `check_references` + the old `check_collection` both return `[]`
# for these rows, since the fixture FILE collects fine -- only the cited
# FUNCTION is skipped/failing). `check_execution` (this PR) closes that gap.

SKIPPED_TEST_ROW = (
    "| W3-996 | fixture requirement | `fre.example.Symbol` | "
    "`tests/fixtures/verify_matrix_scenarios.py::test_fixture_skipped_case` | N/A | N/A "
    "| `deadbeef` (C10) |"
)
FAILING_TEST_ROW = (
    "| W3-995 | fixture requirement | `fre.example.Symbol` | "
    "`tests/fixtures/verify_matrix_scenarios.py::test_fixture_failing_case` | N/A | N/A "
    "| `deadbeef` (C10) |"
)
PASSING_TEST_ROW = (
    "| W3-994 | fixture requirement | `fre.example.Symbol` | "
    "`tests/fixtures/verify_matrix_scenarios.py::test_fixture_passing_case` | N/A | N/A "
    "| `deadbeef` (C10) |"
)


@pytest.mark.unit
def test_a_matrix_citing_a_skipped_test_is_rejected(tmp_path: Path) -> None:
    """A row citing a `@pytest.mark.skip`'d test must be REJECTED: a skip
    proves nothing about the acceptance claim it is cited as evidence for."""
    path = _write_matrix(tmp_path, SKIPPED_TEST_ROW)
    references = parse_matrix(path)
    assert check_references(references) == []  # file/function genuinely exist
    findings = check_execution(references)
    assert len(findings) == 1
    assert findings[0].row_id == "W3-996"
    assert "SKIPPED" in findings[0].reason


@pytest.mark.unit
def test_a_matrix_citing_a_failing_test_is_rejected(tmp_path: Path) -> None:
    """A row citing an actively-failing test must be REJECTED."""
    path = _write_matrix(tmp_path, FAILING_TEST_ROW)
    references = parse_matrix(path)
    assert check_references(references) == []
    findings = check_execution(references)
    assert len(findings) == 1
    assert findings[0].row_id == "W3-995"
    assert "FAILED" in findings[0].reason


@pytest.mark.unit
def test_a_matrix_citing_a_genuinely_passing_test_is_accepted(tmp_path: Path) -> None:
    """Control case: a row citing a real, passing test must NOT be flagged --
    proving `check_execution` distinguishes pass from skip/fail rather than
    rejecting everything indiscriminately."""
    path = _write_matrix(tmp_path, PASSING_TEST_ROW)
    references = parse_matrix(path)
    assert check_execution(references) == []


@pytest.mark.unit
def test_main_exits_nonzero_end_to_end_for_a_skipped_test_citation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End-to-end: the CLI entry point itself must reject a skipped-test
    citation, matching the exact scenario the review found live: 'a row
    citing an @pytest.mark.skip'd test... passes the checker cleanly.'"""
    path = _write_matrix(tmp_path, SKIPPED_TEST_ROW)
    exit_code = main(["verify_requirements_matrix.py", str(path)])
    assert exit_code != 0
    captured = capsys.readouterr()
    assert "W3-996" in captured.err


# --- Finding H: a collection error in one file must not blanket-fail rows
# citing OTHER, unaffected files in the same matrix run ---

BROKEN_FILE_ROW = (
    "| W3-993 | fixture requirement | `fre.example.Symbol` | "
    "`tests/fixtures/verify_matrix_broken_import.py::test_never_collected` | N/A | N/A "
    "| `deadbeef` (C10) |"
)


@pytest.mark.unit
def test_a_broken_import_in_one_file_does_not_fail_rows_citing_other_files(
    tmp_path: Path,
) -> None:
    """A matrix with two rows -- one citing a file that fails to import,
    one citing an unrelated, healthy, passing test -- must flag ONLY the
    row citing the broken file. Before this PR's fix, the batched
    `pytest --collect-only` invocation ran across every cited file in one
    subprocess call; a single import error anywhere in that batch caused
    pytest's non-(0,5) exit code to blanket-fail EVERY row across EVERY
    file in the batch, not just the rows citing the broken one."""
    combined = "# Fixture matrix\n\n" + HEADER + BROKEN_FILE_ROW + "\n" + PASSING_TEST_ROW + "\n"
    path = tmp_path / "matrix.md"
    path.write_text(combined)
    references = parse_matrix(path)
    assert {ref.row_id for ref in references} == {"W3-993", "W3-994"}

    findings = check_execution(references)
    flagged_rows = {finding.row_id for finding in findings}
    assert "W3-993" in flagged_rows, "the row citing the broken-import file must be flagged"
    assert "W3-994" not in flagged_rows, (
        "the row citing the unrelated, healthy, passing file must NOT be flagged "
        "merely because another file in the same batch failed to import"
    )


# --- Finding B: the checked-in matrix must not drift from the generator ---


@pytest.mark.unit
def test_checked_in_matrix_matches_the_generator_today() -> None:
    """The real, checked-in matrix must match
    `scripts/generate_requirements_matrix.py`'s current output exactly."""
    from scripts.verify_requirements_matrix import DEFAULT_MATRIX

    assert check_matrix_matches_generator(DEFAULT_MATRIX) == []


@pytest.mark.unit
def test_a_hand_edited_drift_from_the_generator_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decisive drift test: point the checker's default-matrix path at a
    scratch file, write the generator's real output there (must pass), then
    hand-edit it (simulating a checked-in matrix that has drifted from its
    generator) and confirm the drift check now fails."""
    from scripts.generate_requirements_matrix import render

    scratch_matrix = tmp_path / "matrix.md"
    monkeypatch.setattr(vrm, "DEFAULT_MATRIX", scratch_matrix)

    scratch_matrix.write_text(render())
    assert check_matrix_matches_generator(scratch_matrix) == []

    scratch_matrix.write_text(render() + "\n<!-- hand-edited, now drifted -->\n")
    findings = check_matrix_matches_generator(scratch_matrix)
    assert len(findings) == 1
    assert findings[0].row_id == "MATRIX-DRIFT"

    # Restore: confirm the drift check passes again once content matches.
    scratch_matrix.write_text(render())
    assert check_matrix_matches_generator(scratch_matrix) == []
