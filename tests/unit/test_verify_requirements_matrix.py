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

from scripts.verify_requirements_matrix import (
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
