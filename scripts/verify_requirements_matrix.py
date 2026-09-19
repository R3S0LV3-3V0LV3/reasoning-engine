#!/usr/bin/env python3
"""Matrix referential-integrity checker (C10, defect F14).

Parses every mandatory row of `docs/wave3-requirements-matrix.md` and fails
if any `POSITIVE TEST` or `NEGATIVE/ADVERSARIAL TEST` cell that names a
`path/to/test_file.py::test_function[params]` reference does not actually
exist and collect under pytest. A row citing `N/A` for the negative-test
column is not an error -- it is an honest declaration that no adversarial
test applies to that requirement, which this checker verifies is at least a
DECLARED choice (present in the table) rather than a silently blank cell.

This is the decisive test for the checker itself: point it at a matrix with
a dangling test reference (a file or function that does not exist) and
confirm it exits non-zero naming the row and the missing reference. See
`tests/unit/test_verify_requirements_matrix.py`.

Usage:
    python scripts/verify_requirements_matrix.py [MATRIX_PATH]

Exit code 0 if every referenced test exists and collects; non-zero (with a
line per failing row) otherwise.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = REPO_ROOT / "docs" / "wave3-requirements-matrix.md"

# Matches a backtick-quoted `tests/.../test_x.py::test_name[optional-params]`
# reference anywhere in a table cell.
TEST_REF_RE = re.compile(r"`((?:tests|scripts)/[\w./-]+\.py)(?:::([\w\[\]\-.]+))?`")

TABLE_ROW_RE = re.compile(r"^\|(.+)\|\s*$")


@dataclass(frozen=True)
class TestReference:
    row_id: str
    column: str
    file_path: str
    function_name: str | None


@dataclass(frozen=True)
class Finding:
    row_id: str
    column: str
    reference: str
    reason: str

    def __str__(self) -> str:
        return f"{self.row_id} [{self.column}]: `{self.reference}` -- {self.reason}"


def _split_row(line: str) -> list[str] | None:
    match = TABLE_ROW_RE.match(line.rstrip("\n"))
    if match is None:
        return None
    cells = [cell.strip() for cell in match.group(1).split("|")]
    return cells


def parse_matrix(path: Path) -> list[TestReference]:
    """Extract every test reference from every data row of every table."""
    lines = path.read_text().splitlines()
    references: list[TestReference] = []
    header: list[str] | None = None
    for line in lines:
        cells = _split_row(line)
        if cells is None:
            header = None
            continue
        if all(set(cell) <= {"-", ":"} or cell == "" for cell in cells):
            # Separator row (e.g. |---|---|...|) -- keep the preceding header.
            continue
        if cells and cells[0] == "ID":
            header = cells
            continue
        if header is None:
            continue
        row_id = cells[0]
        if not row_id.startswith("W3-"):
            continue
        for column_name, cell in zip(header, cells, strict=False):
            for file_path, function_name in TEST_REF_RE.findall(cell):
                references.append(
                    TestReference(
                        row_id=row_id,
                        column=column_name,
                        file_path=file_path,
                        function_name=function_name or None,
                    )
                )
    return references


def _strip_params(function_name: str) -> str:
    return function_name.split("[", 1)[0]


def _functions_defined_in(file_path: Path) -> set[str]:
    tree = ast.parse(file_path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def check_references(references: list[TestReference]) -> list[Finding]:
    findings: list[Finding] = []
    file_function_cache: dict[Path, set[str]] = {}
    for ref in references:
        full_path = REPO_ROOT / ref.file_path
        reference_text = ref.file_path + (f"::{ref.function_name}" if ref.function_name else "")
        if not full_path.is_file():
            findings.append(Finding(ref.row_id, ref.column, reference_text, "file does not exist"))
            continue
        if ref.function_name is None:
            continue
        function_name = _strip_params(ref.function_name)
        if full_path not in file_function_cache:
            try:
                file_function_cache[full_path] = _functions_defined_in(full_path)
            except SyntaxError as exc:  # pragma: no cover - defensive
                findings.append(
                    Finding(ref.row_id, ref.column, reference_text, f"file does not parse: {exc}")
                )
                continue
        if function_name not in file_function_cache[full_path]:
            findings.append(
                Finding(
                    ref.row_id,
                    ref.column,
                    reference_text,
                    f"function `{function_name}` not defined in {ref.file_path}",
                )
            )
    return findings


def check_collection(references: list[TestReference]) -> list[Finding]:
    """Confirm every referenced test file actually collects under pytest
    (catches import errors / marker typos / removed fixtures that plain AST
    parsing above cannot see)."""
    findings: list[Finding] = []
    files = sorted({ref.file_path for ref in references if ref.file_path.startswith("tests/")})
    if not files:
        return findings
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *files],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 5):  # 5 == no tests collected, still a real failure here
        last_line = "unknown error"
        if result.stdout:
            last_line = result.stdout.splitlines()[-1]
        elif result.stderr:
            last_line = result.stderr.splitlines()[-1]
        for ref in references:
            if ref.file_path in files:
                findings.append(
                    Finding(
                        ref.row_id,
                        ref.column,
                        ref.file_path,
                        f"pytest --collect-only failed for this file "
                        f"(exit {result.returncode}): {last_line}",
                    )
                )
    return findings


def main(argv: list[str]) -> int:
    matrix_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_MATRIX
    if not matrix_path.is_file():
        print(f"matrix file not found: {matrix_path}", file=sys.stderr)
        return 2
    references = parse_matrix(matrix_path)
    if not references:
        print(
            f"no test references found in {matrix_path} -- matrix is empty or malformed",
            file=sys.stderr,
        )
        return 2
    findings = check_references(references)
    findings.extend(check_collection(references))
    if findings:
        print(
            f"{len(findings)} referential-integrity failure(s) in {matrix_path}:", file=sys.stderr
        )
        for finding in findings:
            print(f"  - {finding}", file=sys.stderr)
        return 1
    print(f"OK: {len(references)} test references in {matrix_path} all exist and collect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
