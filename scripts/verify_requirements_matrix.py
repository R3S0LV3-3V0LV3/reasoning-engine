#!/usr/bin/env python3
"""Matrix referential-integrity checker (C10, defect F14; PR #25 findings A/B/H).

Parses every mandatory row of `docs/wave3-requirements-matrix.md` and fails
if any `POSITIVE TEST` or `NEGATIVE/ADVERSARIAL TEST` cell that names a
`path/to/test_file.py::test_function[params]` reference does not actually
exist, does not collect, or does not PASS under pytest. A row citing a
skipped or xfail'd test is rejected too -- that is not decisive passing
evidence for the acceptance claim it is cited for (finding A). A row citing
`N/A` for the negative-test column is not an error -- it is an honest
declaration that no adversarial test applies to that requirement, which this
checker verifies is at least a DECLARED choice (present in the table) rather
than a silently blank cell.

This checker also fails if the checked-in matrix has drifted from what
`scripts/generate_requirements_matrix.py` currently produces (finding B) --
the generator is the source of truth, and a hand-edited or stale checked-in
file must never silently diverge from it.

This is the decisive test for the checker itself: point it at a matrix with
a dangling test reference (a file or function that does not exist), a
skipped test, or a failing test, and confirm it exits non-zero naming the
row and the reference. See `tests/unit/test_verify_requirements_matrix.py`.

Usage:
    python scripts/verify_requirements_matrix.py [MATRIX_PATH]

Exit code 0 if every referenced test exists, collects, and passes, and the
checked-in matrix (when MATRIX_PATH is the default) matches the generator;
non-zero (with a line per failing row) otherwise.
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


NODE_OUTCOME_RE = re.compile(
    r"^(?P<nodeid>\S+::\S+)\s+(?P<outcome>PASSED|FAILED|SKIPPED|XFAIL|XPASS|ERROR)\b"
)

# Outcomes that do NOT count as decisive passing evidence for a cited
# acceptance test. A skipped or xfail'd test proves nothing about the
# acceptance claim it is cited for -- it must be treated as a failure of
# that row, not silently accepted the way a bare `--collect-only` check
# would (finding A). XPASS (a test marked xfail that unexpectedly passed)
# is also rejected: an inconsistent expectation is not reliable evidence.
_NOT_PASSING = {"SKIPPED", "FAILED", "XFAIL", "XPASS", "ERROR"}


def _node_id(ref: TestReference) -> str:
    return f"{ref.file_path}::{ref.function_name}" if ref.function_name else ref.file_path


def _parse_outcomes(stdout: str) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    for line in stdout.splitlines():
        match = NODE_OUTCOME_RE.match(line)
        if match:
            outcomes[match.group("nodeid")] = match.group("outcome")
    return outcomes


def _collect_node_ids(file_path: str) -> tuple[set[str], subprocess.CompletedProcess[str]]:
    """The real, exact node ids pytest collects for `file_path` (e.g. real
    parametrize ids like `test_x[case0]`, distinct from a matrix cell's
    bracket suffix, which is sometimes purely documentation -- e.g.
    `test_m01_to_m02_axes_are_monotone[consequence]` names which axis a
    single, non-parametrized hypothesis test decisively covers, not a real
    pytest selector)."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            file_path,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    ids = {line.strip() for line in result.stdout.splitlines() if f"{file_path}::" in line}
    return ids, result


def _resolve_node_ids(ref: TestReference, collected: set[str]) -> list[str] | None:
    """Map a cited reference onto the real, collectible pytest node id(s) it
    denotes, or `None` if it does not correspond to any. All returned ids
    must pass for the reference to count as decisive evidence. Handles three
    shapes, in order:

    1. The exact cited id (a genuine, correctly-cited parametrize id).
    2. The id with any bracket suffix stripped (a documentation-only bracket
       decorating a non-parametrized test -- e.g. a single hypothesis test
       whose citation names which axis it covers, not a real pytest
       selector), consistent with `_strip_params` in `check_references`.
    3. A bare, un-bracketed citation of a function that IS genuinely
       parametrized under pytest (every one of its real variants is
       required to pass, since the row's citation did not narrow to one).
    """
    if ref.function_name is None:
        return [ref.file_path]
    exact = f"{ref.file_path}::{ref.function_name}"
    if exact in collected:
        return [exact]
    base = f"{ref.file_path}::{_strip_params(ref.function_name)}"
    if base in collected:
        return [base]
    variants = sorted(node_id for node_id in collected if node_id.startswith(f"{base}["))
    if variants:
        return variants
    return None


def check_execution(references: list[TestReference]) -> list[Finding]:
    """Confirm every referenced test actually RUNS and PASSES under pytest --
    not merely that the file/function exists and collects (finding A). A row
    citing a `@pytest.mark.skip`'d or `xfail`'d test, or a test that
    currently fails, is a false acceptance claim and must be rejected here,
    the same as a dangling reference.

    Batched by file (one `--collect-only` plus one execution subprocess per
    distinct cited file, covering every cited node id in that file at once)
    for performance, while keeping per-file attribution: a collection error
    in one file (e.g. an import error) fails only the rows citing THAT file,
    never rows citing other, unaffected files in the same matrix (finding
    H) -- the previous whole-batch `--collect-only` invocation blanket-
    failed every row across every file whenever any single file failed to
    collect.
    """
    findings: list[Finding] = []
    refs_by_file: dict[str, list[TestReference]] = {}
    for ref in references:
        if not ref.file_path.startswith("tests/"):
            continue
        if not (REPO_ROOT / ref.file_path).is_file():
            continue  # already reported by check_references; do not double-report
        refs_by_file.setdefault(ref.file_path, []).append(ref)

    for file_path, refs in sorted(refs_by_file.items()):
        collected, collect_result = _collect_node_ids(file_path)
        if not collected:
            last_line = "unknown error"
            if collect_result.stdout:
                last_line = collect_result.stdout.strip().splitlines()[-1]
            elif collect_result.stderr:
                last_line = collect_result.stderr.strip().splitlines()[-1]
            for ref in refs:
                reference_text = ref.file_path + (
                    f"::{ref.function_name}" if ref.function_name else ""
                )
                findings.append(
                    Finding(
                        ref.row_id,
                        ref.column,
                        reference_text,
                        f"pytest --collect-only found nothing in {file_path} "
                        f"(exit {collect_result.returncode}): {last_line}",
                    )
                )
            continue

        resolved: dict[TestReference, list[str] | None] = {
            ref: _resolve_node_ids(ref, collected) for ref in refs
        }
        run_ids = sorted({node_id for ids in resolved.values() if ids for node_id in ids})
        outcomes: dict[str, str] = {}
        if run_ids:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-v",
                    "--no-header",
                    "-p",
                    "no:cacheprovider",
                    *run_ids,
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            outcomes = _parse_outcomes(result.stdout)

        for ref in refs:
            reference_text = ref.file_path + (f"::{ref.function_name}" if ref.function_name else "")
            node_ids = resolved[ref]
            if node_ids is None:
                findings.append(
                    Finding(
                        ref.row_id,
                        ref.column,
                        reference_text,
                        f"no real pytest node id in {file_path} matches this reference "
                        f"(not among the {len(collected)} ids pytest actually collects)",
                    )
                )
                continue
            for node_id in node_ids:
                outcome = outcomes.get(node_id)
                if outcome is None:
                    findings.append(
                        Finding(
                            ref.row_id,
                            ref.column,
                            reference_text,
                            f"pytest did not report a pass/fail outcome for `{node_id}` -- "
                            f"treating as not decisive passing evidence",
                        )
                    )
                elif outcome in _NOT_PASSING:
                    findings.append(
                        Finding(
                            ref.row_id,
                            ref.column,
                            reference_text,
                            f"cited test `{node_id}` is {outcome} under pytest -- not "
                            f"decisive passing evidence for this acceptance row",
                        )
                    )
    return findings


def check_matrix_matches_generator(matrix_path: Path) -> list[Finding]:
    """Fail if the checked-in matrix has drifted from
    `scripts/generate_requirements_matrix.py`'s current output (finding B).

    Only meaningful for the real, checked-in default matrix -- a scratch/
    fixture matrix passed explicitly on the command line (as the decisive
    tests for this checker do) has no corresponding generator output to
    compare against, so this check is skipped for any non-default path.
    """
    if matrix_path.resolve() != DEFAULT_MATRIX.resolve():
        return []
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts.generate_requirements_matrix import render
    finally:
        sys.path.pop(0)
    expected = render()
    actual = matrix_path.read_text()
    if expected == actual:
        return []
    return [
        Finding(
            row_id="MATRIX-DRIFT",
            column="(whole file)",
            reference=str(matrix_path),
            reason=(
                "checked-in matrix does not match `python scripts/generate_requirements_matrix.py` "
                "output -- regenerate it (the generator's ROWS/CROSS_CUTTING data is the source of "
                "truth; never hand-edit the checked-in markdown file)"
            ),
        )
    ]


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
    findings.extend(check_execution(references))
    findings.extend(check_matrix_matches_generator(matrix_path))
    if findings:
        print(
            f"{len(findings)} referential-integrity failure(s) in {matrix_path}:", file=sys.stderr
        )
        for finding in findings:
            print(f"  - {finding}", file=sys.stderr)
        return 1
    print(
        f"OK: {len(references)} test references in {matrix_path} all exist, collect, and pass; "
        "checked-in matrix matches the generator."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
