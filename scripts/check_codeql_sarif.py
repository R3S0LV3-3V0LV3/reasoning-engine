"""Fail when a local CodeQL analysis reports a finding or malformed output."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def inspect_sarif(directory: Path) -> tuple[int, tuple[str, ...]]:
    paths = tuple(sorted(directory.rglob("*.sarif")))
    if not paths:
        raise ValueError(f"no SARIF files found under {directory}")

    finding_count = 0
    summaries: list[str] = []
    for path in paths:
        document: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or not isinstance(document.get("runs"), list):
            raise ValueError(f"{path}: malformed SARIF document")
        for run in document["runs"]:
            if not isinstance(run, dict):
                raise ValueError(f"{path}: malformed SARIF run")
            results = run.get("results", [])
            if not isinstance(results, list):
                raise ValueError(f"{path}: malformed SARIF results")
            finding_count += len(results)
            for result in results:
                rule_id = result.get("ruleId", "unknown") if isinstance(result, dict) else "unknown"
                summaries.append(f"{path.name}:{rule_id}")
    return finding_count, tuple(summaries)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_codeql_sarif.py DIRECTORY")
    directory = Path(sys.argv[1])
    try:
        count, summaries = inspect_sarif(directory)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    if count:
        raise SystemExit(f"CodeQL reported {count} finding(s): {', '.join(summaries)}")
    print(f"validated CodeQL SARIF under {directory}: 0 findings")


if __name__ == "__main__":
    main()
