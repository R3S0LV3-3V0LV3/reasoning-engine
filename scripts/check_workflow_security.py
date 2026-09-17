"""Fail CI when GitHub workflow policy regresses."""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOW_DIR = Path(".github/workflows")
USES_PATTERN = re.compile(r"^\s*-?\s*uses:\s*([^\s#]+)", re.MULTILINE)
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def main() -> None:
    failures: list[str] = []
    workflows = sorted((*WORKFLOW_DIR.glob("*.yml"), *WORKFLOW_DIR.glob("*.yaml")))
    if not workflows:
        raise SystemExit("no GitHub workflows found")

    for workflow in workflows:
        text = workflow.read_text(encoding="utf-8")
        if re.search(r"^\s*pull_request_target\s*:", text, re.MULTILINE):
            failures.append(f"{workflow}: pull_request_target is prohibited")

        for use in USES_PATTERN.findall(text):
            if use.startswith("./"):
                continue
            if "@" not in use:
                failures.append(f"{workflow}: action is unpinned: {use}")
                continue
            action, revision = use.rsplit("@", 1)
            if not FULL_SHA.fullmatch(revision):
                failures.append(f"{workflow}: {action} is not pinned to a full commit SHA")

    if failures:
        raise SystemExit("\n".join(failures))

    print(f"validated {len(workflows)} workflow files")


if __name__ == "__main__":
    main()
