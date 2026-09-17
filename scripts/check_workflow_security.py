"""Fail CI when GitHub workflow policy regresses."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

WORKFLOW_DIR = Path(".github/workflows")
FULL_SHA_LENGTH = 40
PROHIBITED_TRIGGERS = frozenset({"pull_request_target", "workflow_run"})


def _walk(value: object) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _walk(child)


def _trigger_names(value: object) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, Mapping):
        return {str(key) for key in value}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {str(item) for item in value}
    return set()


def _permission_failures(value: object, location: str) -> list[str]:
    if value is None:
        return [f"{location}: pull_request workflows must declare permissions explicitly"]
    if isinstance(value, str):
        if value == "read-all":
            return []
        if value == "write-all":
            return [f"{location}: write-all is prohibited for pull_request workflows"]
        return [f"{location}: unsupported permissions value {value!r}"]
    if not isinstance(value, Mapping):
        return [f"{location}: permissions must be a mapping, read-all, or write-all"]

    failures: list[str] = []
    for permission, level in value.items():
        if str(level) == "write":
            failures.append(
                f"{location}: {permission}: write is prohibited for pull_request workflows"
            )
        elif str(level) not in {"read", "write", "none"}:
            failures.append(f"{location}: invalid {permission} permission level {level!r}")
    return failures


def _is_full_sha(revision: str) -> bool:
    return len(revision) == FULL_SHA_LENGTH and all(
        character in "0123456789abcdef" for character in revision
    )


def validate_workflow(path: Path) -> list[str]:
    try:
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    except yaml.YAMLError as error:
        return [f"{path}: invalid YAML: {error}"]
    if not isinstance(document, Mapping):
        return [f"{path}: workflow must be a YAML mapping"]

    failures: list[str] = []
    triggers = _trigger_names(document.get("on"))
    for trigger in sorted(triggers & PROHIBITED_TRIGGERS):
        failures.append(f"{path}: {trigger} is prohibited")

    pull_request_workflow = "pull_request" in triggers
    if pull_request_workflow:
        failures.extend(_permission_failures(document.get("permissions"), str(path)))

    jobs = document.get("jobs")
    if not isinstance(jobs, Mapping):
        failures.append(f"{path}: jobs must be a mapping")
        return failures

    if pull_request_workflow:
        for job_name, job in jobs.items():
            if isinstance(job, Mapping) and "permissions" in job:
                failures.extend(
                    _permission_failures(
                        job.get("permissions"),
                        f"{path}: job {job_name}",
                    )
                )

    for node in _walk(document):
        use = node.get("uses")
        if not isinstance(use, str):
            continue
        if use.startswith("./"):
            failures.append(f"{path}: local actions are prohibited: {use}")
            continue
        if "@" not in use:
            failures.append(f"{path}: action is unpinned: {use}")
            continue
        action, revision = use.rsplit("@", 1)
        normalized_action = action.casefold()
        if not _is_full_sha(revision):
            failures.append(f"{path}: {action} is not pinned to a full commit SHA")

        inputs = node.get("with")
        if pull_request_workflow and normalized_action == "actions/checkout":
            persist_credentials = (
                inputs.get("persist-credentials") if isinstance(inputs, Mapping) else None
            )
            if persist_credentials != "false":
                failures.append(
                    f"{path}: actions/checkout must set persist-credentials: false "
                    "for pull_request workflows"
                )
        if pull_request_workflow and normalized_action == "github/codeql-action/analyze":
            upload = inputs.get("upload") if isinstance(inputs, Mapping) else None
            if upload != "never":
                failures.append(
                    f"{path}: CodeQL analyze must set upload: never for pull_request workflows"
                )

    return failures


def main() -> None:
    workflows = sorted((*WORKFLOW_DIR.glob("*.yml"), *WORKFLOW_DIR.glob("*.yaml")))
    if not workflows:
        raise SystemExit("no GitHub workflows found")

    failures = [failure for workflow in workflows for failure in validate_workflow(workflow)]
    if failures:
        raise SystemExit("\n".join(failures))

    print(f"validated {len(workflows)} workflow files")


if __name__ == "__main__":
    main()
