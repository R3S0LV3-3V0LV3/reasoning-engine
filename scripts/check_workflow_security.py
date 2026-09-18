"""Fail CI when GitHub workflow policy regresses."""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

WORKFLOW_DIR = Path(".github/workflows")
FULL_SHA_LENGTH = 40
PROHIBITED_TRIGGERS = frozenset({"pull_request_target", "workflow_run"})
SHELL_CONTROL_CHARACTERS = frozenset(";&|()")
ALLOWED_WORKFLOW_COMMANDS = frozenset(
    {
        ("python", "scripts/check_codeql_sarif.py", "codeql-results"),
        ("python", "scripts/verify_package.py"),
        ("uv", "build", "--no-build-isolation"),
        (
            "uv",
            "pip",
            "install",
            "--python",
            ".venv/bin/python",
            "--no-deps",
            ".",
            "--no-build-isolation",
        ),
        ("uv", "run", "--no-sync", "mypy", "--strict", "src", "tests"),
        ("uv", "run", "--no-sync", "pytest"),
        ("uv", "run", "--no-sync", "pytest", "tests/integration"),
        ("uv", "run", "--no-sync", "pytest", "tests/property"),
        ("uv", "run", "--no-sync", "pytest", "tests/unit"),
        (
            "uv",
            "run",
            "--no-sync",
            "pytest",
            "tests/unit/test_foundation_freeze.py",
            "tests/integration/test_wave1_gate.py",
            "tests/integration/test_wave2_gate.py",
            "tests/integration/test_transactional_preduction.py",
        ),
        ("uv", "run", "--no-sync", "python", "scripts/check_workflow_security.py"),
        ("uv", "run", "--no-sync", "ruff", "check", "."),
        ("uv", "run", "--no-sync", "ruff", "format", "--check", "."),
        ("uv", "sync", "--locked", "--all-groups", "--no-install-project"),
    }
)
ALLOWED_ENVIRONMENT_ENTRIES = {
    "GITHUB_TOKEN": "${{ secrets.GITHUB_TOKEN }}",
    "PYTHON_VERSION": "3.12",
    "UV_VERSION": "0.11.16",
}
ALLOWED_ACTIONS = frozenset(
    {
        ("actions/checkout", "11d5960a326750d5838078e36cf38b85af677262"),
        ("actions/dependency-review-action", "a1d282b36b6f3519aa1f3fc636f609c47dddb294"),
        ("actions/setup-python", "a26af69be951a213d495a4c3e4e4022e16d87065"),
        ("astral-sh/setup-uv", "d0cc045d04ccac9d8b7881df0226f9e82c39688e"),
        ("github/codeql-action/analyze", "faaca9a8f6edddba5725ffe5adefdab6669a2eca"),
        ("github/codeql-action/init", "faaca9a8f6edddba5725ffe5adefdab6669a2eca"),
        ("gitleaks/gitleaks-action", "e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e"),
    }
)
ALLOWED_ACTION_INPUTS: dict[str, tuple[dict[str, object], ...]] = {
    "actions/checkout": (
        {"persist-credentials": "false"},
        {"fetch-depth": "0", "persist-credentials": "false"},
    ),
    "actions/dependency-review-action": ({"fail-on-severity": "high"},),
    "actions/setup-python": (
        {"python-version": "${{ env.PYTHON_VERSION }}"},
        {"python-version": "3.12"},
    ),
    "astral-sh/setup-uv": ({"enable-cache": "true", "version": "${{ env.UV_VERSION }}"},),
    "github/codeql-action/analyze": ({"output": "codeql-results", "upload": "never"},),
    "github/codeql-action/init": ({"languages": "python", "queries": "security-extended"},),
    "gitleaks/gitleaks-action": ({},),
}
ALLOWED_CONDITIONAL_STEPS = frozenset(
    {
        (
            "actions/dependency-review-action@a1d282b36b6f3519aa1f3fc636f609c47dddb294",
            "github.event_name == 'pull_request'",
        ),
    }
)
REQUIRED_WORKFLOW_DOCUMENT_HASHES = {
    # These are hashes of the parsed YAML documents, not the source text. Comments and
    # formatting may change, but triggers, jobs, steps, controls, and inputs may not be
    # removed, added, or altered without an explicit policy review and hash update.
    "ci.yml": "e8ff9bb3041e05fc455617d89eb2701073748f2887e94d11ef9ee5bacca657e6",
    "codeql.yml": "ac32728591b1e37d0140c7c29acd80d6793973993f435c48400a3ec9fe984b91",
}


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


def _workflow_document_hash(document: Mapping[object, object]) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_workflow_inventory(workflows: Sequence[Path]) -> list[str]:
    found = {path.name for path in workflows}
    required = set(REQUIRED_WORKFLOW_DOCUMENT_HASHES)
    failures = [
        *(f"missing protected workflow: {name}" for name in sorted(required - found)),
        *(f"unapproved workflow file: {name}" for name in sorted(found - required)),
    ]
    return failures


def _permission_failures(value: object, location: str) -> list[str]:
    if value is None:
        return [f"{location}: workflows must declare permissions explicitly"]
    if isinstance(value, str):
        if value == "read-all":
            return []
        if value == "write-all":
            return [f"{location}: write-all is prohibited for repository workflows"]
        return [f"{location}: unsupported permissions value {value!r}"]
    if not isinstance(value, Mapping):
        return [f"{location}: permissions must be a mapping, read-all, or write-all"]

    failures: list[str] = []
    for permission, level in value.items():
        if str(level) == "write":
            failures.append(
                f"{location}: {permission}: write is prohibited for repository workflows"
            )
        elif str(level) not in {"read", "write", "none"}:
            failures.append(f"{location}: invalid {permission} permission level {level!r}")
    return failures


def _is_full_sha(revision: str) -> bool:
    return len(revision) == FULL_SHA_LENGTH and all(
        character in "0123456789abcdef" for character in revision
    )


def _normalized_inputs(
    value: object,
    path: Path,
    action: str,
) -> tuple[dict[str, object], list[str]]:
    if not isinstance(value, Mapping):
        return {}, []

    normalized: dict[str, object] = {}
    failures: list[str] = []
    for key, input_value in value.items():
        normalized_key = str(key).casefold()
        if normalized_key in normalized:
            failures.append(
                f"{path}: {action} declares duplicate case-insensitive input {normalized_key!r}"
            )
            continue
        normalized[normalized_key] = input_value
    return normalized, failures


def _normalized_action(action: str, path: Path) -> tuple[str, list[str]]:
    segments = action.split("/")
    if len(segments) < 2 or any(segment in {"", ".", ".."} for segment in segments):
        return action.casefold(), [f"{path}: action has an ambiguous path: {action}"]
    return "/".join(segments).casefold(), []


def _run_command_failures(value: object, path: Path) -> list[str]:
    if not isinstance(value, str):
        return []

    failures: list[str] = []
    for line in value.splitlines():
        command = line.strip()
        if not command:
            continue
        if "`" in command or "$(" in command:
            failures.append(f"{path}: command substitution is prohibited: {command}")
            continue
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        lexer.commenters = ""
        try:
            tokens = list(lexer)
        except ValueError as error:
            failures.append(f"{path}: cannot parse workflow command {command!r}: {error}")
            continue

        if any(token and set(token) <= SHELL_CONTROL_CHARACTERS for token in tokens):
            failures.append(f"{path}: compound shell commands are prohibited: {command}")
            continue
        if tuple(tokens) not in ALLOWED_WORKFLOW_COMMANDS:
            failures.append(f"{path}: unapproved workflow command: {command}")
    return failures


def _execution_context_failures(node: Mapping[str, Any], path: Path) -> list[str]:
    failures: list[str] = []
    if "shell" in node:
        failures.append(f"{path}: custom workflow shells are prohibited")
    if "working-directory" in node:
        failures.append(f"{path}: non-root workflow working directories are prohibited")
    if "continue-on-error" in node:
        failures.append(f"{path}: continue-on-error is prohibited")
    if "if" in node:
        conditional_step = (node.get("uses"), node.get("if"))
        if conditional_step not in ALLOWED_CONDITIONAL_STEPS:
            failures.append(f"{path}: unapproved workflow condition")

    environment = node.get("env")
    if environment is None:
        return failures
    if not isinstance(environment, Mapping):
        failures.append(f"{path}: workflow env must be a mapping")
        return failures

    seen: set[str] = set()
    for key, value in environment.items():
        name = str(key)
        normalized_name = name.casefold()
        if normalized_name in seen:
            failures.append(f"{path}: duplicate case-insensitive environment key {name!r}")
            continue
        seen.add(normalized_name)
        if name not in ALLOWED_ENVIRONMENT_ENTRIES:
            failures.append(f"{path}: unapproved workflow environment variable: {name}")
            continue
        if value != ALLOWED_ENVIRONMENT_ENTRIES[name]:
            failures.append(f"{path}: unapproved value for workflow environment variable {name}")
    return failures


def validate_workflow(path: Path) -> list[str]:
    try:
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    except yaml.YAMLError as error:
        return [f"{path}: invalid YAML: {error}"]
    if not isinstance(document, Mapping):
        return [f"{path}: workflow must be a YAML mapping"]

    failures: list[str] = []
    expected_hash = REQUIRED_WORKFLOW_DOCUMENT_HASHES.get(path.name)
    if expected_hash is not None and _workflow_document_hash(document) != expected_hash:
        failures.append(
            f"{path}: workflow does not match its required protected trigger/job/step shape"
        )
    triggers = _trigger_names(document.get("on"))
    for trigger in sorted(triggers & PROHIBITED_TRIGGERS):
        failures.append(f"{path}: {trigger} is prohibited")

    failures.extend(_permission_failures(document.get("permissions"), str(path)))

    jobs = document.get("jobs")
    if not isinstance(jobs, Mapping):
        failures.append(f"{path}: jobs must be a mapping")
        return failures

    for job_name, job in jobs.items():
        if not isinstance(job, Mapping):
            failures.append(f"{path}: job {job_name} must be a mapping")
            continue
        if job.get("runs-on") != "ubuntu-latest":
            failures.append(f"{path}: job {job_name} must run on ubuntu-latest")
        if "container" in job or "services" in job:
            failures.append(f"{path}: job {job_name} may not define containers or services")
        if "permissions" in job:
            failures.extend(
                _permission_failures(
                    job.get("permissions"),
                    f"{path}: job {job_name}",
                )
            )

    for node in _walk(document):
        failures.extend(_execution_context_failures(node, path))
        failures.extend(_run_command_failures(node.get("run"), path))
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
        normalized_action, action_failures = _normalized_action(action, path)
        failures.extend(action_failures)
        if not _is_full_sha(revision):
            failures.append(f"{path}: {action} is not pinned to a full commit SHA")

        inputs, input_failures = _normalized_inputs(node.get("with"), path, action)
        failures.extend(input_failures)
        if (normalized_action, revision) not in ALLOWED_ACTIONS:
            failures.append(f"{path}: action is not in the trusted allowlist: {use}")
        allowed_input_profiles = ALLOWED_ACTION_INPUTS.get(normalized_action)
        if allowed_input_profiles is None or inputs not in allowed_input_profiles:
            failures.append(f"{path}: unapproved inputs for action {action}")
        if normalized_action == "actions/checkout":
            persist_credentials = inputs.get("persist-credentials")
            if persist_credentials != "false":
                failures.append(
                    f"{path}: actions/checkout must set persist-credentials: false "
                    "for repository workflows"
                )
        if normalized_action == "github/codeql-action/analyze":
            upload = inputs.get("upload")
            if upload != "never":
                failures.append(
                    f"{path}: CodeQL analyze must set upload: never for repository workflows"
                )

    return failures


def main() -> None:
    workflows = sorted((*WORKFLOW_DIR.glob("*.yml"), *WORKFLOW_DIR.glob("*.yaml")))
    if not workflows:
        raise SystemExit("no GitHub workflows found")

    failures = [
        *validate_workflow_inventory(workflows),
        *(failure for workflow in workflows for failure in validate_workflow(workflow)),
    ]
    if failures:
        raise SystemExit("\n".join(failures))

    print(f"validated {len(workflows)} workflow files")


if __name__ == "__main__":
    main()
