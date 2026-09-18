from pathlib import Path

from scripts.check_workflow_security import validate_workflow


def _workflow(tmp_path: Path, body: str, name: str = "workflow.yml") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_rejects_inline_pull_request_target(tmp_path: Path) -> None:
    path = _workflow(
        tmp_path,
        """
on: [pull_request_target]
permissions: read-all
jobs: {}
""",
    )

    assert any("pull_request_target is prohibited" in item for item in validate_workflow(path))


def test_rejects_workflow_run_mapping(tmp_path: Path) -> None:
    path = _workflow(
        tmp_path,
        """
on:
  workflow_run:
    workflows: [CI]
    types: [completed]
permissions:
  contents: write
jobs: {}
""",
    )

    assert any("workflow_run is prohibited" in item for item in validate_workflow(path))


def test_rejects_inline_unpinned_action(tmp_path: Path) -> None:
    path = _workflow(
        tmp_path,
        """
on: pull_request
permissions: read-all
jobs:
  test:
    steps: [{uses: owner/action@main}]
""",
    )

    assert any("not pinned to a full commit SHA" in item for item in validate_workflow(path))


def test_rejects_write_permission_on_pull_request(tmp_path: Path) -> None:
    path = _workflow(
        tmp_path,
        """
on: pull_request
permissions:
  contents: write
jobs:
  test:
    steps: []
""",
    )

    assert any("contents: write is prohibited" in item for item in validate_workflow(path))


def test_rejects_write_permission_and_persisted_checkout_on_branch_push(tmp_path: Path) -> None:
    path = _workflow(
        tmp_path,
        """
on: push
permissions: write-all
jobs:
  test:
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262
""",
    )

    failures = validate_workflow(path)
    assert any("write-all is prohibited" in item for item in failures)
    assert any("persist-credentials: false" in item for item in failures)


def test_rejects_codeql_write_permission_in_codeql_workflow(tmp_path: Path) -> None:
    path = _workflow(
        tmp_path,
        """
on: pull_request
permissions:
  security-events: write
jobs:
  test:
    steps: []
""",
    )

    assert any("security-events: write is prohibited" in item for item in validate_workflow(path))


def test_rejects_local_composite_action(tmp_path: Path) -> None:
    path = _workflow(
        tmp_path,
        """
on: pull_request
permissions: read-all
jobs:
  test:
    steps:
      - uses: ./.github/actions/helper
""",
    )

    assert any("local actions are prohibited" in item for item in validate_workflow(path))


def test_rejects_persisted_checkout_credentials(tmp_path: Path) -> None:
    path = _workflow(
        tmp_path,
        """
on: pull_request
permissions:
  contents: read
jobs:
  test:
    steps:
      - uses: Actions/Checkout@11d5960a326750d5838078e36cf38b85af677262
""",
    )

    assert any("persist-credentials: false" in item for item in validate_workflow(path))


def test_rejects_dot_path_alias_for_checkout(tmp_path: Path) -> None:
    path = _workflow(
        tmp_path,
        """
on: pull_request
permissions:
  contents: read
jobs:
  test:
    steps:
      - uses: actions/checkout/.@11d5960a326750d5838078e36cf38b85af677262
""",
    )

    failures = validate_workflow(path)
    assert any("action has an ambiguous path: actions/checkout/." in item for item in failures)


def test_rejects_case_insensitive_duplicate_checkout_input(tmp_path: Path) -> None:
    path = _workflow(
        tmp_path,
        """
on: pull_request
permissions:
  contents: read
jobs:
  test:
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262
        with:
          persist-credentials: false
          PERSIST-CREDENTIALS: true
""",
    )

    failures = validate_workflow(path)
    assert any(
        "duplicate case-insensitive input 'persist-credentials'" in item for item in failures
    )


def test_rejects_codeql_upload_from_pull_request(tmp_path: Path) -> None:
    path = _workflow(
        tmp_path,
        """
on: pull_request
permissions: read-all
jobs:
  test:
    steps:
      - uses: github/codeql-action/analyze@faaca9a8f6edddba5725ffe5adefdab6669a2eca
""",
    )

    assert any("CodeQL analyze must set upload: never" in item for item in validate_workflow(path))


def test_rejects_case_insensitive_duplicate_codeql_upload_input(tmp_path: Path) -> None:
    path = _workflow(
        tmp_path,
        """
on: pull_request
permissions: read-all
jobs:
  test:
    steps:
      - uses: github/codeql-action/analyze@faaca9a8f6edddba5725ffe5adefdab6669a2eca
        with:
          upload: never
          UPLOAD: always
""",
    )

    failures = validate_workflow(path)
    assert any("duplicate case-insensitive input 'upload'" in item for item in failures)


def test_allows_case_insensitive_control_input_names(tmp_path: Path) -> None:
    path = _workflow(
        tmp_path,
        """
on: pull_request
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262
        with: {PERSIST-CREDENTIALS: false}
      - uses: github/codeql-action/analyze@faaca9a8f6edddba5725ffe5adefdab6669a2eca
        with: {OUTPUT: codeql-results, UPLOAD: never}
""",
    )

    assert validate_workflow(path) == []


def test_rejects_implicit_or_isolated_project_build_commands(tmp_path: Path) -> None:
    path = _workflow(
        tmp_path,
        """
on: pull_request
permissions: read-all
jobs:
  test:
    steps:
      - run: uv sync --locked --all-groups
      - run: uv run --locked pytest
      - run: uv run pytest
      - run: uv sync --locked --no-install-project && uv run pytest
      - run: uv build
      - run: uv pip install --python .venv/bin/python --no-deps .
""",
    )

    failures = validate_workflow(path)
    assert sum("unapproved workflow command" in item for item in failures) == 5
    assert any("compound shell commands are prohibited" in item for item in failures)


def test_rejects_shell_and_dynamic_python_evaluators(tmp_path: Path) -> None:
    path = _workflow(
        tmp_path,
        """
on: pull_request
permissions: read-all
jobs:
  test:
    steps:
      - run: bash -c 'uv run pytest'
      - run: eval 'uv run pytest'
      - run: python -c 'import subprocess; subprocess.run(["uv", "run", "pytest"])'
      - run: uv run --no-sync bash -c 'uv run pytest'
      - run: uv run --no-sync python -c 'print("unsafe")'
""",
    )

    failures = validate_workflow(path)
    assert sum("unapproved workflow command" in item for item in failures) == 5


def test_rejects_substitution_path_spoofing_and_unlocked_installs(tmp_path: Path) -> None:
    path = _workflow(
        tmp_path,
        """
on: pull_request
permissions: read-all
jobs:
  test:
    steps:
      - run: uv sync --locked --all-groups --no-install-project `uv run pytest`
      - run: ./uv sync --locked --all-groups --no-install-project
      - run: uv run --no-sync ./pytest
      - run: uv sync --all-groups --no-install-project
      - run: uv pip install some-unlocked-package
      - run: uv pip install --python .venv/bin/python --no-deps ./ --no-build-isolation
""",
    )

    failures = validate_workflow(path)
    assert any("command substitution is prohibited" in item for item in failures)
    assert sum("unapproved workflow command" in item for item in failures) == 5


def test_rejects_untrusted_execution_context(tmp_path: Path) -> None:
    path = _workflow(
        tmp_path,
        """
on: pull_request
permissions: read-all
env:
  PATH: payload:/usr/bin:/bin
defaults:
  run:
    shell: python payload.py {0}
    working-directory: payload
jobs:
  test:
    runs-on: self-hosted
    container: attacker/image
    steps:
      - shell: python payload.py {0}
        working-directory: payload
        run: uv sync --locked --all-groups --no-install-project
""",
    )

    failures = validate_workflow(path)
    assert any("custom workflow shells are prohibited" in item for item in failures)
    assert any("non-root workflow working directories are prohibited" in item for item in failures)
    assert any("unapproved workflow environment variable: PATH" in item for item in failures)
    assert any("job test must run on ubuntu-latest" in item for item in failures)
    assert any("job test may not define containers or services" in item for item in failures)


def test_rejects_untrusted_pinned_actions_and_action_inputs(tmp_path: Path) -> None:
    path = _workflow(
        tmp_path,
        """
on: pull_request
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: attacker/path-mutator@1111111111111111111111111111111111111111
      - uses: astral-sh/setup-uv@d0cc045d04ccac9d8b7881df0226f9e82c39688e
        with:
          version: ${{ env.UV_VERSION }}
          enable-cache: true
          tool-dir: payload
""",
    )

    failures = validate_workflow(path)
    assert any("action is not in the trusted allowlist" in item for item in failures)
    assert any("unapproved inputs for action attacker/path-mutator" in item for item in failures)
    assert any("unapproved inputs for action astral-sh/setup-uv" in item for item in failures)


def test_allows_pinned_read_only_workflow_and_local_codeql_analysis(tmp_path: Path) -> None:
    path = _workflow(
        tmp_path,
        """
on:
  pull_request:
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262
        with: {persist-credentials: false}
      - uses: github/codeql-action/analyze@faaca9a8f6edddba5725ffe5adefdab6669a2eca
        with: {output: codeql-results, upload: never}
""",
    )

    assert validate_workflow(path) == []
