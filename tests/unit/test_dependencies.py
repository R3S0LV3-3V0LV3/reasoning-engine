import ast
from pathlib import Path

import pytest


@pytest.mark.unit
def test_domain_and_modules_do_not_import_concrete_adapters() -> None:
    forbidden_by_root = {
        Path("src/fre/domain"): ("fre.adapters", "fre.runtime", "fre.modules"),
        Path("src/fre/modules"): ("fre.adapters",),
    }
    for root, forbidden in forbidden_by_root.items():
        for path in root.glob("**/*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = [
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module is not None
            ]
            assert all(
                not any(name.startswith(prefix) for prefix in forbidden) for name in imports
            ), path


@pytest.mark.unit
def test_deterministic_replay_core_has_no_provider_or_network_imports() -> None:
    forbidden = (
        "fre.adapters.models",
        "fre.adapters.tools",
        "fre.ports.models",
        "fre.ports.tools",
        "httpx",
        "requests",
        "urllib",
        "socket",
    )
    for root in (Path("src/fre/runtime"), Path("src/fre/modules"), Path("src/fre/projections")):
        for path in root.glob("**/*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = [
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module is not None
            ] + [
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            ]
            assert all(not name.startswith(forbidden) for name in imports), path
