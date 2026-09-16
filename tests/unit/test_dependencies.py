import ast
from pathlib import Path

import pytest


@pytest.mark.unit
def test_domain_and_modules_do_not_import_concrete_adapters() -> None:
    forbidden = "fre.adapters"
    for root in (Path("src/fre/domain"), Path("src/fre/modules")):
        for path in root.glob("**/*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = [
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module is not None
            ]
            assert all(not name.startswith(forbidden) for name in imports), path
