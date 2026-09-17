"""Verify the built wheel as an installed consumer sees it."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main() -> None:
    wheels = sorted(Path("dist").glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one wheel, found {len(wheels)}")

    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    if "fre/py.typed" not in names:
        raise SystemExit("built wheel does not contain fre/py.typed")

    with tempfile.TemporaryDirectory(prefix="fre-package-") as directory:
        venv = Path(directory) / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        python = venv / "bin" / "python"
        subprocess.run([str(python), "-m", "pip", "install", str(wheel)], check=True)
        subprocess.run(
            [
                str(python),
                "-c",
                "import fre; from fre.prompts.registry import default_prompt_registry; "
                "default_prompt_registry()",
            ],
            check=True,
        )

    print(f"verified {wheel}")


if __name__ == "__main__":
    main()
