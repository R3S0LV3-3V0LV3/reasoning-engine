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
        requirements = Path(directory) / "requirements.txt"
        venv = Path(directory) / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        python = venv / "bin" / "python"
        subprocess.run(
            [
                "uv",
                "export",
                "--locked",
                "--no-dev",
                "--no-emit-project",
                "--output-file",
                str(requirements),
            ],
            check=True,
        )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--require-hashes",
                "--requirements",
                str(requirements),
            ],
            check=True,
        )
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)],
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "-c",
                "import fre; from fre.prompts.registry import default_prompt_registry; "
                "default_prompt_registry()",
            ],
            check=True,
        )
        # C09 (F08): the coordinator (`fre.composition.Wave3Engine`) is the
        # one authoritative, replayable Wave 3 front-end path -- verify it is
        # actually importable and callable end-to-end from this isolated
        # installed wheel, not merely from the source checkout's own test
        # suite. A zero-call (`allow_model=False`) run exercises the full
        # M01 -> M02 -> M03 -> M04 -> M12 sequence without any network access
        # or provider dependency, which is exactly what a packaging smoke
        # test can run unconditionally.
        subprocess.run(
            [str(python), str(Path(__file__).with_name("_verify_coordinator.py"))], check=True
        )

    print(f"verified {wheel}")


if __name__ == "__main__":
    main()
