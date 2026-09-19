#!/usr/bin/env python3
"""Regenerate `tests/fixtures/golden/*.json` from the golden fixture suite.

Run this ONLY after a genuine, reviewed behavioural change to the Wave 3
front end changes what a golden fixture (A-L) legitimately produces.
`git diff tests/fixtures/golden/` afterwards and review every changed byte
before committing -- an unreviewed diff here silently launders a real
regression into "the new checked-in normal", which is precisely the
overclaiming failure mode C10 exists to close (defect F14).
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    env = dict(os.environ)
    env["FRE_GOLDEN_UPDATE"] = "1"
    return subprocess.call(
        [sys.executable, "-m", "pytest", "tests/golden", "-m", "golden", "-q"], env=env
    )


if __name__ == "__main__":
    raise SystemExit(main())
