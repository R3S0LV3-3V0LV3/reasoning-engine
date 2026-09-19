import json
from pathlib import Path

import pytest

from scripts.check_codeql_sarif import inspect_sarif


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "python.sarif"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_accepts_empty_codeql_results(tmp_path: Path) -> None:
    _write(tmp_path, {"runs": [{"results": []}]})

    assert inspect_sarif(tmp_path) == (0, ())


def test_reports_codeql_findings_without_emitting_source_content(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "runs": [
                {
                    "results": [
                        {
                            "ruleId": "py/example",
                            "message": {"text": "sensitive source excerpt"},
                        }
                    ]
                }
            ]
        },
    )

    assert inspect_sarif(tmp_path) == (1, ("python.sarif:py/example",))


def test_rejects_missing_or_malformed_sarif(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no SARIF"):
        inspect_sarif(tmp_path)
    _write(tmp_path, {"runs": "invalid"})
    with pytest.raises(ValueError, match="malformed SARIF"):
        inspect_sarif(tmp_path)
