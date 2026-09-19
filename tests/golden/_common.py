"""Shared harness for the Wave 3 golden fixtures A-L (C10, defect F14).

Every fixture in `test_golden_fixtures.py` drives its scenario through the
real C09 coordinator (`fre.composition.Wave3Engine.execute_front_end` or its
individually-public, resumable step methods) -- never a hand-wired module
call -- and then records the coordinator's persisted, replayable output
(the exact event sequence, artifact hashes, projections, blockers/
availability, budget totals, and final state hash) as a checked-in golden
file under `tests/fixtures/golden/<name>.json`.

This module deliberately mirrors -- field-for-field -- the exact scenario-
construction helpers already exercised by the coordinator's own decisive
integration gate (`tests/integration/test_wave3_gate.py`) rather than
re-deriving a second, differently-shaped set of fixtures: a golden fixture
that built its inputs differently from the suite proving the coordinator
correct would not be testing the same claim. (A direct cross-package
`tests.integration.test_wave3_gate` import was deliberately avoided here:
with no `tests/__init__.py` in this repository, mypy resolves that file
under two different module identities -- `test_wave3_gate` when collected
by pytest's rootdir-relative import mode, and `tests.integration.
test_wave3_gate` when imported by dotted path from here -- and refuses to
type-check the result. Duplicating these small, stable builders locally
avoids that ambiguity entirely.)

Confirmed experimentally (2026-09, Wave 3 post-freeze cleanup pass, C10
EU-44): adding a bare `tests/__init__.py` (with `pythonpath = ["."]` in
`pyproject.toml`, `testpaths = ["tests"]`) does NOT reproduce the collision
under pytest -- the full suite still collects and passes -- but it does
break `mypy --strict src tests`, which then fails with:
`tests/golden/test_golden_fixtures.py:35: error: Cannot find implementation
or library stub for module named "_common"  [import-not-found]`. That is
because turning `tests/` into a package changes how mypy resolves the
sibling-relative `from _common import ...` in `test_golden_fixtures.py`;
fixing it would require converting every such import to a fully-qualified
`tests.golden._common` form throughout the golden suite (and possibly
elsewhere), which is exactly the broader import-mode change this module's
docstring already judged out of scope for a mechanical duplication fix.
The experimental `tests/__init__.py` was discarded; nothing from it landed.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

from fre.adapters.artifacts_local import LocalArtifactStore
from fre.adapters.storage_sqlite import SQLiteStore
from fre.adapters.testing import FakeClock, FakeUUIDFactory
from fre.domain.common import JsonValue, OutputContract, PermissionSet
from fre.domain.semantic import (
    SemanticCallUsage,
    StructuredModelRequest,
    StructuredModelResult,
    StructuredModelStatus,
)
from fre.domain.task import TaskEnvelope
from fre.engine import FrontierReasoningEngine
from fre.runtime.budget_meter import BudgetMeter

GOLDENS = Path(__file__).resolve().parents[1] / "fixtures" / "golden"


class QueueModel:
    """A structured-model port whose responses are drained one call at a time."""

    def __init__(self, responses: list[StructuredModelResult]) -> None:
        self.responses = iter(responses)
        self.calls: list[str] = []

    async def generate(self, request: StructuredModelRequest) -> StructuredModelResult:
        self.calls.append(request.idempotency_key)
        return next(self.responses)


def _uuids(count: int = 400) -> list[UUID]:
    return [UUID(int=index) for index in range(1, count + 1)]


def make_engine(tmp_path: Path) -> FrontierReasoningEngine:
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=i) for i in range(2000))
    return FrontierReasoningEngine(
        SQLiteStore(tmp_path / "events.db"),
        LocalArtifactStore(tmp_path / "artifacts"),
        clock,
        FakeUUIDFactory(_uuids()),
    )


def make_task(task_id: int = 900, text: str = "Choose a safe option.") -> TaskEnvelope:
    return TaskEnvelope(
        task_id=UUID(int=task_id),
        text=text,
        requested_output=OutputContract(form="TEXT"),
        execution_permissions=PermissionSet(),
    )


def classification_response(*, usage: SemanticCallUsage | None = None) -> StructuredModelResult:
    usage = usage or SemanticCallUsage(input_tokens=100, output_tokens=50)
    anchor: JsonValue = {
        "source_kind": "TASK_TEXT",
        "source_ref": {"object_type": "TaskEnvelope", "object_id": str(UUID(int=900))},
        "selector": "/text",
        "char_start": 0,
        "char_end": len("Choose a safe option."),
    }
    valid: dict[str, JsonValue] = {
        "task_type": {
            "estimate": "DECISION",
            "confidence": 0.9,
            "anchors": [anchor],
            "rationale": "fixture",
        },
        "consequence": {
            "estimate": "LOW",
            "confidence": 0.9,
            "conservative_upper": "MEDIUM",
            "anchors": [anchor],
            "rationale": "fixture",
        },
        "reversibility": {
            "estimate": "HIGH",
            "confidence": 0.9,
            "conservative_upper": "HIGH",
            "anchors": [anchor],
            "rationale": "fixture",
        },
        "ambiguity": {
            "estimate": "MEDIUM",
            "confidence": 0.9,
            "conservative_upper": "MEDIUM",
            "anchors": [anchor],
            "rationale": "fixture",
        },
        "evidence_scarcity": {
            "estimate": "MEDIUM",
            "confidence": 0.9,
            "conservative_upper": "MEDIUM",
            "anchors": [anchor],
            "rationale": "fixture",
        },
        "search_space": {
            "estimate": "BOUNDED",
            "confidence": 0.9,
            "anchors": [anchor],
            "rationale": "fixture",
        },
        "horizon": {
            "estimate": "SHORT",
            "confidence": 0.9,
            "anchors": [anchor],
            "rationale": "fixture",
        },
    }
    return StructuredModelResult(
        status=StructuredModelStatus.SUCCESS,
        adapter_id="fake",
        model_id="fixture",
        raw_response=json.dumps(valid).encode(),
        decoded=valid,
        usage=usage,
    )


def unavailable_response() -> StructuredModelResult:
    return StructuredModelResult(
        status=StructuredModelStatus.UNAVAILABLE,
        adapter_id="fake",
        model_id="fixture",
        raw_response=b"",
        diagnostics=("no model configured",),
    )


def invalid_response() -> StructuredModelResult:
    return StructuredModelResult(
        status=StructuredModelStatus.INVALID_STRUCTURED_OUTPUT,
        adapter_id="fake",
        model_id="fixture",
        raw_response=b"{bad",
        diagnostics=("invalid",),
    )


def formalisation_response(
    items: list[dict[str, JsonValue]] | None = None,
) -> StructuredModelResult:
    payload: JsonValue = {"items": [dict(item) for item in items] if items is not None else []}
    return StructuredModelResult(
        status=StructuredModelStatus.SUCCESS,
        adapter_id="fake",
        model_id="fixture",
        raw_response=json.dumps(payload).encode(),
        decoded=payload,
        usage=SemanticCallUsage(input_tokens=100, output_tokens=50),
    )


# Set FRE_GOLDEN_UPDATE=1 (via `python scripts/update_golden_fixtures.py`) to
# regenerate the checked-in golden files after a genuine, reviewed
# behavioural change. Never set this to silently launder an unreviewed
# regression -- review the resulting diff before committing it.
_UPDATE = os.environ.get("FRE_GOLDEN_UPDATE") == "1"


def event_log(engine: FrontierReasoningEngine, run_id: UUID) -> list[dict[str, Any]]:
    """The run's real, versioned, persisted event sequence."""
    return [
        {"sequence": event.sequence, "module_id": event.module_id, "event_type": event.event_type}
        for event in engine.store.load(run_id)
    ]


def golden_record(engine: FrontierReasoningEngine, run_id: UUID, **extra: Any) -> dict[str, Any]:
    """Build the canonical evidence record for one golden fixture.

    Every field here is derived from the run's real, already-applied,
    persisted state (`engine.inspect`/`engine.store.load`/`engine.snapshot`)
    -- never from an in-flight value the test itself computed -- so this
    record is exactly what an independent auditor replaying the same run
    from its event log would also observe.
    """
    state = engine.inspect(run_id)
    remaining = BudgetMeter().remaining(state.budget)
    last_packet = state.context_packets[-1] if state.context_packets else None
    wave3_ctx = last_packet.wave3_context if last_packet is not None else None
    plan = state.representation_plan_v2
    record: dict[str, Any] = {
        "run_id": str(run_id),
        "version": state.version,
        "final_state_hash": engine.snapshot(run_id),
        "events": event_log(engine, run_id),
        "artifacts": sorted(state.artifacts),
        "model_calls": len(state.model_calls),
        "classification": {
            "present": state.task_signature is not None,
            "mode": (state.classification_record.mode if state.classification_record else None),
            "fallback_used": (
                state.classification_record.fallback_used if state.classification_record else None
            ),
        },
        "problem_spec_present": state.problem_spec is not None,
        "unknowns": sorted(
            u.id for u in (state.problem_spec.unknowns if state.problem_spec else ())
        ),
        "assumptions": sorted(
            a.id for a in (state.problem_spec.assumption_items if state.problem_spec else ())
        ),
        "blockers": {
            "all": sorted(b.blocker_id for b in state.problem_blockers),
            "material": sorted(b.blocker_id for b in state.problem_blockers if not b.resolvable),
        },
        "contradictions": len(state.problem_contradictions),
        "representation_plan": (
            {
                "views": [view.kind.value for view in plan.views],
                "tie_triggered": plan.tie_triggered,
                "adjudication_record_ref": plan.adjudication_record_ref,
                "fallback_used": plan.fallback_used,
            }
            if plan is not None
            else None
        ),
        "context_packets": len(state.context_packets),
        "last_packet": (
            {
                "packet_hash": last_packet.packet_hash,
                "profile": last_packet.profile.value,
                "availability": (wave3_ctx.availability.value if wave3_ctx is not None else None),
            }
            if last_packet is not None
            else None
        ),
        "budget_remaining": {
            "iterations": remaining.resources.iterations,
            "active_concurrent_actions": remaining.active_concurrent_actions,
        },
        "terminal_context_packet_hash": state.terminal_context_packet_hash,
        "stop_decisions": [
            {
                "disposition": decision.disposition.value,
                "reason_codes": sorted(code.value for code in decision.reason_codes),
            }
            for decision in state.stop_decisions
        ],
    }
    record.update(extra)
    return record


def assert_matches_golden(name: str, record: dict[str, Any]) -> None:
    """Compare `record` against the checked-in golden file `name`.json.

    With `FRE_GOLDEN_UPDATE=1` set, (re)writes the golden file instead of
    comparing -- only ever run intentionally via
    `scripts/update_golden_fixtures.py`, and only after reviewing the diff.
    """
    path = GOLDENS / f"{name}.json"
    actual = json.loads(json.dumps(record, sort_keys=True, default=str))
    if _UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        return
    if not path.exists():
        raise AssertionError(
            f"golden fixture file missing: {path}. Run "
            "`python scripts/update_golden_fixtures.py`, review the diff, and commit the "
            "generated file -- a golden fixture with no checked-in evidence proves nothing."
        )
    expected = json.loads(path.read_text())
    assert actual == expected, (
        f"golden fixture '{name}' drifted from its checked evidence at {path}. If this is a "
        "genuine, reviewed behavioural change to the Wave 3 front end, regenerate it with "
        "scripts/update_golden_fixtures.py and review the diff before committing; otherwise "
        "this is a real regression."
    )
