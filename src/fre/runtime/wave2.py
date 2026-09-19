"""Minimal Wave 2 lifecycle coordinator; deliberately not a general orchestrator."""

from uuid import UUID

from fre.domain.common import ArtifactRef
from fre.domain.context import CompilerProfile
from fre.domain.stop import StopDecision, StopDecisionRecord, StopDisposition
from fre.engine import FrontierReasoningEngine
from fre.modules.m12_context import ContextCompiler
from fre.runtime.budget_meter import BudgetMeter
from fre.runtime.events import (
    ArtifactRegistered,
    ContextCompiled,
    RunStatusChanged,
    StopDecisionRecorded,
    StopDecisionRecordedV2,
    TerminalContextAssociated,
)
from fre.runtime.wave3_context import Wave3ContextRuntime


class Wave2Runtime:
    def __init__(
        self,
        engine: FrontierReasoningEngine,
        compiler: ContextCompiler | None = None,
        *,
        wave3_context_runtime: Wave3ContextRuntime | None = None,
    ) -> None:
        self.engine = engine
        self.compiler = compiler or ContextCompiler()
        # C08 (F12/F04) remediation, finding E: prior to this, `Wave3
        # ContextRuntime.compile_and_persist` had zero production callers --
        # F12 was closed only in the sense that the wiring existed, never
        # that any real execution path actually invoked it. When this is
        # provided (see `fre.composition.compose_wave3`, which now wires the
        # composed `Wave3ContextRuntime` in), a real terminal `finalize` call
        # also persists a typed Wave 3 `ContextCompiledV2` packet for runs
        # that have a `ProblemSpec` formalised, alongside the pre-existing v1
        # `ContextCompiled` this method has always emitted.
        self.wave3_context_runtime = wave3_context_runtime

    def record_decision(self, run_id: UUID, decision: StopDecision) -> int:
        state = self.engine.inspect(run_id)
        remaining = BudgetMeter().remaining(state.budget)
        record = StopDecisionRecord(
            decision=decision,
            evaluated_state_version=state.version,
            evaluated_state_hash=state.state_hash,
            budget_projection_hash=remaining.projection_hash,
        )
        event = self.engine.make_event(
            run_id, StopDecisionRecordedV2(record=record), module_id="M13"
        )
        return self.engine.append(run_id, state.version, (event,))[-1].sequence

    def finalize(self, run_id: UUID, decision: StopDecision) -> str:
        if decision.disposition is StopDisposition.CONTINUE or decision.context_request is None:
            raise ValueError("only terminal stop decisions can be finalized")
        state = self.engine.inspect(run_id)
        if not state.stop_decisions or state.stop_decisions[-1] != decision:
            raise ValueError("terminal stop decision must be recorded before finalization")
        remaining = BudgetMeter().remaining(state.budget)
        if remaining.projection_hash != decision.budget_projection_hash:
            raise ValueError("terminal stop decision is stale relative to the budget projection")
        stored_events = self.engine.store.load(run_id)
        decision_sequence: int | None = None
        latest_record: StopDecisionRecord | None = None
        for event in reversed(stored_events):
            event_payload = event.validated_payload()
            if isinstance(event_payload, StopDecisionRecordedV2):
                if event_payload.record.decision != decision:
                    raise ValueError("latest recorded stop decision does not match finalization")
                decision_sequence = event.sequence
                latest_record = event_payload.record
                break
            if isinstance(event_payload, StopDecisionRecorded):
                if event_payload.decision != decision:
                    raise ValueError("latest recorded stop decision does not match finalization")
                decision_sequence = event.sequence
                break
        if decision_sequence is None:
            raise ValueError("terminal stop decision event is missing")
        # decision_sequence == latest_record.evaluated_state_version + 1 is not
        # an assumption that this event is alone in its append batch. Sequence
        # numbers are assigned per event (SQLiteStore._append_locked's
        # `expected_version + offset`, not once per batch), and
        # FrontierReasoningEngine.append previews every event of a batch through
        # RunReducer.apply before persisting it. The reducer's own
        # StopDecisionRecordedV2 handling requires
        # `record.evaluated_state_version == state.version` at the moment this
        # specific event is applied, and its general contiguous-sequence check
        # requires `event.sequence == state.version + 1` at that same moment --
        # together they force this equality for ANY successfully applied
        # StopDecisionRecordedV2 event, regardless of how many other events
        # share its batch or whether it is first, so long as the caller binds
        # the record to the state as it will exist after any earlier same-batch
        # events (see test_finalize_accepts_stop_decision_recorded_inside_
        # multi_event_batch in tests/integration/test_wave2_gate.py).
        if latest_record is not None and (
            not state.stop_decision_records
            or state.stop_decision_records[-1] != latest_record
            or decision_sequence != latest_record.evaluated_state_version + 1
        ):
            raise ValueError("terminal stop decision state binding is invalid")
        non_authoritative = (ArtifactRegistered,)
        if any(
            not isinstance(event.validated_payload(), non_authoritative)
            for event in stored_events
            if event.sequence > decision_sequence
        ):
            raise ValueError("terminal stop decision is stale after authoritative state changes")
        compiled = self.compiler.compile(
            run_id=run_id,
            snapshot_version=state.version,
            ledger=state.ledger,
            budget_remaining=remaining,
            profile=CompilerProfile.HANDOFF,
            size_target=decision.context_request.size_target,
            terminal_disposition=decision.disposition,
        )
        json_artifact = self.engine.store_artifact(
            compiled.canonical_bytes, media_type="application/json"
        )
        markdown_artifact = self.engine.store_artifact(
            compiled.markdown.encode(), media_type="text/markdown"
        )
        payloads = (
            ContextCompiled(
                packet=compiled.packet,
                json_artifact=ArtifactRef(
                    artifact_id=json_artifact.id, sha256=json_artifact.sha256
                ),
                markdown_artifact=ArtifactRef(
                    artifact_id=markdown_artifact.id, sha256=markdown_artifact.sha256
                ),
            ),
            TerminalContextAssociated(
                stop_disposition=decision.disposition,
                packet_hash=compiled.packet.packet_hash,
            ),
            RunStatusChanged(status=decision.disposition, reason="Wave 2 terminal lifecycle"),
        )
        events = tuple(
            self.engine.make_event(run_id, payload, module_id="wave2-runtime")
            for payload in payloads
        )
        self.engine.append(run_id, state.version, events)
        # Finding E (C08 remediation): the real wiring. `state` here was
        # captured before the v1 `ContextCompiled`/terminal-status batch above
        # was appended, so `self.engine.inspect(run_id)` inside
        # `compile_and_persist` re-reads the run AFTER that batch landed --
        # the freshest, actually-applied state, never a stale snapshot. This
        # only runs for runs that have a Wave 3 `ProblemSpec` formalised;
        # runs with no Wave 3 semantic state at all have nothing for
        # `Wave3ContextRuntime.compile_and_persist` to compile and it would
        # otherwise raise.
        if self.wave3_context_runtime is not None:
            refreshed = self.engine.inspect(run_id)
            if refreshed.problem_spec is not None:
                self.wave3_context_runtime.compile_and_persist(
                    run_id,
                    profile=CompilerProfile.HANDOFF,
                    size_target=decision.context_request.size_target,
                    terminal_disposition=decision.disposition,
                )
        return compiled.packet.packet_hash
