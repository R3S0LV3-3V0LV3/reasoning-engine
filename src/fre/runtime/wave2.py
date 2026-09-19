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
        # W3 final-gate fix #7: compile-and-persist the typed Wave 3 context
        # BEFORE appending the v1 terminal batch below, not after.
        #
        # Before this fix, the v1 `ContextCompiled`/`TerminalContextAssociated`/
        # `RunStatusChanged` batch was appended FIRST -- making the run
        # terminal -- and `Wave3ContextRuntime.compile_and_persist` was called
        # only afterward. If that second call raised for any reason (a
        # transient artifact-store failure, an unexpected `ValueError` from
        # `compile_semantic`'s own validation, ...), the run was left
        # permanently stuck: already terminal (no further `StopDecision`/
        # M13 evaluation is possible on a terminal run), yet with no
        # `ContextCompiledV2` ever persisted and no way to retry the failed
        # step, since `finalize` itself is not re-enterable once the run is
        # terminal.
        #
        # Reordering is safe because `compile_and_persist` reads its own
        # fresh `self.engine.inspect(run_id)` and depends only on
        # `problem_spec`/`budget`/`ledger`/representation state -- none of
        # which the v1 terminal batch below ever touches (it only appends
        # `ContextCompiled`/`TerminalContextAssociated`/`RunStatusChanged`,
        # none of which mutate problem/budget/ledger/representation
        # projections) -- so calling it first changes nothing about what it
        # computes. With this reordering, a failure in Wave 3 context
        # compilation now leaves the run in its prior, non-terminal,
        # genuinely retryable state instead of a stuck terminal-without-
        # context one: `finalize` can simply be called again once the
        # underlying problem (e.g. a transient artifact-store outage) is
        # resolved, since neither the terminal batch nor this call has
        # landed yet.
        #
        # EU-42 (w3-cleanup) cross-reference, still accurate after this
        # reordering: this call and `Wave3Engine.compile_context`'s own
        # 7-key-deduped call (`composition.py`) avoid double-compiling a
        # packet only because their `(profile, terminal_disposition)` values
        # happen to differ today -- this call always passes
        # `profile=CompilerProfile.HANDOFF` with a real, non-None
        # `terminal_disposition`, while `compile_context`'s own default is
        # `profile=CompilerProfile.STANDARD` with `terminal_disposition=None`
        # unless a caller explicitly overrides it. That is incidental
        # avoidance, not a structural guarantee: this call performs no dedup
        # scan of its own (unlike `compile_context`), so a future change to
        # either caller's `profile`/`terminal_disposition` values -- e.g. a
        # caller invoking `compile_context` with `profile=HANDOFF` and a
        # matching `terminal_disposition` after this method already ran --
        # could silently reintroduce double-persistence of a context packet.
        # A narrower fix, if ever pursued, would move `compile_context`'s
        # dedup scan into a method on `Wave3ContextRuntime` itself (e.g.
        # `find_matching_packet(...)`) and have this call go through the same
        # method first; that is deliberately not attempted in this pass.
        if self.wave3_context_runtime is not None and state.problem_spec is not None:
            # ARCH-DEFER (C08 remediation, EU-32): this call persists the
            # typed v2 `wave3_context` (via `ContextCompiledV2`), but no
            # terminal-disposition consumer reads it back today -- the
            # actual disposition decision below is driven entirely by the
            # v1 `ContextCompiled`/`TerminalContextAssociated` pair appended
            # after this call, keyed off `ContextPacket.terminal_disposition`;
            # `m13_stop.py` has zero references to
            # `wave3_context`/`ContextCompiledV2`. There is no near-term plan
            # to change that here: teaching finalize/M13 to read
            # `wave3_context` for real terminal-disposition decisions is new
            # coordinator work (likely a C09-adjacent follow-up), not a
            # cleanup item, and is intentionally out of scope for this pass.
            self.wave3_context_runtime.compile_and_persist(
                run_id,
                profile=CompilerProfile.HANDOFF,
                size_target=decision.context_request.size_target,
                terminal_disposition=decision.disposition,
            )
        # `compile_and_persist` above (when it ran) appended its own atomic
        # batch, advancing the run's real committed version past `state.
        # version` captured at the top of this method -- the v1 terminal
        # batch below must therefore be appended against the run's CURRENT
        # version, not that now-stale snapshot, or the optimistic-concurrency
        # check in `FrontierReasoningEngine.append` would reject it.
        current_version = self.engine.inspect(run_id).version
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
        self.engine.append(run_id, current_version, events)
        return compiled.packet.packet_hash
