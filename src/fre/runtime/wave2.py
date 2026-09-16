"""Minimal Wave 2 lifecycle coordinator; deliberately not a general orchestrator."""

from uuid import UUID

from fre.domain.common import ArtifactRef
from fre.domain.context import CompilerProfile
from fre.domain.stop import StopDecision, StopDisposition
from fre.engine import FrontierReasoningEngine
from fre.modules.m12_context import ContextCompiler
from fre.runtime.budget_meter import BudgetMeter
from fre.runtime.events import (
    ContextCompiled,
    RunStatusChanged,
    StopDecisionRecorded,
    TerminalContextAssociated,
)


class Wave2Runtime:
    def __init__(
        self, engine: FrontierReasoningEngine, compiler: ContextCompiler | None = None
    ) -> None:
        self.engine = engine
        self.compiler = compiler or ContextCompiler()

    def record_decision(self, run_id: UUID, decision: StopDecision) -> int:
        state = self.engine.inspect(run_id)
        event = self.engine.make_event(
            run_id, StopDecisionRecorded(decision=decision), module_id="M13"
        )
        return self.engine.append(run_id, state.version, (event,))[-1].sequence

    def finalize(self, run_id: UUID, decision: StopDecision) -> str:
        if decision.disposition is StopDisposition.CONTINUE or decision.context_request is None:
            raise ValueError("only terminal stop decisions can be finalized")
        state = self.engine.inspect(run_id)
        remaining = BudgetMeter().remaining(state.budget)
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
        return compiled.packet.packet_hash
