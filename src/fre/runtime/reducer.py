"""Pure reducer protocol and foundational run reducer."""

from typing import Protocol, TypeVar
from uuid import UUID

from pydantic import Field

from fre.domain.budget import BudgetProjection
from fre.domain.common import FrozenModel, JsonValue, canonical_hash, canonical_json
from fre.domain.context import ContextCompilationRecord, ContextPacket
from fre.domain.ledger import LedgerProjection
from fre.domain.stop import StopDecision
from fre.modules.m02_budget import BudgetAllocator
from fre.modules.m09_ledger import EpistemicLedger
from fre.runtime.budget_meter import BudgetMeter
from fre.runtime.events import (
    ArtifactRegistered,
    BudgetAllocated,
    BudgetConsumed,
    BudgetReservationReleased,
    BudgetReservationSettled,
    BudgetReserved,
    BudgetRevised,
    ContextCompiled,
    LedgerContradictionResolved,
    LedgerDependentsMarkedStale,
    LedgerEdgeAdded,
    LedgerNodeAdded,
    LedgerNodeRevised,
    LedgerNodeStatusChanged,
    RunCreated,
    RunStatusChanged,
    StopDecisionRecorded,
    StoredEvent,
    TerminalContextAssociated,
    TestValueSet,
)

StateT = TypeVar("StateT")


class Reducer(Protocol[StateT]):
    version: str

    def initial(self, run_id: UUID) -> StateT: ...

    def apply(self, state: StateT, event: StoredEvent) -> StateT: ...


class RunState(FrozenModel):
    run_id: UUID
    version: int = Field(ge=0)
    status: str = "NEW"
    config_hash: str | None = None
    artifacts: tuple[str, ...] = ()
    values: dict[str, JsonValue] = Field(default_factory=dict)
    ledger: LedgerProjection = LedgerProjection()
    budget: BudgetProjection = BudgetProjection()
    context_packets: tuple[ContextPacket, ...] = ()
    context_compilations: tuple[ContextCompilationRecord, ...] = ()
    stop_decisions: tuple[StopDecision, ...] = ()
    terminal_context_packet_hash: str | None = None
    terminal_context_disposition: str | None = None

    def snapshot_payload(self) -> dict[str, object]:
        """Return the hash payload, retaining Wave 1 shape for untouched streams."""
        payload: dict[str, object] = self.model_dump(mode="json")
        if (
            self.ledger == LedgerProjection()
            and self.budget == BudgetProjection()
            and not self.context_packets
            and not self.context_compilations
            and not self.stop_decisions
            and self.terminal_context_packet_hash is None
            and self.terminal_context_disposition is None
        ):
            for key in (
                "ledger",
                "budget",
                "context_packets",
                "context_compilations",
                "stop_decisions",
                "terminal_context_packet_hash",
                "terminal_context_disposition",
            ):
                payload.pop(key)
        return payload

    @property
    def state_hash(self) -> str:
        return canonical_hash(self.snapshot_payload())


class RunReducer:
    version = "1.0"

    def initial(self, run_id: UUID) -> RunState:
        return RunState(run_id=run_id, version=0)

    def apply(self, state: RunState, event: StoredEvent) -> RunState:
        if event.run_id != state.run_id or event.sequence != state.version + 1:
            raise ValueError("event is not the next contiguous event for this state")
        payload = event.validated_payload()
        changes: dict[str, object] = {"version": event.sequence}
        if isinstance(payload, RunCreated):
            if state.version != 0:
                raise ValueError("RunCreated must be the first event")
            changes.update(status="CREATED", config_hash=payload.config_hash)
        elif isinstance(payload, RunStatusChanged):
            changes["status"] = payload.status
        elif isinstance(payload, ArtifactRegistered):
            changes["artifacts"] = (*state.artifacts, payload.artifact.sha256)
        elif isinstance(payload, TestValueSet):
            changes["values"] = {**state.values, payload.key: payload.value}
        elif isinstance(payload, LedgerNodeAdded):
            changes["ledger"] = EpistemicLedger().append_node(state.ledger, payload.node)
        elif isinstance(payload, LedgerEdgeAdded):
            changes["ledger"] = EpistemicLedger().append_edge(state.ledger, payload.edge)
        elif isinstance(payload, LedgerNodeRevised):
            changes["ledger"] = EpistemicLedger().revise_node(state.ledger, payload.successor)
        elif isinstance(payload, LedgerNodeStatusChanged):
            changes["ledger"] = EpistemicLedger().mark_status(
                state.ledger, payload.node_ref, payload.status
            )
        elif isinstance(payload, LedgerDependentsMarkedStale):
            expected = tuple(
                sorted(
                    (
                        envelope.affected_node_ref
                        for envelope in state.ledger.stale_envelopes
                        if payload.source_ref in envelope.changed_dependency_refs
                    ),
                    key=lambda ref: (str(ref.node_id), ref.revision),
                )
            )
            supplied = tuple(
                sorted(
                    payload.affected_refs,
                    key=lambda ref: (str(ref.node_id), ref.revision),
                )
            )
            if supplied != expected:
                raise ValueError("stale-dependent marker does not match ledger projection")
        elif isinstance(payload, LedgerContradictionResolved):
            changes["ledger"] = EpistemicLedger().resolve_contradiction(
                state.ledger, payload.resolution, payload.resolution_edge
            )
        elif isinstance(payload, BudgetRevised):
            if payload.policy_version != payload.plan.policy_version:
                raise ValueError("budget revision policy version does not match realized plan")
            changes["budget"] = BudgetAllocator().revise(
                state.budget, payload.plan, payload.policy_hash
            )
        elif isinstance(payload, BudgetAllocated):
            if state.budget.plan is not None:
                raise ValueError("budget is already allocated")
            if payload.policy_version != payload.plan.policy_version:
                raise ValueError("budget allocation policy version does not match realized plan")
            changes["budget"] = BudgetProjection(plan=payload.plan, policy_hash=payload.policy_hash)
        elif isinstance(payload, BudgetConsumed):
            changes["budget"] = BudgetMeter().consume(state.budget, payload.usage)
        elif isinstance(payload, BudgetReserved):
            changes["budget"] = BudgetMeter().reserve(state.budget, payload.reservation)
        elif isinstance(payload, BudgetReservationSettled):
            changes["budget"] = BudgetMeter().settle(
                state.budget, payload.reservation_id, payload.actual_usage
            )
        elif isinstance(payload, BudgetReservationReleased):
            changes["budget"] = BudgetMeter().release(state.budget, payload.reservation_id)
        elif isinstance(payload, ContextCompiled):
            if payload.packet.terminal_disposition is not None and (
                payload.json_artifact is None or payload.markdown_artifact is None
            ):
                raise ValueError("terminal context requires durable JSON and Markdown artifacts")
            expected_packet_hash = canonical_hash(
                payload.packet.model_dump(exclude={"packet_hash"})
            )
            if payload.packet.packet_hash != expected_packet_hash:
                raise ValueError("context packet hash does not match canonical preimage")
            changes["context_packets"] = (*state.context_packets, payload.packet)
            changes["context_compilations"] = (
                *state.context_compilations,
                ContextCompilationRecord(
                    packet_hash=payload.packet.packet_hash,
                    profile=payload.packet.profile,
                    compiler_version=payload.packet.compiler_version,
                    compression_policy_version=payload.packet.compression_policy_version,
                    applied_rule_ids=payload.packet.applied_rule_ids,
                    renderer_version=payload.renderer_version,
                    canonical_byte_size=len(canonical_json(payload.packet)),
                    json_artifact_sha256=payload.json_artifact.sha256
                    if payload.json_artifact
                    else None,
                    markdown_artifact_sha256=payload.markdown_artifact.sha256
                    if payload.markdown_artifact
                    else None,
                ),
            )
        elif isinstance(payload, StopDecisionRecorded):
            changes["stop_decisions"] = (*state.stop_decisions, payload.decision)
        elif isinstance(payload, TerminalContextAssociated):
            matching_packet = next(
                (
                    packet
                    for packet in state.context_packets
                    if packet.packet_hash == payload.packet_hash
                    and packet.profile.value == "HANDOFF"
                ),
                None,
            )
            latest_decision = state.stop_decisions[-1] if state.stop_decisions else None
            if matching_packet is None:
                raise ValueError("terminal context association requires a persisted HANDOFF packet")
            if (
                latest_decision is None
                or latest_decision.disposition.value != payload.stop_disposition
                or matching_packet.terminal_disposition != payload.stop_disposition
            ):
                raise ValueError("terminal context must match the latest recorded stop decision")
            changes["terminal_context_packet_hash"] = payload.packet_hash
            changes["terminal_context_disposition"] = payload.stop_disposition
        if (
            isinstance(payload, RunStatusChanged)
            and payload.status
            in {
                "COMPLETE",
                "PARTIAL_BUDGET",
                "BLOCKED",
                "FAILED_INVARIANT",
                "CANCELLED",
            }
            and (
                state.terminal_context_packet_hash is None
                or state.terminal_context_disposition != payload.status
            )
        ):
            raise ValueError("terminal status requires terminal context association")
        return RunState.model_validate(
            {**state.model_dump(mode="json"), **changes},
            strict=False,
        )

    def reduce(self, run_id: UUID, events: tuple[StoredEvent, ...]) -> RunState:
        state = self.initial(run_id)
        for event in events:
            state = self.apply(state, event)
        return state
