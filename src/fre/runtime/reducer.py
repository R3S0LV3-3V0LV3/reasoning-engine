"""Pure reducer protocol and foundational run reducer."""

from typing import Protocol, TypeVar
from uuid import UUID

from pydantic import Field

from fre.domain.budget import BudgetProjection
from fre.domain.common import FrozenModel, JsonValue, canonical_hash
from fre.domain.context import ContextPacket
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
    stop_decisions: tuple[StopDecision, ...] = ()
    terminal_context_packet_hash: str | None = None

    def snapshot_payload(self) -> dict[str, object]:
        """Return the hash payload, retaining Wave 1 shape for untouched streams."""
        payload: dict[str, object] = self.model_dump(mode="json")
        if (
            self.ledger == LedgerProjection()
            and self.budget == BudgetProjection()
            and not self.context_packets
            and not self.stop_decisions
            and self.terminal_context_packet_hash is None
        ):
            for key in (
                "ledger",
                "budget",
                "context_packets",
                "stop_decisions",
                "terminal_context_packet_hash",
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
        elif isinstance(payload, LedgerContradictionResolved):
            changes["ledger"] = EpistemicLedger().resolve_contradiction(
                state.ledger, payload.resolution, payload.resolution_edge
            )
        elif isinstance(payload, BudgetRevised):
            changes["budget"] = BudgetAllocator().revise(
                state.budget, payload.plan, payload.policy_hash
            )
        elif isinstance(payload, BudgetAllocated):
            if state.budget.plan is not None:
                raise ValueError("budget is already allocated")
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
            changes["context_packets"] = (*state.context_packets, payload.packet)
        elif isinstance(payload, StopDecisionRecorded):
            changes["stop_decisions"] = (*state.stop_decisions, payload.decision)
        elif isinstance(payload, TerminalContextAssociated):
            if not any(
                packet.packet_hash == payload.packet_hash and packet.profile.value == "HANDOFF"
                for packet in state.context_packets
            ):
                raise ValueError("terminal context association requires a persisted HANDOFF packet")
            changes["terminal_context_packet_hash"] = payload.packet_hash
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
            and state.terminal_context_packet_hash is None
        ):
            raise ValueError("terminal status requires terminal context association")
        return state.model_copy(update=changes)

    def reduce(self, run_id: UUID, events: tuple[StoredEvent, ...]) -> RunState:
        state = self.initial(run_id)
        for event in events:
            state = self.apply(state, event)
        return state
