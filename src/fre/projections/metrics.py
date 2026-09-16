"""Deterministic Wave 2 observability derived from canonical projections."""

from fre.domain.budget import BudgetBurnRate, BudgetRemaining
from fre.domain.common import FrozenModel
from fre.domain.context import ContextPacket
from fre.domain.ledger import EpistemicStatus, LedgerProjection
from fre.domain.stop import StopDecision
from fre.modules.m09_ledger import EpistemicLedger


class Wave2Metrics(FrozenModel):
    ledger_node_count: int
    ledger_edge_count: int
    epistemic_status_counts: dict[str, int]
    budget_remaining: BudgetRemaining | None
    budget_burn_rate: BudgetBurnRate | None
    latest_context_profile: str | None
    latest_context_hash: str | None
    latest_context_size: int | None
    latest_stop_disposition: str | None
    latest_stop_reason_codes: tuple[str, ...]


def project_metrics(
    ledger: LedgerProjection,
    *,
    budget_remaining: BudgetRemaining | None = None,
    burn_rate: BudgetBurnRate | None = None,
    context: ContextPacket | None = None,
    stop: StopDecision | None = None,
) -> Wave2Metrics:
    service = EpistemicLedger()
    counts = {status.value: 0 for status in EpistemicStatus}
    for node in ledger.nodes:
        counts[service.effective_status(ledger, node.ref).value] += 1
    return Wave2Metrics(
        ledger_node_count=len(ledger.nodes),
        ledger_edge_count=len(ledger.edges),
        epistemic_status_counts=counts,
        budget_remaining=budget_remaining,
        budget_burn_rate=burn_rate,
        latest_context_profile=context.profile if context else None,
        latest_context_hash=context.packet_hash if context else None,
        latest_context_size=None,
        latest_stop_disposition=stop.disposition if stop else None,
        latest_stop_reason_codes=tuple(code.value for code in stop.reason_codes) if stop else (),
    )
