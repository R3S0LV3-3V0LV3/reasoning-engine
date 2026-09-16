"""Deterministic Wave 2 observability derived from canonical projections."""

from fre.domain.budget import (
    RESOURCE_NAMES,
    BudgetBurnRate,
    BudgetProjection,
    BudgetRemaining,
    ResourceVector,
)
from fre.domain.common import FrozenModel, canonical_json
from fre.domain.context import ContextCompilationRecord, ContextDeltaPacket, ContextPacket
from fre.domain.ledger import EpistemicStatus, LedgerProjection
from fre.domain.stop import StopDecision
from fre.modules.m09_ledger import EpistemicLedger


class Wave2Metrics(FrozenModel):
    ledger_node_count: int
    ledger_edge_count: int
    epistemic_status_counts: dict[str, int]
    budget_remaining: BudgetRemaining | None
    budget_allocated: ResourceVector | None
    budget_committed: ResourceVector | None
    budget_reserved: ResourceVector | None
    budget_policy_version: str | None
    budget_policy_hash: str | None
    budget_burn_rate: BudgetBurnRate | None
    latest_context_profile: str | None
    latest_context_hash: str | None
    latest_context_size: int | None
    latest_context_artifact_hash: str | None
    latest_compression_policy_version: str | None
    latest_applied_compression_rules: tuple[str, ...]
    latest_delta_hash: str | None
    latest_delta_size: int | None
    latest_stop_disposition: str | None
    latest_stop_reason_codes: tuple[str, ...]


def project_metrics(
    ledger: LedgerProjection,
    *,
    budget: BudgetProjection | None = None,
    budget_remaining: BudgetRemaining | None = None,
    burn_rate: BudgetBurnRate | None = None,
    context: ContextPacket | None = None,
    compilation: ContextCompilationRecord | None = None,
    delta: ContextDeltaPacket | None = None,
    stop: StopDecision | None = None,
) -> Wave2Metrics:
    service = EpistemicLedger()
    counts = {status.value: 0 for status in EpistemicStatus}
    for node in ledger.nodes:
        counts[service.effective_status(ledger, node.ref).value] += 1
    reserved = ResourceVector()
    if budget is not None:
        reserved = ResourceVector(
            **{
                name: sum(getattr(item.resources, name) for item in budget.reservations)
                for name in RESOURCE_NAMES
            }
        )
    return Wave2Metrics(
        ledger_node_count=len(ledger.nodes),
        ledger_edge_count=len(ledger.edges),
        epistemic_status_counts=counts,
        budget_remaining=budget_remaining,
        budget_allocated=budget.plan.limits.resources()
        if budget is not None and budget.plan is not None
        else None,
        budget_committed=budget.committed if budget is not None else None,
        budget_reserved=reserved if budget is not None else None,
        budget_policy_version=budget.plan.policy_version
        if budget is not None and budget.plan is not None
        else None,
        budget_policy_hash=budget.policy_hash if budget is not None else None,
        budget_burn_rate=burn_rate,
        latest_context_profile=context.profile if context else None,
        latest_context_hash=context.packet_hash if context else None,
        latest_context_size=len(canonical_json(context)) if context else None,
        latest_context_artifact_hash=compilation.json_artifact_sha256 if compilation else None,
        latest_compression_policy_version=context.compression_policy_version if context else None,
        latest_applied_compression_rules=context.applied_rule_ids if context else (),
        latest_delta_hash=delta.delta_hash if delta else None,
        latest_delta_size=len(canonical_json(delta)) if delta else None,
        latest_stop_disposition=stop.disposition if stop else None,
        latest_stop_reason_codes=tuple(code.value for code in stop.reason_codes) if stop else (),
    )
