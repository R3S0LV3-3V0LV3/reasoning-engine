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
from fre.runtime.reducer import RunState


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


class Wave3Metrics(FrozenModel):
    classification_mode: str | None
    classification_fallback_used: bool
    model_call_count: int
    repair_call_count: int
    schema_validation_failures: int
    problem_item_counts: dict[str, int]
    hard_constraint_status_counts: dict[str, int]
    blocker_count: int
    representation_scores: dict[str, float]
    selected_view_count: int
    fallback_representation_count: int


def project_wave3_metrics(state: RunState) -> Wave3Metrics:
    problem = state.problem_spec
    origin_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    if problem:
        groups = (
            problem.objectives,
            problem.constraints,
            problem.decision_variables,
            problem.fixed_parameters,
            problem.unknowns,
            problem.observables,
            problem.acceptance_criteria,
            problem.assumption_items,
        )
        for group in groups:
            for item in group:
                provenance = item.provenance
                if provenance:
                    origin_counts[provenance.origin] = origin_counts.get(provenance.origin, 0) + 1
        for constraint in problem.constraints:
            if constraint.kind == "HARD":
                key = f"{constraint.verification_mode}:{constraint.verification_status}"
                status_counts[key] = status_counts.get(key, 0) + 1
    views = state.representation_plan.views if state.representation_plan else ()
    return Wave3Metrics(
        classification_mode=state.classification_record.mode
        if state.classification_record
        else None,
        classification_fallback_used=state.classification_record.fallback_used
        if state.classification_record
        else False,
        model_call_count=len(state.model_calls),
        repair_call_count=sum(record.repair_parent_key is not None for record in state.model_calls),
        schema_validation_failures=sum(
            record.status.value == "INVALID_STRUCTURED_OUTPUT" for record in state.model_calls
        ),
        problem_item_counts=origin_counts,
        hard_constraint_status_counts=status_counts,
        blocker_count=len(state.problem_blockers),
        representation_scores={view.kind: view.compatibility_score for view in views},
        selected_view_count=len(views),
        fallback_representation_count=sum(
            artifact.actual_kind.value == "TEXT_TABLE_FALLBACK"
            and artifact.requested_kind != artifact.actual_kind
            for artifact in state.representation_artifacts
        ),
    )


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
