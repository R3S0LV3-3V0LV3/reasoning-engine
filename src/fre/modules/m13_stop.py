"""M13 deterministic confidence-bounded marginal-value stop controller."""

from fre.domain.context import CompilerProfile, ContextCompilationRequest
from fre.domain.stop import (
    AcceptanceStatus,
    MissingRequiredContext,
    StopDecision,
    StopDisposition,
    StopInputs,
    StopPolicy,
    StopReasonCode,
    ValidationStatus,
)

REASON_TEXT = {
    StopReasonCode.CANCELLED: "Cancellation was explicitly requested.",
    StopReasonCode.INVARIANT_FAILURE: "An unrecoverable invariant failed.",
    StopReasonCode.HARD_BUDGET_EXHAUSTED: "A hard budget ceiling prevents required continuation.",
    StopReasonCode.UNRESOLVABLE_BLOCKER: "A required blocker has no permitted resolution.",
    StopReasonCode.MANDATORY_ACTION: "A mandatory action remains.",
    StopReasonCode.MANDATORY_VALIDATION: "Mandatory validation remains incomplete.",
    StopReasonCode.RESOLVABLE_UNKNOWN: "A required unknown has an executable resolution.",
    StopReasonCode.ACCEPTANCE_FAILED: "A required acceptance criterion is not satisfied.",
    StopReasonCode.ACCEPTANCE_COMPLETE: "Acceptance and validation are complete.",
    StopReasonCode.OPTIONAL_WORK_VALUABLE: "The next optional action is definitely valuable.",
    StopReasonCode.OPTIONAL_WORK_UNCERTAIN: "The next action's value and cost intervals overlap.",
    StopReasonCode.OPTIONAL_WORK_NOT_VALUABLE: (
        "The next optional action is definitely not valuable."
    ),
    StopReasonCode.NO_EXECUTABLE_WORK: "No executable work remains.",
}

EPISTEMIC_REASONS = {
    StopReasonCode.UNRESOLVABLE_BLOCKER,
    StopReasonCode.RESOLVABLE_UNKNOWN,
}


class StopController:
    def evaluate(self, inputs: StopInputs, policy: StopPolicy) -> StopDecision:
        disposition: StopDisposition
        reasons: set[StopReasonCode] = set()
        exhausted = inputs.budget.resources.iterations == 0
        if inputs.cancelled:
            disposition, reasons = StopDisposition.CANCELLED, {StopReasonCode.CANCELLED}
        elif inputs.invariant_failure:
            disposition, reasons = (
                StopDisposition.FAILED_INVARIANT,
                {StopReasonCode.INVARIANT_FAILURE},
            )
        elif (
            inputs.acceptance is AcceptanceStatus.SATISFIED
            and inputs.validation in {ValidationStatus.COMPLETE, ValidationStatus.NOT_APPLICABLE}
            and not inputs.required_work
            and inputs.stable
        ):
            disposition, reasons = StopDisposition.COMPLETE, {StopReasonCode.ACCEPTANCE_COMPLETE}
        elif exhausted and inputs.required_work:
            disposition, reasons = (
                StopDisposition.PARTIAL_BUDGET,
                {StopReasonCode.HARD_BUDGET_EXHAUSTED},
            )
        elif inputs.blocker_required and not inputs.blocker_resolvable:
            disposition, reasons = StopDisposition.BLOCKED, {StopReasonCode.UNRESOLVABLE_BLOCKER}
        elif inputs.mandatory_action:
            disposition, reasons = StopDisposition.CONTINUE, {StopReasonCode.MANDATORY_ACTION}
        elif inputs.validation is ValidationStatus.REQUIRED:
            disposition, reasons = StopDisposition.CONTINUE, {StopReasonCode.MANDATORY_VALIDATION}
        elif inputs.blocker_resolvable:
            disposition, reasons = StopDisposition.CONTINUE, {StopReasonCode.RESOLVABLE_UNKNOWN}
        elif inputs.acceptance is AcceptanceStatus.FAILED:
            if exhausted:
                disposition, reasons = (
                    StopDisposition.PARTIAL_BUDGET,
                    {StopReasonCode.HARD_BUDGET_EXHAUSTED, StopReasonCode.ACCEPTANCE_FAILED},
                )
            elif inputs.executable_work:
                disposition, reasons = StopDisposition.CONTINUE, {StopReasonCode.ACCEPTANCE_FAILED}
            else:
                disposition, reasons = (
                    StopDisposition.BLOCKED,
                    {StopReasonCode.ACCEPTANCE_FAILED, StopReasonCode.NO_EXECUTABLE_WORK},
                )
        elif inputs.value is not None and inputs.cost is not None and inputs.executable_work:
            if inputs.value.lower >= inputs.cost.upper + policy.stop_threshold:
                disposition, reasons = (
                    StopDisposition.CONTINUE,
                    {StopReasonCode.OPTIONAL_WORK_VALUABLE},
                )
            elif inputs.value.upper < inputs.cost.lower + policy.stop_threshold:
                disposition, reasons = (
                    StopDisposition.COMPLETE,
                    {StopReasonCode.OPTIONAL_WORK_NOT_VALUABLE},
                )
            elif policy.continue_on_overlap:
                disposition, reasons = (
                    StopDisposition.CONTINUE,
                    {StopReasonCode.OPTIONAL_WORK_UNCERTAIN},
                )
            else:
                disposition, reasons = (
                    StopDisposition.COMPLETE,
                    {StopReasonCode.OPTIONAL_WORK_UNCERTAIN},
                )
        elif inputs.executable_work:
            disposition, reasons = (
                StopDisposition.CONTINUE,
                {StopReasonCode.OPTIONAL_WORK_UNCERTAIN},
            )
        elif inputs.acceptance is AcceptanceStatus.SATISFIED and inputs.stable:
            disposition, reasons = StopDisposition.COMPLETE, {StopReasonCode.ACCEPTANCE_COMPLETE}
        else:
            disposition, reasons = StopDisposition.BLOCKED, {StopReasonCode.NO_EXECUTABLE_WORK}
        if reasons & EPISTEMIC_REASONS and not inputs.epistemic_trigger_refs:
            raise MissingRequiredContext(
                "epistemic stop reasons require concrete LedgerNodeRef provenance"
            )
        ordered = tuple(sorted(reasons, key=lambda item: item.value))
        terminal = disposition is not StopDisposition.CONTINUE
        return StopDecision(
            disposition=disposition,
            policy_version=policy.version,
            reason_codes=ordered,
            reasons=tuple(REASON_TEXT[item] for item in ordered),
            ledger_trigger_refs=tuple(
                sorted(
                    inputs.epistemic_trigger_refs, key=lambda ref: (str(ref.node_id), ref.revision)
                )
            ),
            budget_projection_hash=inputs.budget.projection_hash,
            burn_rate_projection_hash=inputs.burn_rate.projection_hash
            if inputs.burn_rate
            else None,
            marginal_value_method=inputs.value.method_version if inputs.value else None,
            marginal_value_source_refs=tuple(sorted(inputs.value.source_refs))
            if inputs.value
            else (),
            marginal_value_ledger_refs=tuple(
                sorted(
                    inputs.value.ledger_refs,
                    key=lambda ref: (str(ref.node_id), ref.revision),
                )
            )
            if inputs.value
            else (),
            cost_method=inputs.cost.method_version if inputs.cost else None,
            cost_source_refs=tuple(sorted(inputs.cost.source_refs)) if inputs.cost else (),
            context_request=(
                ContextCompilationRequest(
                    profile=CompilerProfile.HANDOFF, size_target=32768, terminal=True
                )
                if terminal
                else None
            ),
        )
