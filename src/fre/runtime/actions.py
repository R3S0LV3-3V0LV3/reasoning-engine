"""Shared provider-neutral module execution contracts."""

from uuid import UUID

from fre.domain.common import ArtifactDescriptor, Diagnostic, FrozenModel, ObjectRef
from fre.runtime.events import UncommittedEvent


class CostEstimate(FrozenModel):
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    runtime_seconds: float = 0


class CostRecord(CostEstimate):
    monetary_cost: float | None = None


class ActionProposal(FrozenModel):
    action_id: UUID
    module_id: str
    operation: str
    input_refs: tuple[ObjectRef, ...] = ()
    prerequisites: tuple[ObjectRef, ...] = ()
    mandatory: bool = False
    blocking: bool = False
    expected_decision_impact: float | None = None
    estimated_cost: CostEstimate
    execution_capabilities: frozenset[str] = frozenset()
    reason: str


class ModuleResult(FrozenModel):
    action_id: UUID
    events: tuple[UncommittedEvent, ...] = ()
    artifacts: tuple[ArtifactDescriptor, ...] = ()
    actual_cost: CostRecord
    warnings: tuple[Diagnostic, ...] = ()
    proposed_followups: tuple[ActionProposal, ...] = ()
