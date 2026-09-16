"""Deterministic context packet, compression, and delta contracts."""

from enum import StrEnum
from uuid import UUID

from pydantic import Field

from fre.domain.budget import BudgetRemaining
from fre.domain.common import FrozenModel, JsonValue
from fre.domain.ledger import LedgerNodeRef


class Availability(StrEnum):
    AVAILABLE = "AVAILABLE"
    NOT_PRODUCED = "NOT_PRODUCED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class CompilerProfile(StrEnum):
    FULL = "FULL"
    STANDARD = "STANDARD"
    HANDOFF = "HANDOFF"


class ContextCompressionPolicy(FrozenModel):
    compression_policy_version: str = "1.0"
    ordered_rule_ids: tuple[str, ...] = ("C01", "C02", "C03", "C04", "C05")
    mandatory_retention_classes: tuple[str, ...] = (
        "hard_constraints",
        "stale_items",
        "unresolved_blockers",
        "rejection_reasons",
        "decision_provenance",
        "budget_remaining",
        "next_or_terminal_action",
    )


class ContextCompilationRequest(FrozenModel):
    profile: CompilerProfile
    size_target: int = Field(ge=1)
    terminal: bool = False


class ContextItem(FrozenModel):
    ref: LedgerNodeRef
    kind: str
    status: str
    content: JsonValue
    provenance_refs: tuple[str, ...] = ()


class RejectedItem(FrozenModel):
    ref: str
    reason: str
    provenance_refs: tuple[str, ...] = ()


class ContextPacket(FrozenModel):
    run_id: UUID
    snapshot_version: int
    profile: CompilerProfile
    compiler_version: str
    compression_policy_version: str
    applied_rule_ids: tuple[str, ...]
    objective: JsonValue | None = None
    output_contract: JsonValue | None = None
    hard_constraints: tuple[JsonValue, ...] = ()
    ledger_items: tuple[ContextItem, ...] = ()
    rejected_items: tuple[RejectedItem, ...] = ()
    unresolved_blockers: tuple[str, ...] = ()
    budget_remaining: BudgetRemaining
    candidate_projection_availability: Availability = Availability.NOT_PRODUCED
    decision_projection_availability: Availability = Availability.NOT_PRODUCED
    acquisition_projection_availability: Availability = Availability.NOT_PRODUCED
    next_action: str | None = None
    terminal_disposition: str | None = None
    packet_hash: str


class ContextCompilationResult(FrozenModel):
    packet: ContextPacket
    canonical_bytes: bytes
    markdown: str
    canonical_byte_size: int


class ContextOverflowDiagnostic(FrozenModel):
    required_bytes: int
    target_bytes: int
    mandatory_classes: tuple[str, ...]


class DeltaOperationType(StrEnum):
    ADD = "ADD"
    REMOVE = "REMOVE"
    REPLACE = "REPLACE"


class DeltaOperation(FrozenModel):
    operation: DeltaOperationType
    path: str
    value: JsonValue | None = None


class ContextDeltaPacket(FrozenModel):
    base_packet_hash: str
    target_packet_hash: str
    compiler_version: str
    compression_policy_version: str
    profile: CompilerProfile
    operations: tuple[DeltaOperation, ...]
    delta_hash: str


class ContextError(ValueError):
    pass


class ContextOverflow(ContextError):
    def __init__(self, diagnostic: ContextOverflowDiagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__("mandatory context exceeds the configured byte target")


class InvalidCompilerProfile(ContextError):
    pass


class InvalidContextDelta(ContextError):
    pass


class ContextDeltaBaseMismatch(InvalidContextDelta):
    pass
