"""Stop decision contract (policy arrives in Wave 2)."""

from enum import StrEnum

from fre.domain.common import ArtifactRef, FrozenModel, ObjectRef


class StopDisposition(StrEnum):
    CONTINUE = "CONTINUE"
    COMPLETE = "COMPLETE"
    PARTIAL_BUDGET = "PARTIAL_BUDGET"
    BLOCKED = "BLOCKED"
    FAILED_INVARIANT = "FAILED_INVARIANT"
    CANCELLED = "CANCELLED"


class StopDecision(FrozenModel):
    disposition: StopDisposition
    reasons: tuple[str, ...]
    triggering_refs: tuple[ObjectRef, ...] = ()
    final_context_packet_ref: ArtifactRef | None = None
