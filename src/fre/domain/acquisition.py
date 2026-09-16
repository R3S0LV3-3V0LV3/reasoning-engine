"""Information-acquisition contracts."""

from enum import StrEnum

from fre.domain.common import ArtifactRef, FrozenModel


class SideEffectLevel(StrEnum):
    NONE = "NONE"
    READ_ONLY = "READ_ONLY"
    EXTERNAL_WRITE = "EXTERNAL_WRITE"


class AcquisitionActionType(StrEnum):
    RETRIEVE = "RETRIEVE"
    WEB_RESEARCH = "WEB_RESEARCH"
    DATABASE_QUERY = "DATABASE_QUERY"
    CODE_EXPERIMENT = "CODE_EXPERIMENT"
    SIMULATION = "SIMULATION"
    FILE_ANALYSIS = "FILE_ANALYSIS"
    HUMAN_QUERY = "HUMAN_QUERY"
    EXTERNAL_TOOL = "EXTERNAL_TOOL"


class AcquisitionAction(FrozenModel):
    action_type: AcquisitionActionType
    target_unknown_ids: tuple[str, ...]
    expected_decision_impact: float
    estimated_cost: float


class ExperimentPlan(FrozenModel):
    question: str
    target_unknown_ids: tuple[str, ...]
    method: str
    inputs: tuple[ArtifactRef, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    validator_ref: str
    stopping_rule: str
    side_effect_level: SideEffectLevel
