import pytest
from pydantic import ValidationError

from fre.domain.evaluation import ConstraintStatus
from fre.domain.problem import ConstraintSpec


@pytest.mark.unit
def test_semantic_hard_constraint_remains_hard_and_unknown_is_distinct() -> None:
    constraint = ConstraintSpec(
        id="safety",
        description="Safe under review",
        kind="HARD",
        verification_mode="HUMAN",
    )
    assert constraint.kind == "HARD"
    assert {ConstraintStatus.UNKNOWN.value, ConstraintStatus.FAIL.value} == {"UNKNOWN", "FAIL"}
    with pytest.raises(ValidationError):
        ConstraintSpec.model_validate(
            {"id": "bad", "description": "bad", "kind": "HARD", "verification_mode": "AUTOMATIC"}
        )


@pytest.mark.unit
def test_domain_models_are_immutable() -> None:
    constraint = ConstraintSpec(
        id="x", description="x", kind="HARD", verification_mode="UNAVAILABLE"
    )
    field = "kind"
    with pytest.raises(ValidationError):
        setattr(constraint, field, "SOFT")
