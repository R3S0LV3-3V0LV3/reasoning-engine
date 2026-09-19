#!/usr/bin/env python3
"""Regenerate `docs/wave3-requirements-matrix.md` from the row data below.

C10 (defect F14) remediation: the previous matrix mapped all 110 numbered
acceptance rows (plus 5 cross-cutting rows) onto the SAME four generic cells
("Wave 3 unit/property/integration gate" / "Typed contracts" / "See event
mapping" / "Implemented") -- a many-to-one mapping that proved nothing about
any individually named semantic. Every row below instead names: the real
implementation symbol/file, a real positive test (`file.py::test_name`,
verified by `scripts/verify_requirements_matrix.py` to actually exist and
collect), a negative/adversarial test where one exists (many genuinely have
none -- e.g. a pure structural/typing acceptance -- and this is recorded
honestly as "N/A", never papered over with an unrelated test), replay/
property/golden evidence where applicable, and the exact commit SHA of the
PR whose remediation round actually introduced/decisively proved that row
(C02-C09; see the header comment in `docs/wave3-review-register.md` for the
full SHA list). Multiple rows legitimately citing the SAME test are not
themselves the defect this closes -- the defect was rows whose cited
"evidence" did not test the named semantic at all. Where one parametrized
test (e.g. `test_m01_to_m02_axes_are_monotone`) genuinely proves several
adjacent rows (one per axis), that is recorded honestly, not hidden.

Run `python scripts/generate_requirements_matrix.py` after editing ROWS
below, then `python scripts/verify_requirements_matrix.py` to check
referential integrity before committing.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

# Evidence commit SHAs for each corrective child PR on
# codex/implement-wave-3-of-frontier-reasoning-engine, as merged.
SHA = {
    "C02": "d7a192bcea9de8f98c81d3364220a0f311748527",
    "C03": "a1f3a17d531c067db5ab571dcd3d9c220da578ea",
    "C04": "e37d59dc2ee84ecdf8e617b255dc3256b1c4c58b",
    "C05": "c68a86eedcbcd76fa46cb7169052f1b67cea0489",
    "C06": "9c6571dcb4d51821e2e20e5015e63948c71ee030",
    "C07": "fd486ddf071e39c0681a213851a9ecab7c7c955a",
    "C08": "e142274f9e0daea6adbf234c630b5c60968b09d7",
    "C09": "7402ac84cadb4f7c162a73d2b545ef1c597d25d0",
    # C10 (this PR, #25) is the acceptance-evidence/matrix-infrastructure PR
    # itself (the matrix, the referential-integrity checker, and golden
    # fixtures A-L) rather than a new implementation round -- no row below
    # currently cites C10 as its OWN implementation evidence (every row's
    # implementation symbol was introduced by C02-C09; C10 only adds the
    # machinery that verifies those rows). C10's own evidence SHA (the
    # squash-merge commit of this PR onto the integration branch) therefore
    # cannot be known until after this PR merges, and is deliberately NOT
    # recorded here as an unresolved placeholder -- see the merged PR #25 on
    # `codex/implement-wave-3-of-frontier-reasoning-engine` (and this file's
    # own git blame at that merge) for that SHA. If a future row ever needs
    # to cite C10 itself as implementation evidence, add it as a new SHA
    # entry keyed to that real, by-then-known merge commit -- never as a
    # bare "PENDING".
    "C10": "see PR #25 merge commit on integration branch (no row cites C10 as its own evidence)",
}

NA = "N/A"

# Status vocabulary matches
# FRE_WAVE3_C01_C10_EXECUTION_COMPLETION_AND_VALIDATION_REGISTER.md section 0
# ("implemented, locally tested, independently reviewed, merged, and frozen
# states"). Every row's cited SHA (C02-C09) is a commit that has completed
# its own independent review, remediation, and squash-merge onto the
# integration branch -- "frozen" is deliberately NOT used here, since
# freezing Waves 1-3 is Phase 9's decision, not any individual child PR's.
_STATUS = "independently reviewed, remediated, and merged"


class MatrixRow(NamedTuple):
    """One requirements-matrix row.

    EU-45 (Wave 3 post-freeze cleanup, C10): replaces the previous plain
    7-tuple so a mis-ordered row fails fast (wrong keyword name / mypy
    field-order error) instead of silently rendering into the wrong column
    or crashing with an unpacking `ValueError` deep inside `render()`.
    """

    row_id: str
    requirement: str
    impl: str
    positive_test: str
    negative_test: str
    evidence: str
    sha_key: str


ROWS: list[MatrixRow] = [
    MatrixRow(
        row_id="W3-001",
        requirement="deterministic explicit metadata extraction",
        impl="fre.modules.source_anchors.validate_source_anchor",
        positive_test="tests/unit/test_source_anchors_m01.py::test_m01_retains_valid_anchors_orders_evidence_and_preserves_explicit_values",
        negative_test="tests/unit/test_source_anchors_m01.py::test_m01_rejects_any_invalid_model_claimed_anchor",
        evidence="golden B (tests/golden/test_golden_fixtures.py)",
        sha_key="C05",
    ),
    MatrixRow(
        row_id="W3-002",
        requirement="complete valid TaskSignature",
        impl="fre.domain.task.TaskSignature",
        positive_test="tests/unit/test_m01_classification_contract.py::test_all_eight_axes_carry_full_provenance",
        negative_test="tests/unit/test_m01_classification_contract.py::test_missing_rationale_on_material_dimension_is_blocked",
        evidence="golden B/I/J",
        sha_key="C05",
    ),
    MatrixRow(
        row_id="W3-003",
        requirement="reversibility to irreversibility mapping",
        impl="fre.modules.m01_classifier (reversibility axis)",
        positive_test="tests/property/test_wave3_properties.py::test_reversibility_transform_is_involution",
        negative_test=NA,
        evidence="property (hypothesis)",
        sha_key="C05",
    ),
    MatrixRow(
        row_id="W3-004",
        requirement="consequence harder-is-higher",
        impl="fre.modules.m01_classifier (M01->M02 axis monotonicity)",
        positive_test="tests/property/test_wave3_properties.py::test_m01_to_m02_axes_are_monotone[consequence]",
        negative_test=NA,
        evidence="property (hypothesis)",
        sha_key="C05",
    ),
    MatrixRow(
        row_id="W3-005",
        requirement="irreversibility harder-is-higher",
        impl="fre.modules.m01_classifier (M01->M02 axis monotonicity)",
        positive_test="tests/property/test_wave3_properties.py::test_m01_to_m02_axes_are_monotone[reversibility]",
        negative_test=NA,
        evidence="property (hypothesis)",
        sha_key="C05",
    ),
    MatrixRow(
        row_id="W3-006",
        requirement="ambiguity harder-is-higher",
        impl="fre.modules.m01_classifier (M01->M02 axis monotonicity)",
        positive_test="tests/property/test_wave3_properties.py::test_m01_to_m02_axes_are_monotone[ambiguity]",
        negative_test=NA,
        evidence="property (hypothesis)",
        sha_key="C05",
    ),
    MatrixRow(
        row_id="W3-007",
        requirement="evidence scarcity harder-is-higher",
        impl="fre.modules.m01_classifier (M01->M02 axis monotonicity)",
        positive_test="tests/property/test_wave3_properties.py::test_m01_to_m02_axes_are_monotone[evidence_scarcity]",
        negative_test=NA,
        evidence="property (hypothesis)",
        sha_key="C05",
    ),
    MatrixRow(
        row_id="W3-008",
        requirement="search-space monotonicity",
        impl="fre.modules.m01_classifier (M01->M02 axis monotonicity)",
        positive_test="tests/property/test_wave3_properties.py::test_m01_to_m02_axes_are_monotone[search_space]",
        negative_test=NA,
        evidence="property (hypothesis)",
        sha_key="C05",
    ),
    MatrixRow(
        row_id="W3-009",
        requirement="deterministic risk floor",
        impl="fre.modules.m01_classifier.TaskClassifier (floor overrides)",
        positive_test="tests/unit/test_m01_classification_contract.py::test_ambiguity_floor_overrides_a_confident_low_self_report",
        negative_test="tests/unit/test_m01_classification_contract.py::test_evidence_scarcity_floor_overrides_a_confident_low_self_report",
        evidence=NA,
        sha_key="C05",
    ),
    MatrixRow(
        row_id="W3-010",
        requirement="model cannot lower risk floor",
        impl="fre.domain.task.FloorOverrideRecord",
        positive_test="tests/unit/test_c05_remediation.py::test_floor_override_record_with_wrong_previous_or_new_floor_is_rejected",
        negative_test=NA,
        evidence=NA,
        sha_key="C05",
    ),
    MatrixRow(
        row_id="W3-011",
        requirement="weighted score cannot lower floor",
        impl="fre.modules.m01_classifier.TaskClassifier (floor no-op guard)",
        positive_test="tests/unit/test_m01_classification_contract.py::test_ambiguity_and_evidence_scarcity_floors_are_no_ops_when_signals_are_absent",
        negative_test=NA,
        evidence=NA,
        sha_key="C05",
    ),
    MatrixRow(
        row_id="W3-012",
        requirement="low-confidence escalation",
        impl="fre.modules.m01_classifier.TaskClassifier (horizon escalation)",
        positive_test="tests/unit/test_m01_classification_contract.py::test_low_horizon_confidence_escalates_to_conservative_default",
        negative_test=NA,
        evidence=NA,
        sha_key="C05",
    ),
    MatrixRow(
        row_id="W3-013",
        requirement="high-confidence retained",
        impl="fre.modules.m01_classifier.TaskClassifier",
        positive_test="tests/unit/test_m01_classification_contract.py::test_all_eight_axes_carry_full_provenance",
        negative_test=NA,
        evidence=NA,
        sha_key="C05",
    ),
    MatrixRow(
        row_id="W3-014",
        requirement="classification refs resolve",
        impl="fre.modules.source_anchors.validate_source_anchor",
        positive_test="tests/unit/test_source_anchors_m01.py::test_m01_retains_valid_anchors_orders_evidence_and_preserves_explicit_values",
        negative_test="tests/unit/test_source_anchors_m01.py::test_m01_rejects_model_anchor_for_unattached_artifact",
        evidence=NA,
        sha_key="C05",
    ),
    MatrixRow(
        row_id="W3-015",
        requirement="deterministic precedence",
        impl="fre.modules.m01_classifier.TaskClassifier (fallback precedence)",
        positive_test="tests/unit/test_m01_classification_contract.py::test_horizon_low_confidence_fallback_is_independent_of_no_proposal_fallback",
        negative_test=NA,
        evidence=NA,
        sha_key="C05",
    ),
    MatrixRow(
        row_id="W3-016",
        requirement="valid structured output accepted",
        impl="fre.semantic_runtime.SemanticModelRuntime._invoke",
        positive_test="tests/unit/test_semantic_runtime.py::test_normal_usage_settlement_preserves_accepted_charge_behavior",
        negative_test=NA,
        evidence="golden B/E",
        sha_key="C04",
    ),
    MatrixRow(
        row_id="W3-017",
        requirement="malformed output rejected",
        impl="fre.semantic_runtime.SemanticModelRuntime._invoke (INVALID_STRUCTURED_OUTPUT)",
        positive_test="tests/golden/test_golden_fixtures.py::test_golden_d_schema_invalid_output_falls_back_after_exhausted_repair",
        negative_test=NA,
        evidence="golden D",
        sha_key="C04",
    ),
    MatrixRow(
        row_id="W3-018",
        requirement="at most one repair",
        impl="fre.semantic_runtime.SemanticModelRuntime (maximum_repair_attempts)",
        positive_test="tests/unit/test_semantic_runtime.py::test_zero_repairs_and_repair_budget_refusal_preserve_initial_record",
        negative_test=NA,
        evidence="golden D",
        sha_key="C04",
    ),
    MatrixRow(
        row_id="W3-019",
        requirement="valid repair revalidated",
        impl="fre.semantic_runtime.SemanticModelRuntime (repair path)",
        positive_test="tests/unit/test_semantic_runtime.py::test_repair_executes_distinct_render_and_validated_artifact_reuses",
        negative_test="tests/integration/test_wave3_gate.py::test_coordinator_schema_repair_success_path_reuses_identical_repaired_proposal",
        evidence=NA,
        sha_key="C04",
    ),
    MatrixRow(
        row_id="W3-020",
        requirement="invalid repair falls back",
        impl="fre.semantic_runtime.SemanticModelRuntime (repair exhausted)",
        positive_test="tests/integration/test_wave3_gate.py::test_coordinator_schema_invalid_path_falls_back_after_exhausted_repair",
        negative_test=NA,
        evidence="golden D",
        sha_key="C04",
    ),
    MatrixRow(
        row_id="W3-021",
        requirement="unavailable fallback",
        impl="fre.modules.m01_classifier.TaskClassifier (allow_model=False path)",
        positive_test="tests/integration/test_wave3_gate.py::test_coordinator_zero_call_fallback_is_first_class",
        negative_test=NA,
        evidence="golden A",
        sha_key="C09",
    ),
    MatrixRow(
        row_id="W3-022",
        requirement="fallback preserves explicit metadata",
        impl="fre.modules.m01_classifier.TaskClassifier.classify (fallback)",
        positive_test="tests/unit/test_source_anchors_m01.py::test_m01_retains_valid_anchors_orders_evidence_and_preserves_explicit_values",
        negative_test=NA,
        evidence="golden A",
        sha_key="C05",
    ),
    MatrixRow(
        row_id="W3-023",
        requirement="fallback confidence unknown",
        impl="fre.domain.task.ClassificationRecord (fallback_used)",
        positive_test="tests/integration/test_wave3_gate.py::test_coordinator_zero_call_fallback_is_first_class",
        negative_test=NA,
        evidence="golden A",
        sha_key="C09",
    ),
    MatrixRow(
        row_id="W3-024",
        requirement="stored result determinism",
        impl="fre.semantic_runtime.SemanticModelRuntime (idempotency)",
        positive_test="tests/unit/test_semantic_runtime.py::test_concurrent_same_identity_cannot_oversubscribe_or_duplicate_call",
        negative_test=NA,
        evidence=NA,
        sha_key="C04",
    ),
    MatrixRow(
        row_id="W3-025",
        requirement="classification replay",
        impl="fre.engine.FrontierReasoningEngine.replay",
        positive_test="tests/integration/test_wave3_gate.py::test_coordinator_normal_semantic_path_and_replay",
        negative_test=NA,
        evidence="golden B/K (3-way replay equality)",
        sha_key="C09",
    ),
    MatrixRow(
        row_id="W3-026",
        requirement="M01-M02 monotonicity",
        impl="fre.modules.m02_budget.BudgetAllocator.revise",
        positive_test="tests/property/test_wave3_properties.py::test_m01_to_m02_axes_are_monotone",
        negative_test=NA,
        evidence="property (hypothesis)",
        sha_key="C05",
    ),
    MatrixRow(
        row_id="W3-027",
        requirement="harder signature cannot lower tier",
        impl="fre.modules.m02_budget.BudgetAllocator.revise",
        positive_test="tests/unit/test_m01_classification_contract.py::test_bootstrap_to_final_revision_cannot_fall_below_committed_plus_reserved",
        negative_test=NA,
        evidence=NA,
        sha_key="C05",
    ),
    MatrixRow(
        row_id="W3-028",
        requirement="lower confidence cannot lower tier",
        impl="fre.modules.m01_classifier.TaskClassifier (confidence threshold)",
        positive_test="tests/unit/test_m01_classification_contract.py::test_preliminary_signature_is_distinct_from_and_never_more_permissive_than_final",
        negative_test=NA,
        evidence=NA,
        sha_key="C05",
    ),
    MatrixRow(
        row_id="W3-029",
        requirement="explicit input anchored",
        impl="fre.modules.m03_formaliser.ProblemFormaliser (EXPLICIT_INPUT anchor requirement)",
        positive_test="tests/unit/test_c06_remediation.py::test_reducer_rejects_explicit_input_node_with_no_anchors_at_all",
        negative_test=NA,
        evidence=NA,
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-030",
        requirement="unanchored item rejected",
        impl="fre.modules.m03_formaliser.ProblemFormaliser",
        positive_test="tests/unit/test_c06_remediation.py::test_reducer_rejects_explicit_input_node_with_unregistered_artifact_anchor",
        negative_test=NA,
        evidence=NA,
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-031",
        requirement="inference remains inference",
        impl="fre.modules.m03_formaliser.ProblemFormaliser (SUPPORTED_INFERENCE)",
        positive_test="tests/unit/test_c06_remediation.py::test_supported_inference_never_promoted_to_fact_by_string_presence",
        negative_test=NA,
        evidence="golden E",
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-032",
        requirement="assumption remains assumption",
        impl="fre.modules.m03_formaliser.ProblemFormaliser (ASSUMPTION)",
        positive_test="tests/unit/test_c06_remediation.py::test_unknown_and_assumption_fields_round_trip_through_engine",
        negative_test=NA,
        evidence="golden G",
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-033",
        requirement="unresolved remains unresolved",
        impl="fre.modules.m03_formaliser.ProblemFormaliser (UNRESOLVED)",
        positive_test="tests/unit/test_wave3.py::test_m03_preserves_hard_unknown_and_epistemic_labels",
        negative_test=NA,
        evidence="golden G",
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-034",
        requirement="contradicted represented",
        impl="fre.modules.m03_formaliser.ProblemFormaliser (CONTRADICTED origin)",
        positive_test="tests/unit/test_c06_remediation.py::test_contradicted_origin_without_a_contradicts_relation_is_rejected",
        negative_test=NA,
        evidence="golden H",
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-035",
        requirement="contradiction diagnostic",
        impl="fre.domain.problem.ContradictionDiagnostic",
        positive_test="tests/unit/test_wave3.py::test_m03_contradiction_maps_to_frozen_m09_events",
        negative_test=NA,
        evidence="golden H",
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-036",
        requirement="contradiction not resolved",
        impl="fre.modules.m03_formaliser.ProblemFormaliser (contradiction persists until resolved)",
        positive_test="tests/integration/test_wave3_gate.py::test_coordinator_m03_contradiction_atomicity_stop_wiring_and_blocker_resolution",
        negative_test=NA,
        evidence="golden H",
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-037",
        requirement="M09 contradiction relation",
        impl="fre.modules.m09_ledger (CONTRADICTS edge)",
        positive_test="tests/unit/test_wave3.py::test_m03_material_contradiction_contests_supported_endpoints",
        negative_test=NA,
        evidence="golden H",
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-038",
        requirement="hard deterministic constraint",
        impl="fre.domain.problem.ConstraintSpec (verification_mode=DETERMINISTIC)",
        positive_test="tests/unit/test_wave3.py::test_m03_only_trusted_named_deterministic_verifier_establishes_status",
        negative_test=NA,
        evidence=NA,
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-039",
        requirement="hard model constraint",
        impl="fre.domain.problem.ConstraintSpec (verification_mode=MODEL)",
        positive_test="tests/unit/test_wave3.py::test_m03_model_constraint_statuses_begin_unknown",
        negative_test=NA,
        evidence=NA,
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-040",
        requirement="hard human constraint",
        impl="fre.domain.problem.ConstraintSpec (verification_mode=HUMAN)",
        positive_test="tests/unit/test_wave3.py::test_m03_model_constraint_statuses_begin_unknown",
        negative_test=NA,
        evidence=NA,
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-041",
        requirement="hard unavailable constraint",
        impl="fre.domain.problem.ConstraintSpec (verification_mode=UNAVAILABLE)",
        positive_test="tests/unit/test_wave3.py::test_m03_model_constraint_statuses_begin_unknown",
        negative_test=NA,
        evidence=NA,
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-042",
        requirement="HARD plus UNKNOWN",
        impl="fre.modules.m03_formaliser.ProblemFormaliser (material UNKNOWN blocker)",
        positive_test="tests/unit/test_c06_remediation.py::test_material_unresolved_unknown_produces_blocker_non_material_does_not",
        negative_test=NA,
        evidence="golden H",
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-043",
        requirement="ERROR not FAIL",
        impl="fre.domain.problem (VerificationStatus.ERROR distinct from FAIL)",
        positive_test="tests/unit/test_wave3.py::test_m03_only_trusted_named_deterministic_verifier_establishes_status",
        negative_test=NA,
        evidence=NA,
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-044",
        requirement="supported objective direction",
        impl="fre.domain.problem.ObjectiveSpec (direction)",
        positive_test="tests/unit/test_wave3.py::test_m03_fallback_provenance_objectives_and_unavailable_acceptance_blocker",
        negative_test=NA,
        evidence="golden B/I",
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-045",
        requirement="unresolved direction not invented",
        impl="fre.modules.m03_formaliser.ProblemFormaliser (direction fallback)",
        positive_test="tests/unit/test_wave3.py::test_m03_fallback_provenance_objectives_and_unavailable_acceptance_blocker",
        negative_test=NA,
        evidence=NA,
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-046",
        requirement="variable vs fixed",
        impl="fre.domain.problem.DecisionVariable / FixedParameter",
        positive_test="tests/unit/test_domain.py::test_semantic_hard_constraint_remains_hard_and_unknown_is_distinct",
        negative_test=NA,
        evidence="golden B",
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-047",
        requirement="unknown explicit",
        impl="fre.domain.problem.UnknownSpec",
        positive_test="tests/unit/test_c06_remediation.py::test_unknown_and_assumption_fields_round_trip_through_engine",
        negative_test=NA,
        evidence="golden G",
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-048",
        requirement="observable explicit",
        impl="fre.domain.problem.ObservableSpec",
        positive_test="tests/unit/test_m12_wave3_context.py::test_semantic_json_and_markdown_exact_goldens",
        negative_test=NA,
        evidence=NA,
        sha_key="C08",
    ),
    MatrixRow(
        row_id="W3-049",
        requirement="criterion route or blocker",
        impl="fre.modules.m03_formaliser.ProblemFormaliser (AcceptanceCriterion blocker)",
        positive_test="tests/unit/test_wave3.py::test_m03_fallback_provenance_objectives_and_unavailable_acceptance_blocker",
        negative_test=NA,
        evidence=NA,
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-050",
        requirement="ambiguity unresolved",
        impl="fre.modules.m03_formaliser.ProblemFormaliser (UNKNOWN materiality default)",
        positive_test="tests/unit/test_c06_remediation.py::test_material_defaults_true_when_no_explicit_signal_is_given",
        negative_test=NA,
        evidence=NA,
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-051",
        requirement="authorized assumption labelled",
        impl="fre.domain.semantic.EpistemicItemProvenance (WORKING_ASSUMPTION)",
        positive_test="tests/unit/test_c06_remediation.py::test_unknown_and_assumption_fields_round_trip_through_engine",
        negative_test=NA,
        evidence="golden G",
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-052",
        requirement="valid formalisation",
        impl="fre.modules.m03_formaliser.ProblemFormaliser.canonical_events",
        positive_test="tests/integration/test_wave3_gate.py::test_coordinator_normal_semantic_path_and_replay",
        negative_test=NA,
        evidence="golden B",
        sha_key="C09",
    ),
    MatrixRow(
        row_id="W3-053",
        requirement="one formalisation repair",
        impl="fre.semantic_runtime.SemanticModelRuntime (module-agnostic repair budget)",
        positive_test="tests/unit/test_semantic_runtime.py::test_repair_executes_distinct_render_and_validated_artifact_reuses",
        negative_test=NA,
        evidence=NA,
        sha_key="C04",
    ),
    MatrixRow(
        row_id="W3-054",
        requirement="semantic invalid rejected",
        impl="fre.modules.m03_formaliser.ProblemFormaliser (invalid proposal)",
        positive_test="tests/unit/test_c06_remediation.py::test_reducer_and_formalise_share_identical_support_ref_error_taxonomy",
        negative_test=NA,
        evidence="golden F",
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-055",
        requirement="repair fallback",
        impl="fre.semantic_runtime.SemanticModelRuntime (repair exhausted, module-agnostic)",
        positive_test="tests/integration/test_wave3_gate.py::test_coordinator_schema_invalid_path_falls_back_after_exhausted_repair",
        negative_test=NA,
        evidence="golden D",
        sha_key="C04",
    ),
    MatrixRow(
        row_id="W3-056",
        requirement="unavailable fallback",
        impl="fre.modules.m03_formaliser.ProblemFormaliser (allow_model=False)",
        positive_test="tests/integration/test_wave3_gate.py::test_coordinator_zero_call_fallback_is_first_class",
        negative_test=NA,
        evidence="golden A",
        sha_key="C09",
    ),
    MatrixRow(
        row_id="W3-057",
        requirement="fallback honest",
        impl="fre.modules.m03_formaliser.ProblemFormaliser (diagnostics honesty)",
        positive_test="tests/unit/test_wave3.py::test_m03_fallback_provenance_objectives_and_unavailable_acceptance_blocker",
        negative_test=NA,
        evidence=NA,
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-058",
        requirement="ProblemSpec replay",
        impl="fre.engine.FrontierReasoningEngine.replay (ProblemSpec)",
        positive_test="tests/integration/test_wave3_gate.py::test_coordinator_normal_semantic_path_and_replay",
        negative_test=NA,
        evidence="golden B",
        sha_key="C09",
    ),
    MatrixRow(
        row_id="W3-059",
        requirement="replay no model",
        impl="fre.composition.Wave3Engine.execute_front_end (resumed run)",
        positive_test="tests/integration/test_wave3_gate.py::test_coordinator_normal_semantic_path_and_replay",
        negative_test=NA,
        evidence="golden K",
        sha_key="C09",
    ),
    MatrixRow(
        row_id="W3-060",
        requirement="blocker LedgerNodeRef",
        impl="fre.domain.problem.ProblemBlocker.ledger_ref",
        positive_test="tests/integration/test_wave3_gate.py::test_coordinator_m03_contradiction_atomicity_stop_wiring_and_blocker_resolution",
        negative_test=NA,
        evidence="golden H",
        sha_key="C06",
    ),
    MatrixRow(
        row_id="W3-061",
        requirement="registry validates",
        impl="fre.domain.representation_registry.default_registry_v2",
        positive_test="tests/unit/test_c07_remediation.py::test_v2_registry_declares_every_representation_kind",
        negative_test=NA,
        evidence=NA,
        sha_key="C07",
    ),
    MatrixRow(
        row_id="W3-062",
        requirement="feature extraction deterministic",
        impl="fre.modules.m04_representation.RepresentationSelector.select_bound",
        positive_test="tests/unit/test_c07_remediation.py::test_selection_is_deterministic_for_identical_registry_and_inputs",
        negative_test=NA,
        evidence="golden I",
        sha_key="C07",
    ),
    MatrixRow(
        row_id="W3-063",
        requirement="scores deterministic",
        impl="fre.modules.m04_representation.RepresentationSelector.select_bound",
        positive_test="tests/unit/test_c07_remediation.py::test_selection_is_deterministic_for_identical_registry_and_inputs",
        negative_test=NA,
        evidence="golden I",
        sha_key="C07",
    ),
    MatrixRow(
        row_id="W3-064",
        requirement="dependency compatibility",
        impl="fre.modules.m04_representation (compatibility scoring)",
        positive_test="tests/unit/test_m04_representations.py::test_all_registered_builders_are_deterministic_and_do_not_run_wave4_algorithms",
        negative_test=NA,
        evidence=NA,
        sha_key="C07",
    ),
    MatrixRow(
        row_id="W3-065",
        requirement="temporal compatibility",
        impl="fre.modules.m04_representation (compatibility scoring)",
        positive_test="tests/unit/test_m04_representations.py::test_all_registered_builders_are_deterministic_and_do_not_run_wave4_algorithms",
        negative_test=NA,
        evidence=NA,
        sha_key="C07",
    ),
    MatrixRow(
        row_id="W3-066",
        requirement="causal compatibility",
        impl="fre.modules.m04_representation (compatibility scoring)",
        positive_test="tests/unit/test_m04_representations.py::test_all_registered_builders_are_deterministic_and_do_not_run_wave4_algorithms",
        negative_test=NA,
        evidence=NA,
        sha_key="C07",
    ),
    MatrixRow(
        row_id="W3-067",
        requirement="multi-objective compatibility",
        impl="fre.modules.m04_representation (compatibility scoring)",
        positive_test="tests/unit/test_m04_representations.py::test_all_registered_builders_are_deterministic_and_do_not_run_wave4_algorithms",
        negative_test=NA,
        evidence=NA,
        sha_key="C07",
    ),
    MatrixRow(
        row_id="W3-068",
        requirement="primary purpose",
        impl="fre.domain.representation.RepresentationView (role=PRIMARY)",
        positive_test="tests/golden/test_golden_fixtures.py::test_golden_i_deterministic_m04_selection_outside_tie_band",
        negative_test=NA,
        evidence="golden I",
        sha_key="C07",
    ),
    MatrixRow(
        row_id="W3-069",
        requirement="auxiliary purpose",
        impl="fre.domain.representation.RepresentationView (role=AUXILIARY)",
        positive_test="tests/unit/test_c07_remediation.py::test_selection_is_deterministic_for_identical_registry_and_inputs",
        negative_test=NA,
        evidence=NA,
        sha_key="C07",
    ),
    MatrixRow(
        row_id="W3-070",
        requirement="limitations",
        impl="fre.domain.representation.RepresentationView.limitations",
        positive_test="tests/unit/test_m04_representations.py::test_unavailable_builder_remains_explicit_in_wire_representation",
        negative_test=NA,
        evidence=NA,
        sha_key="C07",
    ),
    MatrixRow(
        row_id="W3-071",
        requirement="selection provenance",
        impl="fre.domain.representation.RepresentationPlanV2.selection_basis",
        positive_test="tests/unit/test_c07_remediation.py::test_bound_plan_and_artifact_round_trip_through_the_real_engine",
        negative_test=NA,
        evidence="golden I/J",
        sha_key="C07",
    ),
    MatrixRow(
        row_id="W3-072",
        requirement="source refs",
        impl="fre.domain.representation.RepresentationArtifactV2",
        positive_test="tests/unit/test_c07_remediation.py::test_bound_plan_and_artifact_round_trip_through_the_real_engine",
        negative_test=NA,
        evidence="golden I/J",
        sha_key="C07",
    ),
    MatrixRow(
        row_id="W3-073",
        requirement="view budget",
        impl="fre.domain.budget (max_representation_views)",
        positive_test="tests/unit/test_c07_remediation.py::test_tie_band_boundary_is_consistent_at_all_call_sites",
        negative_test=NA,
        evidence="golden J",
        sha_key="C07",
    ),
    MatrixRow(
        row_id="W3-074",
        requirement="two-view tie",
        impl="fre.modules.m04_representation.RepresentationSelector (tie_band)",
        positive_test="tests/golden/test_golden_fixtures.py::test_golden_j_adjudication_inside_tie_band_and_fallback_builder_identity",
        negative_test=NA,
        evidence="golden J",
        sha_key="C07",
    ),
    MatrixRow(
        row_id="W3-075",
        requirement="one-view tie",
        impl="fre.modules.m04_representation.RepresentationSelector (single-candidate tie)",
        positive_test="tests/unit/test_m04_representations.py::test_adjudication_is_disabled_by_default_and_valid_enabled_reorder_is_bounded",
        negative_test=NA,
        evidence=NA,
        sha_key="C07",
    ),
    MatrixRow(
        row_id="W3-076",
        requirement="registry-order invariant",
        impl="fre.modules.m04_representation.RepresentationSelector.select_bound",
        positive_test="tests/unit/test_m04_representations.py::test_sparse_problem_falls_back_and_registry_order_is_irrelevant",
        negative_test=NA,
        evidence=NA,
        sha_key="C07",
    ),
    MatrixRow(
        row_id="W3-077",
        requirement="no unregistered adjudication",
        impl="fre.modules.m04_representation (adjudication_record_ref binding)",
        positive_test="tests/unit/test_c07_remediation.py::test_reject_adjudication_reference_without_matching_model_call",
        negative_test=NA,
        evidence=NA,
        sha_key="C07",
    ),
    MatrixRow(
        row_id="W3-078",
        requirement="no incompatible adjudication",
        impl="fre.modules.m04_representation (tie_band binding)",
        positive_test="tests/unit/test_c07_remediation.py::test_reject_adjudication_reference_outside_declared_tie_band",
        negative_test=NA,
        evidence=NA,
        sha_key="C07",
    ),
    MatrixRow(
        row_id="W3-079",
        requirement="adjudication fallback",
        impl="fre.modules.m04_representation.RepresentationSelector (adjudication failure)",
        positive_test="tests/unit/test_m04_representations.py::test_adjudication_refusal_or_invalidity_preserves_deterministic_plan",
        negative_test=NA,
        evidence=NA,
        sha_key="C07",
    ),
    MatrixRow(
        row_id="W3-080",
        requirement="builder fallback",
        impl="fre.modules.m04_representation (requested vs actual builder)",
        positive_test="tests/unit/test_c07_remediation.py::test_fallback_records_actual_builder_not_requested_builder",
        negative_test="tests/unit/test_c07_remediation.py::test_no_fallback_keeps_requested_and_actual_builder_identical",
        evidence=NA,
        sha_key="C07",
    ),
    MatrixRow(
        row_id="W3-081",
        requirement="artifact hash deterministic",
        impl="fre.modules.m04_representation.build_bound (content hash)",
        positive_test="tests/unit/test_c07_remediation.py::test_reject_forged_content_hash_trusting_computed_bytes_instead",
        negative_test=NA,
        evidence=NA,
        sha_key="C07",
    ),
    MatrixRow(
        row_id="W3-082",
        requirement="immutable artifact",
        impl="fre.domain.representation.RepresentationArtifactV2",
        positive_test="tests/unit/test_c07_remediation.py::test_v2_artifact_cannot_be_constructed_with_forged_attribution",
        negative_test=NA,
        evidence=NA,
        sha_key="C07",
    ),
    MatrixRow(
        row_id="W3-083",
        requirement="staleness",
        impl="fre.runtime.reducer (representation plan / ProblemSpec binding)",
        positive_test="tests/unit/test_c09_remediation.py::test_reformalisation_invalidates_bound_legacy_v1_representation_plan",
        negative_test="tests/unit/test_c07_remediation.py::test_reject_mismatched_source_snapshot_version",
        evidence=NA,
        sha_key="C09",
    ),
    MatrixRow(
        row_id="W3-084",
        requirement="projection not authority",
        impl="fre.modules.m12_context.derive_wave3_availability",
        positive_test="tests/unit/test_c08_remediation.py::test_reducer_rejects_forged_availability_disagreeing_with_recomputation",
        negative_test=NA,
        evidence="golden H",
        sha_key="C08",
    ),
    MatrixRow(
        row_id="W3-085",
        requirement="representation replay",
        impl="fre.engine.FrontierReasoningEngine.replay (RepresentationPlanV2)",
        positive_test="tests/unit/test_c07_remediation.py::test_bound_plan_and_artifact_round_trip_through_the_real_engine",
        negative_test=NA,
        evidence="golden I/J",
        sha_key="C07",
    ),
    MatrixRow(
        row_id="W3-086",
        requirement="prompt registry deterministic",
        impl="fre.prompts.registry.PromptRegistry",
        positive_test="tests/unit/test_wave3.py::test_prompt_and_schema_registries_are_versioned_and_integrity_checked",
        negative_test=NA,
        evidence=NA,
        sha_key="C04",
    ),
    MatrixRow(
        row_id="W3-087",
        requirement="prompt provenance",
        impl="fre.prompts.registry.PromptRegistry",
        positive_test="tests/unit/test_wave3.py::test_prompt_and_schema_registries_are_versioned_and_integrity_checked",
        negative_test=NA,
        evidence=NA,
        sha_key="C04",
    ),
    MatrixRow(
        row_id="W3-088",
        requirement="prompt identity integrity",
        impl="fre.prompts.schemas.canonical_schema_hash",
        positive_test="tests/unit/test_output_schema_binding.py::test_structured_model_request_carries_the_canonical_schema_hash",
        negative_test="tests/unit/test_output_schema_binding.py::test_stale_request_schema_hash_is_rejected_before_provider_invocation",
        evidence=NA,
        sha_key="C04",
    ),
    MatrixRow(
        row_id="W3-089",
        requirement="call idempotency",
        impl="fre.semantic_runtime.SemanticModelRuntime (idempotency_key)",
        positive_test="tests/unit/test_semantic_runtime.py::test_concurrent_same_identity_cannot_oversubscribe_or_duplicate_call",
        negative_test=NA,
        evidence=NA,
        sha_key="C04",
    ),
    MatrixRow(
        row_id="W3-090",
        requirement="raw artifact separate",
        impl="fre.semantic_runtime.SemanticModelRuntime (raw vs proposal artifacts)",
        positive_test="tests/unit/test_semantic_batch_admission.py::test_missing_registered_artifact_is_rejected",
        negative_test=NA,
        evidence=NA,
        sha_key="C04",
    ),
    MatrixRow(
        row_id="W3-091",
        requirement="proposal validation gate",
        impl="fre.runtime.reducer.RunReducer (model-call admission)",
        positive_test="tests/unit/test_c04_remediation.py::test_forged_success_with_schema_invalid_artifact_bytes_is_rejected",
        negative_test=NA,
        evidence=NA,
        sha_key="C04",
    ),
    MatrixRow(
        row_id="W3-092",
        requirement="invalid raw persisted only",
        impl="fre.semantic_runtime.SemanticModelRuntime (INVALID_STRUCTURED_OUTPUT persistence)",
        positive_test="tests/unit/test_c04_remediation.py::test_forged_success_with_mismatched_schema_hash_is_rejected",
        negative_test=NA,
        evidence="golden D",
        sha_key="C04",
    ),
    MatrixRow(
        row_id="W3-093",
        requirement="repair budgeted",
        impl="fre.semantic_runtime.SemanticModelRuntime (repair reservation)",
        positive_test="tests/unit/test_semantic_runtime.py::test_zero_repairs_and_repair_budget_refusal_preserve_initial_record",
        negative_test=NA,
        evidence=NA,
        sha_key="C04",
    ),
    MatrixRow(
        row_id="W3-094",
        requirement="repair refused without budget",
        impl="fre.semantic_runtime.SemanticModelRuntime (repair reservation refusal)",
        positive_test="tests/unit/test_semantic_runtime.py::test_zero_repairs_and_repair_budget_refusal_preserve_initial_record",
        negative_test=NA,
        evidence=NA,
        sha_key="C04",
    ),
    MatrixRow(
        row_id="W3-095",
        requirement="unavailable not evidence",
        impl="fre.semantic_runtime.SemanticModelRuntime (StructuredModelStatus.UNAVAILABLE)",
        positive_test="tests/unit/test_semantic_runtime.py::test_async_statuses_persist_and_release_reservation",
        negative_test=NA,
        evidence=NA,
        sha_key="C04",
    ),
    MatrixRow(
        row_id="W3-096",
        requirement="failure not evidence",
        impl="fre.semantic_runtime.SemanticModelRuntime (provider exception path)",
        positive_test="tests/unit/test_semantic_runtime.py::test_interruption_and_provider_exception_before_usage_release_reservation",
        negative_test=NA,
        evidence=NA,
        sha_key="C04",
    ),
    MatrixRow(
        row_id="W3-097",
        requirement="provider neutral",
        impl="fre.ports.models.StructuredModelPort",
        positive_test="tests/unit/test_dependencies.py::test_deterministic_replay_core_has_no_provider_or_network_imports",
        negative_test=NA,
        evidence=NA,
        sha_key="C04",
    ),
    MatrixRow(
        row_id="W3-098",
        requirement="Wave 1 replay",
        impl="fre.engine.FrontierReasoningEngine.replay (Wave 1 fixture)",
        positive_test="tests/integration/test_wave1_gate.py::test_complete_replay_concurrency_and_deduplication_gate",
        negative_test=NA,
        evidence="tests/fixtures/wave1_history.json",
        sha_key="C09",
    ),
    MatrixRow(
        row_id="W3-099",
        requirement="Wave 2 replay",
        impl="fre.engine.FrontierReasoningEngine.replay (Wave 2 fixture)",
        positive_test="tests/integration/test_wave2_gate.py::test_historic_fixture_replays_and_snapshot_verifies",
        negative_test=NA,
        evidence="tests/fixtures/wave2_history.json",
        sha_key="C09",
    ),
    MatrixRow(
        row_id="W3-100",
        requirement="Wave 1 snapshots",
        impl="fre.engine.FrontierReasoningEngine.snapshot (Wave 1)",
        positive_test="tests/integration/test_wave1_gate.py::test_snapshot_tampering_is_detected",
        negative_test=NA,
        evidence=NA,
        sha_key="C09",
    ),
    MatrixRow(
        row_id="W3-101",
        requirement="Wave 2 snapshots",
        impl="fre.engine.FrontierReasoningEngine.snapshot (Wave 2)",
        positive_test="tests/integration/test_wave2_gate.py::test_snapshot_assisted_path_replays_only_post_snapshot_events",
        negative_test=NA,
        evidence=NA,
        sha_key="C09",
    ),
    MatrixRow(
        row_id="W3-102",
        requirement="historic hashes",
        impl="fre.runtime.reducer.RunState.snapshot_payload (historic hash stability)",
        positive_test="tests/unit/test_foundation_freeze.py::test_policy_hash_is_stable_and_projection_is_historic_event_data",
        negative_test=NA,
        evidence=NA,
        sha_key="C09",
    ),
    MatrixRow(
        row_id="W3-103",
        requirement="pure-SQLite replay",
        impl="fre.engine.FrontierReasoningEngine.replay (reopened store)",
        positive_test="tests/integration/test_wave3_gate.py::test_coordinator_normal_semantic_path_and_replay",
        negative_test=NA,
        evidence="golden B/K",
        sha_key="C09",
    ),
    MatrixRow(
        row_id="W3-104",
        requirement="snapshot-full replay",
        impl="fre.engine.FrontierReasoningEngine.replay_from_snapshot",
        positive_test="tests/integration/test_wave3_gate.py::test_coordinator_normal_semantic_path_and_replay",
        negative_test=NA,
        evidence="golden B/K",
        sha_key="C09",
    ),
    MatrixRow(
        row_id="W3-105",
        requirement="zero-call replay",
        impl="fre.composition.Wave3Engine.execute_front_end (completed-run resume)",
        positive_test="tests/integration/test_wave3_gate.py::test_coordinator_normal_semantic_path_and_replay",
        negative_test=NA,
        evidence="golden K",
        sha_key="C09",
    ),
    MatrixRow(
        row_id="W3-106",
        requirement="M12 Wave 3 packet",
        impl="fre.modules.m12_context.Wave3ContextCompiler.compile_semantic",
        positive_test="tests/unit/test_c08_remediation.py::test_compile_semantic_populates_typed_wave3_context",
        negative_test=NA,
        evidence="golden K",
        sha_key="C08",
    ),
    MatrixRow(
        row_id="W3-107",
        requirement="M13 provenance",
        impl="fre.composition.Wave3Engine.evaluate_stop",
        positive_test="tests/integration/test_wave3_gate.py::test_coordinator_m13_decision_recomputed_equal_after_unrelated_state",
        negative_test=NA,
        evidence="golden L",
        sha_key="C09",
    ),
    MatrixRow(
        row_id="W3-108",
        requirement="M09 regressions",
        impl="fre.modules.m09_ledger.EpistemicLedger (Wave 2 regression suite)",
        positive_test="tests/integration/test_wave2_gate.py::test_combined_wave2_lifecycle_and_differential_replay",
        negative_test=NA,
        evidence=NA,
        sha_key="C09",
    ),
    MatrixRow(
        row_id="W3-109",
        requirement="M02-budget regressions",
        impl="fre.modules.m02_budget.BudgetAllocator (Wave 2 regression suite)",
        positive_test="tests/integration/test_wave2_gate.py::test_historic_budget_replay_ignores_changed_live_policy",
        negative_test=NA,
        evidence=NA,
        sha_key="C09",
    ),
    MatrixRow(
        row_id="W3-110",
        requirement="all foundation regressions",
        impl="fre.runtime.reducer.RunReducer (Waves 1-2 foundation freeze)",
        positive_test="tests/unit/test_foundation_freeze.py::test_reducer_outputs_remain_recursively_immutable",
        negative_test=NA,
        evidence="docs/waves1-2-foundation-freeze-matrix.md",
        sha_key="C09",
    ),
]

CROSS_CUTTING: list[MatrixRow] = [
    MatrixRow(
        row_id="W3-PROMPT",
        requirement=(
            "Prompt/schema integrity, semantic idempotency, raw artifacts, budgeting, one "
            "repair, fallback"
        ),
        impl="fre.semantic_runtime.SemanticModelRuntime",
        positive_test="tests/unit/test_semantic_runtime.py (11 decisive cases)",
        negative_test="tests/unit/test_c04_remediation.py (9 adversarial cases)",
        evidence="golden B/C/D/E",
        sha_key="C04",
    ),
    MatrixRow(
        row_id="W3-GOLDEN",
        requirement="Deterministic fake-model scenarios exercised through the real coordinator",
        impl="fre.composition.Wave3Engine.execute_front_end",
        positive_test="tests/golden/test_golden_fixtures.py (fixtures A-L, 12 cases)",
        negative_test="tests/integration/test_wave3_gate.py (16 coordinator-level cases)",
        evidence="golden A-L",
        sha_key="C09",
    ),
    MatrixRow(
        row_id="W3-DEPS",
        requirement="Provider-neutral boundaries and pure replay/builders",
        impl="fre.ports.models.StructuredModelPort",
        positive_test="tests/unit/test_dependencies.py::test_deterministic_replay_core_has_no_provider_or_network_imports",
        negative_test="tests/unit/test_dependencies.py::test_domain_and_modules_do_not_import_concrete_adapters",
        evidence=NA,
        sha_key="C09",
    ),
    MatrixRow(
        row_id="W3-METRICS",
        requirement="Derived metrics are non-authoritative",
        impl="fre.projections (Wave3Metrics)",
        positive_test="tests/unit/test_foundation_freeze.py::test_wave2_metrics_are_complete_projection_derived_and_non_mutating",
        negative_test=NA,
        evidence=NA,
        sha_key="C09",
    ),
    MatrixRow(
        row_id="W3-NOW4",
        requirement="No Wave 4 or later behavior",
        impl="repository scope (no optimization/search algorithms)",
        positive_test="tests/unit/test_m04_representations.py::test_all_registered_builders_are_deterministic_and_do_not_run_wave4_algorithms",
        negative_test=NA,
        evidence=NA,
        sha_key="C07",
    ),
]


def _row(cols: tuple[str, ...]) -> str:
    return "| " + " | ".join(cols) + " |"


_TABLE_HEADER = (
    "| ID | REQUIREMENT | IMPLEMENTATION SYMBOL/FILE | POSITIVE TEST | "
    "NEGATIVE/ADVERSARIAL TEST | REPLAY/PROPERTY/GOLDEN EVIDENCE | EVIDENCE SHA | STATUS |"
)
_TABLE_SEPARATOR = "|---|---|---|---|---|---|---|---|"


def _render_table(rows: list[MatrixRow], *, heading: str | None = None) -> list[str]:
    """Render a matrix table (main ROWS or CROSS_CUTTING) as markdown lines.

    Shared by both call sites in `render()` -- the only structural difference
    between the main table and the cross-cutting table is the optional
    section heading emitted before the column header row.
    """
    lines: list[str] = []
    if heading is not None:
        lines += ["", f"## {heading}", ""]
    lines += [_TABLE_HEADER, _TABLE_SEPARATOR]
    for row in rows:
        lines.append(
            _row(
                (
                    row.row_id,
                    row.requirement,
                    f"`{row.impl}`",
                    f"`{row.positive_test}`",
                    row.negative_test if row.negative_test == NA else f"`{row.negative_test}`",
                    row.evidence,
                    f"`{SHA[row.sha_key]}` ({row.sha_key})",
                    _STATUS,
                )
            )
        )
    return lines


def render() -> str:
    lines = [
        "# Wave 3 requirements matrix",
        "",
        "C10 (defect F14) remediation: this matrix was rebuilt from the actual, current",
        "codebase state at commit `7402ac84cadb4f7c162a73d2b545ef1c597d25d0` (the C09",
        "integration head this PR branches from) plus this PR's own additions. Every",
        "mandatory row below carries a stable ID, the exact requirement, the real",
        "implementation symbol/file, a real positive test, a real negative/adversarial",
        "test where one exists (`N/A` where none does -- recorded honestly, never",
        "papered over), replay/property/golden evidence where applicable, and the exact",
        "commit SHA of the corrective PR that introduced/decisively proved it, and an",
        "explicit STATUS using the vocabulary defined by",
        "`FRE_WAVE3_C01_C10_EXECUTION_COMPLETION_AND_VALIDATION_REGISTER.md` (implemented,",
        "locally tested, independently reviewed, merged, frozen): every row currently cites",
        "a C02-C09 commit that has been independently reviewed, remediated, and merged onto",
        "the integration branch -- none is `frozen`, since freezing Waves 1-3 is a Phase 9",
        "decision, not any individual child PR's. Every cell in the first six columns is",
        "checked by `scripts/verify_requirements_matrix.py` (run in CI) to reference a test",
        "file/function that actually exists, actually collects, and actually PASSES under",
        "pytest (a skipped/xfailed cited test is treated as a failure of the row's",
        "acceptance claim) -- no row may cite evidence that does not exist, does not run, or",
        "does not pass. The same script also fails CI if this checked-in file drifts from",
        "what `scripts/generate_requirements_matrix.py` currently produces.",
        "",
        "This file is generated by `scripts/generate_requirements_matrix.py` -- edit the",
        "`ROWS`/`CROSS_CUTTING` data there, not this file directly.",
        "",
    ]
    lines += _render_table(ROWS)
    lines += _render_table(CROSS_CUTTING, heading="Cross-cutting governed requirements")
    lines += [
        "",
        "## Known, non-blocking, tracked gaps",
        "",
        "This matrix reports `PASS`-with-evidence rows only; it does not claim zero",
        "known issues in the wider codebase. See",
        "`WAVE3_DEFERRED_CLEANUP_REGISTER.md` at the repository root for 28+ Tier 4-7",
        "findings (code-quality/duplication/naming observations and a small number of",
        "documented, non-blocking residual limitations) raised by each phase's",
        "independent review and explicitly deferred -- by standing policy -- to a single",
        "cleanup pass after C10, rather than blocking any individual phase's merge. None",
        "of those findings are Tier 1-3 (which would have blocked merge), and none",
        "invalidate any row above.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "docs" / "wave3-requirements-matrix.md"
    out.write_text(render())
    print(f"wrote {out}")
