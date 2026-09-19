"""M12 deterministic context compiler, Markdown renderer, and deltas."""

import copy
from typing import cast
from uuid import UUID

from fre.domain.budget import RESOURCE_NAMES, BudgetPlan, BudgetRemaining
from fre.domain.common import (
    JsonValue,
    bind_hash,
    canonical_hash,
    canonical_json,
    canonical_unordered,
)
from fre.domain.context import (
    CompilerProfile,
    ContextCompilationResult,
    ContextCompressionPolicy,
    ContextDeltaBaseMismatch,
    ContextDeltaPacket,
    ContextItem,
    ContextOverflow,
    ContextOverflowDiagnostic,
    ContextPacket,
    DeltaOperation,
    DeltaOperationType,
    RejectedItem,
    UnresolvedUnknownRef,
    Wave3ContextAvailability,
    Wave3SemanticContext,
)
from fre.domain.ledger import EpistemicStatus, LedgerNode, LedgerNodeRef, LedgerProjection
from fre.domain.problem import ProblemBlocker, ProblemSpec
from fre.domain.representation import (
    RepresentationArtifact,
    RepresentationArtifactV2,
    RepresentationPlan,
    RepresentationPlanV2,
)
from fre.domain.task import TaskSignature
from fre.modules.m09_ledger import EpistemicLedger

DEFAULT_TARGETS = {
    CompilerProfile.FULL: 262144,
    CompilerProfile.STANDARD: 65536,
    CompilerProfile.HANDOFF: 32768,
}


def derive_wave3_availability(
    *,
    problem_blockers: tuple[ProblemBlocker, ...],
    representation: RepresentationPlan | None,
    representation_artifacts: tuple[RepresentationArtifact, ...] = (),
    representation_v2: RepresentationPlanV2 | None = None,
    representation_artifacts_v2: tuple[RepresentationArtifactV2, ...] = (),
    budget_remaining: BudgetRemaining,
) -> tuple[Wave3ContextAvailability, str]:
    """Pure derivation of Wave 3 semantic availability from real state.

    This is the SAME function `Wave3ContextCompiler.compile_semantic` calls to
    populate `Wave3SemanticContext.availability`/`availability_reason` and
    that `RunReducer.apply` calls again, independently, from the actually-
    applied run state at `ContextCompiled@2.0` time (see the critical lesson
    in the Phase 6/C08 plan: a claim is only trustworthy when the reducer
    recomputes it from ground truth rather than checking it only against
    itself). Sharing one function closes the drift risk C07 named explicitly
    for `bind_hash` (three independently hand-maintained recomputations that
    could silently disagree).

    C08 remediation findings A/B/F: two independent defects in the
    multi-view validation branch below were closed together.

    Finding A: the original check used ``any(... for view in views for
    artifact in artifacts)`` -- true the moment ANY single view anywhere had
    a matching artifact, even if every OTHER declared view had none. A
    2-view plan with only one view's artifact present incorrectly reported
    ``AVAILABLE``. The check is now ``all(any(...) for view in views)``:
    EVERY declared view must independently have at least one matching
    artifact.

    Finding B: `RepresentationArtifact` (v1) carries only `problem_spec_hash`,
    not a binding to the exact `RepresentationPlan` that selected it. Two
    different plans selected against the same `ProblemSpec` (e.g. re-selected
    after a budget or registry change) share `problem_spec_hash`, so a stale
    artifact from an old plan would satisfy validation for a brand new plan
    that happens to want the same `RepresentationKind`. `RepresentationArtifactV2`
    (C07) does not have this gap: `RepresentationArtifactV2.plan_hash` binds
    to the exact `RepresentationPlanV2.plan_hash` that selected it, so two
    different plans can never share a matching bound artifact by accident.

    Finding F: this function now prefers the v2 (bound) representation state
    when present -- `representation_v2`/`representation_artifacts_v2` -- and
    only falls back to the legacy v1 fields when no v2 plan was selected on
    this run. When v2 is used, finding B is fully closed (plan-hash-bound
    matching). When only v1 state exists (no v2 selection ever ran on this
    run), the same view-matching logic runs against `representation_artifacts`
    with no plan-revision binding -- this is a known, documented residual
    limitation of the legacy v1 representation contract, not something this
    function can close purely with the data v1 provides. Callers that need
    the stronger guarantee should migrate to `select_bound`/`build_bound`.
    """
    material_blockers = tuple(blocker for blocker in problem_blockers if not blocker.resolvable)
    if material_blockers:
        ids = ", ".join(sorted(blocker.blocker_id for blocker in material_blockers))
        return (
            Wave3ContextAvailability.PARTIAL_BLOCKED,
            f"material (non-resolvable) problem blocker(s) unresolved: {ids}",
        )
    if representation is None and representation_v2 is None:
        return (
            Wave3ContextAvailability.NOT_REQUESTED,
            "representation plan was not requested for this compilation",
        )
    resources = budget_remaining.resources
    if all(getattr(resources, name) <= 0 for name in RESOURCE_NAMES):
        return (
            Wave3ContextAvailability.UNAVAILABLE_BUDGET,
            "remaining budget is exhausted across every countable resource",
        )
    bound_artifacts: tuple[RepresentationArtifact, ...] | tuple[RepresentationArtifactV2, ...]
    if representation_v2 is not None:
        views = representation_v2.views
        bound_artifacts = tuple(
            artifact
            for artifact in representation_artifacts_v2
            if artifact.plan_hash == representation_v2.plan_hash
        )
    else:
        views = representation.views if representation is not None else ()
        bound_artifacts = representation_artifacts
    if views and not all(
        any(
            artifact.requested_kind == view.kind or artifact.actual_kind == view.kind
            for artifact in bound_artifacts
        )
        for view in views
    ):
        return (
            Wave3ContextAvailability.UNAVAILABLE_VALIDATION,
            "representation plan selects view(s) with no corresponding validated artifact",
        )
    return (
        Wave3ContextAvailability.AVAILABLE,
        "problem, blockers, and representation are all available and mutually consistent",
    )


class ContextCompiler:
    compiler_version = "1.0"
    renderer_version = "1.0"

    def __init__(self, policy: ContextCompressionPolicy | None = None) -> None:
        self.policy = policy or ContextCompressionPolicy()
        self.ledger = EpistemicLedger()

    def compile(
        self,
        *,
        run_id: UUID,
        snapshot_version: int,
        ledger: LedgerProjection,
        budget_remaining: BudgetRemaining,
        profile: CompilerProfile,
        size_target: int | None = None,
        rejected_items: tuple[RejectedItem, ...] = (),
        hard_constraints: tuple[JsonValue, ...] = (),
        unresolved_blockers: tuple[str, ...] = (),
        next_action: str | None = None,
        terminal_disposition: str | None = None,
        objective: JsonValue | None = None,
        output_contract: JsonValue | None = None,
        wave3_context: "Wave3SemanticContext | None" = None,
    ) -> ContextCompilationResult:
        eligible_rules = self.policy.profile_rule_eligibility[profile]
        applied: list[str] = ["C01"] if "C01" in eligible_rules else []
        deduplicated_constraints = canonical_unordered(hard_constraints)
        deduplicated_rejections = canonical_unordered(rejected_items)
        deduplicated_blockers = tuple(sorted(set(unresolved_blockers)))
        current_by_id: dict[UUID, LedgerNode] = {}
        for node in ledger.nodes:
            if (
                node.node_id not in current_by_id
                or node.revision > current_by_id[node.node_id].revision
            ):
                current_by_id[node.node_id] = node
        if profile is CompilerProfile.FULL:
            nodes = ledger.nodes
        else:
            nodes = tuple(current_by_id.values())
            if "C02" in eligible_rules:
                applied.append("C02")
        items = tuple(
            sorted(
                (
                    ContextItem(
                        ref=node.ref,
                        kind=node.node_type,
                        status=self.ledger.effective_status(ledger, node.ref),
                        content=node.content,
                        provenance_refs=node.provenance_refs,
                        predecessor_ref=(
                            LedgerNodeRef(node_id=node.node_id, revision=node.revision - 1)
                            if node.revision > 1
                            else None
                        ),
                    )
                    for node in nodes
                    if profile is not CompilerProfile.HANDOFF
                    or self.ledger.effective_status(ledger, node.ref)
                    in {
                        EpistemicStatus.SUPPORTED,
                        EpistemicStatus.STALE,
                        EpistemicStatus.UNRESOLVED,
                    }
                ),
                key=lambda item: (str(item.ref.node_id), item.ref.revision),
            )
        )
        if "C04" in eligible_rules:
            applied.append("C04")
        if "C05" in eligible_rules:
            applied.append("C05")
        packet = ContextPacket(
            run_id=run_id,
            snapshot_version=snapshot_version,
            profile=profile,
            compiler_version=self.compiler_version,
            compression_policy_version=self.policy.compression_policy_version,
            applied_rule_ids=tuple(applied),
            objective=objective,
            output_contract=output_contract,
            hard_constraints=deduplicated_constraints,
            ledger_items=items,
            rejected_items=deduplicated_rejections,
            unresolved_blockers=deduplicated_blockers,
            budget_remaining=budget_remaining,
            next_action=next_action,
            terminal_disposition=terminal_disposition,
            wave3_context=wave3_context,
            packet_hash="0" * 64,
        )
        # Finding H (C08 remediation): use the single shared `bind_hash`
        # helper (C07 finding M) instead of a hand-duplicated
        # `canonical_hash(model.model_dump(exclude={...}))` preimage
        # computation, closing the same drift risk C07 named for
        # `RepresentationPlanV2.plan_hash`.
        packet = packet.model_copy(
            update={"packet_hash": bind_hash(packet, exclude={"packet_hash"})}
        )
        packet_bytes = canonical_json(packet)
        target = size_target or DEFAULT_TARGETS[profile]
        if len(packet_bytes) > target:
            raise ContextOverflow(
                ContextOverflowDiagnostic(
                    required_bytes=len(packet_bytes),
                    target_bytes=target,
                    mandatory_classes=self.policy.mandatory_retention_classes,
                )
            )
        return ContextCompilationResult(
            packet=packet,
            canonical_bytes=packet_bytes,
            markdown=self.render_markdown(packet),
            canonical_byte_size=len(packet_bytes),
        )

    @staticmethod
    def render_markdown(packet: ContextPacket) -> str:
        lines = [
            "# Frontier Reasoning Engine Context",
            "",
            f"- Profile: `{packet.profile}`",
            f"- Snapshot: `{packet.snapshot_version}`",
            f"- Packet hash: `{packet.packet_hash}`",
            "",
            "## Ledger",
        ]
        for item in packet.ledger_items:
            lines.append(
                f"- `{item.ref.node_id}@{item.ref.revision}` [{item.status}] "
                f"{canonical_json(item.content).decode()}"
            )
        lines.extend(("", "## Rejections"))
        lines.extend(f"- `{item.ref}` — {item.reason}" for item in packet.rejected_items)
        lines.extend(
            (
                "",
                "## Budget",
                f"- Remaining: `{canonical_json(packet.budget_remaining.resources).decode()}`",
                "",
                "## Next / terminal action",
                f"- {packet.terminal_disposition or packet.next_action or 'NOT_PRODUCED'}",
                "",
            )
        )
        return "\n".join(lines)


class Wave3ContextCompiler(ContextCompiler):
    """Compiler v2.0 used only when Wave 3 semantic state is incorporated."""

    compiler_version = "2.0"

    def compile_semantic(
        self,
        *,
        problem: ProblemSpec,
        representation: RepresentationPlan | None,
        problem_blockers: tuple[ProblemBlocker, ...] = (),
        representation_artifacts: tuple[RepresentationArtifact, ...] = (),
        representation_v2: RepresentationPlanV2 | None = None,
        representation_artifacts_v2: tuple[RepresentationArtifactV2, ...] = (),
        task_signature: TaskSignature | None = None,
        budget_plan: BudgetPlan | None = None,
        budget_policy_hash: str | None = None,
        prompt_version: str | None = None,
        model_identity: str | None = None,
        **kwargs: object,
    ) -> ContextCompilationResult:
        if representation is not None and representation.problem_spec_hash != canonical_hash(
            problem
        ):
            raise ValueError("representation plan does not bind current ProblemSpec")
        blocker_descriptions = tuple(blocker.description for blocker in problem_blockers)
        supplied_blockers = cast(
            tuple[str, ...], kwargs.pop("unresolved_blockers", blocker_descriptions)
        )
        if tuple(sorted(set(supplied_blockers))) != tuple(sorted(set(blocker_descriptions))):
            raise ValueError("unresolved_blockers does not match the supplied problem_blockers")
        ledger_projection = cast(LedgerProjection, kwargs["ledger"])
        budget_remaining = cast(BudgetRemaining, kwargs["budget_remaining"])
        availability, availability_reason = derive_wave3_availability(
            problem_blockers=problem_blockers,
            representation=representation,
            representation_artifacts=representation_artifacts,
            representation_v2=representation_v2,
            representation_artifacts_v2=representation_artifacts_v2,
            budget_remaining=budget_remaining,
        )
        wave3_context = Wave3SemanticContext(
            availability=availability,
            availability_reason=availability_reason,
            problem_spec_ref=canonical_hash(problem),
            task_signature_ref=canonical_hash(task_signature) if task_signature else None,
            budget_plan_ref=canonical_hash(budget_plan) if budget_plan else None,
            budget_policy_hash=budget_policy_hash,
            blocker_refs=tuple(sorted(blocker.blocker_id for blocker in problem_blockers)),
            ledger_root=canonical_hash(ledger_projection),
            ledger_version=cast(int, kwargs["snapshot_version"]),
            # W3 final-gate fix #5: prefer the v2 (bound) representation
            # state when present, mirroring `derive_wave3_availability`'s own
            # v2-preferred/v1-fallback rule (finding F) above. Before this
            # fix, these two ref fields were computed ONLY from the legacy
            # v1 `representation`/`representation_artifacts` -- so a run
            # using only v2 selection (`representation_v2` populated,
            # `representation` always `None`) got `representation_plan_ref
            # =None` and `representation_artifact_refs=()` even when a real,
            # bound v2 plan and its artifacts existed. Only artifacts whose
            # `plan_hash` actually matches `representation_v2.plan_hash` are
            # included, same as `derive_wave3_availability`'s own
            # `bound_artifacts` filter, so a stale artifact from a
            # superseded v2 plan is never counted.
            representation_plan_ref=(
                canonical_hash(representation_v2)
                if representation_v2 is not None
                else (canonical_hash(representation) if representation else None)
            ),
            representation_artifact_refs=(
                tuple(
                    sorted(
                        canonical_hash(artifact)
                        for artifact in representation_artifacts_v2
                        if artifact.plan_hash == representation_v2.plan_hash
                    )
                )
                if representation_v2 is not None
                else tuple(
                    sorted(canonical_hash(artifact) for artifact in representation_artifacts)
                )
            ),
            prompt_version=prompt_version,
            model_identity=model_identity,
            unresolved_unknowns=tuple(
                UnresolvedUnknownRef(
                    id=item.id,
                    description=item.description,
                    domain=item.domain,
                    rationale=item.rationale,
                    impact=item.impact,
                    decision_relevance=item.decision_relevance,
                    resolvable=item.resolvable,
                    candidate_actions=item.candidate_actions,
                )
                for item in problem.unknowns
            ),
        )
        semantic_summary: JsonValue = {
            "objectives": [item.model_dump(mode="json") for item in problem.objectives],
            "hard_constraints": [
                item.model_dump(mode="json") for item in problem.constraints if item.kind == "HARD"
            ],
            "soft_preferences": [
                item.model_dump(mode="json") for item in problem.constraints if item.kind == "SOFT"
            ],
            "decision_variables": [
                item.model_dump(mode="json") for item in problem.decision_variables
            ],
            "fixed_parameters": [item.model_dump(mode="json") for item in problem.fixed_parameters],
            "unknowns": [item.model_dump(mode="json") for item in problem.unknowns],
            "observables": [item.model_dump(mode="json") for item in problem.observables],
            "assumptions": {
                "statements": list(problem.assumptions),
                "items": [item.model_dump(mode="json") for item in problem.assumption_items],
            },
            "acceptance_criteria": [
                item.model_dump(mode="json") for item in problem.acceptance_criteria
            ],
            "output_requirements": problem.output_contract.model_dump(mode="json"),
            "relations": [item.model_dump(mode="json") for item in problem.relations],
            "explicit_blockers": [item.model_dump(mode="json") for item in problem_blockers],
            "representation": representation.model_dump(mode="json") if representation else None,
        }
        constraints: tuple[JsonValue, ...] = tuple(
            item.model_dump(mode="json") for item in problem.constraints if item.kind == "HARD"
        )
        return self.compile(
            objective=semantic_summary,
            output_contract=problem.output_contract.model_dump(mode="json"),
            hard_constraints=constraints,
            unresolved_blockers=blocker_descriptions,
            wave3_context=wave3_context,
            **kwargs,  # type: ignore[arg-type]
        )

    @staticmethod
    def render_markdown(packet: ContextPacket) -> str:
        semantic = packet.objective if isinstance(packet.objective, dict) else {}
        lines = [
            "# Frontier Reasoning Engine Context",
            "",
            f"- Profile: `{packet.profile}`",
            f"- Snapshot: `{packet.snapshot_version}`",
            f"- Packet hash: `{packet.packet_hash}`",
        ]
        lines.extend(("", "## Wave 3 semantic availability"))
        if packet.wave3_context is None:
            lines.append("- NOT_REQUESTED (no typed Wave 3 semantic context was compiled)")
        else:
            wave3 = packet.wave3_context
            lines.extend(
                (
                    f"- Availability: `{wave3.availability}` — {wave3.availability_reason}",
                    f"- Problem spec ref: `{wave3.problem_spec_ref}`",
                    f"- Ledger root: `{wave3.ledger_root}` @ version `{wave3.ledger_version}`",
                    f"- Blocker refs: `{', '.join(wave3.blocker_refs) or 'none'}`",
                    "- Representation plan ref: `"
                    f"{wave3.representation_plan_ref or 'NOT_PRODUCED'}`",
                    "- Representation artifact refs: `"
                    f"{', '.join(wave3.representation_artifact_refs) or 'none'}`",
                    f"- Unresolved UNKNOWNs: `{len(wave3.unresolved_unknowns)}`",
                )
            )
        sections = (
            ("Objectives", "objectives"),
            ("Hard constraints", "hard_constraints"),
            ("Soft preferences", "soft_preferences"),
            ("Decision variables", "decision_variables"),
            ("Fixed parameters", "fixed_parameters"),
            ("Unknowns", "unknowns"),
            ("Observables", "observables"),
            ("Assumptions", "assumptions"),
            ("Acceptance criteria", "acceptance_criteria"),
            ("Output requirements", "output_requirements"),
            ("Relations", "relations"),
            ("Explicit blockers", "explicit_blockers"),
            ("Current representation", "representation"),
        )
        for title, key in sections:
            lines.extend(("", f"## {title}"))
            value = semantic.get(key)
            if isinstance(value, list):
                lines.extend(f"- `{canonical_json(item).decode()}`" for item in value)
                if not value:
                    lines.append("- None")
            else:
                lines.append(f"- `{canonical_json(value).decode()}`")
        lines.extend(("", "## Ledger"))
        for item in packet.ledger_items:
            lines.append(
                f"- `{item.ref.node_id}@{item.ref.revision}` [{item.status}] "
                f"{canonical_json(item.content).decode()}"
            )
        lines.extend(("", "## Rejections"))
        lines.extend(f"- `{item.ref}` — {item.reason}" for item in packet.rejected_items)
        lines.extend(
            (
                "",
                "## Budget",
                f"- Remaining: `{canonical_json(packet.budget_remaining.resources).decode()}`",
                "",
                "## Next / terminal action",
                f"- {packet.terminal_disposition or packet.next_action or 'NOT_PRODUCED'}",
                "",
            )
        )
        return "\n".join(lines)


def generate_delta(base: ContextPacket, target: ContextPacket) -> ContextDeltaPacket:
    base_data = base.model_dump(mode="json")
    target_data = target.model_dump(mode="json")
    operations: list[DeltaOperation] = []
    for key in sorted(set(base_data) | set(target_data)):
        path = f"/{key}"
        if key not in target_data:
            operations.append(DeltaOperation(operation=DeltaOperationType.REMOVE, path=path))
        elif key not in base_data:
            operations.append(
                DeltaOperation(
                    operation=DeltaOperationType.ADD,
                    path=path,
                    value=cast(JsonValue, target_data[key]),
                )
            )
        elif base_data[key] != target_data[key]:
            operations.append(
                DeltaOperation(
                    operation=DeltaOperationType.REPLACE,
                    path=path,
                    value=cast(JsonValue, target_data[key]),
                )
            )
    preimage = {
        "base_packet_hash": base.packet_hash,
        "target_packet_hash": target.packet_hash,
        "compiler_version": target.compiler_version,
        "compression_policy_version": target.compression_policy_version,
        "profile": target.profile,
        "operations": tuple(operations),
    }
    return ContextDeltaPacket(
        base_packet_hash=base.packet_hash,
        target_packet_hash=target.packet_hash,
        compiler_version=target.compiler_version,
        compression_policy_version=target.compression_policy_version,
        profile=target.profile,
        operations=tuple(operations),
        delta_hash=canonical_hash(preimage),
    )


def apply_delta(base: ContextPacket, delta: ContextDeltaPacket) -> ContextPacket:
    delta_preimage = delta.model_dump(exclude={"delta_hash"})
    if canonical_hash(delta_preimage) != delta.delta_hash:
        raise ValueError("delta hash does not match its canonical preimage")
    if base.packet_hash != delta.base_packet_hash:
        raise ContextDeltaBaseMismatch("delta base hash does not match the supplied packet")
    data = copy.deepcopy(base.model_dump(mode="json"))
    for operation in delta.operations:
        key = operation.path.removeprefix("/")
        if "/" in key or not key:
            raise ValueError("Wave 2 delta paths address canonical top-level packet fields")
        if operation.operation is DeltaOperationType.REMOVE:
            data.pop(key, None)
        else:
            data[key] = operation.value
    result = ContextPacket.model_validate(data, strict=False)
    recomputed_target_hash = canonical_hash(result.model_dump(exclude={"packet_hash"}))
    if (
        result.packet_hash != delta.target_packet_hash
        or recomputed_target_hash != delta.target_packet_hash
    ):
        raise ValueError("delta did not reconstruct the declared target hash")
    return result
