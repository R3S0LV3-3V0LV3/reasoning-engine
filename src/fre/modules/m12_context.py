"""M12 deterministic context compiler, Markdown renderer, and deltas."""

import copy
from typing import cast
from uuid import UUID

from fre.domain.budget import BudgetRemaining
from fre.domain.common import JsonValue, canonical_hash, canonical_json, canonical_unordered
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
)
from fre.domain.ledger import EpistemicStatus, LedgerNode, LedgerNodeRef, LedgerProjection
from fre.domain.problem import ProblemBlocker, ProblemSpec
from fre.domain.representation import RepresentationPlan
from fre.modules.m09_ledger import EpistemicLedger

DEFAULT_TARGETS = {
    CompilerProfile.FULL: 262144,
    CompilerProfile.STANDARD: 65536,
    CompilerProfile.HANDOFF: 32768,
}


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
            packet_hash="0" * 64,
        )
        packet = packet.model_copy(
            update={"packet_hash": canonical_hash(packet.model_dump(exclude={"packet_hash"}))}
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
        **kwargs: object,
    ) -> ContextCompilationResult:
        if representation is not None and representation.problem_spec_hash != canonical_hash(
            problem
        ):
            raise ValueError("representation plan does not bind current ProblemSpec")
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
