"""Wave 3 context compilation runtime: the real caller that persists it.

C08 (F12, downstream F04) remediation: prior to this module, `Wave3
ContextCompiler.compile_semantic()` was constructed in `compose_wave3` and
never invoked by any production path -- it returned an in-memory
`ContextCompilationResult` only, and the sole real `ContextCompiled` emission
(`Wave2Runtime.finalize`) always used the plain Wave 2 `ContextCompiler`
with no semantic arguments. `Wave3ContextRuntime.compile_and_persist` is the
missing wiring: it reads ONLY already-applied, persisted `RunState` (never
caller-supplied claims about state), compiles the typed Wave 3 semantic
packet from it, and appends `ArtifactRegistered` (json + markdown) and
`ContextCompiledV2` atomically in one `engine.append` batch -- so a durably
persisted Wave 3 context packet and its two artifacts either all land
together or none do.
"""

from uuid import UUID

from fre.domain.budget import BudgetPlan
from fre.domain.common import ArtifactRef
from fre.domain.context import CompilerProfile, Wave3SemanticContext
from fre.engine import FrontierReasoningEngine
from fre.modules.m12_context import Wave3ContextCompiler
from fre.runtime.budget_meter import BudgetMeter
from fre.runtime.events import ArtifactRegistered, ContextCompiledV2


class Wave3ContextRuntime:
    """Compiles the current run's typed Wave 3 semantic context and persists it."""

    def __init__(
        self, engine: FrontierReasoningEngine, compiler: Wave3ContextCompiler | None = None
    ) -> None:
        self.engine = engine
        self.compiler = compiler or Wave3ContextCompiler()

    def compile_and_persist(
        self,
        run_id: UUID,
        *,
        profile: CompilerProfile,
        size_target: int | None = None,
        next_action: str | None = None,
        terminal_disposition: str | None = None,
        budget_plan: BudgetPlan | None = None,
        budget_policy_hash: str | None = None,
        prompt_version: str | None = None,
        model_identity: str | None = None,
    ) -> str:
        """Compile the CURRENT, already-applied Wave 3 state and persist it.

        Every input to `compile_semantic` below is read from `state` -- the
        result of replaying this run's own committed event history -- never
        from a caller-supplied claim about what that state is. This is what
        makes the reducer's independent `ContextCompiledV2` ground-truth
        verification meaningful in production, not merely in tests that hand-
        construct a state and payload directly.
        """
        state = self.engine.inspect(run_id)
        if state.problem_spec is None:
            raise ValueError(
                "Wave 3 context compilation requires a ProblemSpec already formalised on this run"
            )
        remaining = BudgetMeter().remaining(state.budget)
        compiled = self.compiler.compile_semantic(
            problem=state.problem_spec,
            representation=state.representation_plan,
            problem_blockers=state.problem_blockers,
            representation_artifacts=state.representation_artifacts,
            task_signature=state.task_signature,
            budget_plan=budget_plan if budget_plan is not None else state.budget.plan,
            budget_policy_hash=(
                budget_policy_hash if budget_policy_hash is not None else state.budget.policy_hash
            ),
            prompt_version=prompt_version,
            model_identity=model_identity,
            run_id=run_id,
            snapshot_version=state.version,
            ledger=state.ledger,
            budget_remaining=remaining,
            profile=profile,
            size_target=size_target,
            next_action=next_action,
            terminal_disposition=terminal_disposition,
        )
        json_artifact = self.engine.store_artifact(
            compiled.canonical_bytes, media_type="application/json"
        )
        markdown_artifact = self.engine.store_artifact(
            compiled.markdown.encode(), media_type="text/markdown"
        )
        json_ref = ArtifactRef(artifact_id=json_artifact.id, sha256=json_artifact.sha256)
        markdown_ref = ArtifactRef(
            artifact_id=markdown_artifact.id, sha256=markdown_artifact.sha256
        )
        payloads = (
            ArtifactRegistered(
                artifact=json_ref,
                media_type="application/json",
                byte_size=len(compiled.canonical_bytes),
            ),
            ArtifactRegistered(
                artifact=markdown_ref,
                media_type="text/markdown",
                byte_size=len(compiled.markdown.encode()),
            ),
            ContextCompiledV2(
                packet=compiled.packet,
                json_artifact=json_ref,
                markdown_artifact=markdown_ref,
                renderer_version=self.compiler.renderer_version,
            ),
        )
        events = tuple(
            self.engine.make_event(run_id, payload, module_id="M12") for payload in payloads
        )
        self.engine.append(run_id, state.version, events)
        return compiled.packet.packet_hash


__all__ = ["Wave3ContextRuntime", "Wave3SemanticContext"]
