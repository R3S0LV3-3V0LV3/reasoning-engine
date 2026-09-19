"""The narrow composition root for configured Wave 3 services.

C09 (F08) remediation: `Wave3Engine` used to compose and version-pin every
Wave 3 collaborator (M01, M02, M04, M12, ...) without ever sequencing them --
its only method was `create_run()`. The M01 -> M02 -> M03 -> M04 -> M12 flow
that actually executes the Wave 3 front end existed only as hand-wired,
test-only code (`tests/integration/test_wave3_gate.py`), constructing every
module directly and never going through `compose_wave3`.

The methods added to `Wave3Engine` below are now the one public, authoritative,
replayable path that executes that flow: `classify_task`/`formalise_problem`/
`select_representation`/`compile_context` are individually resumable (each is
a no-op once its own persisted state already exists, so a retry after
interruption never re-invokes the provider or re-emits a duplicate event
batch), and `execute_front_end` sequences all four plus an optional M13 stop
evaluation. Every module's own reducer-level admission checks (C04-C08) remain
the actual enforcement point -- this coordinator only ever builds real events
through each module's own `canonical_events`/`select_bound`/`build_bound`
surface and appends them through `FrontierReasoningEngine.append`, which
previews every event through the same `RunReducer` production/replay always
uses. No ordering invariant here is enforced merely by this file's calling
convention: every cross-module reference this coordinator creates (budget
revision <- classification, representation plan <- ProblemSpec, context
packet <- ledger/budget/representation) is independently re-verified by the
reducer from already-applied state, not merely self-consistent within the
event this coordinator constructed.
"""

from dataclasses import dataclass
from uuid import UUID

from fre.config import Wave3Config
from fre.domain.budget import DeploymentLimits, TierPolicy
from fre.domain.common import ArtifactRef, FrozenModel, canonical_hash
from fre.domain.context import CompilerProfile
from fre.domain.problem import ProblemBlocker, ProblemSpec
from fre.domain.representation import RepresentationPlanV2
from fre.domain.representation_registry import default_registry_v2
from fre.domain.stop import (
    AcceptanceStatus,
    CostEstimateInterval,
    MarginalValueEstimate,
    StopDecision,
    StopDisposition,
    StopInputs,
    StopPolicy,
    ValidationStatus,
)
from fre.domain.task import ClassificationRecord, TaskEnvelope, TaskSignature
from fre.engine import FrontierReasoningEngine, RunHandle
from fre.modules.m01_classifier import ClassificationPolicy, TaskClassifier
from fre.modules.m02_budget import BudgetAllocator, default_tier_policy
from fre.modules.m03_formaliser import ProblemFormaliser
from fre.modules.m04_representation import (
    RepresentationSelectionPolicy,
    RepresentationSelector,
    default_registry,
)
from fre.modules.m12_context import Wave3ContextCompiler
from fre.modules.m13_stop import StopController
from fre.ports.models import StructuredModelPort
from fre.prompts.registry import PromptRegistry, default_prompt_registry
from fre.prompts.schemas import (
    ClassificationOutput,
    OutputSchemaRegistry,
    ProblemFormalisationOutput,
    RepresentationAdjudicationOutput,
    default_output_schema_registry,
)
from fre.runtime.budget_meter import BudgetMeter
from fre.runtime.events import (
    ArtifactRegistered,
    BudgetAllocated,
    BudgetRevised,
    EventPayload,
    RepresentationArtifactCompiledV2,
    RepresentationPlanSelectedV2,
    TaskClassified,
    TaskPreliminarilyClassified,
)
from fre.runtime.wave2 import Wave2Runtime
from fre.runtime.wave3_context import Wave3ContextRuntime
from fre.semantic_runtime import SemanticModelRuntime, SemanticRuntimePolicy


class UnsupportedWave3Configuration(ValueError):
    """A configured version has no implementation in this build."""


class EffectiveWave3Policy(FrozenModel):
    """Complete, canonical identity of the policy graph that will execute."""

    schema_version: str
    classification: ClassificationPolicy
    semantic_runtime: SemanticRuntimePolicy
    representation_selection: RepresentationSelectionPolicy
    prompt_registry_version: str
    output_schema_registry_version: str
    context_compiler_version: str

    @property
    def policy_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class Wave3Components:
    """Configured services, all derived from one effective policy identity."""

    policy: EffectiveWave3Policy
    classifier: TaskClassifier
    semantic_runtime: SemanticModelRuntime
    # C09 (F08): M03's collaborator was previously absent from this
    # composition entirely -- every caller had to construct its own bare
    # `ProblemFormaliser()` (as the pre-coordinator test file did). It carries
    # no policy/config of its own (see `ProblemFormaliser`'s docstring), so
    # adding it here is purely a wiring completion, not a new policy surface.
    formaliser: ProblemFormaliser
    representation_selector: RepresentationSelector
    context_compiler: Wave3ContextCompiler
    # C08 (F12): the real caller that persists a compiled Wave 3 semantic
    # context packet (`ContextCompiled@2.0` + its two durable artifacts,
    # atomically) instead of `context_compiler.compile_semantic()` ever being
    # invoked only to produce an in-memory-only result nothing durably reads.
    context_runtime: Wave3ContextRuntime
    # C08 (F12) remediation, finding E: the real production call site.
    # `Wave2Runtime.finalize` is the sole terminal-disposition lifecycle
    # entrypoint; wiring `context_runtime` into it here (rather than leaving
    # `context_runtime` reachable only from ad-hoc/test code) is what closes
    # F12 on an actual execution path.
    wave2_runtime: Wave2Runtime
    prompts: PromptRegistry
    schemas: OutputSchemaRegistry


class Wave3PipelineResult(FrozenModel):
    """Identifiers/projections of one `execute_front_end` call, derived only
    from persisted state (never from any in-flight, unpersisted value)."""

    run_id: UUID
    version: int
    task_signature: TaskSignature | None = None
    classification_record: ClassificationRecord | None = None
    problem_spec: ProblemSpec | None = None
    problem_blockers: tuple[ProblemBlocker, ...] = ()
    representation_plan: RepresentationPlanV2 | None = None
    context_packet_hash: str | None = None
    blocked: bool = False


class Wave3Engine:
    """Owning façade that persists the exact effective policy at run creation
    and sequences the real M01 -> M02 -> M03 -> M04 -> M12 Wave 3 front end.

    Every method below re-derives its own precondition from
    `self.engine.inspect(run_id)` -- never from a caller-supplied claim about
    what has already happened -- and is a no-op (returning the already
    -persisted value) once its own step's product already exists on the run.
    This makes every step, and `execute_front_end` as their composition,
    safely retryable/resumable after interruption at any boundary: retrying
    never re-invokes the semantic-model provider and never re-emits a
    duplicate event batch for a step that already landed.
    """

    def __init__(self, engine: FrontierReasoningEngine, components: Wave3Components) -> None:
        self.engine = engine
        self.components = components
        self.effective_policy = components.policy

    def create_run(self) -> RunHandle:
        return self.engine.create_run(self.effective_policy)

    def finalize(self, run_id: UUID, decision: StopDecision) -> str:
        """Real call site for the composed Wave 2 + Wave 3 terminal lifecycle.

        Delegates to `Wave2Runtime.finalize`, which (now that it is composed
        with `components.context_runtime` here) also persists a typed Wave 3
        `ContextCompiledV2` packet whenever this run has a formalised
        `ProblemSpec` -- see finding E in the C08 remediation.
        """
        return self.components.wave2_runtime.finalize(run_id, decision)

    # ------------------------------------------------------------------
    # C09 (F08): the real, sequenced Wave 3 front end.
    # ------------------------------------------------------------------

    def classify_task(
        self,
        run_id: UUID,
        envelope: TaskEnvelope,
        *,
        proposal: ClassificationOutput | None = None,
        model_call_key: str | None = None,
        tier_policy: TierPolicy | None = None,
        deployment: DeploymentLimits | None = None,
    ) -> TaskSignature:
        """M01 (deterministic-first classification) + M02 (bootstrap/revised budget).

        Resumable at two independent boundaries, not just one:

        1. A run whose `task_signature` is already persisted returns it
           unchanged and appends nothing.
        2. A run that already has a bootstrap budget persisted (`state.budget
           .plan is not None`, `state.task_signature is None`) -- e.g.
           interrupted between the bootstrap append and the final
           classification append below -- resumes from the bootstrap step
           without allocating a second one: `_bootstrap_budget` itself is a
           no-op once `state.budget.plan` already exists.

        `TaskClassifier.canonical_events` is deliberately NOT used as a single
        call here: it would require the real M01 semantic call (issued by
        `classify_task_semantic`, if any) to run BEFORE any budget exists to
        reserve it against, which is backwards -- `SemanticModelRuntime.
        _invoke` reserves budget by appending `BudgetReserved` against the
        run's *already-persisted* plan, so the bootstrap budget must be
        persisted first, then the semantic call made, then the final
        classification (which mirrors the second half of `canonical_events`'
        logic exactly, via `_finalize_classification`) persisted last.
        """
        state = self.engine.inspect(run_id)
        if state.task_signature is not None:
            return state.task_signature
        self._bootstrap_budget(run_id, envelope, tier_policy, deployment)
        return self._finalize_classification(
            run_id, envelope, proposal, model_call_key, tier_policy, deployment
        )

    def _bootstrap_budget(
        self,
        run_id: UUID,
        envelope: TaskEnvelope,
        tier_policy: TierPolicy | None,
        deployment: DeploymentLimits | None,
    ) -> None:
        state = self.engine.inspect(run_id)
        if state.budget.plan is not None:
            return
        policy = self.components.policy.classification
        bootstrap_signature = self.components.classifier.bootstrap_signature(envelope, policy)
        bootstrap_plan, bootstrap_hash = BudgetAllocator().allocate(
            bootstrap_signature,
            tier_policy or default_tier_policy(),
            deployment or DeploymentLimits(),
        )
        events = (
            TaskPreliminarilyClassified(signature=bootstrap_signature),
            BudgetAllocated(
                plan=bootstrap_plan,
                policy_version=bootstrap_plan.policy_version,
                policy_hash=bootstrap_hash,
            ),
        )
        stored = tuple(self.engine.make_event(run_id, item, module_id="M01") for item in events)
        self.engine.append(run_id, state.version, stored)

    def _finalize_classification(
        self,
        run_id: UUID,
        envelope: TaskEnvelope,
        proposal: ClassificationOutput | None,
        model_call_key: str | None,
        tier_policy: TierPolicy | None,
        deployment: DeploymentLimits | None,
    ) -> TaskSignature:
        state = self.engine.inspect(run_id)
        if state.task_signature is not None:
            return state.task_signature
        classifier = self.components.classifier
        policy = self.components.policy.classification
        signature, record = classifier.classify(
            envelope, proposal, policy, model_call_key=model_call_key
        )
        plan, policy_hash = BudgetAllocator().allocate(
            signature, tier_policy or default_tier_policy(), deployment or DeploymentLimits()
        )
        # Validate-only: proves the bootstrap -> final handoff never violates
        # M02's monotone-tier/validation-floor revision invariants before any
        # event is emitted -- mirrors `TaskClassifier.canonical_events`.
        BudgetAllocator().revise(state.budget, plan, policy_hash)
        events: list[EventPayload] = list(
            classifier.provenance_events(
                record, created_at=self.engine.clock.now(), uuids=self.engine.uuids
            )
        )
        events.append(TaskClassified(signature=signature, record=record))
        events.append(
            BudgetRevised(plan=plan, policy_version=plan.policy_version, policy_hash=policy_hash)
        )
        stored = tuple(self.engine.make_event(run_id, item, module_id="M01") for item in events)
        self.engine.append(run_id, state.version, stored)
        result = self.engine.inspect(run_id).task_signature
        assert result is not None
        return result

    async def classify_task_semantic(
        self,
        run_id: UUID,
        envelope: TaskEnvelope,
        *,
        allow_model: bool = True,
        tier_policy: TierPolicy | None = None,
        deployment: DeploymentLimits | None = None,
    ) -> TaskSignature:
        """`classify_task`, optionally with a real M01 semantic call in between
        the bootstrap budget and the final classification.

        `allow_model=False` makes the zero-call deterministic-fallback path a
        first-class, directly selectable configuration (not merely a
        test-only shortcut): `proposal` stays `None` and
        `TaskClassifier.classify` runs its honest, fully-floored fallback.
        The bootstrap budget is always persisted first (see `classify_task`'s
        docstring for why), so the semantic call -- when attempted -- reserves
        against a real, already-committed budget exactly as production
        traffic does.
        """
        state = self.engine.inspect(run_id)
        if state.task_signature is not None:
            return state.task_signature
        self._bootstrap_budget(run_id, envelope, tier_policy, deployment)
        proposal: ClassificationOutput | None = None
        model_call_key: str | None = None
        if allow_model:
            execution = await self.components.semantic_runtime.execute(
                run_id=run_id,
                module_id="M01",
                module_version="1.0",
                operation="classify",
                prompt_id="m01.classify",
                prompt_version="1.0",
                canonical_input=envelope.model_dump(mode="json"),
            )
            if isinstance(execution.proposal, ClassificationOutput):
                proposal = execution.proposal
                model_call_key = execution.record.idempotency_key if execution.record else None
        return self._finalize_classification(
            run_id, envelope, proposal, model_call_key, tier_policy, deployment
        )

    async def formalise_problem(
        self,
        run_id: UUID,
        envelope: TaskEnvelope,
        *,
        allow_model: bool = True,
        trusted_verifier_results: dict[str, object] | None = None,
    ) -> ProblemSpec:
        """M03 (problem formalisation) + M09 (ledger/blocker/contradiction provenance).

        Resumable: a run whose `problem_spec` is already persisted returns it
        unchanged and never re-invokes the provider. `known_ledger_refs` is
        always the run's real, already-applied ledger revisions (never an
        empty/caller-guessed set), so a same-proposal `support` reference
        that legitimately targets an already-persisted node (e.g. on a retry
        after a partial interruption) resolves exactly as it would replay.
        """
        state = self.engine.inspect(run_id)
        if state.problem_spec is not None:
            return state.problem_spec
        proposal: ProblemFormalisationOutput | None = None
        if allow_model:
            execution = await self.components.semantic_runtime.execute(
                run_id=run_id,
                module_id="M03",
                module_version="1.0",
                operation="formalise",
                prompt_id="m03.formalise",
                prompt_version="1.0",
                canonical_input=envelope.model_dump(mode="json"),
            )
            if isinstance(execution.proposal, ProblemFormalisationOutput):
                proposal = execution.proposal
        available_artifacts = frozenset(item.sha256 for item in envelope.attachments)
        known_ledger_refs = frozenset((node.node_id, node.revision) for node in state.ledger.nodes)
        events = self.components.formaliser.canonical_events(
            envelope,
            proposal,
            created_at=self.engine.clock.now(),
            uuids=self.engine.uuids,
            available_artifacts=available_artifacts,
            trusted_verifier_results=trusted_verifier_results,  # type: ignore[arg-type]
            known_ledger_refs=known_ledger_refs,
        )
        stored = tuple(self.engine.make_event(run_id, item, module_id="M03") for item in events)
        current = self.engine.inspect(run_id)
        self.engine.append(run_id, current.version, stored)
        result = self.engine.inspect(run_id).problem_spec
        assert result is not None
        return result

    async def select_representation(
        self, run_id: UUID, *, allow_adjudication: bool = True
    ) -> RepresentationPlanV2:
        """M04 bound (v2) representation selection + optional budgeted adjudication + artifacts.

        Resumable: a run whose `representation_plan_v2` is already persisted
        returns it unchanged. Adjudication (a real semantic call, charged
        through the composed `SemanticModelRuntime` exactly like every other
        Wave 3 model call) is only ever attempted when the deterministic
        selection is genuinely tied within its own declared `tie_band` --
        never a floating condition outside that one declared policy gate --
        mirroring `RepresentationSelector.select_with_adjudication_bound`.
        This method does not call that helper directly because a real
        semantic call can itself append events (advancing the run's
        committed version) between the deterministic selection and the final
        persisted batch; the plan's `source_snapshot_version` is therefore
        always re-derived against the run's version as it stands immediately
        before the final append, never a value captured before a possible
        adjudication call landed -- otherwise the reducer's binding check
        (`RepresentationPlanSelectedV2`'s `source_snapshot_version != state.version`)
        would reject every adjudicated plan.
        """
        state = self.engine.inspect(run_id)
        if state.representation_plan_v2 is not None:
            return state.representation_plan_v2
        if state.problem_spec is None:
            raise ValueError("representation selection requires a formalised ProblemSpec")
        if state.task_signature is None or state.budget.plan is None:
            raise ValueError(
                "representation selection requires an authoritative classification and an "
                "allocated budget"
            )
        selector = self.components.representation_selector
        policy = self.components.policy.representation_selection
        registry = default_registry_v2()
        deterministic = selector.select_bound(
            state.problem_spec,
            state.task_signature,
            state.budget.plan,
            state.version,
            registry,
            policy,
        )
        plan = deterministic
        candidates = tuple(view for view in deterministic.views if view.builder_available)
        if (
            allow_adjudication
            and policy.model_adjudication_enabled
            and deterministic.tie_triggered
            and len(candidates) >= 2
        ):
            execution = await self.components.semantic_runtime.execute(
                run_id=run_id,
                module_id="M04",
                module_version="1.0",
                operation="adjudicate",
                prompt_id="m04.adjudicate",
                prompt_version="1.0",
                canonical_input={
                    "problem_spec_hash": deterministic.problem_spec_hash,
                    "candidates": [
                        {
                            "kind": view.kind.value,
                            "compatibility_score": view.compatibility_score,
                            "score_components": [
                                item.model_dump(mode="json") for item in view.score_components
                            ],
                            "purpose": view.purpose,
                            "limitations": list(view.limitations),
                        }
                        for view in candidates
                    ],
                    "problem_summary": {
                        "objectives": [
                            item.model_dump(mode="json") for item in state.problem_spec.objectives
                        ],
                        "constraints": [
                            item.model_dump(mode="json") for item in state.problem_spec.constraints
                        ],
                        "unknowns": [
                            item.model_dump(mode="json") for item in state.problem_spec.unknowns
                        ],
                        "relations": [
                            item.model_dump(mode="json") for item in state.problem_spec.relations
                        ],
                    },
                    "view_limit": state.budget.plan.search.max_representation_views,
                },
            )
            if execution.record is not None:
                refreshed = self.engine.inspect(run_id)
                assert refreshed.problem_spec is not None
                assert refreshed.task_signature is not None
                assert refreshed.budget.plan is not None
                rescoped = selector.select_bound(
                    refreshed.problem_spec,
                    refreshed.task_signature,
                    refreshed.budget.plan,
                    refreshed.version,
                    registry,
                    policy,
                )
                proposal = (
                    execution.proposal
                    if isinstance(execution.proposal, RepresentationAdjudicationOutput)
                    else None
                )
                outcome = selector.apply_adjudication_v2(
                    rescoped,
                    proposal,
                    allowed_kinds=frozenset(view.kind for view in candidates),
                    view_limit=state.budget.plan.search.max_representation_views,
                    adjudication_record_ref=execution.record.idempotency_key,
                )
                plan = outcome.plan
        final_state = self.engine.inspect(run_id)
        if plan.source_snapshot_version != final_state.version:
            # Should be unreachable given the re-derivation above; fail loud
            # rather than let a stale binding reach the reducer, which would
            # reject it anyway but with a less specific diagnostic.
            raise ValueError(
                "representation plan source_snapshot_version drifted before persistence"
            )
        stored_bytes: list[tuple[ArtifactRef, int]] = []

        def writer(content_bytes: bytes) -> ArtifactRef:
            descriptor = self.engine.store_artifact(content_bytes, media_type="application/json")
            ref = ArtifactRef(artifact_id=descriptor.id, sha256=descriptor.sha256)
            stored_bytes.append((ref, len(content_bytes)))
            return ref

        assert final_state.problem_spec is not None
        built_artifacts = tuple(
            selector.build_bound(view, final_state.problem_spec, plan, writer, registry=registry)
            for view in plan.views
        )
        # `RepresentationPlanSelectedV2` must be the FIRST event in this batch:
        # `FrontierReasoningEngine.append` assigns sequence numbers (and hence
        # the reducer-visible `state.version`) per EVENT, not once per batch
        # (see `engine.py`'s own docstring on this), so any event ahead of it
        # here would advance `state.version` past `plan.source_snapshot_version`
        # (bound to `final_state.version`, captured before this batch) before
        # the plan's own binding check ever runs. The `ArtifactRegistered`
        # events must, in turn, precede the `RepresentationArtifactCompiledV2`
        # events that require their sha256 already present in `state.artifacts`.
        events: list[EventPayload] = [RepresentationPlanSelectedV2(plan=plan)]
        events.extend(
            ArtifactRegistered(artifact=ref, media_type="application/json", byte_size=size)
            for ref, size in stored_bytes
        )
        events.extend(
            RepresentationArtifactCompiledV2(artifact=artifact) for artifact in built_artifacts
        )
        stored_events = tuple(
            self.engine.make_event(run_id, item, module_id="M04") for item in events
        )
        self.engine.append(run_id, final_state.version, stored_events)
        result = self.engine.inspect(run_id).representation_plan_v2
        assert result is not None
        return result

    def compile_context(
        self,
        run_id: UUID,
        *,
        profile: CompilerProfile = CompilerProfile.STANDARD,
        terminal_disposition: str | None = None,
    ) -> str:
        """M12: persist a typed Wave 3 context packet reflecting the run's
        current, already-applied M01-M04/M09 state.

        Resumable/idempotent: if the run already carries a `ContextCompiledV2`
        packet whose `wave3_context` refs (`problem_spec_ref`, `ledger_root`,
        `budget_plan_ref`) already match the run's CURRENT referenced state --
        at the same `(profile, terminal_disposition)` -- that packet's hash is
        returned and nothing new is appended. Comparing `packet.
        snapshot_version == state.version` directly would never match on a
        genuine resume: compiling and persisting a packet itself appends the
        `ArtifactRegistered`/`ContextCompiledV2` events that advance
        `state.version` past the packet's own `snapshot_version` (which is
        pinned to the version immediately BEFORE that batch) the moment it
        first succeeds -- so this compares the real, independently-hashed
        referenced state instead of a version counter that can never recur.
        """
        state = self.engine.inspect(run_id)
        if state.problem_spec is None:
            raise ValueError("context compilation requires a formalised ProblemSpec")
        problem_ref = canonical_hash(state.problem_spec)
        ledger_root = canonical_hash(state.ledger)
        budget_plan_ref = (
            canonical_hash(state.budget.plan) if state.budget.plan is not None else None
        )
        for packet in reversed(state.context_packets):
            wave3 = packet.wave3_context
            if (
                wave3 is not None
                and packet.profile == profile
                and packet.terminal_disposition == terminal_disposition
                and wave3.problem_spec_ref == problem_ref
                and wave3.ledger_root == ledger_root
                and wave3.budget_plan_ref == budget_plan_ref
            ):
                return packet.packet_hash
        return self.components.context_runtime.compile_and_persist(
            run_id, profile=profile, terminal_disposition=terminal_disposition
        )

    async def execute_front_end(
        self,
        run_id: UUID,
        envelope: TaskEnvelope,
        *,
        allow_model: bool = True,
        allow_adjudication: bool = True,
        tier_policy: TierPolicy | None = None,
        deployment: DeploymentLimits | None = None,
        context_profile: CompilerProfile = CompilerProfile.STANDARD,
    ) -> Wave3PipelineResult:
        """The one public, authoritative, replayable Wave 3 front-end path.

        Sequences M01 -> M02 -> M03 -> M04 -> M12 through the resumable steps
        above. A run that has already completed some prefix of these steps
        (e.g. resumed after interruption) skips straight past them -- no step
        here re-invokes the provider or re-allocates a bootstrap budget once
        its own product is already persisted. A completed run (every step's
        product already present, including a context packet compiled at the
        run's current version) replays this entire method with zero provider
        invocations.
        """
        await self.classify_task_semantic(
            run_id,
            envelope,
            allow_model=allow_model,
            tier_policy=tier_policy,
            deployment=deployment,
        )
        await self.formalise_problem(run_id, envelope, allow_model=allow_model)
        await self.select_representation(run_id, allow_adjudication=allow_adjudication)
        packet_hash = self.compile_context(run_id, profile=context_profile)
        state = self.engine.inspect(run_id)
        material_blockers = tuple(
            blocker for blocker in state.problem_blockers if not blocker.resolvable
        )
        return Wave3PipelineResult(
            run_id=run_id,
            version=state.version,
            task_signature=state.task_signature,
            classification_record=state.classification_record,
            problem_spec=state.problem_spec,
            problem_blockers=state.problem_blockers,
            representation_plan=state.representation_plan_v2,
            context_packet_hash=packet_hash,
            blocked=bool(material_blockers),
        )

    # ------------------------------------------------------------------
    # M13: stop evaluation/finalisation over the coordinator's own real state.
    # ------------------------------------------------------------------

    def evaluate_stop(
        self,
        run_id: UUID,
        *,
        acceptance: AcceptanceStatus = AcceptanceStatus.PENDING,
        validation: ValidationStatus = ValidationStatus.NOT_APPLICABLE,
        mandatory_action: bool = False,
        cancelled: bool = False,
        invariant_failure: bool = False,
        stable: bool = True,
        value: MarginalValueEstimate | None = None,
        cost: CostEstimateInterval | None = None,
        policy: StopPolicy | None = None,
    ) -> StopDecision:
        """Evaluate M13 over this run's real, already-applied blockers/budget.

        `blocker_required`/`blocker_resolvable`/`epistemic_trigger_refs` are
        always derived from `state.problem_blockers` as actually persisted by
        `formalise_problem` -- never from a caller's separate claim about
        whether a blocker exists. This is what makes the decision recomputed
        here genuinely equal across two calls separated only by unrelated
        state changes (e.g. an unrelated context compilation): the inputs
        that matter to the disposition are unchanged, so the decision is too,
        even though `evaluated_state_version`/`evaluated_state_hash` (bound
        into the persisted `StopDecisionRecord`, not into `StopDecision`
        itself) differ.
        """
        state = self.engine.inspect(run_id)
        remaining = BudgetMeter().remaining(state.budget)
        material_blockers = tuple(
            blocker for blocker in state.problem_blockers if not blocker.resolvable
        )
        resolvable_blockers = tuple(
            blocker for blocker in state.problem_blockers if blocker.resolvable
        )
        blocker_required = bool(state.problem_blockers)
        blocker_resolvable = bool(resolvable_blockers) and not material_blockers
        inputs = StopInputs(
            budget=remaining,
            acceptance=acceptance,
            validation=validation,
            cancelled=cancelled,
            invariant_failure=invariant_failure,
            required_work=blocker_required or mandatory_action,
            executable_work=not blocker_required and not mandatory_action,
            blocker_required=blocker_required,
            blocker_resolvable=blocker_resolvable,
            mandatory_action=mandatory_action,
            stable=stable,
            value=value,
            cost=cost,
            epistemic_trigger_refs=tuple(blocker.ledger_ref for blocker in state.problem_blockers),
        )
        decision = StopController().evaluate(inputs, policy or StopPolicy(version="wave3-m13/1.0"))
        self.components.wave2_runtime.record_decision(run_id, decision)
        return decision

    def finalize_if_terminal(self, run_id: UUID, decision: StopDecision) -> str | None:
        """Finalize (persisting a terminal Wave 2 + Wave 3 context) only when terminal."""
        if decision.disposition is StopDisposition.CONTINUE:
            return None
        return self.finalize(run_id, decision)


def _require(name: str, actual: str, supported: str) -> None:
    if actual != supported:
        raise UnsupportedWave3Configuration(
            f"unsupported {name} {actual!r}; supported version is {supported!r}"
        )


def compose_wave3(
    engine: FrontierReasoningEngine,
    model: StructuredModelPort,
    config: Wave3Config | None = None,
) -> Wave3Engine:
    """Translate configuration once and inject the resulting policies everywhere."""
    configured = config or Wave3Config()
    _require("Wave3 configuration schema", configured.schema_version, "1.0")
    _require("classification policy", configured.classification_policy_version, "wave3-m01/1.0")
    _require(
        "semantic runtime policy",
        configured.semantic_runtime_policy_version,
        "wave3-semantic-runtime/2.0",
    )
    _require(
        "representation selection policy",
        configured.representation_selection_policy_version,
        "wave3-m04/1.0",
    )
    _require(
        "representation registry",
        configured.representation_registry_version,
        "wave3-m04-registry/1.0",
    )
    _require("prompt registry", configured.prompt_registry_version, "wave3-prompts/1.0")
    _require(
        "output schema registry", configured.output_schema_registry_version, "wave3-schemas/1.0"
    )
    _require("Wave3 context compiler", configured.wave3_context_compiler_version, "2.0")

    classification = ClassificationPolicy(
        version=configured.classification_policy_version,
        confidence_threshold=configured.confidence_threshold,
    )
    semantic = SemanticRuntimePolicy(
        version=configured.semantic_runtime_policy_version,
        maximum_repair_attempts=configured.maximum_repair_attempts,
        reserve_input_tokens=configured.reserve_input_tokens,
        reserve_output_tokens=configured.reserve_output_tokens,
    )
    representation = RepresentationSelectionPolicy(
        version=configured.representation_selection_policy_version,
        registry_version=configured.representation_registry_version,
        tie_band=configured.representation_tie_band,
        minimum_compatibility=configured.representation_minimum_compatibility,
        model_adjudication_enabled=configured.model_adjudication_enabled,
    )
    effective = EffectiveWave3Policy(
        schema_version=configured.schema_version,
        classification=classification,
        semantic_runtime=semantic,
        representation_selection=representation,
        prompt_registry_version=configured.prompt_registry_version,
        output_schema_registry_version=configured.output_schema_registry_version,
        context_compiler_version=configured.wave3_context_compiler_version,
    )
    prompts = default_prompt_registry()
    schemas = default_output_schema_registry()
    context_compiler = Wave3ContextCompiler()
    context_runtime = Wave3ContextRuntime(engine, context_compiler)
    components = Wave3Components(
        policy=effective,
        classifier=TaskClassifier(classification),
        semantic_runtime=SemanticModelRuntime(
            model,
            engine,
            prompts,
            schemas,
            semantic,
            execution_config_hash=effective.policy_hash,
        ),
        formaliser=ProblemFormaliser(),
        representation_selector=RepresentationSelector(representation, default_registry()),
        context_compiler=context_compiler,
        context_runtime=context_runtime,
        # Finding E (C08 remediation): compose `Wave2Runtime` with this run's
        # `Wave3ContextRuntime` so `Wave3Engine.finalize` (and any other real
        # caller of this composed `Wave2Runtime`) actually persists a typed
        # Wave 3 context packet on every terminal disposition, not merely
        # returns one from an in-memory-only call nothing durable reads.
        wave2_runtime=Wave2Runtime(engine, wave3_context_runtime=context_runtime),
        prompts=prompts,
        schemas=schemas,
    )
    return Wave3Engine(engine, components)
