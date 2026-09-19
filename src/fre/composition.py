"""The narrow composition root for configured Wave 3 services.

C09 (F08) remediation: `Wave3Engine` used to compose and version-pin every
Wave 3 collaborator (M01, M02, M04, M12, ...) without ever sequencing them --
its only method was `create_run()`. The M01 -> M02 -> M03 -> M04 -> M12 flow
that actually executes the Wave 3 front end existed only as hand-wired,
test-only code (`tests/integration/test_wave3_gate.py`), constructing every
module directly and never going through `compose_wave3`.

The methods added to `Wave3Engine` below are now the one public, RECOMMENDED,
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

Independent-review remediation (PR #24 finding F): `execute_front_end` is
RECOMMENDED, not the ONLY callable path, and this module previously overclaimed
otherwise ("the one public, authoritative... path"). Every individual step
method on `Wave3Engine` (`classify_task`, `classify_task_semantic`,
`formalise_problem`, `select_representation`, `compile_context`,
`evaluate_stop`, `finalize_if_terminal`) remains independently public and
callable on its own, and so do the underlying `wave3.components.<module>`
collaborators (`components.classifier`, `components.formaliser`,
`components.representation_selector`, `components.context_compiler`, ...) --
`Wave3Components` is a plain `@dataclass`, not a private/sealed object, and
nothing in this codebase makes its module attributes non-public. Calling a
module directly (bypassing this coordinator) is a real, supported
capability (every pre-C09 test in this repo did exactly that), but it loses
whichever of this coordinator's OWN guards are not independently
re-enforced by that module's own reducer-level checks. Concretely, as of the
C09 (F08) remediation in PR #24 (finding D):
  - The precondition guards on `compile_context`/`evaluate_stop`/
    `finalize_if_terminal` (requiring M03, or M01+M02+M03, or a matching
    recorded `StopDecision`, respectively) ARE enforced structurally by the
    reducer/`BudgetMeter` beneath them too (calling the underlying module
    directly with genuinely missing prerequisite state still fails loudly,
    just with a less specific message coming from a deeper layer) -- so
    bypassing this coordinator does not silently corrupt state on these axes.
  - What IS lost by bypassing `classify_task`/`formalise_problem`/
    `select_representation`/`compile_context` specifically is this
    coordinator's OWN resumability/idempotency discipline: the "already
    persisted, no-op, no duplicate provider call" checks each of those
    methods performs by re-inspecting `self.engine.inspect(run_id)` before
    acting are convention enforced by THIS FILE, not by the reducer -- a
    caller hand-wiring a module directly (as `select_with_adjudication_bound`
    still does not fully replicate -- see finding H) can re-invoke a real
    semantic-model call or re-append an event batch that the coordinator
    path would have skipped. `execute_front_end` is the only path that gets
    that discipline for free across the whole M01-M04/M12 sequence.
"""

from dataclasses import dataclass
from uuid import UUID

from fre.config import Wave3Config
from fre.domain.budget import DeploymentLimits, TierPolicy
from fre.domain.common import ArtifactRef, FrozenModel, canonical_hash
from fre.domain.context import CompilerProfile
from fre.domain.problem import ProblemBlocker, ProblemSpec
from fre.domain.representation import RepresentationPlanV2
from fre.domain.representation_registry import RepresentationDefinition, default_registry_v2
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
    default_output_schema_registry,
)
from fre.runtime.budget_meter import BudgetMeter
from fre.runtime.reducer import RunState
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

        EU-36 (w3-cleanup) cross-reference: see `TaskClassifier.
        canonical_events`'s own docstring (finding #9) for the mirror image
        of this note -- it documents why it is designed for exactly one
        calling context (a run's initial classification with no prior budget
        activity) and must never be called by this orchestrator.
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

        Independent-review remediation (PR #24 finding C): `known_ledger_refs`
        used to be captured from the `state` fetched BEFORE the `await`
        above (when `allow_model=True`). `SemanticModelRuntime.execute` can
        itself append real events (e.g. `BudgetReserved`/`BudgetConsumed`,
        and in principle any other concurrent ledger mutation applied to this
        run during that await window) -- so building `known_ledger_refs` from
        the pre-await `state` risked resolving `support` references against a
        ledger snapshot that was already stale by the time this method's own
        batch is appended, exactly the class of bug `select_representation`
        was already careful to avoid for its own `source_snapshot_version`
        (see that method's docstring). Mirroring that correct pattern: after
        the semantic call (if any) completes, `known_ledger_refs` -- like
        `available_artifacts` -- is rebuilt from a FRESH `self.engine.
        inspect(run_id)` call, not the snapshot taken before the await.
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
        # Finding C: always re-inspect AFTER the (possible) await above so
        # `known_ledger_refs` reflects the run's real, current ledger state,
        # never a snapshot captured before a concurrent mutation could have
        # landed during the semantic call.
        current = self.engine.inspect(run_id)
        available_artifacts = frozenset(item.sha256 for item in envelope.attachments)
        known_ledger_refs = frozenset(
            (node.node_id, node.revision) for node in current.ledger.nodes
        )
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
        never a floating condition outside that one declared policy gate.

        Independent-review remediation (PR #24 finding H): this method used
        to duplicate `RepresentationSelector.select_with_adjudication_bound`'s
        entire adjudication sequence inline (~90 lines), specifically because
        that shared helper had its own stale-`source_snapshot_version` bug --
        it bound the final plan to the version captured BEFORE `runtime.
        execute`'s `await`, even though that call can itself append real
        events (advancing the run's committed version) before the final
        persisted batch. That helper now re-derives its own deterministic
        plan from a fresh `runtime.engine.inspect(run_id)` after the semantic
        call lands (see its own docstring), so this method calls it directly
        instead of maintaining a second, duplicate implementation of the same
        fix. `allow_adjudication=False` is expressed by passing `runtime=
        None` -- `select_with_adjudication_bound` treats a `None` runtime
        exactly like "adjudication is not possible" and returns the
        deterministic plan untouched, without ever constructing a semantic
        call.
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
        registry = default_registry_v2()
        plan, final_state = await self._select_and_verify_plan(
            run_id, state, allow_adjudication=allow_adjudication, registry=registry
        )
        return self._build_and_persist_artifacts(run_id, plan, final_state, registry=registry)

    async def _select_and_verify_plan(
        self,
        run_id: UUID,
        state: RunState,
        *,
        allow_adjudication: bool,
        registry: tuple[RepresentationDefinition, ...],
    ) -> tuple[RepresentationPlanV2, RunState]:
        """Steps 2-3 of `select_representation`: deterministic+adjudication
        selection, then binding-drift verification against a fresh state.

        Mirrors `_bootstrap_budget`'s role in `classify_task`'s own
        decomposition -- everything up to (but not including) building and
        persisting the resulting artifacts.
        """
        assert state.problem_spec is not None
        assert state.task_signature is not None
        assert state.budget.plan is not None
        selector = self.components.representation_selector
        policy = self.components.policy.representation_selection
        outcome = await selector.select_with_adjudication_bound(
            state.problem_spec,
            state.task_signature,
            state.budget.plan,
            state.version,
            runtime=self.components.semantic_runtime if allow_adjudication else None,
            run_id=run_id,
            registry=registry,
            policy=policy,
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
        return plan, final_state

    def _build_and_persist_artifacts(
        self,
        run_id: UUID,
        plan: RepresentationPlanV2,
        final_state: RunState,
        *,
        registry: tuple[RepresentationDefinition, ...],
    ) -> RepresentationPlanV2:
        """Steps 4-6 of `select_representation`: build the plan's artifacts,
        construct and append the event batch, and read back the result.

        Mirrors `_finalize_classification`'s role in `classify_task`'s own
        decomposition.
        """
        selector = self.components.representation_selector
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
        packet that already reflects the run's CURRENT referenced state -- at
        the same `(profile, terminal_disposition)` -- that packet's hash is
        returned and nothing new is appended. Comparing `packet.
        snapshot_version == state.version` directly would never match on a
        genuine resume: compiling and persisting a packet itself appends the
        `ArtifactRegistered`/`ContextCompiledV2` events that advance
        `state.version` past the packet's own `snapshot_version` (which is
        pinned to the version immediately BEFORE that batch) the moment it
        first succeeds -- so this compares the real, independently-hashed
        referenced state instead of a version counter that can never recur.

        Independent-review remediation (PR #24 finding B): the match key
        previously covered only `problem_spec_ref`/`ledger_root`/
        `budget_plan_ref`. Two real staleness gaps followed from that:

        1. A packet compiled before `select_representation` landed (so its
           `wave3_context` carries no representation state) was
           indistinguishable, on those three fields alone, from a packet
           compiled after -- `select_representation` never touches
           `problem_spec`/`ledger`/`budget.plan`. `compile_context` would
           therefore return the STALE, pre-representation packet forever,
           even after a representation plan was selected in between. Fixed
           below by comparing `state.representation_plan_v2.
           source_snapshot_version` (when a v2 plan exists) against the
           candidate packet's own `snapshot_version`: a packet only reflects
           a v2 plan that was selected strictly BEFORE that packet was
           compiled (`plan.source_snapshot_version < packet.snapshot_version`)
           -- this is a purely state-derived check, not a new stored/trusted
           packet field, so it needs no reducer or schema change. (`wave3_
           context.representation_plan_ref` is NOT used for this: it is only
           ever populated from the legacy v1 `representation` argument to
           `compile_semantic`, which this v2-only coordinator path never
           supplies, so that field is always `None` here and cannot
           distinguish pre- from post-representation-selection packets.)
        2. `task_signature` (classification) could change on a run (e.g. a
           mid-run revision) without moving `problem_spec_ref`/`ledger_root`/
           `budget_plan_ref` at all. `wave3_context.task_signature_ref` IS
           populated correctly for the real `state.task_signature` (see
           `Wave3ContextCompiler.compile_semantic`), so it is now compared
           directly.
        3. `budget_plan_ref` only hashed `state.budget.plan`'s STRUCTURE
           (limits/policy), not how much of it had actually been consumed or
           reserved -- a packet compiled before a `BudgetReserved`/
           `BudgetConsumed` event would be wrongly treated as still current
           after one landed, even though the run's real remaining budget
           changed. Every packet already carries its own compile-time
           `budget_remaining` (`BudgetMeter().remaining(state.budget)` at
           that moment, via `Wave3ContextRuntime.compile_and_persist`), so
           that full projection is now compared as well, not just the plan
           shape.

        Independent-review remediation (PR #24 finding J): `budget_policy_
        hash`/`prompt_version`/`model_identity`/`size_target`/`next_action`
        are DELIBERATELY still not part of this match key -- verified safe,
        not merely overlooked:
          - `budget_policy_hash` is `state.budget.policy_hash`, which
            `fre.modules.m02_budget.BudgetAllocator.allocate`/`.revise`
            ALWAYS sets in the same atomic `BudgetAllocated`/`BudgetRevised`
            event as `state.budget.plan` itself (see those events' shared
            payload) -- there is no code path in this codebase that changes
            one without the other. `budget_plan_ref` (already compared
            above) can therefore never go stale while `budget_policy_hash`
            silently changes underneath it.
          - `prompt_version`/`model_identity`/`size_target`/`next_action` are
            optional keyword arguments of `Wave3ContextRuntime.
            compile_and_persist` that THIS method never passes -- every call
            `compile_context` ever makes leaves all four at their `None`
            default, unconditionally. They cannot vary across two calls made
            through this method, so they cannot cause a false match here.
            (A caller invoking `compile_and_persist` directly, bypassing this
            coordinator, could vary them -- but that is finding F's
            documented, structurally-unenforced bypass case, not a gap in
            this method's own dedup key.)
        """
        state = self.engine.inspect(run_id)
        # Finding D (precondition guard): M03 is the only genuine hard
        # prerequisite here -- `Wave3ContextRuntime.compile_and_persist`
        # accepts `task_signature`/`representation_v2` as optional (M01/M04
        # need not have run; `derive_wave3_availability` reports that
        # honestly via `Wave3ContextAvailability`, it does not require it).
        #
        # EU-40 (w3-cleanup): investigated removing this guard as a duplicate
        # of `Wave3ContextRuntime.compile_and_persist`'s own identical
        # `problem_spec is None` check. That removal is NOT safe: this method
        # does genuine work between its own guard and the delegate call --
        # notably `BudgetMeter().remaining(state.budget)` below, which raises
        # its own (less specific, differently-worded) error when
        # `state.budget.plan` is `None`, as it always is on a fresh run that
        # has not even reached M01/M02 yet. Removing this guard would
        # therefore surface a confusing "budget has not been allocated"
        # failure instead of this method's own clear, correctly-named
        # precondition message for a run that simply never formalised a
        # `ProblemSpec` -- so both guards are kept; this one stays the
        # earliest, most specific diagnostic for its own precondition, and
        # `compile_and_persist`'s copy remains the single source of truth for
        # any caller that reaches it directly, bypassing this coordinator.
        if state.problem_spec is None:
            raise ValueError("context compilation requires a formalised ProblemSpec")
        problem_ref = canonical_hash(state.problem_spec)
        ledger_root = canonical_hash(state.ledger)
        budget_plan_ref = (
            canonical_hash(state.budget.plan) if state.budget.plan is not None else None
        )
        budget_remaining = BudgetMeter().remaining(state.budget)
        task_signature_ref = (
            canonical_hash(state.task_signature) if state.task_signature is not None else None
        )
        representation_plan_v2 = state.representation_plan_v2
        # EU-35 (w3-cleanup): this scan is reverse-ordered specifically to
        # short-circuit on the common "already compiled recently" case (the
        # most-recent-match, if any, is the first candidate examined). The
        # worst-case O(n) miss-path is bounded by the number of context
        # packets compiled on this run, which is typically small -- accepted
        # as a measured-acceptable cost, not a real bottleneck, pending
        # evidence otherwise. If `state.context_packets` is ever observed
        # growing large enough for this to matter, the fix is a derived index
        # keyed on the match tuple below (a dict overwrite naturally gives
        # "most recent per key"), populated either via a new `RunState` field
        # (an event-sourced schema change) or a per-call local index built
        # from this same tuple (which would not reduce asymptotic cost, only
        # clarify the code) -- scope that as its own unit if/when evidence
        # justifies it.
        for packet in reversed(state.context_packets):
            wave3 = packet.wave3_context
            if (
                wave3 is None
                or packet.profile != profile
                or packet.terminal_disposition != terminal_disposition
                or wave3.problem_spec_ref != problem_ref
                or wave3.ledger_root != ledger_root
                or wave3.budget_plan_ref != budget_plan_ref
                or packet.budget_remaining != budget_remaining
                or wave3.task_signature_ref != task_signature_ref
            ):
                continue
            if (
                representation_plan_v2 is not None
                and representation_plan_v2.source_snapshot_version >= packet.snapshot_version
            ):
                # This packet was compiled at or before the version the
                # current v2 representation plan was selected against, so it
                # cannot reflect that plan -- stale, keep searching/recompile.
                continue
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
        """The RECOMMENDED, replayable Wave 3 front-end path -- not the ONLY
        callable path (see the module docstring's C09/PR #24 finding F
        remediation: every individual step method and the underlying
        `wave3.components.<module>` collaborators remain independently
        public and callable on their own).

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
        material_blockers = _material_blockers(state.problem_blockers)
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

        Independent-review remediation (PR #24 finding D): explicit
        precondition guards, raising a clear `ValueError` naming the missing
        prerequisite, rather than silently proceeding on incomplete state (or
        surfacing only as an unrelated-looking downstream exception).
        `evaluate_stop` genuinely needs both M01+M02 (a budget must actually
        be allocated for `BudgetMeter().remaining` to mean anything -- it
        previously only failed indirectly via `BudgetExceeded` several lines
        below, with no reference to M13/evaluate_stop in the message) and M03
        (formalisation must have run for `state.problem_blockers` to be a
        real, considered judgement rather than merely "empty because M03
        never ran" -- the two are otherwise indistinguishable and this method
        would silently treat "not yet formalised" as "no blockers exist").
        """
        state = self.engine.inspect(run_id)
        if state.budget.plan is None:
            raise ValueError(
                "evaluate_stop requires M01/M02 classification and budget allocation to have "
                "run on this run before a stop decision can be evaluated"
            )
        if state.problem_spec is None:
            raise ValueError(
                "evaluate_stop requires M03 problem formalisation to have run on this run "
                "before its blockers can be evaluated"
            )
        remaining = BudgetMeter().remaining(state.budget)
        material_blockers = _material_blockers(state.problem_blockers)
        resolvable_blockers = _resolvable_blockers(state.problem_blockers)
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
        """Finalize (persisting a terminal Wave 2 + Wave 3 context) only when terminal.

        Independent-review remediation (PR #24 finding D): `finalize`/
        `Wave2Runtime.finalize` genuinely require `evaluate_stop` (which
        itself now guards M01/M02/M03 -- see that method's docstring) to have
        already been called and its exact `decision` already recorded via
        `record_decision` on this run; `Wave2Runtime.finalize` does check
        this, but several lines deep and with a message that does not name
        `evaluate_stop` as the missing step. That is surfaced here, one call
        earlier, with a clearer message naming the actual prerequisite.
        """
        if decision.disposition is StopDisposition.CONTINUE:
            return None
        state = self.engine.inspect(run_id)
        if not state.stop_decisions or state.stop_decisions[-1] != decision:
            raise ValueError(
                "finalize_if_terminal requires this exact StopDecision to have already been "
                "recorded via evaluate_stop on this run before it can be finalized"
            )
        return self.finalize(run_id, decision)


def _material_blockers(
    problem_blockers: tuple[ProblemBlocker, ...],
) -> tuple[ProblemBlocker, ...]:
    """Blockers that are not resolvable -- shared by `execute_front_end` and
    `evaluate_stop`, which both need this exact filter (EU-37)."""
    return tuple(blocker for blocker in problem_blockers if not blocker.resolvable)


def _resolvable_blockers(
    problem_blockers: tuple[ProblemBlocker, ...],
) -> tuple[ProblemBlocker, ...]:
    """The inverse of `_material_blockers`, kept alongside it for symmetry."""
    return tuple(blocker for blocker in problem_blockers if blocker.resolvable)


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
