# Frontier Reasoning Engine — Build Architecture v1.0

**Status:** build-authoritative baseline  
**Primary implementation target:** local-first, provider-neutral Python control plane  
**Initial delivery surfaces:** Python SDK and CLI; thin MCP/service adapters after the core is stable  
**Source basis:** the supplied *Frontier Reasoning Engine — Technical Introduction* and the supplied technical pass over Modules 1–13.

---

## 1. Binding architectural decision

The Frontier Reasoning Engine (FRE) will be implemented as a **modular reasoning control plane**, not as one monolithic agent and not as a fixed 13-step prompt chain.

The core engine owns:

1. canonical task and run state;
2. action selection and budget enforcement;
3. immutable events, provenance, and replay;
4. deterministic algorithms for constraints, Pareto analysis, sensitivity, state compilation, and stopping;
5. ports for language models, tools, retrieval, code execution, storage, and human input;
6. the thirteen reasoning modules as independently testable operators.

Language models propose structured semantic judgements, candidates, and challenges. Deterministic code validates schemas, enforces budgets and invariants, computes graph and optimisation operations, persists state, and decides which actions are legally available. No model response may mutate canonical state directly.

### 1.1 Core design principles

- **Canonical state before conversational history.** Conversation text is an input artifact, not the runtime database.
- **Events before mutation.** Modules return typed events; the orchestrator commits them atomically.
- **Partial truth, not forced booleans.** Constraint and evidence states include `UNKNOWN`, `CONTESTED`, and `ERROR`; unknown never means false.
- **Immutable candidates and claims.** Repairs create successor revisions; they do not rewrite history.
- **Representations are projections.** A graph, state machine, decision table, or hypergraph is a view over canonical state, never an alternative source of truth.
- **Deterministic core, model-assisted semantics.** Arithmetic, dominance, dependency checks, budget accounting, replay, and schema validation remain deterministic.
- **Provider neutrality.** Modules request capabilities such as `structured_generation`, `divergent_generation`, `critique`, `retrieval`, or `code_execution`; deployment adapters map these to models and surfaces.
- **Hard resource ceilings.** Soft marginal-value stopping improves efficiency; hard iteration and budget ceilings guarantee termination.
- **Auditability without chain-of-thought dependence.** Persist structured rationales, evidence references, decisions, and concise justifications—not hidden reasoning traces.

---

## 2. Binding refinements to the two drafts

The supplied drafts establish the correct control architecture, but the following refinements are mandatory for implementation.

### 2.1 Non-decidable hard constraints are not demoted

A hard constraint can remain hard even when it requires semantic, model-assisted, or human verification. The implementation records:

```text
constraint_kind = HARD | SOFT
verification_mode = DETERMINISTIC | MODEL | HUMAN | UNAVAILABLE
status = PASS | FAIL | UNKNOWN | ERROR
```

Only a verified `FAIL` makes a candidate infeasible. `UNKNOWN` blocks an unconditional conclusion but does not silently become a soft preference.

### 2.2 Missing objective values do not permit dominance

Pareto pruning applies only across comparable, sufficiently evaluated objective vectors. Missing or materially uncertain values make candidates incomparable unless robust interval dominance can be established.

### 2.3 Confidence is not bounded by the weakest dependency in all cases

Independent evidence can increase confidence. The ledger therefore does not enforce `child_confidence <= min(parent_confidence)` universally. It records a confidence method, evidence independence groups, and provenance. When an upstream claim changes, descendants become `STALE` and are recomputed or revalidated.

### 2.4 The ledger is a typed property graph, not one globally acyclic graph

`supports`, `depends_on`, and `derived_from` edges must form an acyclic support graph for each claim revision. `contradicts` edges may be bidirectional. `supersedes` edges form a temporal revision chain.

### 2.5 Soft stop logic is heuristic; hard caps guarantee termination

Frontier stability and estimated value of information are useful control signals, not a proof of finite convergence in an open-ended model-generated search space. Every run has hard ceilings for iterations, calls, tool actions, and expenditure.

### 2.6 Value-of-information estimates are ranking heuristics unless a calibrated decision model exists

FRE may compute formal EVPI only when probabilities and utilities are explicitly available and validated. Otherwise it uses a clearly labelled heuristic priority score based on likely decision impact, resolvability, reliability, and acquisition cost.

### 2.7 Multi-objective frontiers do not always have a scalar “leader”

Falsification and stopping target the non-dominated frontier, the provisional recommendation under the active decision policy, and candidates whose uncertainty could alter that recommendation. A generic global score is not assumed.

### 2.8 Reproducibility means replay reproducibility

The same stored model/tool outputs and the same reducer/compiler versions must replay to the same state. Re-running a nondeterministic remote model is not guaranteed to reproduce identical content.

---

## 3. System boundary

### 3.1 In scope for v1

- typed ingestion of a task, attachments, constraints, and output contract;
- adaptive reasoning tier selection;
- structured problem formalisation;
- representation planning;
- independent candidate generation with progressive widening;
- four-valued constraint evaluation and dominance pruning;
- pairwise and bounded higher-order synthesis;
- targeted falsification;
- an auditable epistemic ledger;
- Pareto, sensitivity, ablation, regret, and robustness analysis;
- decision-sensitive information acquisition through pluggable tools;
- deterministic context packet compilation;
- marginal-value and hard-budget stopping;
- local persistence, replay, inspection, and resumption;
- SDK, CLI, and later MCP/service exposure.

### 3.2 Explicit non-goals for v1

- training or fine-tuning models;
- a general autonomous agent with unrestricted side effects;
- a distributed workflow cluster;
- exact probabilistic inference where no calibrated probability model exists;
- unrestricted exhaustive combinatorial search;
- a graphical user interface;
- hard-coded dependence on Chat, Work, Codex, or one provider API;
- treating model-written prose as an authoritative state store.

---

## 4. High-level component architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                           Integration Surfaces                       │
│       Python SDK · CLI · MCP adapter · optional HTTP service         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                         Run Orchestrator                             │
│  policy loop · action scheduler · budget meter · retries · commits  │
└───────────────┬────────────────────┬───────────────────┬────────────┘
                │                    │                   │
        ┌───────▼────────┐   ┌───────▼────────┐  ┌──────▼──────────┐
        │ 13 Reasoning   │   │ Deterministic  │  │ Capability /    │
        │ Modules        │   │ Algorithms     │  │ Surface Router  │
        └───────┬────────┘   └───────┬────────┘  └──────┬──────────┘
                │                    │                   │
┌───────────────▼────────────────────▼───────────────────▼────────────┐
│                              Ports                                  │
│ structured model · tools · retrieval · sandbox · human · storage    │
└───────────────┬────────────────────┬───────────────────┬────────────┘
                │                    │                   │
        ┌───────▼────────┐   ┌───────▼────────┐  ┌──────▼──────────┐
        │ SQLite Event   │   │ Artifact Store │  │ Provider / Tool │
        │ Store + Views  │   │ content-hashed │  │ Adapters        │
        └────────────────┘   └────────────────┘  └─────────────────┘
```

### 4.1 Why the core is a control plane

The thirteen modules have different computational semantics. Some are deterministic, some model-assisted, some hybrid, and some can execute in parallel. A control-plane architecture gives them one canonical run state, common budget accounting, shared provenance, and a single invariant-enforcing commit path.

### 4.2 Deployment mapping

The domain model uses generic execution capabilities:

```text
STRUCTURED_MODEL
DIVERGENT_MODEL
CRITIC_MODEL
DETERMINISTIC_COMPUTE
CODE_SANDBOX
RETRIEVAL
EXTERNAL_TOOL
HUMAN_INPUT
```

A deployment adapter may map those capabilities to Chat, Work, Codex, a local model, a web research tool, or another execution system. Those product names must not leak into domain logic.

---

## 5. Runtime and orchestration model

### 5.1 Append-only run events

Every successful module execution returns events rather than mutating shared objects. The orchestrator performs an optimistic-version transaction:

```python
result = await module.execute(action, context)
event_store.append(
    run_id=run_id,
    expected_version=snapshot.version,
    events=result.events,
)
new_snapshot = reducer.apply(snapshot, result.events)
```

If the version has changed, the action is re-evaluated against the new snapshot rather than blindly committed.

### 5.2 Module protocol

```python
class ReasoningModule(Protocol):
    module_id: ModuleId

    def propose_actions(
        self,
        snapshot: RunSnapshot,
        policy: RuntimePolicy,
    ) -> Sequence[ActionProposal]: ...

    async def execute(
        self,
        action: ActionProposal,
        context: ModuleContext,
    ) -> ModuleResult: ...
```

```python
class ActionProposal(BaseModel):
    action_id: UUID
    module_id: ModuleId
    operation: str
    input_refs: list[ObjectRef]
    prerequisites: list[PredicateRef]
    mandatory: bool
    blocking: bool
    expected_decision_impact: float | None
    estimated_cost: CostEstimate
    execution_capabilities: set[Capability]
    reason: str
```

```python
class ModuleResult(BaseModel):
    action_id: UUID
    events: list[DomainEvent]
    artifacts: list[ArtifactDescriptor]
    actual_cost: CostRecord
    warnings: list[Diagnostic]
    proposed_followups: list[ActionProposal]
```

### 5.3 Scheduler order

The default scheduler selects actions in this order:

1. invariant-preserving mandatory actions;
2. unresolved blockers affecting hard constraints or acceptance criteria;
3. actions required by the active tier policy;
4. highest estimated decision impact per normalised cost;
5. context compilation checkpoints;
6. stop evaluation.

Expected impact and cost are only ranking signals. They do not override hard constraints, permissions, safety, or budget ceilings.

### 5.4 Default control loop

```text
INGEST
  → M01 CLASSIFY
  → M02 ALLOCATE
  → M03 FORMALISE
  → M04 SELECT REPRESENTATIONS
  → M05 GENERATE
  → M06 PROPAGATE / PRUNE
  → optional M07 SYNTHESISE → M06
  → M10 PRELIMINARY DECISION / SENSITIVITY
  → M08 FALSIFY decision-relevant survivors → M06 → M10
  → M11 ACQUIRE if decision-sensitive
  → affected modules re-run
  → M13 STOP CHECK
  → M12 COMPILE FINAL CONTEXT
```

This is a default policy, not a hard pipeline. Modules can be skipped, repeated, or run in parallel where their preconditions permit it.

### 5.5 Re-entry rules

- New candidates from M05 or M07 always re-enter M06.
- Verified falsification from M08 invalidates or contests affected candidates and triggers M06/M10.
- New observations from M11 mark dependent evaluations stale and trigger the smallest affected recomputation set.
- A changed hard constraint from M03 invalidates all dependent candidate evaluations.
- M13 can request another M05, M08, or M11 action, but only within the remaining tier budget.

### 5.6 Idempotency

Every action is keyed by:

```text
(run_id, module_id, operation, canonical_input_hash, module_version, policy_hash)
```

Repeated execution with the same key returns the stored result unless explicitly run in `refresh` mode.

---

## 6. Canonical domain contracts

All persistent objects carry:

```text
id · schema_version · created_at · created_by_action · provenance_refs
```

### 6.1 Task envelope

```python
class TaskEnvelope(BaseModel):
    task_id: UUID
    text: str
    attachments: list[ArtifactRef]
    explicit_constraints: list[str]
    requested_output: OutputContract
    user_metadata: dict[str, JsonValue]
    execution_permissions: PermissionSet
```

### 6.2 Task signature

```python
class TaskSignature(BaseModel):
    task_type: TaskType
    consequence: Ordinal4
    irreversibility: Ordinal4
    ambiguity: Ordinal4
    search_space: SearchSpaceClass
    evidence_scarcity: Ordinal4
    horizon: HorizonClass
    output_form: OutputForm
    dimension_confidence: dict[str, float]
    evidence_refs: list[LedgerRef]
```

The implementation uses `irreversibility` rather than an inverted `reversibility` scale so all difficulty axes increase in the same direction.

### 6.3 Budget plan and meter

```python
class BudgetLimits(BaseModel):
    max_iterations: int
    max_llm_calls: int
    max_tool_calls: int
    max_input_tokens: int | None
    max_output_tokens: int | None
    max_candidates: int
    max_concurrent_actions: int
    max_runtime_seconds: int | None

class SearchPolicy(BaseModel):
    initial_candidate_count: int
    max_candidate_depth: int
    max_representation_views: int
    synthesis_order_limit: int
    falsifier_operator_limit: int
    independent_validation_branches: int
    sensitivity_samples: int

class BudgetPlan(BaseModel):
    tier: ReasoningTier
    limits: BudgetLimits
    search: SearchPolicy
    acquisition_threshold: float
    stop_threshold: float
    policy_version: str
```

`BudgetMeter` is a deterministic projection over committed cost events. It is the only authority on remaining resources.

### 6.4 Problem specification

```python
class ObjectiveSpec(BaseModel):
    id: str
    name: str
    direction: Literal["MIN", "MAX", "TARGET", "LEXICOGRAPHIC"]
    unit: str | None
    priority: int | None
    evaluator_ref: str | None
    description: str

class ConstraintSpec(BaseModel):
    id: str
    description: str
    kind: Literal["HARD", "SOFT"]
    verification_mode: Literal["DETERMINISTIC", "MODEL", "HUMAN", "UNAVAILABLE"]
    verifier_ref: str | None
    source_refs: list[LedgerRef]

class UnknownSpec(BaseModel):
    id: str
    description: str
    domain: JsonValue | None
    decision_relevance: float | None
    resolvable: bool | None
    candidate_actions: list[str]

class AcceptanceCriterion(BaseModel):
    id: str
    predicate_description: str
    verification_mode: str
    required: bool

class ProblemSpec(BaseModel):
    decision_variables: list[DecisionVariable]
    objectives: list[ObjectiveSpec]
    constraints: list[ConstraintSpec]
    assumptions: list[LedgerRef]
    unknowns: list[UnknownSpec]
    observables: list[ObservableSpec]
    acceptance_criteria: list[AcceptanceCriterion]
    output_contract: OutputContract
```

### 6.5 Representation plan

```python
class RepresentationView(BaseModel):
    id: str
    kind: RepresentationKind
    role: Literal["PRIMARY", "AUXILIARY"]
    compatibility_score: float
    expected_value: str
    builder_ref: str
    source_object_refs: list[ObjectRef]

class RepresentationPlan(BaseModel):
    views: list[RepresentationView]
    selection_basis: list[LedgerRef]
```

### 6.6 Candidate and lineage

```python
class CandidateComponent(BaseModel):
    id: str
    kind: str
    configuration: dict[str, JsonValue]
    dependencies: list[str]
    interfaces: list[str]

class Candidate(BaseModel):
    candidate_id: UUID
    revision: int
    parent_ids: list[UUID]
    generator_operator: str
    representation_refs: list[str]
    title: str
    summary: str
    components: list[CandidateComponent]
    assumptions: list[LedgerRef]
    predicted_effects: list[PredictedEffect]
    status: CandidateStatus
    structural_fingerprint: str
```

Candidates are immutable. A repair or refinement creates a new revision with `parent_ids`.

### 6.7 Evaluations

```python
class ConstraintEvaluation(BaseModel):
    candidate_id: UUID
    constraint_id: str
    status: Literal["PASS", "FAIL", "UNKNOWN", "ERROR"]
    confidence: ConfidenceAssessment
    evidence_refs: list[LedgerRef]
    verifier_version: str

class ObjectiveEstimate(BaseModel):
    candidate_id: UUID
    objective_id: str
    point: float | None
    lower: float | None
    upper: float | None
    distribution_ref: ArtifactRef | None
    quality: EvidenceQuality
    evidence_refs: list[LedgerRef]
```

### 6.8 Epistemic graph

```python
class LedgerNode(BaseModel):
    node_id: UUID
    revision: int
    node_type: LedgerNodeType
    content: JsonValue
    epistemic_status: EpistemicStatus
    confidence: ConfidenceAssessment | None
    source_refs: list[ArtifactRef]

class LedgerEdge(BaseModel):
    edge_id: UUID
    source_id: UUID
    target_id: UUID
    relation: LedgerRelation
    independence_group: str | None
    metadata: dict[str, JsonValue]
```

`LedgerNodeType` includes:

```text
FACT · OBSERVATION · INFERENCE · ASSUMPTION · HYPOTHESIS · PROPOSAL
DECISION · UNKNOWN · CHALLENGE · EVIDENCE · ARTIFACT
```

`EpistemicStatus` includes:

```text
SUPPORTED · PROVISIONAL · CONTESTED · REFUTED · UNRESOLVED · STALE
```

### 6.9 Decision output

```python
class DecisionResult(BaseModel):
    feasible_candidates: list[UUID]
    conditional_candidates: list[UUID]
    pareto_front: list[UUID]
    recommendation: UUID | None
    selection_policy: str | None
    tradeoffs: list[Tradeoff]
    sensitivity_findings: list[SensitivityFinding]
    ablation_findings: list[AblationFinding]
    regret_findings: list[RegretFinding]
    unstable_assumptions: list[LedgerRef]
    status: Literal["RESOLVED", "FRONTIER", "BLOCKED", "INSUFFICIENT_EVIDENCE"]
```

### 6.10 Context packet and stop decision

```python
class ContextPacket(BaseModel):
    run_id: UUID
    snapshot_version: int
    objective: list[ObjectiveSpec]
    hard_constraints: list[ConstraintSpec]
    decisions_fixed: list[LedgerRef]
    verified_facts: list[LedgerRef]
    important_assumptions: list[LedgerRef]
    active_candidates: list[UUID]
    rejected_candidates: list[RejectedCandidate]
    current_frontier: list[UUID]
    unresolved_high_value_questions: list[UnknownSpec]
    budget_remaining: BudgetRemaining
    next_action: ActionProposal | None
    compiler_version: str
    packet_hash: str

class StopDecision(BaseModel):
    disposition: Literal[
        "CONTINUE", "COMPLETE", "PARTIAL_BUDGET", "BLOCKED",
        "FAILED_INVARIANT", "CANCELLED"
    ]
    reasons: list[str]
    triggering_refs: list[ObjectRef]
    final_context_packet_ref: ArtifactRef | None
```

---

## 7. Ports, adapters, and execution routing

### 7.1 Structured model port

```python
class StructuredModelPort(Protocol):
    async def generate(
        self,
        *,
        role: ModelRole,
        messages: list[Message],
        output_schema: type[T],
        limits: ModelCallLimits,
        idempotency_key: str,
    ) -> ModelCallResult[T]: ...
```

Requirements:

- provider-native structured output when available;
- strict schema validation;
- at most one schema-repair attempt;
- persisted request template version, model identifier, usage, parsed result, and raw-response artifact;
- transport retries only for classified transient failures;
- no model call can bypass the budget meter.

### 7.2 Tool and acquisition port

```python
class ToolPort(Protocol):
    async def execute(self, request: ToolRequest) -> ToolResult: ...
```

All tools declare:

```text
capabilities · side_effect_level · permissions · cost model · result schema
```

v1 defaults to read-only tools and sandboxed code execution. Any external side effect requires an explicit permission token.

### 7.3 Storage ports

```text
EventStore       append/load events with optimistic versioning
SnapshotStore    read/write periodic state snapshots
ArtifactStore    content-addressed files and large payloads
ProjectionStore  queryable ledger/candidate/call projections
CacheStore       idempotent action and model-call cache
```

### 7.4 Capability router

The router resolves an action's required capability set against available adapters, policy, privacy, and cost. It returns a routing decision or an explicit `UNAVAILABLE` result. Modules do not contain provider-specific branching.

---

# 8. Architecture of the thirteen modules

Each module section below is binding for its public contract. Internal algorithms may evolve if tests preserve these semantics.

## M01 — Task Classifier

### Responsibility

Compile the task envelope into a typed task signature used by budget and policy decisions. It does not solve the task and does not select a model by name.

### Execution mode

**Hybrid:** deterministic extraction of explicit metadata plus one constrained structured-model call when semantic classification is required.

### Consumes

- `TaskEnvelope`
- configured classification policy
- explicit domain/risk rules

### Produces

- `TaskSignature`
- ledger nodes for classification evidence and assumptions
- diagnostics for low-confidence dimensions

### Algorithm

1. Extract explicit output form, attachment types, permissions, and user-declared constraints deterministically.
2. Apply policy rules that establish risk floors; for example, configured high-consequence domains cannot be classified below the policy minimum.
3. Request a structured semantic classification for task type, ambiguity, search-space class, evidence scarcity, and horizon.
4. Merge deterministic and model-derived fields with deterministic rules taking precedence.
5. Validate every enum and range.
6. Increase ambiguity or evidence scarcity when dimension confidence is below threshold; never fabricate certainty.

### Invariants

- every field is present and typed;
- all difficulty axes point in the harder-is-higher direction;
- risk floors cannot be lowered by a model response;
- classification evidence is traceable to the task or a named policy rule.

### Failure behaviour

- malformed model output: one repair attempt, then deterministic conservative fallback;
- unavailable model: rule-only classification with lowered confidence;
- invariant violation: stop the run as `FAILED_INVARIANT`.

### Acceptance tests

- equivalent explicit tasks classify identically under a fake deterministic model;
- adding consequence or irreversibility cannot reduce the downstream tier;
- missing evidence raises, rather than lowers, evidence scarcity.

---

## M02 — Budget Allocator

### Responsibility

Convert the task signature and deployment limits into a bounded reasoning policy. It controls breadth, depth, validation, acquisition, and hard resource ceilings.

### Execution mode

**Deterministic.**

### Consumes

- `TaskSignature`
- deployment ceilings
- tier policy configuration
- optional previous budget revision and committed usage

### Produces

- `BudgetPlan`
- `BudgetAllocated` or `BudgetRevised` event

### Algorithm

1. Convert ordinal axes into a monotone difficulty vector.
2. Compute a configurable base score.
3. Apply hard tier floors and ceilings.
4. Select the tier policy row.
5. Clamp it to deployment limits.
6. Preserve already consumed resources when revising a plan.

A reference form is:

```text
difficulty = wc*consequence
           + wi*irreversibility
           + wa*ambiguity
           + ws*search_space
           + we*evidence_scarcity
```

The linear form is a configurable baseline, not a learned truth. Policy floors handle cases where a weighted sum would be unsafe.

### Dynamic revision

The allocator may raise a tier when M03–M06 reveal hidden complexity. It may reduce future optional work but cannot retroactively alter usage or reduce required validation below a consequence floor.

### Invariants

- monotonicity across harder task signatures;
- hard ceilings are never exceeded;
- tier changes are evented and justified;
- no module can spend outside the plan.

### Acceptance tests

- property-based monotonicity;
- boundary tests for every threshold;
- budget revision preserves consumed counts;
- concurrent actions cannot overspend through a race.

---

## M03 — Problem Formaliser

### Responsibility

Compile task language into an explicit problem specification while separating user-supplied facts, inferred assumptions, unknowns, objectives, hard constraints, soft objectives, and acceptance criteria.

### Execution mode

**Hybrid:** structured model extraction plus deterministic validation and source-span anchoring.

### Consumes

- `TaskEnvelope`
- `TaskSignature`
- relevant ledger nodes and artifact references

### Produces

- `ProblemSpec`
- claim/assumption/unknown ledger nodes
- contradiction and incompleteness diagnostics

### Algorithm

1. Extract candidate decision variables, objectives, constraints, unknowns, observables, and acceptance criteria.
2. Link each item to an explicit source span or label it as inference/assumption.
3. Normalise objective direction and units.
4. Assign each constraint a verification mode without changing whether it is hard or soft.
5. Detect incompatible constraints, missing objective evaluators, and underspecified acceptance criteria.
6. Resolve low-impact ambiguity through an explicitly logged assumption when policy permits.
7. Emit a blocker for material ambiguity that cannot safely be inferred.

### Constraint policy

- `HARD + DETERMINISTIC`: executable predicate.
- `HARD + MODEL`: semantic evaluation, with evidence and confidence.
- `HARD + HUMAN`: remains unresolved until human confirmation or a policy-authorised assumption.
- `HARD + UNAVAILABLE`: blocks an unconditional final answer.

### Invariants

- no inferred statement is stored as a fact;
- hard/soft status is preserved;
- every required acceptance criterion is evaluable or explicitly blocked;
- objectives have a direction or are marked qualitative.

### Acceptance tests

- semantic hard constraints remain hard;
- contradictory constraints are surfaced;
- source-derived and inferred items are distinguishable;
- formalisation round-trips through JSON without semantic loss.

---

## M04 — Representation Selector

### Responsibility

Choose one primary and, when justified, one or more auxiliary problem representations that lower reasoning cost or expose structure.

### Execution mode

**Hybrid:** deterministic feature scoring with optional structured-model adjudication for close cases.

### Representation registry

The v1 registry supports:

```text
DECISION_TABLE · DEPENDENCY_DAG · STATE_MACHINE · EVENT_LOG
KNOWLEDGE_GRAPH · CAUSAL_DAG · CSP · ILP · SCENARIO_TREE
HYPERGRAPH · MORPHOLOGICAL_SPACE · TEXT_FALLBACK
```

Every registry entry declares:

```text
required features · supported operations · build cost · query cost
losses/limitations · compiler implementation
```

### Consumes

- `ProblemSpec`
- task signature and budget plan
- registry metadata

### Produces

- `RepresentationPlan`
- compiled representation artifacts

### Algorithm

1. Compute features such as variable discreteness, temporal structure, dependency density, uncertainty, intervention questions, and likely component interactions.
2. Score compatible representations.
3. Select a primary representation under the tier's view limit.
4. Retain an auxiliary representation only when it exposes materially different information.
5. Compile both from canonical objects; do not duplicate authority.

### Invariants

- canonical state remains representation-independent;
- representation compilation is deterministic for a fixed snapshot and compiler version;
- every selected view has a declared purpose;
- view count respects the budget.

### Acceptance tests

- temporal tasks select an event-log-compatible view;
- dependency tasks select a DAG-compatible view;
- close scores can preserve two lenses without arbitrary tie-breaking;
- rebuilding a view from the same snapshot yields the same hash.

---

## M05 — Candidate / Hypothesis Generator

### Responsibility

Generate structurally independent candidate solutions or hypotheses and progressively widen only promising regions of the search space.

### Execution mode

**Model-assisted with deterministic selection, deduplication, lineage, and budget enforcement.**

### Operator registry

The initial ten operators are:

1. objective-emphasis shift;
2. causal-mechanism shift;
3. representation shift;
4. decomposition shift;
5. information-flow shift;
6. time-horizon shift;
7. deterministic-versus-probabilistic shift;
8. centralised-versus-distributed control shift;
9. state-first-versus-event-first shift;
10. exploration-versus-exploitation shift.

### Consumes

- `ProblemSpec`
- `RepresentationPlan`
- budget/search policy
- active frontier for later widening rounds

### Produces

- immutable `Candidate` objects;
- lineage and operator ledger nodes;
- diversity and cheap-screening metrics.

### Generation protocol

**Seed round:** selected operators execute in isolated calls so one candidate does not anchor the others.  
**Expansion rounds:** only eligible frontier or high-novelty candidates are expanded.  
**Deduplication:** use structural fingerprints first, then semantic duplicate adjudication only for borderline cases.  
**Cheap screening:** estimate acceptance coverage, likely feasibility, novelty, and implementation cost without substituting for M06/M10.

A default progressive widening rule is:

```text
width(t) = min(tier_max_candidates, ceil(initial_width * t^alpha))
0 < alpha < 1
```

The scheduler may stop widening earlier when the frontier stabilises.

### Invariants

- every candidate has an operator and lineage;
- seed candidates are generated independently;
- no duplicate candidate is admitted under the same structural fingerprint;
- candidates are not mutated after commitment;
- cheap screening cannot mark a candidate finally selected or infeasible.

### Failure behaviour

One failed operator does not fail the round if minimum diversity is still met. If all configured operators fail, the run becomes `BLOCKED` or falls back to a single direct candidate according to tier policy.

### Acceptance tests

- candidate lineage is complete;
- seed generation does not contain prior candidates in its prompt payload;
- widening respects tier width and depth;
- repeated identical output is deduplicated.

---

## M06 — Constraint Propagator and Dominance Pruner

### Responsibility

Evaluate hard constraints, preserve conditional candidates, and remove only candidates that are verified infeasible or validly dominated.

### Execution mode

**Hybrid evaluators; deterministic aggregation and pruning.**

### Consumes

- candidates;
- problem constraints and objectives;
- representation views;
- evaluator registry;
- existing evidence and estimates.

### Produces

- `ConstraintEvaluation` records;
- objective estimates where cheap evaluators are available;
- feasible, conditional, invalid, dominated, and incomparable sets;
- immutable pruning decisions with reasons.

### Four-valued feasibility

```text
PASS     verified satisfied
FAIL     verified violated
UNKNOWN  not yet known or verifier unavailable
ERROR    verifier failed; treated as unknown plus diagnostic
```

Candidate status:

- `INVALID` only if at least one hard constraint is a verified `FAIL`;
- `VIABLE` if all hard constraints are `PASS`;
- `CONDITIONAL` if none fail and at least one is `UNKNOWN`/`ERROR`.

### Dominance modes

1. **Exact dominance:** deterministic complete objective vectors.
2. **Robust interval dominance:** A's worst case is no worse than B's best case on all objectives, with strict improvement on at least one.
3. **Incomparable:** missing values, overlapping uncertainty, or qualitative objectives without an explicit comparison rule.

Dominance never compensates for hard-constraint failure.

### Invariants

- unknown is never converted to fail;
- every pruned candidate references the precise constraint or dominator;
- pruning is idempotent;
- objective directions and units are normalised before comparison;
- a candidate cannot dominate itself.

### Acceptance tests

- four-valued truth table;
- exact and interval Pareto fixtures;
- property-based idempotence;
- missing objective data prevents invalid dominance;
- a new candidate from M07 re-enters evaluation correctly.

---

## M07 — Interaction / Synthesis Search

### Responsibility

Discover useful combinations of surviving components and create bounded hybrid candidates when interaction value plausibly exceeds additive value.

### Execution mode

**Hybrid:** deterministic compatibility graph construction plus model/simulation-assisted synergy hypotheses.

### Consumes

- non-dominated and conditional candidates;
- candidate component graphs;
- objective evaluators and compatibility rules;
- synthesis budget.

### Produces

- interaction graph/hypergraph artifacts;
- `SynergyHypothesis` ledger nodes;
- new immutable hybrid candidates.

### Search policy

1. Extract reusable components from candidates.
2. Exclude known-incompatible pairs using deterministic rules.
3. Rank pairwise combinations by complementarity, objective coverage, and interface compatibility.
4. Generate and evaluate bounded pairwise hybrids.
5. Escalate to order-three hyperedges only when pairwise models leave a material, explicitly logged interaction hypothesis and the tier permits it.
6. Do not search order greater than three in v1 unless a task-specific plugin overrides the policy.
7. Send every hybrid back through M06.

### Synergy semantics

A synergy claim starts as `HYPOTHESIS`. It becomes `SUPPORTED` only through a named evaluator, simulation, experiment, or evidence. LLM judgement alone remains provisional.

### Invariants

- no combinatorial unbounded expansion;
- every hybrid identifies source components and parents;
- incompatibility rules run before model calls;
- pair/triple limits follow the budget;
- hybrids never bypass M06.

### Acceptance tests

- incompatible components are rejected before generation;
- duplicate hybrids collapse under fingerprints;
- order-three search cannot run below its configured tier;
- synergy status remains provisional without validation.

---

## M08 — Falsifier

### Responsibility

Apply targeted adversarial tests to decision-relevant candidates and claims, producing challenges rather than generic criticism.

### Execution mode

**Model-assisted with deterministic target selection, operator scheduling, challenge verification, and state updates.**

### Initial falsification operators

1. assumption attack;
2. counterexample search;
3. internal contradiction search;
4. causal alternative generation;
5. edge-case/adversarial input search;
6. missing-variable or confounder detection;
7. implementation-failure analysis;
8. distribution-shift analysis;
9. Goodhart/gaming analysis;
10. dependency/provenance attack.

### Consumes

- Pareto/conditional frontier;
- provisional recommendation, if any;
- problem spec, representations, evaluations, and ledger;
- tier falsification limit.

### Produces

- typed `Challenge` nodes;
- candidate status changes when verified;
- optional successor-repair candidate proposals;
- recomputation events.

### Target selection

Targets are selected from:

- the non-dominated frontier;
- the provisional recommendation;
- candidates near a decision boundary under sensitivity analysis;
- high-consequence claims with weak evidence;
- interaction hypotheses on which the recommendation depends.

### Challenge states

```text
NO_FINDING · RISK · COUNTEREXAMPLE · FATAL
UNVERIFIED · VERIFIED · REJECTED
```

Only a verified challenge can invalidate a candidate. A non-fatal challenge may lower evidence quality or move a claim to `CONTESTED`. A repair creates a new candidate revision; the original remains immutable.

### Independence

Where budget permits, falsifier calls use an isolated prompt that contains the candidate, requirements, and evidence but omits prior promotional rationale and rankings.

### Invariants

- every challenge names a target, operator, and evidence;
- model-generated criticism is not automatically verified;
- fatal status requires a hard-constraint or acceptance-criterion consequence;
- downstream stale dependencies are marked.

### Acceptance tests

- falsified leaders are removed from subsequent frontiers;
- unverified criticism cannot invalidate a candidate;
- repaired candidates retain parent lineage;
- target selection works without a scalar global leader.

---

## M09 — Epistemic Ledger

### Responsibility

Provide the authoritative provenance and epistemic-state graph for claims, evidence, assumptions, unknowns, challenges, proposals, and decisions.

### Execution mode

**Deterministic shared subsystem.** It is implemented before the semantic modules but remains numbered M09 to preserve the conceptual architecture.

### Graph model

Node revisions are immutable. Relations include:

```text
SUPPORTS · DEPENDS_ON · DERIVED_FROM · CONTRADICTS · SUPERSEDES
FALSIFIES · RESOLVES · OBSERVED_IN · EVALUATES · SELECTS · REJECTS
```

Rules:

- `SUPPORTS`, `DEPENDS_ON`, and `DERIVED_FROM` must not create a cycle for a claim revision;
- `SUPERSEDES` must follow revision order;
- `CONTRADICTS` may be symmetric;
- dangling references are forbidden;
- large source material remains in the artifact store and is referenced by hash/span.

### Confidence model

```python
class ConfidenceAssessment(BaseModel):
    score: float | None
    level: Literal["LOW", "MEDIUM", "HIGH", "VERY_HIGH"] | None
    method: Literal[
        "DIRECT", "MODEL_JUDGMENT", "RULE", "AGGREGATED",
        "STATISTICAL", "UNSPECIFIED"
    ]
    calibration_group: str | None
    evidence_independence_groups: list[str]
    explanation: str
```

No universal arithmetic propagation is assumed. Upstream changes mark descendants `STALE`; a module-specific evaluator decides how to recompute them.

### Public operations

```text
append_node · append_edge · revise_node · mark_status · descendants
ancestors · contradictions · unresolved_dependencies · query_by_type
materialise_projection · validate_graph
```

### Invariants

- append-only history;
- complete provenance for decisions and prunes;
- support graph acyclicity;
- deterministic replay from events;
- no confidence change without a new assessment event.

### Acceptance tests

- cycle prevention;
- contradiction edges do not trigger false cycle failures;
- upstream revision marks descendants stale;
- event replay reconstructs the same ledger projection and hash.

---

## M10 — Decision Engine

### Responsibility

Compute the feasible decision frontier, test robustness, identify leverage and instability, and select a recommendation only under an explicit decision policy.

### Execution mode

**Deterministic where evaluations are numeric; hybrid only when a configured qualitative comparator is required.**

### Consumes

- viable and conditional candidates;
- objective estimates;
- problem priorities and decision policy;
- ledger assumptions and uncertainty models;
- evaluator registry.

### Produces

- `DecisionResult`;
- Pareto frontier;
- sensitivity, ablation, robustness, and regret findings;
- decision ledger node when a recommendation is justified.

### Decision sequence

1. Remove verified infeasible candidates.
2. Separate viable and conditional candidates.
3. Compute exact or robust Pareto fronts.
4. If one candidate robustly dominates and acceptance criteria are met, recommend it.
5. Otherwise evaluate assumption sensitivity and scenario robustness.
6. Run component ablation only where removal can be meaningfully re-evaluated.
7. Estimate regret across declared scenarios when useful.
8. Apply one explicit single-choice policy only if the output contract requires a single recommendation.

### Selection policy order

Preferred order:

1. robust dominance;
2. lexicographic priorities explicitly present in the problem;
3. epsilon-constraint policy;
4. minimax or expected regret with declared scenarios/probabilities;
5. weighted scalarisation as the final fallback, with sensitivity over weights.

No hidden weights are allowed.

### Ablation semantics

Ablation outcomes are classified as:

```text
MEASURED · SIMULATED · MODEL_ESTIMATED · NOT_EVALUABLE
```

The engine must not present model-estimated ablation as measured fact.

### Invariants

- the active selection policy is named and versioned;
- qualitative and quantitative comparisons are distinguishable;
- conditional candidates remain visible when their unknowns could change the result;
- no recommendation is forced when evidence is insufficient.

### Acceptance tests

- standard Pareto fixtures;
- robust interval dominance;
- sensitivity flips are detected;
- scalarisation runs only after frontier construction;
- identical inputs and policy produce an identical decision result.

---

## M11 — Active Information / Experiment Controller

### Responsibility

Rank and execute only information-acquisition actions whose expected decision impact justifies their cost and permissions.

### Execution mode

**Hybrid:** deterministic prioritisation over model/tool-generated action candidates; execution through capability adapters.

### Consumes

- unresolved unknowns and conditional constraints;
- `DecisionResult` and sensitivity findings;
- available tools and permissions;
- remaining budget.

### Produces

- ranked `AcquisitionAction` objects;
- executed observations and artifacts;
- updates resolving or narrowing unknowns;
- explicit acquisition failures.

### Action types

```text
RETRIEVE · WEB_RESEARCH · DATABASE_QUERY · CODE_EXPERIMENT
SIMULATION · FILE_ANALYSIS · HUMAN_QUERY · EXTERNAL_TOOL
```

### Priority heuristic

Unless a calibrated probabilistic utility model exists, use:

```text
priority = expected_decision_impact
         × probability_of_resolution
         × expected_result_reliability
         ÷ normalised_cost
```

All terms are labelled estimates. M10 sensitivity supplies the decision-impact signal.

### Experiment contract

```python
class ExperimentPlan(BaseModel):
    question: str
    target_unknown_ids: list[str]
    method: str
    inputs: list[ArtifactRef]
    expected_outputs: list[str]
    validator_ref: str
    stopping_rule: str
    side_effect_level: SideEffectLevel
```

### Execution rules

- check permissions before selection;
- prefer read-only, local, and deterministic acquisition when equivalent;
- cap actions by tier and remaining budget;
- store raw results as artifacts and interpreted observations as ledger nodes;
- failed acquisition leaves the unknown unresolved and records the cause;
- if no adapter is available, emit an acquisition plan rather than inventing a result.

### Invariants

- every action targets a named uncertainty or conditional constraint;
- no side effect without permission;
- acquisition results never become facts without source/provenance;
- tool failure cannot be treated as negative evidence.

### Acceptance tests

- high-impact low-cost uncertainty ranks above low-impact high-cost uncertainty;
- unavailable tools yield plans, not fabricated observations;
- result updates mark only dependent state stale;
- permissions block disallowed side effects.

---

## M12 — Context Compiler

### Responsibility

Project canonical run and ledger state into compact, authoritative handoff packets for continuation, inspection, or another execution surface.

### Execution mode

**Deterministic projection and rendering.** Optional prose polish is non-authoritative and must preserve ledger references.

### Consumes

- a versioned `RunSnapshot`;
- ledger/candidate/decision projections;
- token or size target;
- compiler profile.

### Produces

- canonical JSON `ContextPacket`;
- deterministic Markdown rendering;
- packet hash and artifact reference.

### Profiles

```text
FULL       all active decision-relevant state and rejection register
STANDARD   full frontier, decisive facts/assumptions, compact rejection register
HANDOFF    minimum sufficient state for safe continuation
```

### Projection rules

Always preserve:

- objective and output contract;
- hard constraints and their verification states;
- fixed decisions and their bases;
- verified facts and high-leverage assumptions;
- active/conditional frontier;
- rejected candidates with reasons;
- unresolved high-priority questions;
- budget remaining and next action.

Compress first by removing duplicated prose, obsolete revisions, low-leverage rejected detail, and redundant evidence excerpts. Never remove the reason a branch was rejected.

### Reproducibility

The packet is idempotent for:

```text
(snapshot_version, compiler_version, profile, size_target)
```

Its hash is persisted. Recompilation after code changes requires a new compiler version.

### Invariants

- every substantive assertion has a ledger/object reference;
- stale decisions are excluded or labelled;
- deterministic ordering;
- no hidden branch is presented as active;
- packet size respects the profile target or emits an explicit overflow diagnostic.

### Acceptance tests

- idempotent packet hash;
- rejected candidates retain reasons;
- stale claims do not appear as verified;
- full → standard → handoff compression preserves required fields.

---

## M13 — Marginal-Value Stop Controller

### Responsibility

Choose `CONTINUE`, `COMPLETE`, or a transparent partial/blocking disposition using hard ceilings, acceptance status, frontier stability, sensitivity, and remaining acquisition value.

### Execution mode

**Deterministic policy.**

### Consumes

- budget meter;
- action history;
- `DecisionResult`;
- unresolved blockers and ranked acquisition actions;
- acceptance criteria;
- validation requirements;
- system/user cancellation state.

### Produces

- `StopDecision`;
- a final context packet request for every terminal disposition;
- next-action recommendation when continuing.

### Hard terminal conditions

- user/system cancellation;
- budget or iteration ceiling reached;
- unrecoverable invariant failure;
- no executable action remains and the run is blocked;
- all required acceptance criteria are met and no mandatory validation remains.

### Mandatory-continue conditions, subject to hard ceilings

- an unexecuted mandatory tier action remains;
- a decision-relevant hard constraint is unknown and a permitted resolving action exists;
- a high-consequence recommendation lacks required independent validation;
- a remaining acquisition action exceeds the configured priority threshold;
- frontier or recommendation remains materially unstable under sensitivity analysis.

### Soft completion conditions

A run may complete when:

- acceptance criteria are met;
- the recommendation/frontier is stable across the configured checkpoint window;
- no unresolved item is likely to alter the decision above threshold;
- another candidate/falsification/acquisition pass has lower normalised expected value than its cost;
- the tier's required validation is complete.

### Terminal dispositions

- `COMPLETE`: acceptance met; decision or frontier is stable enough.
- `PARTIAL_BUDGET`: hard budget ended before full completion; packet contains the best current state.
- `BLOCKED`: a required uncertainty/permission/capability prevents continuation.
- `FAILED_INVARIANT`: canonical-state or safety invariant failed.
- `CANCELLED`: explicit cancellation.

### Invariants

- hard ceilings guarantee termination;
- every stop/continue decision has named reasons and references;
- no `COMPLETE` status while a required acceptance criterion is known to fail;
- every terminal disposition compiles a context packet.

### Acceptance tests

- finite-budget termination under an adversarial module that keeps proposing work;
- stable frontier plus no high-value action leads to completion;
- unresolved hard blocker leads to `BLOCKED`, not false completion;
- budget exhaustion yields a usable partial packet.

---

## 9. Tier policy baseline

These are implementation defaults, not universal constants. They live in versioned configuration and can be recalibrated empirically.

| Tier | Intended behaviour | Initial candidates | Views | Falsification | Acquisition | Validation | Iteration ceiling |
|---|---|---:|---:|---|---|---|---:|
| T0 | direct inference / deterministic answer | 1 | 0–1 | none | none | schema only | 1 |
| T1 | structured single pass | 1–2 | 1 | basic consistency | none | output checks | 2 |
| T2 | small comparison | 3–4 | 1 | 2 operators | 0–1 | one decision pass | 4 |
| T3 | frontier search | 4–6 | 1–2 | 4 operators | up to 2 | sensitivity + targeted critic | 6 |
| T4 | high-consequence analysis | 6–10 | up to 2 | full applicable suite | up to 5 | robust sensitivity/ablation | 10 |
| T5 | independent adversarial validation | up to 12 per branch | up to 2 | full suite, isolated | up to 10 | two isolated branches + reconciliation | 14 |

Additional configurable limits include LLM calls, tool calls, token ceilings, concurrency, sensitivity samples, candidate depth, and hyperedge order.

### T5 isolation

T5 creates at least two branch IDs. Independent branches share the task and authoritative evidence but do not receive each other's candidates, evaluations, or rationale until reconciliation. The reconciliation pass compares outputs, traces disagreements to assumptions/evidence, and records the resolution or persistent conflict.

---

## 10. Module dependency graph

```text
                 ┌──── M09 Ledger / Event Core ────┐
                 │                                  │
M01 → M02 → M03 → M04 → M05 → M06 → M10 → M11 → M13
                              │      ▲      │       │
                              ▼      │      └───────┘
                             M07 ─────┘       re-entry
                              │
                              ▼
                             M06

M06/M10 → M08 → M06/M10
All modules → M12 checkpoints/final packet
```

Implementation order differs from conceptual number order because M09 and the event/runtime core are foundational.

---

## 11. Persistence architecture

### 11.1 Initial storage choice

Use **SQLite in WAL mode** plus a local content-addressed artifact directory. This is sufficient for a single-user local engine, easy to inspect, and avoids premature service complexity. Storage ports allow later PostgreSQL/object-store adapters.

### 11.2 Required tables

```text
runs
  run_id · status · schema_version · config_hash · current_version · timestamps

events
  run_id · sequence · event_type · module_id · action_id · payload_json
  input_hash · module_version · created_at

snapshots
  run_id · sequence · state_json · state_hash · reducer_version · created_at

artifacts
  artifact_id · sha256 · media_type · byte_size · path · metadata_json

model_calls
  call_id · run_id · action_id · role · provider · model · request_hash
  response_artifact_id · parsed_payload_json · usage_json · status

tool_calls
  call_id · run_id · action_id · tool_id · request_hash · result_artifact_id
  cost_json · side_effect_level · status

ledger_nodes / ledger_edges
  query projections derived from events

candidates / evaluations / decisions
  query projections derived from events
```

The event log is authoritative; projections are rebuildable.

### 11.3 Snapshots

Create snapshots at configurable event intervals and at phase boundaries. A snapshot stores the reducer version and a hash. Replay validation compares a rebuilt state hash with the snapshot hash.

### 11.4 Artifact handling

- content-address by SHA-256;
- never duplicate identical content;
- store text spans and structured extraction references;
- keep raw model/tool outputs separate from parsed canonical objects;
- redact or encrypt sensitive artifacts through an optional storage policy.

---

## 12. Prompt and model-call architecture

### 12.1 Prompt registry

Every model-assisted operation references:

```text
prompt_id · prompt_version · output_schema_version · role · safety profile
```

Prompts are package resources, not inline strings spread through modules.

### 12.2 Context construction

Prompt inputs are compiled from object references and the smallest relevant context packet. Modules must not receive the entire transcript by default.

### 12.3 Structured output rules

- provider-native schema constraints preferred;
- strict Pydantic validation;
- one repair attempt using validation errors only;
- rejected raw outputs retained as artifacts;
- parsed output never bypasses module-level semantic validation.

### 12.4 Model roles

```text
FAST_STRUCTURED      M01 and low-cost extraction
DEEP_STRUCTURED      M03 and complex representation decisions
DIVERGENT_GENERATOR  M05 and M07 candidate generation
CRITIC               M08
QUALITATIVE_JUDGE    only where M06/M10 cannot use deterministic comparators
RENDERER              optional M12 prose rendering
```

A configuration maps roles to adapters/models. Module code names roles, not products.

---

## 13. Error and retry semantics

Errors are classified as:

| Class | Meaning | Default response |
|---|---|---|
| `TRANSIENT_TRANSPORT` | timeout/rate/network | bounded retry with backoff |
| `INVALID_STRUCTURED_OUTPUT` | schema failure | one repair, then module fallback/failure |
| `TOOL_UNAVAILABLE` | no adapter/capability | emit blocker or acquisition plan |
| `PERMISSION_DENIED` | side effect not authorised | block action, do not retry |
| `BUDGET_EXHAUSTED` | hard ceiling | terminal partial packet |
| `DOMAIN_BLOCKED` | required unknown cannot be resolved | `BLOCKED` |
| `INVARIANT_VIOLATION` | corrupted state/illegal transition | terminal failure |
| `MODULE_BUG` | unhandled exception | record diagnostic; fail action; policy decides run status |

Retries must use the same idempotency key. Model retries do not silently increase candidate diversity.

---

## 14. Concurrency

Parallelism is permitted for independent seed candidates, falsifier operators, representation compilation, and read-only acquisition actions.

Rules:

- modules receive immutable snapshots;
- commits use optimistic version checks;
- parallel results are sorted by stable action ID before reduction;
- budget reservations occur before launch and settle after completion;
- cancelled/failed actions release unused reservations;
- concurrency never changes deterministic replay order.

---

## 15. Observability

Every run emits structured logs and metrics keyed by `run_id`, `branch_id`, and `action_id`.

Minimum metrics:

```text
tier · iterations · budget consumed/remaining
model/tool calls and latency
candidate count, diversity, and depth
constraint pass/fail/unknown counts
frontier size and changes
falsification findings by severity
acquisition actions and resolution rate
context packet sizes
stop disposition and reasons
```

A trace viewer is deferred, but CLI inspection must expose this information.

---

## 16. Security and privacy baseline

- local-first persistence by default;
- explicit adapter permissions and side-effect levels;
- secrets from environment/keychain adapters, never persisted in events;
- content hashes rather than copied sensitive text where possible;
- optional redaction hooks before remote model/tool calls;
- sandboxed code execution with file/network policies;
- no external writes in v1 without an explicit permission token;
- audit events for every outbound call.

---

## 17. Public interfaces

### 17.1 Python SDK

```python
engine = FrontierReasoningEngine(config)
run = await engine.start(task_envelope)
result = await engine.execute(run.run_id)
packet = await engine.compile_context(run.run_id, profile="HANDOFF")
```

Required methods:

```text
start · execute · step · resume · cancel · inspect · replay
compile_context · export_run · list_actions · approve_action
```

### 17.2 CLI

```text
fre run TASK.md [--config config.yaml]
fre step RUN_ID
fre resume RUN_ID
fre inspect RUN_ID [--json]
fre frontier RUN_ID
fre ledger RUN_ID [--claim ID]
fre context RUN_ID --profile handoff
fre replay RUN_ID --verify
fre export RUN_ID --output run-bundle.zip
fre approve RUN_ID ACTION_ID
fre cancel RUN_ID
```

### 17.3 MCP/service adapter

Build only after SDK/CLI contracts stabilise. The adapter must remain thin: it translates tool calls to SDK operations and never implements reasoning logic itself.

---

## 18. Repository architecture

```text
frontier-reasoning-engine/
├── pyproject.toml
├── README.md
├── LICENSE
├── configs/
│   ├── default.yaml
│   ├── tiers.yaml
│   └── model_roles.example.yaml
├── docs/
│   ├── FRE_ARCHITECTURE_V1.md
│   ├── adr/
│   └── schemas/
├── src/fre/
│   ├── __init__.py
│   ├── cli.py
│   ├── engine.py
│   ├── domain/
│   │   ├── common.py
│   │   ├── task.py
│   │   ├── budget.py
│   │   ├── problem.py
│   │   ├── representation.py
│   │   ├── candidate.py
│   │   ├── evaluation.py
│   │   ├── ledger.py
│   │   ├── decision.py
│   │   ├── acquisition.py
│   │   ├── context.py
│   │   └── stop.py
│   ├── runtime/
│   │   ├── orchestrator.py
│   │   ├── scheduler.py
│   │   ├── policy.py
│   │   ├── budget_meter.py
│   │   ├── reducer.py
│   │   ├── events.py
│   │   ├── snapshots.py
│   │   └── registry.py
│   ├── modules/
│   │   ├── m01_classifier.py
│   │   ├── m02_budget.py
│   │   ├── m03_formaliser.py
│   │   ├── m04_representation.py
│   │   ├── m05_generator.py
│   │   ├── m06_constraints.py
│   │   ├── m07_synthesis.py
│   │   ├── m08_falsifier.py
│   │   ├── m09_ledger.py
│   │   ├── m10_decision.py
│   │   ├── m11_information.py
│   │   ├── m12_context.py
│   │   └── m13_stop.py
│   ├── algorithms/
│   │   ├── pareto.py
│   │   ├── interval_dominance.py
│   │   ├── sensitivity.py
│   │   ├── ablation.py
│   │   ├── regret.py
│   │   ├── progressive_widening.py
│   │   ├── fingerprints.py
│   │   └── graph_validation.py
│   ├── ports/
│   │   ├── models.py
│   │   ├── tools.py
│   │   ├── storage.py
│   │   ├── sandbox.py
│   │   ├── human.py
│   │   └── clock.py
│   ├── adapters/
│   │   ├── storage_sqlite/
│   │   ├── artifacts_local/
│   │   ├── models_fake/
│   │   ├── models_openai_compatible/
│   │   ├── tools_local/
│   │   └── sandbox_subprocess/
│   ├── prompts/
│   │   ├── registry.py
│   │   └── templates/
│   └── projections/
│       ├── ledger.py
│       ├── candidates.py
│       ├── decisions.py
│       └── metrics.py
└── tests/
    ├── unit/
    ├── property/
    ├── integration/
    ├── golden/
    └── fixtures/
```

No module may import a concrete adapter. Dependencies point inward toward domain and ports.

---

## 19. Build order for Codex

### Wave 0 — Repository and quality foundation

Deliver:

- package skeleton and dependency management;
- formatting, linting, static typing, and test commands;
- CI workflow or local equivalent;
- configuration loader and schema-version utilities;
- architecture document copied into `docs/`.

Gate:

```text
package imports · tests run · lint/type checks run · no module logic yet
```

### Wave 1 — Canonical contracts, events, and storage

Deliver:

- all domain models in Section 6;
- event envelope and reducer protocol;
- SQLite event/snapshot store;
- local artifact store;
- fake clock, fake model, and fake tool adapters;
- run creation, append, replay, and snapshot hash verification.

Gate:

```text
an empty run can be created, evented, snapshotted, reloaded, and replay-verified
```

### Wave 2 — M09, M02, M12, and M13 deterministic spine

Deliver:

- ledger graph and projections;
- budget allocation/meter;
- context compiler;
- stop controller with hard ceilings;
- property tests for monotonicity, graph integrity, idempotence, and termination.

Gate:

```text
a synthetic run can allocate budget, record claims, compile context, and terminate safely
```

### Wave 3 — M01, M03, and M04 structured semantic front end

Deliver:

- prompt registry and structured model port;
- classifier, formaliser, and representation selector;
- fake-model golden fixtures;
- deterministic fallbacks.

Gate:

```text
a task becomes a validated signature, problem spec, ledger state, and representation plan
```

### Wave 4 — M05, M06, and M10 search/decision core

Deliver:

- candidate operator registry and progressive widening;
- structural fingerprints and deduplication;
- constraint evaluator registry and four-valued aggregation;
- exact/interval Pareto algorithms;
- sensitivity, ablation metadata, regret, and decision result.

Gate:

```text
a multi-candidate fixture produces correct viable/conditional sets and a tested frontier
```

### Wave 5 — M07 and M08 synthesis/falsification

Deliver:

- component compatibility graph;
- bounded pair/triple synthesis;
- falsifier operator registry and isolated calls;
- challenge verification and immutable repair lineage.

Gate:

```text
a hybrid can be generated, re-evaluated, challenged, and invalidated without state mutation
```

### Wave 6 — M11 acquisition and capability routing

Deliver:

- acquisition action model and ranking;
- capability/permission router;
- local retrieval/file/code experiment adapters;
- unknown resolution and targeted stale-state propagation.

Gate:

```text
a decision-sensitive unknown triggers a permitted action and updates only dependent state
```

### Wave 7 — Full orchestrator, CLI, and branch validation

Deliver:

- adaptive scheduler and complete default policy loop;
- concurrency reservations and deterministic commit order;
- T5 isolated branches and reconciliation;
- CLI commands;
- exportable run bundle;
- end-to-end golden scenarios.

Gate:

```text
all six golden scenarios pass, replay hashes match, and hard budgets stop adversarial loops
```

### Wave 8 — Thin MCP/service adapter

Deliver only after the SDK is stable:

- MCP or HTTP translation layer;
- authentication/permission mapping;
- integration tests proving parity with SDK operations.

---

## 20. Test architecture and mandatory scenarios

### 20.1 Unit tests

- schema validation and migrations;
- reducers and event application;
- each deterministic algorithm;
- each module's preconditions and fallback.

### 20.2 Property tests

- budget monotonicity;
- event replay equivalence;
- pruning idempotence;
- Pareto non-domination invariants;
- interval dominance soundness;
- support-graph acyclicity;
- context compiler idempotence;
- hard-budget termination.

### 20.3 Integration tests

Use fake deterministic adapters so results are stable. Test storage, scheduler, modules, and projections together.

### 20.4 Golden end-to-end scenarios

1. **Simple direct task:** T0/T1; no candidate explosion; completes after one pass.
2. **Multi-objective architecture choice:** several candidates, one dominated, stable Pareto frontier.
3. **Unknown hard constraint:** candidate remains conditional; engine does not falsely prune or complete.
4. **Falsified provisional recommendation:** verified challenge removes it; a successor/frontier is recomputed.
5. **High-value acquisition flips decision:** tool observation resolves an unknown and changes the recommendation with full provenance.
6. **Adversarial endless proposer:** hard budget stops the run and produces `PARTIAL_BUDGET` plus a valid handoff packet.

### 20.5 Acceptance bar

A wave is not complete because code exists. It is complete when:

- public contracts match this specification;
- tests for invariants and failures pass;
- state can replay deterministically from stored outputs;
- no provider-specific dependency leaks into domain/modules;
- every decision/prune/stop has provenance;
- documentation and examples are updated.

---

## 21. Architecture decision records to create

Codex should create and maintain these ADRs:

```text
ADR-001 Modular control plane over monolithic agent
ADR-002 Lightweight event sourcing and replay
ADR-003 SQLite + content-addressed local artifact store for v1
ADR-004 Typed four-valued constraint semantics
ADR-005 Representations as canonical-state projections
ADR-006 Immutable candidates and claim revisions
ADR-007 Provider-neutral capability ports
ADR-008 Pareto-first decision policy and delayed scalarisation
ADR-009 Hard budget termination plus heuristic marginal-value stopping
ADR-010 SDK/CLI first; MCP/service adapter second
```

---

## 22. Deferred extension points

These are intentionally deferred but architecturally supported:

- PostgreSQL and object-store adapters;
- learned budget/tier calibration from run outcomes;
- calibrated probabilistic confidence and formal EVPI;
- distributed workers;
- richer CSP/ILP/causal-model integrations;
- adaptive operator learning;
- visual graph/frontier inspection;
- domain-specific module packs;
- automatic benchmark and evaluator suites;
- cryptographic run signing.

They must not block v1.

---

## 23. Definition of v1 complete

FRE v1 is complete when a user can submit a task through the SDK or CLI and the engine can:

1. classify and budget it;
2. formalise it with provenance;
3. select useful representations;
4. generate independent candidates;
5. evaluate constraints without conflating unknown and false;
6. compute and maintain a multi-objective frontier;
7. synthesize bounded hybrids;
8. falsify decision-relevant candidates;
9. preserve an auditable epistemic graph;
10. acquire decision-sensitive information through authorised tools;
11. stop within hard limits;
12. emit a compact, deterministic context packet;
13. replay the run from stored events and external-call outputs to the same state hash.

This is the minimum build that realises the supplied Frontier Reasoning Engine rather than merely simulating it through a long prompt.
