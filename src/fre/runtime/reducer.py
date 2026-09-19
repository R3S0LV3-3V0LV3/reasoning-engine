"""Pure reducer protocol and foundational run reducer."""

import hashlib
from collections import deque
from collections.abc import Callable, Mapping
from typing import Protocol, TypeVar
from uuid import UUID

from pydantic import Field, TypeAdapter, ValidationError

from fre.domain.budget import BudgetProjection, ResourceVector
from fre.domain.common import FrozenModel, JsonValue, bind_hash, canonical_hash, canonical_json
from fre.domain.context import ContextCompilationRecord, ContextPacket, UnresolvedUnknownRef
from fre.domain.ledger import EpistemicStatus, LedgerProjection
from fre.domain.problem import ContradictionDiagnostic, ProblemBlocker, ProblemSpec
from fre.domain.representation import (
    RepresentationArtifact,
    RepresentationArtifactV2,
    RepresentationCandidateScore,
    RepresentationPlan,
    RepresentationPlanV2,
)
from fre.domain.representation_registry import (
    ORPHAN_KIND_REASONS,
    default_registry_v2,
    representation_determinism_hash,
    score_details,
)
from fre.domain.semantic import (
    EpistemicOriginLabel,
    SemanticAccountingCondition,
    SemanticModelCallRecord,
    SemanticModelCallRecordV2,
    SourceAnchor,
    StructuredModelStatus,
    SupportRef,
)
from fre.domain.stop import StopDecision, StopDecisionRecord
from fre.domain.task import ClassificationRecord, TaskSignature
from fre.modules.m02_budget import BudgetAllocator
from fre.modules.m09_ledger import EpistemicLedger
from fre.modules.m12_context import Wave3ContextCompiler, derive_wave3_availability
from fre.modules.source_anchors import (
    resolve_support_ref_at_reduction,
    validate_anchor_artifact_registration,
)
from fre.prompts.schemas import OutputSchemaRegistry
from fre.runtime.budget_meter import BudgetMeter
from fre.runtime.events import (
    ArtifactRegistered,
    BudgetAllocated,
    BudgetConsumed,
    BudgetReservationReleased,
    BudgetReservationSettled,
    BudgetReserved,
    BudgetRevised,
    ClassificationDiagnosticRecorded,
    ContextCompiled,
    ContextCompiledV2,
    LedgerContradictionResolved,
    LedgerDependentsMarkedStale,
    LedgerEdgeAdded,
    LedgerNodeAdded,
    LedgerNodeRevised,
    LedgerNodeStatusChanged,
    ModelCallFailed,
    ModelCallFailedV2,
    ModelCallRecorded,
    ModelCallRecordedV2,
    ProblemBlockerRecorded,
    ProblemContradictionRecorded,
    ProblemFormalised,
    RepresentationArtifactCompiled,
    RepresentationArtifactCompiledV2,
    RepresentationPlanSelected,
    RepresentationPlanSelectedV2,
    RunCreated,
    RunStatusChanged,
    StopDecisionRecorded,
    StopDecisionRecordedV2,
    StoredEvent,
    TaskClassified,
    TaskPreliminarilyClassified,
    TerminalContextAssociated,
    TestValueSet,
)

StateT = TypeVar("StateT")


class Reducer(Protocol[StateT]):
    version: str

    def initial(self, run_id: UUID) -> StateT: ...

    def apply(self, state: StateT, event: StoredEvent) -> StateT: ...


class RunState(FrozenModel):
    run_id: UUID
    version: int = Field(ge=0)
    status: str = "NEW"
    config_hash: str | None = None
    artifacts: tuple[str, ...] = ()
    values: dict[str, JsonValue] = Field(default_factory=dict)
    ledger: LedgerProjection = LedgerProjection()
    budget: BudgetProjection = BudgetProjection()
    context_packets: tuple[ContextPacket, ...] = ()
    context_compilations: tuple[ContextCompilationRecord, ...] = ()
    stop_decisions: tuple[StopDecision, ...] = ()
    stop_decision_records: tuple[StopDecisionRecord, ...] = ()
    terminal_context_packet_hash: str | None = None
    terminal_context_disposition: str | None = None
    model_calls: tuple[SemanticModelCallRecordV2 | SemanticModelCallRecord, ...] = ()
    task_signature: TaskSignature | None = None
    # C05 remediation (finding #14, documentation-only): write-only by
    # design -- audit/forensic record of the bootstrap-budget-sizing
    # signature only. Never read back by production code (the bootstrap
    # budget calculation uses the local `bootstrap_signature` variable in
    # `TaskClassifier.canonical_events` directly). See
    # `TaskPreliminarilyClassified`'s docstring (runtime/events.py) for the
    # full rationale, including why removal is out of scope here (all 10
    # golden fixtures carry this field).
    preliminary_task_signature: TaskSignature | None = None
    classification_record: ClassificationRecord | None = None
    classification_diagnostics: tuple[str, ...] = ()
    problem_spec: ProblemSpec | None = None
    problem_blockers: tuple[ProblemBlocker, ...] = ()
    problem_contradictions: tuple[ContradictionDiagnostic, ...] = ()
    representation_plan: RepresentationPlan | None = None
    representation_artifacts: tuple[RepresentationArtifact, ...] = ()
    # C07/M04 bound v2 selection state. Additive, post-Wave-3-freeze fields:
    # see `snapshot_payload` below, which omits both whenever they are empty
    # so every pre-existing sealed snapshot hash is preserved unchanged.
    representation_plan_v2: RepresentationPlanV2 | None = None
    representation_artifacts_v2: tuple[RepresentationArtifactV2, ...] = ()
    # Reducer-local bookkeeping for the F09 reservation-settlement duplicate
    # check embedded in `RunReducer.apply` (see the `ModelCallRecordedV2` /
    # `ModelCallFailedV2` branch below and finding #2 in the C04 remediation
    # round). Populated when a `BudgetReservationSettled` is applied and
    # drained the moment the matching semantic model-call record is applied.
    # `BudgetReservationSettled` is a general budget event, not exclusively a
    # semantic-model-call one (plenty of Wave 2 budget usage settles
    # reservations with no model call ever attached) -- so, unlike every other
    # field here, an entry can legitimately sit unconsumed forever. It is
    # therefore excluded from `model_dump`/the sealed snapshot hash entirely
    # (`exclude=True`): it is a transient computation aid for the single
    # `reduce()`/`apply()` sequence in progress, carried forward explicitly by
    # `apply` (see the top of that method), never persisted, and always
    # rebuilt honestly from genesis on the next full replay -- which is
    # exactly what both real replay and `FrontierReasoningEngine.append`'s own
    # preview loop already do.
    pending_semantic_settlements: dict[str, ResourceVector] = Field(
        default_factory=dict, exclude=True
    )

    def snapshot_payload(self) -> dict[str, object]:
        """Return the hash payload, retaining Wave 1 shape for untouched streams."""
        payload: dict[str, object] = self.model_dump(mode="json")
        if not self.stop_decision_records:
            # V1 snapshots predate state-bound stop records. Keep their sealed
            # payload and hash unchanged even when they contain v1 decisions.
            payload.pop("stop_decision_records")
        wave_2_absent = (
            self.ledger == LedgerProjection()
            and self.budget == BudgetProjection()
            and not self.context_packets
            and not self.context_compilations
            and not self.stop_decisions
            and not self.stop_decision_records
            and self.terminal_context_packet_hash is None
            and self.terminal_context_disposition is None
        )
        wave_3_absent = (
            not self.model_calls
            and self.task_signature is None
            and self.preliminary_task_signature is None
            and self.classification_record is None
            and not self.classification_diagnostics
            and self.problem_spec is None
            and not self.problem_blockers
            and not self.problem_contradictions
            and self.representation_plan is None
            and not self.representation_artifacts
        )
        # Wave 3 did not exist when Wave 2 snapshots were sealed.  Its empty
        # fields must therefore be omitted independently of populated Wave 2 state.
        if wave_3_absent:
            for key in (
                "model_calls",
                "task_signature",
                "preliminary_task_signature",
                "classification_record",
                "classification_diagnostics",
                "problem_spec",
                "problem_blockers",
                "problem_contradictions",
                "representation_plan",
                "representation_artifacts",
            ):
                payload.pop(key)
        if wave_2_absent and wave_3_absent:
            for key in (
                "ledger",
                "budget",
                "context_packets",
                "context_compilations",
                "stop_decisions",
                "terminal_context_packet_hash",
                "terminal_context_disposition",
            ):
                payload.pop(key)
        # C07 (M04 bound v2 representation state) did not exist when every
        # earlier snapshot was sealed. Pop both fields whenever they are
        # empty, independently of the wave_2/wave_3 flags above, so no
        # pre-existing run's hash changes merely because this phase shipped.
        if self.representation_plan_v2 is None and not self.representation_artifacts_v2:
            for key in ("representation_plan_v2", "representation_artifacts_v2"):
                payload.pop(key)
        return payload

    @property
    def state_hash(self) -> str:
        return canonical_hash(self.snapshot_payload())


_SUPPORT_REF_LIST_ADAPTER: TypeAdapter[tuple[SupportRef, ...]] = TypeAdapter(tuple[SupportRef, ...])
_ANCHOR_LIST_ADAPTER: TypeAdapter[tuple[SourceAnchor, ...]] = TypeAdapter(tuple[SourceAnchor, ...])


def _validate_m03_ledger_node_provenance(
    state: "RunState", content: "Mapping[str, JsonValue]"
) -> None:
    """Re-resolve an M03 ledger node's embedded `support` refs and `anchors`
    against state already applied earlier in this same reduction (see the
    long comment on the `LedgerNodeAdded` branch above for why this exists at
    the reducer level, not only in `ProblemFormaliser`).

    C06 remediation, round 2 (findings A, B, C, E, F, G -- see the PR body for
    the full mapping):

    - (G) `support`/`anchors` are parsed back into the SAME typed pydantic
      models (`SupportRef`/`SourceAnchor`) `source_anchors.py` already
      defines, and resolved through the SAME envelope-independent resolution
      helpers formalise-time validation uses
      (`resolve_support_ref_at_reduction`, `validate_anchor_artifact_
      registration`) -- not a second, independently-drifting duck-typed
      reimplementation of "what counts as resolvable evidence".
    - (A) A `SupportProblemItemRef` is resolved only against OTHER M03
      ledger nodes that share this node's own `batch_id` (embedded in
      `content` by `ProblemFormaliser._ledger_events` -- one fresh id per
      `canonical_events`/`_ledger_events` call), never against arbitrary
      all-time ledger history under a matching `id` string. A node with no
      `batch_id` of its own (or one that matches no other applied node's
      `batch_id`) can never resolve a same-proposal item reference this way
      -- this closes the exact exploit: a hand-built `LedgerNodeAdded`
      citing an item-id from an unrelated, already-committed earlier
      proposal is now rejected here exactly as `formalise()` already
      rejects it (its `index` is scoped to the current proposal only).
    - (F) A `content["id"]` colliding with an already-admitted M03 node's id
      from a DIFFERENT batch is rejected outright: only `LedgerNodeRevised`
      represents a legitimate revision of an existing item's content; a
      fresh `LedgerNodeAdded` reusing an id already claimed by another
      batch is always a forged/unintended collision, never an intentional
      revision, and this is a deliberate design choice (documented here,
      not an oversight) -- a future caller needing genuine cross-batch
      "supersede under the same id" semantics would need a new, explicit
      event for it.
    - (B) `anchors` are re-validated too, not only `support`: an
      `EXPLICIT_INPUT`-origin node must carry at least one anchor, and any
      `ARTIFACT`-kind anchor among them is re-resolved against
      `state.artifacts`.
    - (C) `resolve_support_ref_at_reduction` also rejects a
      `SupportProblemItemRef` naming a `RELATION`-kind target here, not only
      in `formalise()` -- a minimal, kind-based relevance safeguard. This
      cannot and does not verify that a resolved target's *content*
      actually substantiates the citing claim; full semantic-relevance
      verification is not mechanically achievable and remains a residual,
      documented limitation of both this reducer check and `formalise()`
      itself.

    Neither this function nor its shared helpers can re-resolve a
    `TASK_FIELD`/`TASK_TEXT`-kind anchor against the originating
    `TaskEnvelope` (the reducer has no envelope in scope) -- that remains
    `formalise()`-only, exactly as before. Their *structural* shape (a
    non-empty, correctly ordered span; a well-formed excerpt hash) is,
    however, guaranteed the moment they are parsed back into the typed
    `SourceAnchor` model below, via its own `valid_range` validator and
    field constraints (finding E).
    """
    own_id = content.get("id")
    own_id_str = own_id if isinstance(own_id, str) else None
    own_batch_id = content.get("batch_id")

    # (F) Cross-batch duplicate id: a fresh LedgerNodeAdded must never reuse
    # an id already admitted under a different batch/proposal. Revision has
    # its own dedicated event (`LedgerNodeRevised`); this path is
    # additions-only.
    if own_id_str is not None:
        for node in state.ledger.nodes:
            if (
                node.producing_module == "M03"
                and isinstance(node.content, dict)
                and node.content.get("id") == own_id_str
                and node.content.get("batch_id") != own_batch_id
            ):
                raise ValueError(
                    "M03 ledger node id collides with an item already admitted under a "
                    "different batch/proposal; a genuine revision requires a "
                    "LedgerNodeRevised event, not a second LedgerNodeAdded"
                )

    # (A) Same-proposal item references resolve only within the batch that
    # produced this very node -- never against arbitrary prior history.
    known_m03_content_by_id: dict[str, Mapping[str, JsonValue]] = {}
    if own_batch_id is not None:
        for node in state.ledger.nodes:
            node_item_id = node.content.get("id") if isinstance(node.content, dict) else None
            if (
                node.producing_module == "M03"
                and isinstance(node.content, dict)
                and isinstance(node_item_id, str)
                and node.content.get("batch_id") == own_batch_id
            ):
                known_m03_content_by_id[node_item_id] = node.content
    known_ledger_revisions = frozenset((node.node_id, node.revision) for node in state.ledger.nodes)
    available_artifacts = frozenset(state.artifacts)

    support = content.get("support")
    if isinstance(support, list):
        try:
            parsed_support = _SUPPORT_REF_LIST_ADAPTER.validate_python(support, strict=False)
        except ValidationError as error:
            raise ValueError("M03 ledger node carries a malformed support reference") from error
        for ref in parsed_support:
            resolve_support_ref_at_reduction(
                ref,
                own_id=own_id_str,
                known_m03_content_by_id=known_m03_content_by_id,
                known_ledger_revisions=known_ledger_revisions,
                available_artifacts=available_artifacts,
            )

    # (B) `anchors` were never re-checked at all before this fix.
    anchors = content.get("anchors")
    parsed_anchors: tuple[SourceAnchor, ...] = ()
    if isinstance(anchors, list):
        try:
            parsed_anchors = _ANCHOR_LIST_ADAPTER.validate_python(anchors, strict=False)
        except ValidationError as error:
            raise ValueError("M03 ledger node carries a malformed anchor") from error
        for anchor in parsed_anchors:
            validate_anchor_artifact_registration(anchor, available_artifacts)
    if content.get("origin") == EpistemicOriginLabel.EXPLICIT_INPUT.value and not parsed_anchors:
        raise ValueError("M03 ledger node claims EXPLICIT_INPUT origin without any anchor")


def _validate_model_call_common(
    state: RunState,
    payload: ModelCallRecorded | ModelCallFailed | ModelCallRecordedV2 | ModelCallFailedV2,
) -> None:
    """Shared validation shape (EU-05, C04 cleanup, item #5) for the checks
    genuinely identical across all four `ModelCall*` event types -- V1/V2,
    Recorded/Failed alike -- called once before each branch's own
    type-specific checks in `RunReducer.apply`.

    Only checks proven byte-identical across every payload type are here:
    idempotency-key uniqueness, the "a successful call must carry both
    artifacts" rule (itself conditioned on the payload being one of the two
    *Recorded* variants, not merged into a false, type-blind shape), and
    artifact registration. The V2-only reservation/settlement matching
    (shared by `ModelCallRecordedV2`/`ModelCallFailedV2` alone, not the V1
    types) and each type's own status/accounting-condition rules remain
    branch-local in `apply` -- they are not "shared-looking" duplication,
    they are genuinely different per type.
    """
    if any(item.idempotency_key == payload.record.idempotency_key for item in state.model_calls):
        raise ValueError("semantic model-call identity already recorded")
    if isinstance(payload, (ModelCallRecorded, ModelCallRecordedV2)) and (
        payload.record.raw_artifact is None or payload.record.proposal_artifact is None
    ):
        raise ValueError("successful semantic model call requires raw and proposal artifacts")
    for artifact in (payload.record.raw_artifact, payload.record.proposal_artifact):
        if artifact is not None and artifact.sha256 not in state.artifacts:
            raise ValueError("semantic model-call artifact is not registered")


class RunReducer:
    version = "2.0"
    compatible_snapshot_versions = frozenset({"1.0", "2.0"})

    def __init__(
        self,
        *,
        artifact_reader: Callable[[str], bytes] | None = None,
        schema_registry: OutputSchemaRegistry | None = None,
        trust_unverified_artifacts: bool = False,
    ) -> None:
        # Both are optional so every existing bare `RunReducer()` construction
        # (tests, snapshot/replay helpers that predate this dependency) keeps
        # working unchanged; without them the content-provenance check below
        # is skipped. `FrontierReasoningEngine` always wires both in its own
        # default reducer so the production append/replay path is fully
        # covered -- see `fre.engine.FrontierReasoningEngine.__init__`.
        self._artifact_reader = artifact_reader
        self._schema_registry = schema_registry
        # Finding I (C07 remediation): without an `artifact_reader`, this
        # reducer previously *silently skipped* the M04 v2 artifact's
        # `content_hash`/`determinism_hash` recomputation -- accepting any
        # caller-asserted hash unverified with no signal that verification
        # never ran. That is now a hard, fail-closed error unless this flag
        # is explicitly set, so "unverified" can never be the silent default
        # -- it must be a deliberate, named opt-in (e.g. a read-only replay
        # context that genuinely has no artifact store wired).
        self._trust_unverified_artifacts = trust_unverified_artifacts

    def accepts_snapshot_version(self, version: str) -> bool:
        return version in self.compatible_snapshot_versions

    def initial(self, run_id: UUID) -> RunState:
        return RunState(run_id=run_id, version=0)

    def apply(self, state: RunState, event: StoredEvent) -> RunState:
        if event.run_id != state.run_id or event.sequence != state.version + 1:
            raise ValueError("event is not the next contiguous event for this state")
        payload = event.validated_payload()
        # `pending_semantic_settlements` is excluded from `model_dump` (see its
        # field docstring), so `RunState.model_validate({**state.model_dump(...),
        # **changes})` below would otherwise silently reset it to empty on every
        # single apply() call. Seed it from the incoming state unconditionally so
        # it survives untouched across events that don't concern it; the branches
        # below overwrite this default whenever they actually change it.
        changes: dict[str, object] = {
            "version": event.sequence,
            "pending_semantic_settlements": state.pending_semantic_settlements,
        }
        if isinstance(payload, RunCreated):
            if state.version != 0:
                raise ValueError("RunCreated must be the first event")
            changes.update(status="CREATED", config_hash=payload.config_hash)
        elif isinstance(payload, RunStatusChanged):
            changes["status"] = payload.status
        elif isinstance(payload, ArtifactRegistered):
            changes["artifacts"] = (*state.artifacts, payload.artifact.sha256)
        elif isinstance(payload, TestValueSet):
            changes["values"] = {**state.values, payload.key: payload.value}
        elif isinstance(payload, LedgerNodeAdded):
            # C06 remediation (F06, lesson learned from C04/C05): the
            # `ProblemFormaliser.ledger_events()` bypass that let a caller
            # emit M03 ledger nodes with unvalidated anchors/support was
            # closed by making it private -- but per the C04/C05 lesson, an
            # atomicity/provenance invariant enforced only by a wrapper's
            # calling convention is not actually closed, only hidden, if a
            # *different* caller can still build the same raw events and
            # hand them straight to `RunReducer.apply` (directly, or via a
            # store that skips `canonical_events`). This check makes that
            # bypass structurally impossible for M03 nodes specifically: it
            # re-resolves every `support` reference embedded in the node's
            # own `content` against state *already applied earlier in this
            # same reduction*, exactly mirroring how `TaskClassified`
            # (finding #2, C05) and `ModelCallRecordedV2` (finding #2, C04)
            # already look backward at sequentially-built state rather than
            # trusting the payload's own shape. It cannot re-validate a
            # `SourceAnchor`'s resolution against the originating
            # `TaskEnvelope` (the reducer has no envelope in scope), so that
            # one check remains `formalise`'s alone -- but a dangling
            # ledger/artifact reference or a same-proposal item reference to
            # a node that was never actually admitted is caught here too.
            if payload.node.producing_module == "M03" and isinstance(payload.node.content, dict):
                _validate_m03_ledger_node_provenance(state, payload.node.content)
            changes["ledger"] = EpistemicLedger().append_node(state.ledger, payload.node)
        elif isinstance(payload, LedgerEdgeAdded):
            changes["ledger"] = EpistemicLedger().append_edge(state.ledger, payload.edge)
        elif isinstance(payload, LedgerNodeRevised):
            changes["ledger"] = EpistemicLedger().revise_node(state.ledger, payload.successor)
        elif isinstance(payload, LedgerNodeStatusChanged):
            changes["ledger"] = EpistemicLedger().mark_status(
                state.ledger, payload.node_ref, payload.status
            )
        elif isinstance(payload, LedgerDependentsMarkedStale):
            ledger = EpistemicLedger()
            # Descendant traversal alone treats an unknown source as a leaf, so
            # resolve it explicitly before accepting an empty affected set.
            ledger.effective_status(state.ledger, payload.source_ref)
            descendants = ledger.descendants(state.ledger, payload.source_ref)
            matching_envelopes = tuple(
                envelope
                for envelope in state.ledger.stale_envelopes
                if payload.source_ref in envelope.changed_dependency_refs
            )
            envelope_affected = set()
            for envelope in matching_envelopes:
                changed = tuple(
                    sorted(
                        envelope.changed_dependency_refs,
                        key=lambda ref: (str(ref.node_id), ref.revision),
                    )
                )
                valid_revision_pair = (
                    len(changed) == 2
                    and changed[0].node_id == changed[1].node_id
                    and changed[1].revision == changed[0].revision + 1
                    and any(
                        edge.relation.value == "SUPERSEDES"
                        and edge.source == changed[1]
                        and edge.target == changed[0]
                        for edge in state.ledger.edges
                    )
                    and envelope.affected_node_ref in ledger.descendants(state.ledger, changed[0])
                )
                if not valid_revision_pair:
                    raise ValueError("malformed stale-dependency revision envelope")
                envelope_affected.add(envelope.affected_node_ref)
            expected = tuple(
                sorted(
                    set(descendants) | envelope_affected,
                    key=lambda ref: (str(ref.node_id), ref.revision),
                )
            )
            supplied = tuple(
                sorted(
                    payload.affected_refs,
                    key=lambda ref: (str(ref.node_id), ref.revision),
                )
            )
            if supplied != expected:
                raise ValueError("stale-dependent marker does not match ledger projection")
            updated_ledger = state.ledger
            for affected_ref in supplied:
                updated_ledger = ledger.mark_status(
                    updated_ledger, affected_ref, EpistemicStatus.STALE
                )
            changes["ledger"] = updated_ledger
        elif isinstance(payload, LedgerContradictionResolved):
            changes["ledger"] = EpistemicLedger().resolve_contradiction(
                state.ledger, payload.resolution, payload.resolution_edge
            )
        elif isinstance(payload, BudgetRevised):
            if payload.policy_version != payload.plan.policy_version:
                raise ValueError("budget revision policy version does not match realized plan")
            # C05 remediation (finding #3): `BudgetRevised` is only ever
            # constructed today as the tail of M01's classification/provenance
            # batch (`TaskClassifier.canonical_events`), revising the M02
            # bootstrap budget up to the tier the *authoritative*
            # classification demands. A `BudgetRevised` with no classification
            # behind it at all -- neither already persisted nor earlier in
            # this same batch -- is an orphaned tier revision with nothing to
            # justify it, and must be rejected here regardless of caller
            # discipline (mirrors the C04 lesson: enforce in the reducer, not
            # only in a wrapper). `state.task_signature` is populated by the
            # `TaskClassified` branch below and, thanks to the sequential
            # per-event application both `FrontierReasoningEngine.append`'s
            # preview loop and real replay perform, is already visible here
            # for any `TaskClassified` earlier in the same batch -- it does
            # not require a prior, separately-persisted batch.
            if state.task_signature is None:
                raise ValueError(
                    "budget revision requires an authoritative TaskClassified to already be "
                    "applied (in this batch or prior state); no classification signature found"
                )
            changes["budget"] = BudgetAllocator().revise(
                state.budget, payload.plan, payload.policy_hash
            )
        elif isinstance(payload, BudgetAllocated):
            if state.budget.plan is not None:
                raise ValueError("budget is already allocated")
            if payload.policy_version != payload.plan.policy_version:
                raise ValueError("budget allocation policy version does not match realized plan")
            changes["budget"] = BudgetProjection(plan=payload.plan, policy_hash=payload.policy_hash)
        elif isinstance(payload, BudgetConsumed):
            changes["budget"] = BudgetMeter().consume(state.budget, payload.usage)
        elif isinstance(payload, BudgetReserved):
            changes["budget"] = BudgetMeter().reserve(state.budget, payload.reservation)
        elif isinstance(payload, BudgetReservationSettled):
            changes["budget"] = BudgetMeter().settle(
                state.budget, payload.reservation_id, payload.actual_usage
            )
            # Duplicate-tracking half of the F09 reducer-level check (finding
            # #2): record that this reservation was genuinely settled, with
            # this exact charged usage, so the semantic model-call record that
            # is always appended alongside it in the same batch can be
            # verified against real, applied evidence rather than its own
            # unverified claim -- see the `ModelCallRecordedV2`/
            # `ModelCallFailedV2` branch below.
            changes["pending_semantic_settlements"] = {
                **state.pending_semantic_settlements,
                payload.reservation_id: payload.actual_usage,
            }
        elif isinstance(payload, BudgetReservationReleased):
            changes["budget"] = BudgetMeter().release(state.budget, payload.reservation_id)
        elif isinstance(payload, ContextCompiled):
            if payload.packet.terminal_disposition is not None and (
                payload.json_artifact is None or payload.markdown_artifact is None
            ):
                raise ValueError("terminal context requires durable JSON and Markdown artifacts")
            # Finding H (C08 remediation): shared `bind_hash` helper (see
            # `ContextCompiler.compile`'s own use of it).
            expected_packet_hash = bind_hash(payload.packet, exclude={"packet_hash"})
            if payload.packet.packet_hash != expected_packet_hash:
                raise ValueError("context packet hash does not match canonical preimage")
            changes["context_packets"] = (*state.context_packets, payload.packet)
            changes["context_compilations"] = (
                *state.context_compilations,
                ContextCompilationRecord(
                    packet_hash=payload.packet.packet_hash,
                    profile=payload.packet.profile,
                    compiler_version=payload.packet.compiler_version,
                    compression_policy_version=payload.packet.compression_policy_version,
                    applied_rule_ids=payload.packet.applied_rule_ids,
                    renderer_version=payload.renderer_version,
                    canonical_byte_size=len(canonical_json(payload.packet)),
                    json_artifact_sha256=payload.json_artifact.sha256
                    if payload.json_artifact
                    else None,
                    markdown_artifact_sha256=payload.markdown_artifact.sha256
                    if payload.markdown_artifact
                    else None,
                ),
            )
        elif isinstance(payload, ContextCompiledV2):
            # C08 (M12, F12/F04 remediation): this is the event that actually
            # PERSISTS a compiled Wave 3 context packet -- unlike the
            # in-memory-only `Wave3ContextCompiler.compile_semantic()` return
            # value, appending this event is what closes F12's "context is
            # never persisted by any production path" gap. Both artifacts are
            # mandatory on the payload type itself (unlike v1 `ContextCompiled`,
            # where they are optional), so atomicity of "packet + both durable
            # artifacts" is enforced structurally, not merely by a runtime
            # `if` check.
            packet = payload.packet
            if packet.wave3_context is None:
                raise ValueError(
                    "ContextCompiled@2.0 requires a packet with a typed wave3_context; use "
                    "the v1 ContextCompiled event for packets with no Wave 3 semantic context"
                )
            wave3 = packet.wave3_context
            # Finding H (C08 remediation): shared `bind_hash` helper.
            expected_packet_hash = bind_hash(packet, exclude={"packet_hash"})
            if packet.packet_hash != expected_packet_hash:
                raise ValueError("context packet hash does not match canonical preimage")
            for context_artifact in (payload.json_artifact, payload.markdown_artifact):
                if context_artifact.sha256 not in state.artifacts:
                    raise ValueError("context compilation artifact is not registered")
            # --- Objective 3 / critical-lesson remediation --------------
            # Everything above only proves the packet is internally self-
            # consistent (its own packet_hash seals its own fields) and that
            # its declared artifacts are real registered bytes. None of it
            # proves `wave3_context`'s claimed refs/availability correspond to
            # the run state that was ACTUALLY applied before this event. Every
            # ref field is independently re-derived/re-checked against that
            # real, already-applied state below -- never trusted merely
            # because it agrees with itself.
            if wave3.schema_version != "1.0":
                raise ValueError("wave3_context schema_version does not match a known contract")
            if state.problem_spec is None or wave3.problem_spec_ref != canonical_hash(
                state.problem_spec
            ):
                raise ValueError(
                    "wave3_context problem_spec_ref does not match the run's current ProblemSpec"
                )
            if wave3.task_signature_ref is not None and (
                state.task_signature is None
                or wave3.task_signature_ref != canonical_hash(state.task_signature)
            ):
                raise ValueError(
                    "wave3_context task_signature_ref does not match the run's current "
                    "TaskSignature"
                )
            if wave3.budget_plan_ref is not None and (
                state.budget.plan is None
                or wave3.budget_plan_ref != canonical_hash(state.budget.plan)
            ):
                raise ValueError(
                    "wave3_context budget_plan_ref does not match the run's current BudgetPlan"
                )
            if (
                wave3.budget_policy_hash is not None
                and wave3.budget_policy_hash != state.budget.policy_hash
            ):
                raise ValueError(
                    "wave3_context budget_policy_hash does not match the run's current budget "
                    "policy_hash"
                )
            # Every declared blocker ref must resolve to an actually-applied
            # ProblemBlocker -- a ref naming a blocker_id this run never
            # recorded is a dangling reference and is rejected outright (the
            # C04-C07-pattern check the plan calls out explicitly).
            real_blocker_ids = {blocker.blocker_id for blocker in state.problem_blockers}
            declared_blocker_ids = set(wave3.blocker_refs)
            if declared_blocker_ids - real_blocker_ids:
                raise ValueError(
                    "wave3_context blocker_refs references a blocker_id absent from the run's "
                    "applied problem_blockers"
                )
            # Finding C (C08 remediation): the check above only proves every
            # DECLARED entry is real -- it never proved every REAL entry was
            # declared. A packet could silently omit a real, currently-applied
            # blocker from `blocker_refs` (understating what this run actually
            # knows) and the check above would never notice. Completeness is
            # required in both directions.
            if real_blocker_ids - declared_blocker_ids:
                raise ValueError(
                    "wave3_context blocker_refs omits a real problem_blocker applied to this run"
                )
            # Finding D (C08 remediation): `prompt_version`/`model_identity`
            # are caller-supplied claims about which semantic model call
            # produced/influenced this compilation. Neither field is otherwise
            # tied to anything real -- a packet could forge any string here.
            # Require them to correspond to an actual, already-applied
            # `SemanticModelCallRecord`(V2) on this run, mirroring how C07's
            # reducer verifies `adjudication_record_ref` against a real
            # applied call.
            if (wave3.prompt_version is not None or wave3.model_identity is not None) and not any(
                call.prompt_version == wave3.prompt_version
                and call.model_id == wave3.model_identity
                for call in state.model_calls
            ):
                raise ValueError(
                    "wave3_context prompt_version/model_identity does not correspond to "
                    "any real SemanticModelCallRecord applied to this run"
                )
            if wave3.ledger_root != canonical_hash(state.ledger):
                raise ValueError(
                    "wave3_context ledger_root does not match the run's actual ledger state"
                )
            # `ledger_version` is pinned to `packet.snapshot_version` at
            # construction time (both come from the exact same compile-time
            # `snapshot_version` value) and can legitimately be BEHIND
            # `state.version` at apply time: a `ContextCompiledV2` batch may
            # (and, in `Wave3ContextRuntime.compile_and_persist`, always does)
            # append its own `ArtifactRegistered` events ahead of itself in
            # the same batch, each advancing `state.version` by one before
            # this event is reached -- none of which changes `state.ledger`
            # itself. It can never legitimately be AHEAD of `state.version`
            # (no packet can be compiled against a ledger revision the run
            # has not yet reached).
            if wave3.ledger_version != packet.snapshot_version:
                raise ValueError(
                    "wave3_context ledger_version does not match this packet's own snapshot_version"
                )
            if wave3.ledger_version > state.version:
                raise ValueError(
                    "wave3_context ledger_version is ahead of the run state actually reached"
                )
            if wave3.representation_plan_ref is not None and (
                state.representation_plan is None
                or wave3.representation_plan_ref != canonical_hash(state.representation_plan)
            ):
                raise ValueError(
                    "wave3_context representation_plan_ref does not match the run's current "
                    "representation plan"
                )
            real_artifact_hashes = {
                canonical_hash(artifact) for artifact in state.representation_artifacts
            }
            declared_artifact_refs = set(wave3.representation_artifact_refs)
            if declared_artifact_refs - real_artifact_hashes:
                raise ValueError(
                    "wave3_context representation_artifact_refs references an artifact absent "
                    "from the run's applied representation_artifacts"
                )
            # Finding C (C08 remediation): completeness in the other direction
            # too -- every real, currently-applied v1 `representation_artifacts`
            # entry (the only entries `Wave3SemanticContext.representation_
            # artifact_refs` can ever represent -- it has no v2-specific ref
            # field) must be declared; a packet omitting one is understating
            # real, already-applied state.
            if real_artifact_hashes - declared_artifact_refs:
                raise ValueError(
                    "wave3_context representation_artifact_refs omits a real "
                    "representation_artifact applied to this run"
                )
            # Sibling enumeration (the C06 lesson applied here): every UNKNOWN
            # surfaced into the permitted view must match, field-for-field, an
            # UNKNOWN the current ProblemSpec actually declares -- not only
            # `support`-style refs, and not only a subset of UnknownSpec's
            # fields.
            real_unknowns_by_id = (
                {item.id: item for item in state.problem_spec.unknowns}
                if state.problem_spec is not None
                else {}
            )
            # Finding I (C08 remediation): rather than a hand-ordered 7-tuple
            # positional comparison (which would silently stop catching a
            # mismatch the moment `UnknownSpec`/`UnresolvedUnknownRef` gained a
            # field that wasn't also added to both tuples here), reconstruct
            # the expected `UnresolvedUnknownRef` using the exact same field
            # mapping `Wave3ContextCompiler.compile_semantic` uses and compare
            # full model instances directly.
            for declared in wave3.unresolved_unknowns:
                real = real_unknowns_by_id.get(declared.id)
                if real is None:
                    raise ValueError(
                        "wave3_context unresolved_unknowns does not match the run's current "
                        "ProblemSpec UNKNOWN metadata exactly"
                    )
                expected_unknown_ref = UnresolvedUnknownRef(
                    id=real.id,
                    description=real.description,
                    domain=real.domain,
                    rationale=real.rationale,
                    impact=real.impact,
                    decision_relevance=real.decision_relevance,
                    resolvable=real.resolvable,
                    candidate_actions=real.candidate_actions,
                )
                if declared != expected_unknown_ref:
                    raise ValueError(
                        "wave3_context unresolved_unknowns does not match the run's current "
                        "ProblemSpec UNKNOWN metadata exactly"
                    )
            # Finding C (C08 remediation): completeness in the other direction
            # -- every real, currently open UNKNOWN on the applied ProblemSpec
            # must be surfaced; a packet omitting one silently understates
            # governed uncertainty this run actually has open.
            declared_unknown_ids = {item.id for item in wave3.unresolved_unknowns}
            if set(real_unknowns_by_id) - declared_unknown_ids:
                raise ValueError(
                    "wave3_context unresolved_unknowns omits a real, currently open UNKNOWN "
                    "from the run's ProblemSpec"
                )
            # Finally: `availability`/`availability_reason` are re-derived from
            # the same pure function the compiler itself calls, applied to the
            # REAL, already-applied state -- not trusted from the packet's own
            # (already ref-verified above) claim. A packet whose refs are all
            # individually genuine could still misclassify overall
            # availability (e.g. omit a real material blocker from its own
            # narrative); this closes that gap.
            recomputed_availability, recomputed_reason = derive_wave3_availability(
                problem_blockers=state.problem_blockers,
                representation=state.representation_plan,
                representation_artifacts=state.representation_artifacts,
                # Finding F (C08 remediation): recompute using the run's C07
                # bound v2 representation state too, exactly like
                # `Wave3ContextRuntime.compile_and_persist` now does -- see
                # `derive_wave3_availability`'s own docstring for why v2 is
                # preferred (finding B is only fully closed on the v2 path).
                representation_v2=state.representation_plan_v2,
                representation_artifacts_v2=state.representation_artifacts_v2,
                budget_remaining=BudgetMeter().remaining(state.budget),
            )
            if wave3.availability != recomputed_availability:
                raise ValueError(
                    "wave3_context availability does not match an independent recomputation "
                    "from the run's actually-applied state"
                )
            if wave3.availability_reason != recomputed_reason:
                raise ValueError(
                    "wave3_context availability_reason does not match the canonical reason for "
                    "the independently recomputed availability"
                )
            # Finding G (C08 remediation): every check above only proves the
            # packet is internally self-consistent and that `wave3_context`'s
            # claims match real, applied run state -- none of it proves the
            # two DECLARED ARTIFACTS actually contain the correct rendering of
            # THIS exact packet (they could be any other artifact this run
            # happened to register earlier, with a colliding claim). Checked
            # last, once the packet is already known-good on every other
            # count: `RunState.artifacts` stores only registered sha256
            # strings, not the underlying bytes, so the reducer cannot re-read
            # and re-hash stored bytes directly; it CAN independently
            # recompute what those bytes must be from `packet` itself (both
            # renderings are pure functions of the packet) and require the
            # claimed sha256 to match that recomputation exactly.
            expected_json_sha256 = hashlib.sha256(canonical_json(packet)).hexdigest()
            if payload.json_artifact.sha256 != expected_json_sha256:
                raise ValueError(
                    "context json artifact sha256 does not match the canonical bytes "
                    "recomputed from the packet itself"
                )
            expected_markdown_sha256 = hashlib.sha256(
                Wave3ContextCompiler.render_markdown(packet).encode()
            ).hexdigest()
            if payload.markdown_artifact.sha256 != expected_markdown_sha256:
                raise ValueError(
                    "context markdown artifact sha256 does not match the markdown "
                    "recomputed from the packet itself"
                )
            changes["context_packets"] = (*state.context_packets, packet)
            changes["context_compilations"] = (
                *state.context_compilations,
                ContextCompilationRecord(
                    packet_hash=packet.packet_hash,
                    profile=packet.profile,
                    compiler_version=packet.compiler_version,
                    compression_policy_version=packet.compression_policy_version,
                    applied_rule_ids=packet.applied_rule_ids,
                    renderer_version=payload.renderer_version,
                    canonical_byte_size=len(canonical_json(packet)),
                    json_artifact_sha256=payload.json_artifact.sha256,
                    markdown_artifact_sha256=payload.markdown_artifact.sha256,
                ),
            )
        elif isinstance(payload, StopDecisionRecordedV2):
            record = payload.record
            remaining = BudgetMeter().remaining(state.budget)
            if record.evaluated_state_version != state.version:
                raise ValueError("stop decision evaluated state version does not match pre-state")
            if record.evaluated_state_hash != state.state_hash:
                raise ValueError("stop decision evaluated state hash does not match pre-state")
            if record.budget_projection_hash != remaining.projection_hash:
                raise ValueError("stop decision budget binding does not match pre-state")
            # Defence-in-depth, not a decisive guarantee: state.version strictly
            # increases by exactly one per applied event (see the contiguous-
            # sequence check above), and the two checks just above already force
            # record.evaluated_state_version == state.version for THIS apply. No
            # earlier record in state.stop_decision_records can therefore ever
            # carry the same evaluated_state_version, so this duplicate-binding
            # branch is unreachable via the normal record_decision -> apply path
            # today. It is kept in case a future replay/retry path resubmits an
            # identical (version, hash, decision) triple through some other
            # route; do not treat its presence as evidence that duplicate
            # submission is exercised or tested.
            if any(
                existing.evaluated_state_version == record.evaluated_state_version
                and existing.evaluated_state_hash == record.evaluated_state_hash
                and existing.decision == record.decision
                for existing in state.stop_decision_records
            ):
                raise ValueError("stop decision evaluation binding is already recorded")
            changes["stop_decisions"] = (*state.stop_decisions, record.decision)
            changes["stop_decision_records"] = (*state.stop_decision_records, record)
        elif isinstance(payload, StopDecisionRecorded):
            changes["stop_decisions"] = (*state.stop_decisions, payload.decision)
        elif isinstance(payload, TerminalContextAssociated):
            matching_packet = next(
                (
                    packet
                    for packet in state.context_packets
                    if packet.packet_hash == payload.packet_hash
                    and packet.profile.value == "HANDOFF"
                ),
                None,
            )
            latest_decision = state.stop_decisions[-1] if state.stop_decisions else None
            if matching_packet is None:
                raise ValueError("terminal context association requires a persisted HANDOFF packet")
            if (
                latest_decision is None
                or latest_decision.disposition.value != payload.stop_disposition
                or matching_packet.terminal_disposition != payload.stop_disposition
            ):
                raise ValueError("terminal context must match the latest recorded stop decision")
            changes["terminal_context_packet_hash"] = payload.packet_hash
            changes["terminal_context_disposition"] = payload.stop_disposition
        elif isinstance(
            payload, (ModelCallRecorded, ModelCallFailed, ModelCallRecordedV2, ModelCallFailedV2)
        ):
            _validate_model_call_common(state, payload)
            # V1 events (`ModelCallRecorded`/`ModelCallFailed`, plain
            # `SemanticModelCallRecord`) predate reservation-linked accounting and
            # carry no `reservation_id` -- they remain decode-only and are not
            # subject to the batch-admission checks below.
            if isinstance(payload, ModelCallRecordedV2):
                if payload.record.status is not StructuredModelStatus.SUCCESS:
                    raise ValueError("a recorded semantic success must carry SUCCESS status")
                if payload.record.accounting_condition is not None:
                    raise ValueError(
                        "a recorded semantic success may not carry an accounting condition"
                    )
                if payload.record.validation_diagnostics:
                    raise ValueError(
                        "a recorded semantic success may not carry validation diagnostics"
                    )
            if isinstance(payload, ModelCallFailedV2):
                if (
                    payload.record.status is StructuredModelStatus.SUCCESS
                    and payload.record.accounting_condition
                    is not SemanticAccountingCondition.PROVIDER_USAGE_EXCEEDED_RESERVATION
                ):
                    raise ValueError(
                        "a semantic failure may not present SUCCESS status without a "
                        "recognised override"
                    )
                # Finding #8 (C04 remediation): PROVIDER_USAGE_EXCEEDED_RESERVATION is
                # only a legitimate accounting condition when the provider actually
                # overran its reservation AND either (a) still returned output that
                # validated (SUCCESS, downgraded to a failure purely by the override)
                # or (b) returned output that failed schema validation
                # (INVALID_STRUCTURED_OUTPUT). No other status can co-occur with this
                # accounting condition -- an arbitrary status (e.g. UNAVAILABLE,
                # TRANSIENT_FAILURE, PERMANENT_FAILURE) paired with it is forged or
                # corrupted data, not a real domain outcome.
                if (
                    payload.record.accounting_condition
                    is SemanticAccountingCondition.PROVIDER_USAGE_EXCEEDED_RESERVATION
                    and payload.record.status
                    not in (
                        StructuredModelStatus.SUCCESS,
                        StructuredModelStatus.INVALID_STRUCTURED_OUTPUT,
                    )
                ):
                    raise ValueError(
                        "provider-usage-exceeded-reservation accounting is only valid "
                        "when the semantic model call succeeded or returned invalid "
                        "structured output"
                    )
            if isinstance(payload, (ModelCallRecordedV2, ModelCallFailedV2)):
                # Finding #2 (C04 remediation): duplicate, reducer-local half of the
                # F09 batch-admission proof. `validate_semantic_reservation_admission`
                # (run by `FrontierReasoningEngine.append` over the whole proposed
                # batch, order-independently -- see its own docstring and finding #5)
                # remains the authoritative pre-persistence gate. This check makes the
                # same property impossible to bypass by calling `RunReducer.apply`
                # directly against a store that skips that gate: a semantic model-call
                # record is only ever admitted here if a matching
                # `BudgetReservationSettled` for its own `reservation_id`, with
                # bit-for-bit matching `actual_usage`, was *already applied earlier in
                # this same reduction* -- `BudgetReservationReleased` never populates
                # `pending_semantic_settlements`, so a record referencing a released
                # (or never-reserved) reservation_id is rejected here too.
                #
                # Unlike the batch-admission gate, this check is strictly sequential
                # (order-dependent): it cannot accept a settlement that is applied
                # *after* its record, because at record-apply time no later event has
                # been seen yet. The one production caller
                # (`SemanticModelRuntime._invoke`) always emits the settlement before
                # the record in its event tuple, so this never rejects real traffic;
                # `engine.append` also applies every event through this same reducer,
                # in submitted order, before persisting (see `FrontierReasoningEngine.
                # append`), so a batch that satisfies the order-independent gate but
                # reverses this order is still rejected end-to-end by that second,
                # stricter pass -- intentionally: see finding #5's test, which
                # exercises `validate_semantic_reservation_admission` directly rather
                # than the full `engine.append` path for exactly this reason.
                pending_usage = state.pending_semantic_settlements.get(
                    payload.record.reservation_id
                )
                if pending_usage is None:
                    raise ValueError(
                        "semantic model-call reservation settlement evidence is missing "
                        "from applied state"
                    )
                if pending_usage != payload.record.charged_usage:
                    raise ValueError(
                        "semantic model-call charged usage does not match its applied "
                        "budget settlement"
                    )
                changes["pending_semantic_settlements"] = {
                    key: value
                    for key, value in state.pending_semantic_settlements.items()
                    if key != payload.record.reservation_id
                }
                # Finding #1 (C04 remediation): content-provenance proof. A SUCCESS
                # record's `proposal_artifact` bytes must actually decode and validate
                # against the exact schema its own `output_schema_id`/`_version`/`_hash`
                # claim -- otherwise a forged record could point at arbitrary,
                # schema-invalid bytes and still pass every structural check above.
                # This also subsumes finding #4 (F13's schema-hash check): re-deriving
                # a successful validation against the registry-resolved model type
                # necessarily reconfirms the schema hash, structurally, not just at the
                # one `_invoke` call site.
                if (
                    payload.record.status is StructuredModelStatus.SUCCESS
                    and payload.record.proposal_artifact is not None
                    and self._artifact_reader is not None
                    and self._schema_registry is not None
                ):
                    definition, model_type = self._schema_registry.get(
                        payload.record.output_schema_id, payload.record.output_schema_version
                    )
                    if definition.schema_hash != payload.record.output_schema_hash:
                        raise ValueError(
                            "semantic model-call output schema hash does not match the "
                            "registered schema"
                        )
                    artifact_bytes = self._artifact_reader(payload.record.proposal_artifact.sha256)
                    try:
                        model_type.model_validate_json(artifact_bytes, strict=True)
                    except (ValidationError, ValueError) as error:
                        raise ValueError(
                            "semantic model-call proposal artifact does not validate "
                            "against its claimed output schema"
                        ) from error
            changes["model_calls"] = (*state.model_calls, payload.record)
        elif isinstance(payload, TaskPreliminarilyClassified):
            changes["preliminary_task_signature"] = payload.signature
        elif isinstance(payload, TaskClassified):
            # C05 remediation (finding #2): a `TaskClassified` carrying a
            # MODEL-basis dimension makes an epistemic claim ("a model
            # proposed this value") that must be backed by real M09
            # provenance, not merely the record's own self-reported `basis`
            # field. `TaskClassifier.canonical_events` emits every MODEL
            # dimension's `LedgerNodeAdded` provenance node *before*
            # `TaskClassified` in its returned batch precisely so this check
            # can be a simple, order-dependent backward look at
            # `state.ledger` -- mirroring the C04 lesson (enforce the
            # invariant in the reducer itself, not only in a wrapper the
            # caller might skip) and its own `ModelCallRecordedV2` reservation
            # check (also a backward look at state built up earlier in the
            # same sequential application). This closes the exact gap a
            # forged batch could otherwise exploit: shipping a
            # `TaskClassified` with `basis="MODEL"` dimensions and never
            # including (or omitting from) the batch a single matching
            # provenance node.
            for name, result in payload.record.dimensions.items():
                if result.basis != "MODEL":
                    continue
                expected_content = {"axis": name, **result.model_dump(mode="json")}
                if not any(
                    node.producing_module == "M01" and node.content == expected_content
                    for node in state.ledger.nodes
                ):
                    raise ValueError(
                        f"TaskClassified dimension '{name}' claims a MODEL basis but has no "
                        "matching M09 LedgerNodeAdded provenance node applied before it"
                    )
            changes["task_signature"] = payload.signature
            changes["classification_record"] = payload.record
        elif isinstance(payload, ClassificationDiagnosticRecorded):
            changes["classification_diagnostics"] = (
                *state.classification_diagnostics,
                f"{payload.code}:{payload.message}",
            )
        elif isinstance(payload, ProblemFormalised):
            changes["problem_spec"] = payload.problem
            # Blockers and contradiction diagnostics are scoped to the problem
            # specification that produced them.  The same atomic batch may add
            # replacements after this event, but stale findings must not survive
            # a reformalisation.
            changes["problem_blockers"] = ()
            changes["problem_contradictions"] = ()
            if state.problem_spec is not None and state.problem_spec != payload.problem:
                changes["representation_plan"] = None
        elif isinstance(payload, ProblemBlockerRecorded):
            # Resolve concrete provenance before accepting a blocker.
            EpistemicLedger().effective_status(state.ledger, payload.blocker.ledger_ref)
            changes["problem_blockers"] = (*state.problem_blockers, payload.blocker)
        elif isinstance(payload, ProblemContradictionRecorded):
            EpistemicLedger().effective_status(state.ledger, payload.diagnostic.left_ref)
            EpistemicLedger().effective_status(state.ledger, payload.diagnostic.right_ref)
            changes["problem_contradictions"] = (*state.problem_contradictions, payload.diagnostic)
        elif isinstance(payload, RepresentationPlanSelected):
            if state.problem_spec is None or (
                payload.plan.problem_spec_hash is not None
                and payload.plan.problem_spec_hash != canonical_hash(state.problem_spec)
            ):
                raise ValueError("representation plan does not bind current ProblemSpec")
            changes["representation_plan"] = payload.plan
        elif isinstance(payload, RepresentationArtifactCompiled):
            if (
                state.problem_spec is None
                or canonical_hash(state.problem_spec) != payload.artifact.problem_spec_hash
            ):
                raise ValueError("representation artifact does not bind current ProblemSpec")
            changes["representation_artifacts"] = (
                *state.representation_artifacts,
                payload.artifact,
            )
        elif isinstance(payload, RepresentationPlanSelectedV2):
            # C07 (M04, F10/F11 remediation): a bound v2 plan must bind to the
            # ProblemSpec and run-state revision that were actually applied
            # BEFORE this event, not merely to whatever the caller claims.
            # `RepresentationPlanV2.bind_plan_hash` already proves the plan's
            # own internal self-consistency (its `plan_hash` seals every other
            # field) at construction time -- this reducer check is the part
            # only sequentially-applied state can prove: that the plan's
            # declared `problem_spec_hash`/`source_snapshot_version` actually
            # match the ProblemSpec and version this run had reached.
            plan = payload.plan
            expected_plan_hash = bind_hash(plan, exclude={"plan_hash"})
            if plan.plan_hash != expected_plan_hash:
                raise ValueError(
                    "representation plan (v2) hash does not match its own canonical preimage"
                )
            if (
                state.problem_spec is None
                or canonical_hash(state.problem_spec) != plan.problem_spec_hash
            ):
                raise ValueError("representation plan (v2) does not bind current ProblemSpec")
            if plan.source_snapshot_version != state.version:
                raise ValueError(
                    "representation plan (v2) source_snapshot_version does not match the "
                    "run state it was actually selected against"
                )
            # --- C07 root-cause remediation (findings A/B/C) -------------
            # Everything above only proves the plan is SELF-consistent (its
            # own plan_hash seals its own fields) and binds to the ProblemSpec
            # this run actually has. None of it proves the plan's claimed
            # `registry_hash`/`candidate_scores`/`tie_triggered` correspond to
            # REALITY -- a plan could self-consistently claim to have scored
            # against a forged registry, forged scores, or a forged tie. The
            # checks below independently recompute each of those three values
            # from the real registry/scoring function and reject any
            # disagreement -- ground truth is never taken on the plan's own
            # word.
            real_registry = default_registry_v2()
            real_registry_hash = canonical_hash(real_registry)
            # Finding A, defense-in-depth: an orphan kind (declared in the
            # vocabulary but explicitly de-registered -- see
            # `ORPHAN_KIND_REASONS`) must never appear as a plan view,
            # unconditionally, regardless of what `registry_hash` claims. This
            # is checked before, and independently of, the registry_hash
            # comparison immediately below so that even a hypothetical future
            # registry that still (incorrectly) marked an orphan as available
            # could not smuggle it through.
            for view in plan.views:
                if view.kind in ORPHAN_KIND_REASONS:
                    raise ValueError(
                        f"representation plan (v2) selects orphan kind {view.kind}, which is "
                        "unconditionally rejected regardless of the plan's claimed registry_hash"
                    )
            # Finding A: the plan's claimed `registry_hash` must match the
            # hash of the REAL, currently-deployed v2 registry -- a plan
            # cannot claim to have been scored against a registry that never
            # existed.
            if plan.registry_hash != real_registry_hash:
                raise ValueError(
                    "representation plan (v2) registry_hash does not match the real, "
                    "currently-deployed representation registry"
                )
            if state.problem_spec is not None:
                # Finding B: recompute every candidate's compatibility score
                # against the ACTUAL current ProblemSpec, using the exact same
                # pure `score_details` function `RepresentationSelector`
                # itself calls (see `fre.domain.representation_registry`), and
                # compare to the plan's claimed `candidate_scores`. A plan
                # cannot claim scores that do not correspond to any real
                # scoring of the real problem.
                scored = [
                    (definition, *score_details(definition, state.problem_spec))
                    for definition in real_registry
                ]
                ordered = sorted(
                    scored, key=lambda item: (-item[1], item[0].priority, item[0].kind.value)
                )
                recomputed_scores = tuple(
                    RepresentationCandidateScore(
                        kind=definition.kind,
                        compatibility_score=score,
                        score_components=components,
                        builder_available=definition.builder_available,
                        cost=definition.cost_weight,
                        expected_benefit=round(definition.expected_benefit_weight * score, 6),
                    )
                    for definition, score, components in ordered
                )
                # Exact equality is expected (scoring is deterministic
                # arithmetic over a fixed, finite set of weights -- no
                # floating-point accumulation that would require a tolerance
                # in practice), but a small epsilon is applied to the score
                # comparison itself so a hypothetical future non-deterministic
                # scoring refinement cannot turn a benign rounding artifact
                # into a false rejection.
                if len(recomputed_scores) != len(plan.candidate_scores) or any(
                    recomputed.kind != claimed.kind
                    or recomputed.builder_available != claimed.builder_available
                    or abs(recomputed.compatibility_score - claimed.compatibility_score) > 1e-9
                    for recomputed, claimed in zip(
                        recomputed_scores, plan.candidate_scores, strict=False
                    )
                ):
                    raise ValueError(
                        "representation plan (v2) candidate_scores do not match an independent "
                        "recomputation against the current ProblemSpec"
                    )
                # Finding C: recompute `tie_triggered` from the (now-verified)
                # scores instead of trusting the plan's self-reported boolean.
                #
                # IMPORTANT (independent-review remediation, PR #24 finding
                # A): `tie_triggered` and `fallback_used` are INDEPENDENT
                # flags on `RepresentationPlanV2` and must be recomputed
                # independently. It is tempting to assume
                # `fallback_used=True` implies "selection fell through to the
                # single typed-fallback candidate with no real competing
                # candidate", which would make `recomputed_tie` trivially
                # `False` whenever `fallback_used` is set. That assumption is
                # WRONG: `RepresentationSelector.select_bound` also sets
                # `fallback_used=True` in a second, legitimate case -- when
                # there ARE >=2 real compatible candidates, they ARE tied
                # within `tie_band` (so `tie_triggered=True`), and the
                # tied second-place candidate that gets admitted as the
                # AUXILIARY view happens to be `TEXT_TABLE_FALLBACK` itself.
                # In that case both flags are `True` simultaneously, and
                # coupling them here would cause `RunReducer.apply` to
                # wrongly reject an entirely legitimate plan. `tie_triggered`
                # is therefore recomputed here PURELY from
                # `candidate_scores`/`tie_band` (top-2 within tie_band among
                # candidates meeting `minimum_compatibility`), with no
                # reference to `fallback_used` at all. Note this recomputation
                # is deliberately consistent with the "no compatible
                # candidate at all" fallback case too: when every candidate
                # scores below `minimum_compatibility`, `compatible_scores`
                # below is empty (length < 2), so `recomputed_tie` is `False`
                # there as well -- the same outcome the old coupled logic
                # produced for that case, without needing to special-case it.
                #
                # NOTE (documented limitation): the plan does not carry the
                # exact `minimum_compatibility` policy threshold that was in
                # effect at selection time (only `tie_band` is a plan field);
                # `RepresentationSelectionPolicy.minimum_compatibility`
                # defaults to 0.20 everywhere in this codebase except tests
                # that construct an ad-hoc policy, so 0.20 is used here as the
                # best available reconstruction of "compatible" candidates.
                # A plan legitimately selected under a non-default
                # `minimum_compatibility` could, in a narrow edge case,
                # disagree with this recomputation; no such policy override
                # is wired anywhere outside test-only construction today.
                compatible_scores = [
                    candidate
                    for candidate in recomputed_scores
                    if candidate.compatibility_score >= 0.20
                ]
                recomputed_tie = len(compatible_scores) > 1 and (
                    abs(
                        compatible_scores[0].compatibility_score
                        - compatible_scores[1].compatibility_score
                    )
                    <= plan.tie_band
                )
                if plan.tie_triggered != recomputed_tie:
                    raise ValueError(
                        "representation plan (v2) tie_triggered does not match an independent "
                        "recomputation from its (verified) candidate_scores and tie_band"
                    )
                # Finding G: independently recompute `input_hash` from the
                # exact (problem, signature, budget) triple `select_bound`
                # seals it from, mirroring `problem_spec_hash`'s existing
                # verification. Documented residual limitation: unlike
                # `problem_spec`, `RunState` does not guarantee
                # `task_signature`/`budget.plan` are populated by the time a
                # v2 plan is applied (M01 classification and M02 budget
                # allocation are not hard prerequisites enforced by this
                # reducer before M04 selection), so when either is absent
                # here, `input_hash` cannot be independently verified and is
                # trusted from the plan's own (already self-consistency
                # checked via `plan_hash`) claim -- this is a real, honest gap
                # or, when both are present, checked exactly.
                if state.task_signature is not None and state.budget.plan is not None:
                    expected_input_hash = canonical_hash(
                        {
                            "problem": state.problem_spec,
                            "signature": state.task_signature,
                            "budget": state.budget.plan,
                        }
                    )
                    if plan.input_hash != expected_input_hash:
                        raise ValueError(
                            "representation plan (v2) input_hash does not match an independent "
                            "recomputation from the current problem/signature/budget"
                        )
            # Objective 3: semantic adjudication is only ever a legitimate
            # input to selection when the plan itself claims it ran inside the
            # declared tie band, AND when it is backed by a real, already-
            # applied semantic model-call record carrying that exact
            # identity -- never a bare, self-reported reference.
            if plan.adjudication_record_ref is not None:
                if not plan.tie_triggered:
                    raise ValueError(
                        "representation plan (v2) references adjudication outside its own "
                        "declared tie band"
                    )
                matching_call = next(
                    (
                        call
                        for call in state.model_calls
                        if call.idempotency_key == plan.adjudication_record_ref
                    ),
                    None,
                )
                if matching_call is None:
                    raise ValueError(
                        "representation plan (v2) adjudication_record_ref has no matching "
                        "semantic model-call record applied in this run"
                    )
                # Finding H: the referenced call must actually BE an M04
                # representation-adjudication call, not merely some call in
                # this run that happens to share the claimed idempotency key
                # (which is itself a content-derived hash, but checking only
                # its presence -- as the pre-remediation code did -- does not
                # confirm its module/operation identity).
                if matching_call.module_id != "M04" or matching_call.operation != "adjudicate":
                    raise ValueError(
                        "representation plan (v2) adjudication_record_ref resolves to a "
                        "semantic model-call record that is not an M04 representation "
                        "adjudication call"
                    )
            changes["representation_plan_v2"] = plan
        elif isinstance(payload, RepresentationArtifactCompiledV2):
            bound_artifact = payload.artifact
            # `RepresentationArtifactV2.fallback_attribution_is_consistent`
            # already makes the F10 exploit shape (fallback content attributed
            # to the requested, non-executing builder) structurally
            # unconstructable. What remains for the reducer -- state this type
            # cannot see on its own -- is: does this artifact actually bind to
            # a plan and ProblemSpec this run really has, and do its claimed
            # bytes/hash actually exist and match what it asserts?
            if (
                state.problem_spec is None
                or canonical_hash(state.problem_spec) != bound_artifact.problem_spec_hash
            ):
                raise ValueError("representation artifact (v2) does not bind current ProblemSpec")
            if (
                state.representation_plan_v2 is None
                or state.representation_plan_v2.plan_hash != bound_artifact.plan_hash
            ):
                raise ValueError(
                    "representation artifact (v2) does not bind an already-applied bound plan"
                )
            # Bound against the SAME declared snapshot revision as its own
            # plan -- not against `state.version` at artifact-application
            # time, which has already advanced past the plan's own value the
            # moment `RepresentationPlanSelectedV2` itself was applied (and
            # may advance further still if unrelated events land between
            # plan selection and artifact compilation).
            if (
                bound_artifact.source_snapshot_version
                != state.representation_plan_v2.source_snapshot_version
            ):
                raise ValueError(
                    "representation artifact (v2) source_snapshot_version does not match the "
                    "bound plan it was actually built against"
                )
            if not any(
                view.kind is bound_artifact.requested_kind
                for view in state.representation_plan_v2.views
            ):
                raise ValueError(
                    "representation artifact (v2) requests a kind absent from its bound plan"
                )
            if bound_artifact.registry_hash != state.representation_plan_v2.registry_hash:
                raise ValueError(
                    "representation artifact (v2) registry_hash does not match its bound plan"
                )
            # Finding F: `registry_hash` alone does not prove the artifact
            # agrees with its plan about WHICH registry/policy generation
            # produced it -- two different (version, policy) pairs could in
            # principle hash identically only by coincidence of the hashed
            # registry content; cross-check the declared version strings
            # too, independently of the hash.
            if bound_artifact.registry_version != state.representation_plan_v2.registry_version:
                raise ValueError(
                    "representation artifact (v2) registry_version does not match its bound plan"
                )
            if (
                bound_artifact.selection_policy_version
                != state.representation_plan_v2.selection_policy_version
            ):
                raise ValueError(
                    "representation artifact (v2) selection_policy_version does not match its "
                    "bound plan"
                )
            # Finding A, defense-in-depth (artifact side): neither the
            # requested nor the actual kind may ever be an explicitly
            # de-registered orphan, unconditionally -- see the identical
            # check on `RepresentationPlanSelectedV2` above.
            for kind in (bound_artifact.requested_kind, bound_artifact.actual_kind):
                if kind in ORPHAN_KIND_REASONS:
                    raise ValueError(
                        f"representation artifact (v2) references orphan kind {kind}, which is "
                        "unconditionally rejected"
                    )
            # Finding E: the artifact's claimed ACTUAL builder identity must
            # match what the real registry actually declares for the actual
            # kind it claims to have built -- a forged builder id/version
            # cannot be smuggled through merely by keeping `registry_hash`
            # self-consistent.
            real_registry = default_registry_v2()
            real_definition = next(
                (
                    definition
                    for definition in real_registry
                    if definition.kind is bound_artifact.actual_kind
                ),
                None,
            )
            if real_definition is None:
                raise ValueError(
                    "representation artifact (v2) actual_kind is not declared by the real "
                    "representation registry"
                )
            if (
                real_definition.builder_id != bound_artifact.actual_builder_id
                or real_definition.builder_version != bound_artifact.actual_builder_version
            ):
                raise ValueError(
                    "representation artifact (v2) actual builder identity does not match what "
                    "the real representation registry declares for its actual_kind"
                )
            if bound_artifact.physical_artifact_ref.sha256 not in state.artifacts:
                raise ValueError("representation artifact (v2) bytes are not registered")
            # F10/Objective 4 decisive check: a caller-asserted `content_hash`
            # is never trusted on its own. The bytes are re-read from the
            # artifact store by their registered sha256 and rehashed; any
            # disagreement -- a forged claim, or bytes that no longer exist --
            # is rejected here, before persistence.
            #
            # Finding I: without an `artifact_reader`, this verification
            # cannot run at all -- the pre-remediation code silently skipped
            # it in that case, accepting any caller-asserted content_hash/
            # determinism_hash unverified with no signal that verification
            # never happened. That silent skip is now a hard, fail-closed
            # error unless the reducer was explicitly constructed with
            # `trust_unverified_artifacts=True` (a deliberate, named opt-in
            # for e.g. a read-only context with no artifact store wired), so
            # "unverified" can never be the accidental default.
            if self._artifact_reader is None:
                if not self._trust_unverified_artifacts:
                    raise ValueError(
                        "representation artifact (v2) content_hash/determinism_hash cannot be "
                        "independently verified because this RunReducer has no artifact_reader; "
                        "construct it with one, or pass trust_unverified_artifacts=True to "
                        "explicitly accept this artifact's bytes/hash unverified"
                    )
            else:
                try:
                    actual_bytes = self._artifact_reader(
                        bound_artifact.physical_artifact_ref.sha256
                    )
                except (KeyError, FileNotFoundError, OSError) as error:
                    raise ValueError(
                        "representation artifact (v2) bytes could not be read from the "
                        "artifact store"
                    ) from error
                if actual_bytes is None:
                    raise ValueError("representation artifact (v2) bytes are missing")
                recomputed = hashlib.sha256(actual_bytes).hexdigest()
                if recomputed != bound_artifact.physical_artifact_ref.sha256:
                    raise ValueError(
                        "representation artifact (v2) stored bytes do not match their "
                        "registered sha256"
                    )
                if recomputed != bound_artifact.content_hash:
                    raise ValueError(
                        "representation artifact (v2) content_hash does not match the "
                        "artifact's actual stored bytes; the caller's claim is rejected"
                    )
                # Finding M: the shared `representation_determinism_hash`
                # preimage function -- the exact same one
                # `RepresentationSelector.build_bound` calls -- rather than a
                # second, hand-copied dict-literal mirror of it.
                expected_determinism_hash = representation_determinism_hash(
                    registry_hash=bound_artifact.registry_hash,
                    actual_kind=bound_artifact.actual_kind,
                    problem_spec_hash=bound_artifact.problem_spec_hash,
                    actual_builder_id=bound_artifact.actual_builder_id,
                    actual_builder_version=bound_artifact.actual_builder_version,
                    content_hash=recomputed,
                )
                if bound_artifact.determinism_hash != expected_determinism_hash:
                    raise ValueError(
                        "representation artifact (v2) determinism_hash does not match its "
                        "declared inputs and actual content"
                    )
            changes["representation_artifacts_v2"] = (
                *state.representation_artifacts_v2,
                bound_artifact,
            )
        if (
            isinstance(payload, RunStatusChanged)
            and payload.status
            in {
                "COMPLETE",
                "PARTIAL_BUDGET",
                "BLOCKED",
                "FAILED_INVARIANT",
                "CANCELLED",
            }
            and (
                state.terminal_context_packet_hash is None
                or state.terminal_context_disposition != payload.status
            )
        ):
            raise ValueError("terminal status requires terminal context association")
        return RunState.model_validate(
            {**state.model_dump(mode="json"), **changes},
            strict=False,
        )

    def reduce(self, run_id: UUID, events: tuple[StoredEvent, ...]) -> RunState:
        state = self.initial(run_id)
        for event in events:
            state = self.apply(state, event)
        return state


def validate_semantic_reservation_admission(events: tuple[StoredEvent, ...]) -> None:
    """Batch-admission proof that every authoritative semantic outcome in
    `events` is backed by real, matching budget-accounting evidence -- from
    within the very same proposed batch, not merely trusted from the caller.

    `SemanticModelRuntime._invoke` always appends a reservation-settlement (or,
    on interruption before a record exists, a release with no paired record)
    together with the `ModelCallRecorded`/`ModelCallFailed` v2 event in one
    `FrontierReasoningEngine.append` call. This function is the batch-scoped
    half of the F09 fix: for every `ModelCallRecordedV2`/`ModelCallFailedV2`
    anywhere in the batch, it requires exactly one still-unconsumed
    `BudgetReservationSettled` (never a `BudgetReservationReleased`) for that
    record's own `reservation_id`, *somewhere in the same batch*, whose
    `actual_usage` matches the record's `charged_usage` bit-for-bit.

    This closes the exact reviewer-flagged gap: a forged `ModelCallRecordedV2`
    paired with a `BudgetReservationReleased` (instead of `Settled`) for its
    reservation, or with a settlement for someone else's reservation_id, is
    rejected here before a single event in the batch is persisted.

    Finding #5 (C04 remediation): this function scans the *whole* batch in two
    passes -- first collecting every settlement/release by `reservation_id`,
    then validating every record against that complete map -- rather than a
    single forward scan that only sees events positioned earlier in the
    tuple. A record is therefore correctly admitted regardless of whether its
    settlement happens to be ordered before or after it within the same
    submitted batch; only membership in the same batch matters, not relative
    position. (`RunReducer.apply`'s own embedded duplicate of this check,
    applied sequentially per event during both this preview and later replay,
    remains strictly order-dependent -- see the long comment on that check for
    why that is intentional and does not regress real traffic.)

    Finding #11 (C04 remediation, documentation-only): a settlement referenced
    by a record MUST be in *this same batch* as that record -- this is a
    deliberate design choice, not an oversight. This function only ever sees
    one proposed `FrontierReasoningEngine.append` batch at a time and proves
    linkage within it; it has no visibility into, and deliberately does not
    trust, any settlement/release from a prior, already-persisted batch. A
    future caller needing cross-batch or streaming settlement (e.g. settling a
    reservation in one call and recording the outcome in a later one) would
    need to extend this function to consult persisted history, not just
    assume today's within-batch proof already covers that case.

    v1 events (`ModelCallRecorded`/`ModelCallFailed`, carrying a plain
    `SemanticModelCallRecord` with no `reservation_id`) predate reservation
    accounting and are intentionally left decode-only: they are ignored here.

    Whether the settled/released reservation itself existed beforehand is
    still enforced independently and unconditionally by `BudgetMeter.settle`/
    `.release` (raising `InvalidReservationSettlement`/`ReservationConflict`)
    the moment the reducer applies that same event -- this function does not
    duplicate that check, only the cross-event linkage `apply()` cannot see
    one event at a time.
    """
    # Pass 1: collect every settlement/release in the batch by reservation_id,
    # in encounter order, regardless of where the corresponding record sits.
    # EU-03 (C04 cleanup, item #3): a `deque` (not `list`) so pass 2's FIFO
    # consumption below is O(1) per record via `popleft()` instead of O(n)
    # via `list.pop(0)` (which shifts every remaining element down). FIFO
    # ordering semantics are identical to a list's for this append/pop-front
    # usage; only the complexity of draining the queue changes.
    pending: dict[str, deque[tuple[str, ResourceVector | None]]] = {}
    for event in events:
        payload = event.validated_payload()
        if isinstance(payload, BudgetReservationSettled):
            pending.setdefault(payload.reservation_id, deque()).append(
                ("SETTLED", payload.actual_usage)
            )
        elif isinstance(payload, BudgetReservationReleased):
            pending.setdefault(payload.reservation_id, deque()).append(("RELEASED", None))
    # Pass 2: validate every record against the complete map built above, so a
    # settlement positioned after its record in the tuple is still found.
    for event in events:
        payload = event.validated_payload()
        if isinstance(payload, (ModelCallRecordedV2, ModelCallFailedV2)):
            queue = pending.get(payload.record.reservation_id)
            if not queue:
                raise ValueError(
                    "semantic model-call reservation settlement evidence is missing from this batch"
                )
            kind, actual_usage = queue.popleft()
            if kind != "SETTLED":
                raise ValueError(
                    "semantic model-call must be paired with a reservation settlement "
                    "in this batch, not a release"
                )
            if actual_usage != payload.record.charged_usage:
                raise ValueError(
                    "semantic model-call charged usage does not match its budget settlement"
                )
