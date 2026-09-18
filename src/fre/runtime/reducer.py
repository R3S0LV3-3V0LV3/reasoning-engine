"""Pure reducer protocol and foundational run reducer."""

from collections.abc import Callable, Mapping
from typing import Protocol, TypeVar
from uuid import UUID

from pydantic import Field, TypeAdapter, ValidationError

from fre.domain.budget import BudgetProjection, ResourceVector
from fre.domain.common import FrozenModel, JsonValue, canonical_hash, canonical_json
from fre.domain.context import ContextCompilationRecord, ContextPacket
from fre.domain.ledger import EpistemicStatus, LedgerProjection
from fre.domain.problem import ContradictionDiagnostic, ProblemBlocker, ProblemSpec
from fre.domain.representation import RepresentationArtifact, RepresentationPlan
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
    RepresentationPlanSelected,
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
    preliminary_task_signature: TaskSignature | None = None
    classification_record: ClassificationRecord | None = None
    classification_diagnostics: tuple[str, ...] = ()
    problem_spec: ProblemSpec | None = None
    problem_blockers: tuple[ProblemBlocker, ...] = ()
    problem_contradictions: tuple[ContradictionDiagnostic, ...] = ()
    representation_plan: RepresentationPlan | None = None
    representation_artifacts: tuple[RepresentationArtifact, ...] = ()
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


class RunReducer:
    version = "2.0"
    compatible_snapshot_versions = frozenset({"1.0", "2.0"})

    def __init__(
        self,
        *,
        artifact_reader: Callable[[str], bytes] | None = None,
        schema_registry: OutputSchemaRegistry | None = None,
    ) -> None:
        # Both are optional so every existing bare `RunReducer()` construction
        # (tests, snapshot/replay helpers that predate this dependency) keeps
        # working unchanged; without them the content-provenance check below
        # is skipped. `FrontierReasoningEngine` always wires both in its own
        # default reducer so the production append/replay path is fully
        # covered -- see `fre.engine.FrontierReasoningEngine.__init__`.
        self._artifact_reader = artifact_reader
        self._schema_registry = schema_registry

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
            expected_packet_hash = canonical_hash(
                payload.packet.model_dump(exclude={"packet_hash"})
            )
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
            if any(
                item.idempotency_key == payload.record.idempotency_key for item in state.model_calls
            ):
                raise ValueError("semantic model-call identity already recorded")
            if isinstance(payload, (ModelCallRecorded, ModelCallRecordedV2)) and (
                payload.record.raw_artifact is None or payload.record.proposal_artifact is None
            ):
                raise ValueError(
                    "successful semantic model call requires raw and proposal artifacts"
                )
            for artifact in (payload.record.raw_artifact, payload.record.proposal_artifact):
                if artifact is not None and artifact.sha256 not in state.artifacts:
                    raise ValueError("semantic model-call artifact is not registered")
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
    pending: dict[str, list[tuple[str, ResourceVector | None]]] = {}
    for event in events:
        payload = event.validated_payload()
        if isinstance(payload, BudgetReservationSettled):
            pending.setdefault(payload.reservation_id, []).append(("SETTLED", payload.actual_usage))
        elif isinstance(payload, BudgetReservationReleased):
            pending.setdefault(payload.reservation_id, []).append(("RELEASED", None))
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
            kind, actual_usage = queue.pop(0)
            if kind != "SETTLED":
                raise ValueError(
                    "semantic model-call must be paired with a reservation settlement "
                    "in this batch, not a release"
                )
            if actual_usage != payload.record.charged_usage:
                raise ValueError(
                    "semantic model-call charged usage does not match its budget settlement"
                )
