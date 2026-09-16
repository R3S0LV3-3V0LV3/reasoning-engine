"""Budgeted, idempotent semantic-model execution outside the pure replay core."""

from pydantic import BaseModel, ValidationError

from fre.adapters.artifacts_local import LocalArtifactStore
from fre.domain.budget import BudgetExceeded, BudgetProjection, BudgetReservation, ResourceVector
from fre.domain.common import ArtifactRef, FrozenModel, JsonValue, canonical_hash, canonical_json
from fre.domain.semantic import (
    SemanticCallCharge,
    SemanticModelCallRecord,
    StructuredModelRequest,
    StructuredModelResult,
    StructuredModelStatus,
)
from fre.ports.clock import UUIDFactory
from fre.ports.models import StructuredModelPort
from fre.prompts.registry import PromptRegistry
from fre.prompts.schemas import OutputSchemaRegistry
from fre.runtime.budget_meter import BudgetMeter
from fre.runtime.events import (
    ArtifactRegistered,
    BudgetReservationSettled,
    BudgetReserved,
    EventPayload,
    ModelCallFailed,
    ModelCallRecorded,
)
from fre.runtime.reducer import RunState


class SemanticRuntimePolicy(FrozenModel):
    version: str = "wave3-semantic-runtime/1.0"
    maximum_repair_attempts: int = 1
    reserve_input_tokens: int = 4096
    reserve_output_tokens: int = 2048

    @property
    def policy_hash(self) -> str:
        return canonical_hash(self)


class SemanticExecution(FrozenModel):
    proposal: BaseModel | None = None
    record: SemanticModelCallRecord | None = None
    event_payloads: tuple[EventPayload, ...] = ()
    reused: bool = False
    repaired: bool = False


class SemanticModelRuntime:
    def __init__(
        self,
        model: StructuredModelPort,
        artifacts: LocalArtifactStore,
        uuids: UUIDFactory,
        prompts: PromptRegistry,
        schemas: OutputSchemaRegistry,
    ) -> None:
        self.model = model
        self.artifacts = artifacts
        self.uuids = uuids
        self.prompts = prompts
        self.schemas = schemas

    def execute(
        self,
        state: RunState,
        *,
        module_id: str,
        module_version: str,
        operation: str,
        prompt_id: str,
        prompt_version: str,
        canonical_input: JsonValue,
        policy: SemanticRuntimePolicy | None = None,
        allow_repair: bool = True,
    ) -> SemanticExecution:
        policy = policy or SemanticRuntimePolicy()
        prompt = self.prompts.get(prompt_id, prompt_version)
        rendered = self.prompts.render(prompt_id, prompt_version, canonical_input)
        schema, model_type = self.schemas.get(prompt.output_schema_id, prompt.output_schema_version)
        identity = canonical_hash(
            {
                "run_id": state.run_id,
                "module_id": module_id,
                "operation": operation,
                "canonical_input_hash": rendered.canonical_input_hash,
                "module_version": module_version,
                "policy_hash": policy.policy_hash,
                "prompt_id": prompt.prompt_id,
                "prompt_version": prompt.prompt_version,
                "template_hash": prompt.template_hash,
                "output_schema_id": schema.schema_id,
                "output_schema_version": schema.schema_version,
                "output_schema_hash": schema.schema_hash,
            }
        )
        initial = next(
            (record for record in state.model_calls if record.idempotency_key == identity), None
        )
        prior_repair = next(
            (
                record
                for record in state.model_calls
                if record.repair_parent_key == identity
                and record.status is StructuredModelStatus.SUCCESS
            ),
            None,
        )
        prior = prior_repair or initial
        if prior is not None:
            proposal: BaseModel | None = None
            if prior.status is StructuredModelStatus.SUCCESS and prior.raw_artifact is not None:
                try:
                    proposal = model_type.model_validate_json(
                        self.artifacts.get(prior.raw_artifact.sha256), strict=True
                    )
                except ValidationError:
                    proposal = None
            return SemanticExecution(
                proposal=proposal,
                record=prior,
                reused=True,
                repaired=prior.repair_parent_key is not None,
            )
        first = self._call(
            state.budget,
            identity,
            module_id,
            module_version,
            operation,
            prompt,
            rendered.messages,
            rendered.canonical_input_hash,
            schema.schema_id,
            schema.schema_version,
            schema.schema_hash,
            model_type,
            policy,
            None,
        )
        if (
            first.proposal is not None
            or not allow_repair
            or first.record is None
            or first.record.status is not StructuredModelStatus.INVALID_STRUCTURED_OUTPUT
        ):
            return first
        projected = self._project_budget(state.budget, first.event_payloads)
        repair_prompt = self.prompts.get(f"{prompt_id}.repair", prompt_version)
        repair_identity = canonical_hash(
            {
                "parent": identity,
                "repair": 1,
                "prompt_id": repair_prompt.prompt_id,
                "template_hash": repair_prompt.template_hash,
            }
        )
        repaired = self._call(
            projected,
            repair_identity,
            module_id,
            module_version,
            f"{operation}-repair",
            repair_prompt,
            rendered.messages,
            rendered.canonical_input_hash,
            schema.schema_id,
            schema.schema_version,
            schema.schema_hash,
            model_type,
            policy,
            identity,
        )
        return SemanticExecution(
            proposal=repaired.proposal,
            record=repaired.record,
            event_payloads=(*first.event_payloads, *repaired.event_payloads),
            repaired=True,
        )

    def _call(
        self,
        budget: BudgetProjection,
        identity: str,
        module_id: str,
        module_version: str,
        operation: str,
        prompt: object,
        messages: tuple[dict[str, JsonValue], ...],
        input_hash: str,
        schema_id: str,
        schema_version: str,
        schema_hash: str,
        model_type: type[BaseModel],
        policy: SemanticRuntimePolicy,
        repair_parent: str | None,
    ) -> SemanticExecution:
        from fre.prompts.registry import PromptDefinition

        assert isinstance(prompt, PromptDefinition)
        reservation_id = str(self.uuids.new())
        reserved = ResourceVector(
            llm_calls=1,
            input_tokens=policy.reserve_input_tokens,
            output_tokens=policy.reserve_output_tokens,
        )
        reservation = BudgetReservation(
            reservation_id=reservation_id, action_id=identity, resources=reserved
        )
        try:
            BudgetMeter().reserve(budget, reservation)
        except (BudgetExceeded, ValueError):
            return SemanticExecution()
        request = StructuredModelRequest(
            role=prompt.model_role,
            messages=messages,
            output_schema_id=schema_id,
            output_schema_version=schema_version,
            max_input_tokens=policy.reserve_input_tokens,
            max_output_tokens=policy.reserve_output_tokens,
            idempotency_key=identity,
        )
        result = self.model.generate(request)
        descriptor = self.artifacts.put(
            result.raw_response,
            media_type="application/octet-stream",
            metadata={"semantic_call": identity, "status": result.status.value},
        )
        artifact = ArtifactRef(artifact_id=descriptor.id, sha256=descriptor.sha256)
        actual = self._charge(result, reserved)
        status = result.status
        proposal: BaseModel | None = None
        if status is StructuredModelStatus.SUCCESS:
            try:
                proposal = model_type.model_validate_json(
                    canonical_json(result.decoded), strict=True
                )
            except ValidationError:
                status = StructuredModelStatus.INVALID_STRUCTURED_OUTPUT
        charge = SemanticCallCharge(
            llm_calls=1,
            input_tokens=actual.input_tokens,
            output_tokens=actual.output_tokens,
            basis="REPORTED_USAGE"
            if result.usage.input_tokens is not None and result.usage.output_tokens is not None
            else "CONSERVATIVE_RESERVED_CAPACITY",
        )
        record = SemanticModelCallRecord(
            call_id=self.uuids.new(),
            idempotency_key=identity,
            module_id=module_id,
            operation=operation,
            module_version=module_version,
            policy_version=policy.version,
            policy_hash=policy.policy_hash,
            prompt_id=prompt.prompt_id,
            prompt_version=prompt.prompt_version,
            template_hash=prompt.template_hash,
            output_schema_id=schema_id,
            output_schema_version=schema_version,
            output_schema_hash=schema_hash,
            canonical_input_hash=input_hash,
            model_role=prompt.model_role,
            adapter_id=result.adapter_id,
            model_id=result.model_id,
            status=status,
            raw_artifact=artifact,
            usage=result.usage,
            policy_charge=charge,
            repair_parent_key=repair_parent,
        )
        record_event: EventPayload = (
            ModelCallRecorded(record=record)
            if status is StructuredModelStatus.SUCCESS
            else ModelCallFailed(
                record=record, reason="; ".join(result.diagnostics) or status.value
            )
        )
        events: tuple[EventPayload, ...] = (
            BudgetReserved(reservation=reservation),
            ArtifactRegistered(
                artifact=artifact,
                media_type="application/octet-stream",
                byte_size=len(result.raw_response),
            ),
            BudgetReservationSettled(reservation_id=reservation_id, actual_usage=actual),
            record_event,
        )
        return SemanticExecution(proposal=proposal, record=record, event_payloads=events)

    @staticmethod
    def _charge(result: StructuredModelResult, reserved: ResourceVector) -> ResourceVector:
        return ResourceVector(
            llm_calls=1,
            input_tokens=result.usage.input_tokens
            if result.usage.input_tokens is not None
            else reserved.input_tokens,
            output_tokens=result.usage.output_tokens
            if result.usage.output_tokens is not None
            else reserved.output_tokens,
        )

    @staticmethod
    def _project_budget(
        budget: BudgetProjection, payloads: tuple[EventPayload, ...]
    ) -> BudgetProjection:
        result = budget
        meter = BudgetMeter()
        for payload in payloads:
            if isinstance(payload, BudgetReserved):
                result = meter.reserve(result, payload.reservation)
            elif isinstance(payload, BudgetReservationSettled):
                result = meter.settle(result, payload.reservation_id, payload.actual_usage)
        return result
