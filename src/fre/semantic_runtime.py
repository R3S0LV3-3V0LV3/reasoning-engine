"""Durable, asynchronous semantic-model execution outside the replay core."""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from fre.domain.budget import BudgetExceeded, BudgetReservation, ResourceVector
from fre.domain.common import ArtifactRef, FrozenModel, JsonValue, canonical_hash, canonical_json
from fre.domain.semantic import (
    SemanticAccountingCondition,
    SemanticCallCharge,
    SemanticChargeBasis,
    SemanticModelCallRecord,
    SemanticModelCallRecordV2,
    StructuredModelRequest,
    StructuredModelResult,
    StructuredModelStatus,
)
from fre.engine import FrontierReasoningEngine
from fre.ports.models import StructuredModelPort
from fre.prompts.registry import PromptDefinition, PromptRegistry
from fre.prompts.schemas import OutputSchemaDefinition, OutputSchemaRegistry
from fre.runtime.events import (
    ArtifactRegistered,
    BudgetReservationReleased,
    BudgetReservationSettled,
    BudgetReserved,
    EventPayload,
    ModelCallFailedV2,
    ModelCallRecordedV2,
)


class SemanticRuntimePolicy(FrozenModel):
    version: str = "wave3-semantic-runtime/2.0"
    maximum_repair_attempts: int = Field(default=1, ge=0, le=1)
    reserve_input_tokens: int = 4096
    reserve_output_tokens: int = 2048

    @property
    def policy_hash(self) -> str:
        return canonical_hash(self)


class SemanticExecution(FrozenModel):
    proposal: BaseModel | None = None
    record: SemanticModelCallRecordV2 | SemanticModelCallRecord | None = None
    event_payloads: tuple[EventPayload, ...] = ()
    reused: bool = False
    repaired: bool = False
    cause: str | None = None


class SemanticOwnershipError(ValueError):
    """The selected prompt/schema does not belong to the requested operation."""


class SemanticSchemaBindingError(ValueError):
    """The request's schema hash does not match the registry's canonical bytes."""


class SemanticPolicyMismatchError(ValueError):
    """The runtime policy graph does not match the run's persisted identity."""


class SemanticModelRuntime:
    def __init__(
        self,
        model: StructuredModelPort,
        engine: FrontierReasoningEngine,
        prompts: PromptRegistry,
        schemas: OutputSchemaRegistry,
        policy: SemanticRuntimePolicy | None = None,
        execution_config_hash: str | None = None,
    ) -> None:
        self.model = model
        self.engine = engine
        self.artifacts = engine.artifacts
        self.prompts = prompts
        self.schemas = schemas
        self.policy = policy or SemanticRuntimePolicy()
        self.execution_config_hash = execution_config_hash

    async def execute(
        self,
        *,
        run_id: object,
        module_id: str,
        module_version: str,
        operation: str,
        prompt_id: str,
        prompt_version: str,
        repair_prompt_version: str | None = None,
        canonical_input: JsonValue,
        policy: SemanticRuntimePolicy | None = None,
        allow_repair: bool = True,
    ) -> SemanticExecution:
        from uuid import UUID

        if not isinstance(run_id, UUID):
            raise TypeError("run_id must be a UUID")
        policy = policy or self.policy
        if self.execution_config_hash is not None:
            if policy != self.policy:
                raise SemanticPolicyMismatchError(
                    "the policy override does not match the composed semantic runtime policy"
                )
            persisted_hash = self.engine.inspect(run_id).config_hash
            if persisted_hash != self.execution_config_hash:
                raise SemanticPolicyMismatchError(
                    "the composed policy hash does not match the run's persisted config_hash: "
                    f"{self.execution_config_hash} != {persisted_hash}"
                )
        prompt = self.prompts.get(prompt_id, prompt_version)
        schema, model_type = self.schemas.get(prompt.output_schema_id, prompt.output_schema_version)
        self._validate_ownership(prompt, schema, module_id, operation)
        rendered = self.prompts.render(prompt_id, prompt_version, canonical_input)
        identity = self._identity(
            run_id=str(run_id),
            module_id=module_id,
            operation=operation,
            module_version=module_version,
            policy_hash=policy.policy_hash,
            prompt=prompt,
            schema=schema,
            input_hash=rendered.canonical_input_hash,
        )
        reused = self._reuse(run_id, identity, model_type)
        if reused is not None:
            return reused
        first = await self._invoke(
            run_id,
            identity,
            module_id,
            module_version,
            operation,
            prompt,
            rendered.messages,
            rendered.canonical_input_hash,
            schema,
            model_type,
            policy,
            None,
        )
        if (
            first.proposal is not None
            or not allow_repair
            or policy.maximum_repair_attempts == 0
            or first.record is None
            or first.record.status is not StructuredModelStatus.INVALID_STRUCTURED_OUTPUT
        ):
            return first

        repair_prompt = self.prompts.get(
            f"{prompt_id}.repair", repair_prompt_version or prompt_version
        )
        self._validate_repair_ownership(repair_prompt, prompt, schema, module_id, operation)
        repair_input: JsonValue = {
            "original_canonical_input": canonical_input,
            "parent_call_identity": identity,
            "invalid_raw_response_artifact": first.record.raw_artifact.model_dump(mode="json")
            if first.record.raw_artifact
            else None,
            "schema_validation_diagnostics": list(first.record.validation_diagnostics),
        }
        repair_render = self.prompts.render(
            repair_prompt.prompt_id, repair_prompt.prompt_version, repair_input
        )
        repair_identity = self._identity(
            run_id=str(run_id),
            module_id=module_id,
            operation=f"{operation}-repair",
            module_version=module_version,
            policy_hash=policy.policy_hash,
            prompt=repair_prompt,
            schema=schema,
            input_hash=repair_render.canonical_input_hash,
            parent=identity,
        )
        repaired = await self._invoke(
            run_id,
            repair_identity,
            module_id,
            module_version,
            f"{operation}-repair",
            repair_prompt,
            repair_render.messages,
            repair_render.canonical_input_hash,
            schema,
            model_type,
            policy,
            identity,
        )
        if repaired.record is None:
            return first.model_copy(
                update={"cause": "REPAIR_BUDGET_UNAVAILABLE", "repaired": False}
            )
        return repaired.model_copy(update={"repaired": True})

    async def _invoke(
        self,
        run_id: object,
        identity: str,
        module_id: str,
        module_version: str,
        operation: str,
        prompt: PromptDefinition,
        messages: tuple[dict[str, JsonValue], ...],
        input_hash: str,
        schema: OutputSchemaDefinition,
        model_type: type[BaseModel],
        policy: SemanticRuntimePolicy,
        repair_parent: str | None,
    ) -> SemanticExecution:
        from uuid import UUID

        assert isinstance(run_id, UUID)
        reservation = BudgetReservation(
            reservation_id=str(self.engine.uuids.new()),
            action_id=identity,
            resources=ResourceVector(
                llm_calls=1,
                input_tokens=policy.reserve_input_tokens,
                output_tokens=policy.reserve_output_tokens,
            ),
        )
        reserved_event = BudgetReserved(reservation=reservation)
        state = self.engine.inspect(run_id)
        try:
            self.engine.append(
                run_id,
                state.version,
                (self.engine.make_event(run_id, reserved_event, module_id="semantic-runtime"),),
            )
        except (BudgetExceeded, ValueError):
            return SemanticExecution(cause="BUDGET_UNAVAILABLE")
        request = StructuredModelRequest(
            role=prompt.model_role,
            messages=messages,
            output_schema_id=schema.schema_id,
            output_schema_version=schema.schema_version,
            output_schema_hash=schema.schema_hash,
            max_input_tokens=policy.reserve_input_tokens,
            max_output_tokens=policy.reserve_output_tokens,
            idempotency_key=identity,
        )
        # Re-check the schema hash against a fresh registry lookup, immediately
        # before the provider is invoked. `schema` was captured earlier in
        # `execute()`; this guards against the request having been mutated,
        # replayed, or constructed by a caller who bypassed the registry lookup
        # above, and against the registry's own integrity check having since
        # started failing (`OutputSchemaRegistry.get` re-validates on every call).
        current_definition, _ = self.schemas.get(
            request.output_schema_id, request.output_schema_version
        )
        if current_definition.schema_hash != request.output_schema_hash:
            self._release(run_id, reservation.reservation_id)
            raise SemanticSchemaBindingError(
                "structured request schema hash does not match the registered schema bytes"
            )
        try:
            result = await self.model.generate(request)
        except BaseException:
            self._release(run_id, reservation.reservation_id)
            raise

        raw = self.artifacts.put(
            result.raw_response,
            media_type="application/octet-stream",
            metadata={"role": "raw-model-response", "semantic_call": identity},
        )
        raw_ref = ArtifactRef(artifact_id=raw.id, sha256=raw.sha256)
        status = result.status
        proposal: BaseModel | None = None
        diagnostics = list(result.diagnostics)
        if status is StructuredModelStatus.SUCCESS:
            try:
                proposal = model_type.model_validate_json(
                    canonical_json(result.decoded), strict=True
                )
            except ValidationError as error:
                status = StructuredModelStatus.INVALID_STRUCTURED_OUTPUT
                diagnostics.extend(self._validation_diagnostics(error))
        proposal_ref: ArtifactRef | None = None
        proposal_bytes: bytes | None = None
        if proposal is not None:
            proposal_bytes = canonical_json(proposal)
            stored = self.artifacts.put(
                proposal_bytes,
                media_type="application/json",
                metadata={"role": "validated-proposal", "semantic_call": identity},
            )
            proposal_ref = ArtifactRef(artifact_id=stored.id, sha256=stored.sha256)
        actual = self._charge(result, reservation.resources)
        over = any(
            getattr(actual, name) > getattr(reservation.resources, name)
            for name in ("llm_calls", "input_tokens", "output_tokens")
        )
        charged = reservation.resources if over else actual
        charge_basis = (
            SemanticChargeBasis.RESERVATION_CAP_ON_PROVIDER_OVERAGE
            if over
            else (
                SemanticChargeBasis.REPORTED_USAGE
                if result.usage.input_tokens is not None and result.usage.output_tokens is not None
                else SemanticChargeBasis.CONSERVATIVE_RESERVED_CAPACITY
            )
        )
        accounting = (
            SemanticAccountingCondition.PROVIDER_USAGE_EXCEEDED_RESERVATION if over else None
        )
        record = SemanticModelCallRecordV2(
            call_id=self.engine.uuids.new(),
            idempotency_key=identity,
            module_id=module_id,
            operation=operation,
            module_version=module_version,
            policy_version=policy.version,
            policy_hash=policy.policy_hash,
            prompt_id=prompt.prompt_id,
            prompt_version=prompt.prompt_version,
            template_hash=prompt.template_hash,
            output_schema_id=schema.schema_id,
            output_schema_version=schema.schema_version,
            output_schema_hash=schema.schema_hash,
            canonical_input_hash=input_hash,
            model_role=prompt.model_role,
            adapter_id=result.adapter_id,
            model_id=result.model_id,
            status=status,
            raw_artifact=raw_ref,
            proposal_artifact=proposal_ref,
            usage=result.usage,
            policy_charge=SemanticCallCharge(
                llm_calls=1,
                input_tokens=charged.input_tokens,
                output_tokens=charged.output_tokens,
                basis=charge_basis.value,
            ),
            reservation_id=reservation.reservation_id,
            reported_usage=result.usage,
            charged_usage=charged,
            charge_basis=charge_basis,
            repair_parent_key=repair_parent,
            accounting_condition=accounting,
            validation_diagnostics=tuple(diagnostics),
        )
        registrations: list[EventPayload] = [
            ArtifactRegistered(
                artifact=raw_ref,
                media_type="application/octet-stream",
                byte_size=len(result.raw_response),
            )
        ]
        if proposal_ref is not None and proposal_bytes is not None:
            registrations.append(
                ArtifactRegistered(
                    artifact=proposal_ref,
                    media_type="application/json",
                    byte_size=len(proposal_bytes),
                )
            )
        accounting_event: EventPayload = BudgetReservationSettled(
            reservation_id=reservation.reservation_id, actual_usage=charged
        )
        recorded: EventPayload = (
            ModelCallRecordedV2(record=record)
            if (status is StructuredModelStatus.SUCCESS and not over)
            else ModelCallFailedV2(
                record=record,
                reason=("; ".join(diagnostics) or accounting.value if accounting else status.value),
            )
        )
        payloads = (*registrations, accounting_event, recorded)
        self._append_current(run_id, payloads)
        return SemanticExecution(
            proposal=proposal if not over else None,
            record=record,
            event_payloads=(reserved_event, *payloads),
        )

    def _append_current(self, run_id: object, payloads: tuple[EventPayload, ...]) -> None:
        from uuid import UUID

        assert isinstance(run_id, UUID)
        state = self.engine.inspect(run_id)
        events = tuple(
            self.engine.make_event(run_id, p, module_id="semantic-runtime") for p in payloads
        )
        self.engine.append(run_id, state.version, events)

    def _release(self, run_id: object, reservation_id: str) -> None:
        self._append_current(run_id, (BudgetReservationReleased(reservation_id=reservation_id),))

    def recover_outstanding_reservations(self, run_id: object) -> tuple[str, ...]:
        """Release calls left in-flight by process interruption, in stable identifier order."""
        state = self.engine.inspect(run_id)  # type: ignore[arg-type]
        reservation_ids = tuple(sorted(item.reservation_id for item in state.budget.reservations))
        if reservation_ids:
            self._append_current(
                run_id,
                tuple(BudgetReservationReleased(reservation_id=item) for item in reservation_ids),
            )
        return reservation_ids

    def _reuse(
        self, run_id: object, identity: str, model_type: type[BaseModel]
    ) -> SemanticExecution | None:
        state = self.engine.inspect(run_id)  # type: ignore[arg-type]
        initial = next((r for r in state.model_calls if r.idempotency_key == identity), None)
        repair = next(
            (
                r
                for r in state.model_calls
                if r.repair_parent_key == identity and r.status is StructuredModelStatus.SUCCESS
            ),
            None,
        )
        record = repair or initial
        if record is None:
            return None
        proposal = None
        if (
            record.status is StructuredModelStatus.SUCCESS
            and record.accounting_condition is None
            and record.proposal_artifact is not None
        ):
            proposal = model_type.model_validate_json(
                self.artifacts.get(record.proposal_artifact.sha256), strict=True
            )
        return SemanticExecution(
            proposal=proposal,
            record=record,
            reused=True,
            repaired=record.repair_parent_key is not None,
        )

    @staticmethod
    def _validate_ownership(
        prompt: PromptDefinition, schema: OutputSchemaDefinition, module: str, operation: str
    ) -> None:
        if (prompt.module_id, prompt.operation) != (module, operation) or (
            schema.module_id,
            schema.operation,
        ) != (module, operation):
            raise SemanticOwnershipError(
                "prompt/schema ownership does not match requested operation"
            )

    @staticmethod
    def _validate_repair_ownership(
        repair: PromptDefinition,
        parent: PromptDefinition,
        schema: OutputSchemaDefinition,
        module: str,
        operation: str,
    ) -> None:
        if (
            repair.module_id != module
            or repair.operation != f"{operation}-repair"
            or (repair.output_schema_id, repair.output_schema_version)
            != (schema.schema_id, schema.schema_version)
            or parent.module_id != module
        ):
            raise SemanticOwnershipError("repair prompt is incompatible with parent operation")

    @staticmethod
    def _identity(
        *, prompt: PromptDefinition, schema: OutputSchemaDefinition, **values: JsonValue
    ) -> str:
        return canonical_hash(
            {
                **values,
                "prompt_id": prompt.prompt_id,
                "prompt_version": prompt.prompt_version,
                "template_hash": prompt.template_hash,
                "output_schema_id": schema.schema_id,
                "output_schema_version": schema.schema_version,
                "output_schema_hash": schema.schema_hash,
            }
        )

    @staticmethod
    def _validation_diagnostics(error: ValidationError) -> tuple[str, ...]:
        return tuple(f"{'.'.join(map(str, item['loc']))}:{item['type']}" for item in error.errors())

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
