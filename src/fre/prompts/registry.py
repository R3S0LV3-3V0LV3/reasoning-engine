"""Versioned deterministic prompt registry and canonical renderer."""

from pydantic import Field, model_validator

from fre.domain.common import FrozenModel, JsonValue, canonical_hash, canonical_json


class PromptIntegrityError(ValueError):
    pass


class PromptVersionConflict(PromptIntegrityError):
    pass


class PromptDefinition(FrozenModel):
    prompt_id: str
    prompt_version: str
    module_id: str
    operation: str
    model_role: str
    input_contract_version: str
    output_schema_id: str
    output_schema_version: str
    template: str
    template_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: str) -> "PromptDefinition":
        return cls(**values, template_hash=canonical_hash(values))

    @model_validator(mode="after")
    def integrity(self) -> "PromptDefinition":
        if canonical_hash(self.model_dump(exclude={"template_hash"})) != self.template_hash:
            raise ValueError("prompt template hash mismatch")
        return self


class PromptRenderResult(FrozenModel):
    prompt_id: str
    prompt_version: str
    template_hash: str
    canonical_input_hash: str
    messages: tuple[dict[str, JsonValue], ...]


class PromptRegistry:
    def __init__(self, definitions: tuple[PromptDefinition, ...] = ()) -> None:
        self._definitions: dict[tuple[str, str], PromptDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: PromptDefinition) -> None:
        key = (definition.prompt_id, definition.prompt_version)
        existing = self._definitions.get(key)
        if existing is not None and existing != definition:
            raise PromptVersionConflict(f"conflicting prompt identity: {key}")
        self._definitions[key] = definition

    def get(self, prompt_id: str, version: str) -> PromptDefinition:
        try:
            definition = self._definitions[(prompt_id, version)]
        except KeyError as error:
            raise PromptIntegrityError(f"unknown prompt: {prompt_id}@{version}") from error
        PromptDefinition.model_validate(definition.model_dump())
        return definition

    def render(self, prompt_id: str, version: str, value: JsonValue) -> PromptRenderResult:
        definition = self.get(prompt_id, version)
        input_hash = canonical_hash(value)
        rendered = definition.template.replace("{canonical_input}", canonical_json(value).decode())
        return PromptRenderResult(
            prompt_id=prompt_id,
            prompt_version=version,
            template_hash=definition.template_hash,
            canonical_input_hash=input_hash,
            messages=({"role": "system", "content": rendered},),
        )


def default_prompt_registry() -> PromptRegistry:
    specs = (
        ("m01.classify", "M01", "classify", "semantic-classifier", "m01.classification-output"),
        (
            "m01.classify.repair",
            "M01",
            "classify-repair",
            "semantic-classifier",
            "m01.classification-output",
        ),
        (
            "m03.formalise",
            "M03",
            "formalise",
            "semantic-formaliser",
            "m03.problem-formalisation-output",
        ),
        (
            "m03.formalise.repair",
            "M03",
            "formalise-repair",
            "semantic-formaliser",
            "m03.problem-formalisation-output",
        ),
        (
            "m04.adjudicate",
            "M04",
            "adjudicate",
            "semantic-adjudicator",
            "m04.representation-adjudication-output",
        ),
        (
            "m04.adjudicate.repair",
            "M04",
            "adjudicate-repair",
            "semantic-adjudicator",
            "m04.representation-adjudication-output",
        ),
    )
    return PromptRegistry(
        tuple(
            PromptDefinition.create(
                prompt_id=p,
                prompt_version="1.0",
                module_id=m,
                operation=o,
                model_role=r,
                input_contract_version="1.0",
                output_schema_id=s,
                output_schema_version="1.0",
                template=(
                    f"Operation {o}. Return only the registered schema. "
                    "Canonical input: {canonical_input}"
                ),
            )
            for p, m, o, r, s in specs
        )
    )
