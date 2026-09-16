"""Deterministic structured-model test adapter."""

from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class FakeStructuredModel:
    def __init__(self, responses: dict[str, BaseModel]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    async def generate(self, *, output_schema: type[T], idempotency_key: str) -> T:
        self.calls.append(idempotency_key)
        response = self._responses[idempotency_key]
        return output_schema.model_validate(response.model_dump())
