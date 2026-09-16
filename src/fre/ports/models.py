"""Structured-model port."""

from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class StructuredModelPort(Protocol):
    async def generate(self, *, output_schema: type[T], idempotency_key: str) -> T: ...
