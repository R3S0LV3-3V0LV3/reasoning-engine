"""Provider-neutral structured-model port."""

from typing import Protocol

from fre.domain.semantic import StructuredModelRequest, StructuredModelResult


class StructuredModelPort(Protocol):
    async def generate(self, request: StructuredModelRequest) -> StructuredModelResult: ...
