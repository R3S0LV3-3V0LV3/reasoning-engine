"""Deterministic provider-neutral structured-model test adapter."""

from fre.domain.semantic import StructuredModelRequest, StructuredModelResult


class FakeStructuredModel:
    def __init__(self, responses: dict[str, StructuredModelResult]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def generate(self, request: StructuredModelRequest) -> StructuredModelResult:
        self.calls.append(request.idempotency_key)
        return self._responses[request.idempotency_key]
