"""Deterministic tool test adapter."""

from fre.ports.tools import ToolRequest, ToolResult


class FakeTool:
    def __init__(self, results: dict[str, ToolResult | Exception]) -> None:
        self._results = results
        self.calls: list[str] = []

    async def execute(self, request: ToolRequest) -> ToolResult:
        self.calls.append(request.tool_id)
        result = self._results[request.tool_id]
        if isinstance(result, Exception):
            raise result
        return result
