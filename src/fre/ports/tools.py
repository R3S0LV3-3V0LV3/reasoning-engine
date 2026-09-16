"""Tool port."""

from typing import Protocol

from fre.domain.common import FrozenModel, JsonValue


class ToolRequest(FrozenModel):
    tool_id: str
    payload: dict[str, JsonValue]


class ToolResult(FrozenModel):
    status: str
    payload: dict[str, JsonValue]


class ToolPort(Protocol):
    async def execute(self, request: ToolRequest) -> ToolResult: ...
