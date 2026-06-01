"""Tool adapter for Codex providers that consume tools through MCP."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from agentify.core.tool import Tool
from agentify.memory.interfaces import MemoryAddress


class CodexMCPToolAdapter:
    """Prepare live Agentify tools for Codex runtime MCP execution."""

    def __init__(
        self,
        *,
        agent_name: str,
        provider: str,
        tool_timeout: int,
        max_tool_iter: int | None,
        callbacks: list[Any],
        execute_tool,
        add_memory,
    ) -> None:
        self.agent_name = agent_name
        self.provider = provider
        self.tool_timeout = tool_timeout
        self.max_tool_iter = max_tool_iter
        self.callbacks = callbacks
        self._execute_tool = execute_tool
        self._add_memory = add_memory
        self._tool_call_count = 0

    def llm_params(self, tools: list[Tool], *, addr: MemoryAddress) -> dict[str, Any]:
        return {
            "agentify_tools": tools,
            "agentify_tool_timeout": self.tool_timeout,
            "agentify_tool_executor": (
                lambda tool_name, arguments: self.execute_tool_call(
                    tool_name,
                    arguments,
                    addr=addr,
                )
            ),
        }

    async def execute_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        addr: MemoryAddress,
    ) -> str:
        """Execute a Codex MCP tool call while preserving Agentify history."""
        tool_call_id = f"mcp_{self.provider[:3]}_{uuid.uuid4().hex[:12]}"
        arguments_json = json.dumps(arguments or {}, ensure_ascii=False)
        tool_call = {
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": arguments_json,
            },
        }

        for cb in self.callbacks:
            cb_func = getattr(cb, "on_assistant_tool_intent", None)
            if callable(cb_func):
                cb_func(self.agent_name, "", [tool_name])

        await self._add_memory(
            role="assistant",
            content="",
            tool_calls=[tool_call],
            metadata={"source": "codex_mcp"},
            addr=addr,
        )

        if self.max_tool_iter is not None and self._tool_call_count >= self.max_tool_iter:
            result_content = json.dumps(
                {
                    "error": (
                        f"Tool iteration limit reached after {self.max_tool_iter} "
                        "tool calls."
                    )
                }
            )
        else:
            self._tool_call_count += 1
            try:
                result_content = await asyncio.wait_for(
                    self._execute_tool(tool_name, arguments or {}),
                    timeout=float(self.tool_timeout),
                )
            except asyncio.TimeoutError:
                result_content = json.dumps(
                    {
                        "error": (
                            f"Tool '{tool_name}' execution timed out after "
                            f"{self.tool_timeout} seconds."
                        )
                    }
                )

        await self._add_memory(
            role="tool",
            content=result_content,
            tool_call_id=tool_call_id,
            name=tool_name,
            metadata={"source": "codex_mcp"},
            addr=addr,
        )
        return result_content
