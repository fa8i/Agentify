"""Runtime MCP bridge for exposing live Agentify tools to Codex."""

from __future__ import annotations

import asyncio
import json
import secrets
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from agentify.core.tool import Tool
from agentify.mcp.server import AgentifyMCPServer


class RuntimeMCPBridge:
    """Expose live Tool objects through a local broker for MCP subprocesses."""

    def __init__(
        self,
        *,
        name: str = "agentify-runtime-tools",
        tool_timeout: float | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Awaitable[str]] | None = None,
    ) -> None:
        self.name = name
        self.tool_timeout = tool_timeout
        self.tool_executor = tool_executor
        self._token = secrets.token_urlsafe(32)
        self._server: asyncio.AbstractServer | None = None
        self._host = "127.0.0.1"
        self._port: int | None = None
        self._agentify_server = AgentifyMCPServer([])
        self._tool_names: list[str] = []
        self._connection_count = 0

    @property
    def connection_count(self) -> int:
        """Number of requests served to the MCP proxy subprocess.

        Zero after a turn means Codex never started (or never reached) the
        Agentify runtime MCP server.
        """
        return self._connection_count

    @property
    def is_running(self) -> bool:
        return self._server is not None and self._port is not None

    async def start(self) -> None:
        if self.is_running:
            return
        self._server = await asyncio.start_server(self._handle_client, self._host, 0)
        sockets = self._server.sockets or []
        if not sockets:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            raise RuntimeError("Runtime MCP bridge failed to bind a local socket.")
        self._port = int(sockets[0].getsockname()[1])

    async def close(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None
        self._port = None

    def update_tools(self, tools: Sequence[Tool]) -> None:
        tool_list = list(tools)
        self._tool_names = [tool.name for tool in tool_list]
        self._agentify_server = AgentifyMCPServer(tool_list)

    def update_tool_timeout(self, tool_timeout: float | None) -> None:
        self.tool_timeout = tool_timeout

    def update_tool_executor(
        self,
        tool_executor: Callable[[str, dict[str, Any]], Awaitable[str]] | None,
    ) -> None:
        self.tool_executor = tool_executor

    def codex_config(self) -> dict[str, Any]:
        if not self.is_running or self._port is None:
            raise RuntimeError("Runtime MCP bridge is not running.")
        return {
            "mcp_servers": {
                self.name: {
                    "command": sys.executable,
                    "args": [
                        "-m",
                        "agentify.mcp.runtime_server",
                        "--host",
                        self._host,
                        "--port",
                        str(self._port),
                        "--token",
                        self._token,
                        "--call-timeout",
                        str(self.call_timeout()),
                    ],
                    "enabled_tools": self._tool_names,
                }
            }
        }

    def call_timeout(self) -> float:
        """Socket read timeout for proxied tool calls.

        Must exceed ``tool_timeout`` so the bridge's own timeout response
        (a structured MCP error) reaches the proxy instead of the socket
        read aborting first.
        """
        if self.tool_timeout is not None and self.tool_timeout > 0:
            return float(self.tool_timeout) + 30.0
        return 600.0

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._connection_count += 1
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=10)
            request = json.loads(line.decode("utf-8"))
            response = await self._dispatch(request)
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}

        writer.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def _dispatch(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if request.get("token") != self._token:
            return {"ok": False, "error": "Unauthorized runtime MCP bridge request."}

        action = request.get("action")
        if action == "list_tools":
            return {
                "ok": True,
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "inputSchema": tool.inputSchema,
                    }
                    for tool in self._agentify_server.list_tools()
                ],
            }

        if action == "call_tool":
            name = str(request.get("name") or "")
            arguments = request.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            tool_call = self._call_tool(name, arguments)
            if self.tool_timeout is not None and self.tool_timeout > 0:
                try:
                    result = await asyncio.wait_for(tool_call, timeout=float(self.tool_timeout))
                except asyncio.TimeoutError:
                    return {
                        "ok": True,
                        "isError": True,
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {"error": f"Tool '{name}' timed out."},
                                    ensure_ascii=False,
                                ),
                            }
                        ],
                    }
            else:
                result = await tool_call
            return {
                "ok": True,
                "isError": bool(getattr(result, "isError", False)),
                "content": [
                    content.model_dump()
                    if hasattr(content, "model_dump")
                    else dict(content)
                    for content in result.content
                ],
            }

        return {"ok": False, "error": f"Unsupported runtime MCP bridge action: {action}"}

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if self.tool_executor is None:
            return await self._agentify_server.call_tool(name, arguments)

        result_text = await self.tool_executor(name, arguments)
        from mcp.types import CallToolResult, TextContent

        return CallToolResult(content=[TextContent(type="text", text=result_text)])
