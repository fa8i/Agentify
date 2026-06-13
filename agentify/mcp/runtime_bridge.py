"""Runtime MCP bridge for exposing live Agentify tools to Codex."""

from __future__ import annotations

import asyncio
import json
import secrets
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agentify.core.tool import Tool
from agentify.mcp.server import AgentifyMCPServer

ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[str]]

_DEFAULT_SESSION = "__default__"


@dataclass
class _SessionEntry:
    """Per-session tool registry so concurrent turns never share executors."""

    server: AgentifyMCPServer = field(default_factory=lambda: AgentifyMCPServer([]))
    tool_names: list[str] = field(default_factory=list)
    tool_timeout: float | None = None
    tool_executor: ToolExecutor | None = None
    connection_count: int = 0


class RuntimeMCPBridge:
    """Expose live Tool objects through a local broker for MCP subprocesses.

    Tools, timeout, and executor are registered **per session** via
    :meth:`register_session`, so concurrent turns for different Agentify
    sessions can safely share one bridge. The legacy single-session API
    (``update_tools`` / ``update_tool_timeout`` / ``update_tool_executor``
    plus ``codex_config()`` without a session) operates on a default session
    and remains supported.
    """

    def __init__(
        self,
        *,
        name: str = "agentify-runtime-tools",
        tool_timeout: float | None = None,
        tool_executor: ToolExecutor | None = None,
    ) -> None:
        self.name = name
        self._token = secrets.token_urlsafe(32)
        self._server: asyncio.AbstractServer | None = None
        self._host = "127.0.0.1"
        self._port: int | None = None
        self._sessions: dict[str, _SessionEntry] = {}
        self._connection_count = 0
        if tool_timeout is not None or tool_executor is not None:
            entry = self._entry(_DEFAULT_SESSION)
            entry.tool_timeout = tool_timeout
            entry.tool_executor = tool_executor

    # ------------------------------------------------------------------
    # Session registry
    # ------------------------------------------------------------------

    def _entry(self, session_id: str | None) -> _SessionEntry:
        key = session_id or _DEFAULT_SESSION
        entry = self._sessions.get(key)
        if entry is None:
            entry = _SessionEntry()
            self._sessions[key] = entry
        return entry

    def register_session(
        self,
        session_id: str,
        *,
        tools: Sequence[Tool],
        tool_timeout: float | None = None,
        tool_executor: ToolExecutor | None = None,
    ) -> None:
        """Register (or refresh) tools, timeout, and executor for a session."""
        entry = self._entry(session_id)
        tool_list = list(tools)
        entry.tool_names = [tool.name for tool in tool_list]
        entry.server = AgentifyMCPServer(tool_list)
        entry.tool_timeout = tool_timeout
        entry.tool_executor = tool_executor

    def unregister_session(self, session_id: str) -> None:
        """Drop a session's tool registry (e.g., when the session ends)."""
        self._sessions.pop(session_id, None)

    @property
    def connection_count(self) -> int:
        """Total requests served to MCP proxy subprocesses.

        Zero after a turn means Codex never started (or never reached) the
        Agentify runtime MCP server.
        """
        return self._connection_count

    def session_connection_count(self, session_id: str | None) -> int:
        """Requests served for one session's MCP proxy."""
        entry = self._sessions.get(session_id or _DEFAULT_SESSION)
        return entry.connection_count if entry is not None else 0

    # Legacy single-session API (default session)

    @property
    def tool_timeout(self) -> float | None:
        return self._entry(_DEFAULT_SESSION).tool_timeout

    @property
    def tool_executor(self) -> ToolExecutor | None:
        return self._entry(_DEFAULT_SESSION).tool_executor

    def update_tools(self, tools: Sequence[Tool]) -> None:
        entry = self._entry(_DEFAULT_SESSION)
        tool_list = list(tools)
        entry.tool_names = [tool.name for tool in tool_list]
        entry.server = AgentifyMCPServer(tool_list)

    def update_tool_timeout(self, tool_timeout: float | None) -> None:
        self._entry(_DEFAULT_SESSION).tool_timeout = tool_timeout

    def update_tool_executor(self, tool_executor: ToolExecutor | None) -> None:
        self._entry(_DEFAULT_SESSION).tool_executor = tool_executor

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Codex configuration
    # ------------------------------------------------------------------

    def codex_config(self, session_id: str | None = None) -> dict[str, Any]:
        if not self.is_running or self._port is None:
            raise RuntimeError("Runtime MCP bridge is not running.")
        entry = self._entry(session_id)
        args = [
            "-m",
            "agentify.mcp.runtime_server",
            "--host",
            self._host,
            "--port",
            str(self._port),
            "--token",
            self._token,
            "--call-timeout",
            str(self.call_timeout(session_id)),
        ]
        if session_id is not None:
            args += ["--session", session_id]
        return {
            "mcp_servers": {
                self.name: {
                    "command": sys.executable,
                    "args": args,
                    "enabled_tools": list(entry.tool_names),
                }
            }
        }

    def call_timeout(self, session_id: str | None = None) -> float:
        """Socket read timeout for proxied tool calls.

        Must exceed ``tool_timeout`` so the bridge's own timeout response
        (a structured MCP error) reaches the proxy instead of the socket
        read aborting first.
        """
        tool_timeout = self._entry(session_id).tool_timeout
        if tool_timeout is not None and tool_timeout > 0:
            return float(tool_timeout) + 30.0
        return 600.0

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

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

        session_key = str(request.get("session") or _DEFAULT_SESSION)
        entry = self._sessions.get(session_key)
        if entry is None:
            return {
                "ok": False,
                "error": f"Unknown runtime MCP bridge session: {session_key}",
            }
        entry.connection_count += 1

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
                    for tool in entry.server.list_tools()
                ],
            }

        if action == "call_tool":
            name = str(request.get("name") or "")
            arguments = request.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            tool_call = self._call_tool(entry, name, arguments)
            if entry.tool_timeout is not None and entry.tool_timeout > 0:
                try:
                    result = await asyncio.wait_for(
                        tool_call, timeout=float(entry.tool_timeout)
                    )
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

    async def _call_tool(
        self,
        entry: _SessionEntry,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        if entry.tool_executor is None:
            return await entry.server.call_tool(name, arguments)

        result_text = await entry.tool_executor(name, arguments)
        from mcp.types import CallToolResult, TextContent

        return CallToolResult(content=[TextContent(type="text", text=result_text)])
