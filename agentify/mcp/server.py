"""Adapters and stdio server for exposing Agentify tools through MCP."""

import argparse
import asyncio
import inspect
import importlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from jsonschema import ValidationError, validate
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool as MCPTool

from agentify.core.tool import Tool


class AgentifyMCPServer:
    """In-process MCP adapter exposing a scoped set of Agentify tools.

    This class is transport-agnostic on purpose. It provides the MCP-facing
    list/call operations that a stdio or HTTP MCP transport can wire into later.
    """

    def __init__(
        self,
        tools: Sequence[Tool],
        *,
        allowlist: Iterable[str] | None = None,
        debug_log: str | Path | None = None,
    ) -> None:
        allowed_names = set(allowlist) if allowlist is not None else None
        self._tools: dict[str, Tool] = {}
        self._debug_log = Path(debug_log) if debug_log else None

        for tool in tools:
            if allowed_names is not None and tool.name not in allowed_names:
                continue
            self._tools[tool.name] = tool
        self._log_debug("tools loaded: " + ",".join(self._tools) if self._tools else "tools loaded: none")

    def list_tools(self) -> list[MCPTool]:
        """Return MCP tool definitions for the exposed Agentify tools."""
        self._log_debug("list_tools called")
        return [agentify_tool_to_mcp_tool(tool) for tool in self._tools.values()]

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> CallToolResult:
        """Execute an exposed Agentify tool and return an MCP call result."""
        self._log_debug(f"call_tool called: {name}")
        tool = self._tools.get(name)
        if tool is None:
            self._log_debug(f"call_tool rejected missing tool: {name}")
            return _error_result(f"Tool '{name}' is not exposed by this Agentify MCP server.")

        args = dict(arguments or {})
        try:
            _validate_tool_arguments(tool, args)
            result = await _call_agentify_tool(tool, args)
            self._log_debug(f"call_tool finished: {name}")
            return _text_result(_serialize_result(result))
        except Exception as exc:
            self._log_debug(f"call_tool error: {name}: {exc}")
            return _error_result(str(exc))

    def _log_debug(self, message: str) -> None:
        if self._debug_log is None:
            return
        self._debug_log.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._debug_log.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {message}\n")


def agentify_tool_to_mcp_tool(tool: Tool) -> MCPTool:
    """Convert an Agentify Tool schema into an MCP Tool definition."""
    return MCPTool(
        name=tool.name,
        description=tool.schema.get("description") or "",
        inputSchema=_tool_input_schema(tool),
    )


def load_tool_registry(registry: str) -> list[Tool]:
    """Load Agentify tools from an importable ``module.path:function`` registry."""
    if ":" not in registry:
        raise ValueError("Registry must use 'module.path:function_name' format.")

    module_path, function_name = registry.split(":", 1)
    if not module_path or not function_name:
        raise ValueError("Registry must use 'module.path:function_name' format.")

    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        raise ValueError(f"Could not import registry module '{module_path}'.") from exc

    factory = getattr(module, function_name, None)
    if not callable(factory):
        raise ValueError(f"Registry function '{registry}' was not found or is not callable.")

    tools = factory()
    if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence):
        raise ValueError("Registry function must return a sequence of Agentify Tool objects.")

    invalid_items = [item for item in tools if not isinstance(item, Tool)]
    if invalid_items:
        raise ValueError("Registry function must return only Agentify Tool objects.")

    return list(tools)


def parse_allowlist(value: str | None) -> list[str] | None:
    """Parse comma-separated tool names for CLI allowlists."""
    if not value:
        return None
    names = [item.strip() for item in value.split(",") if item.strip()]
    return names or None


def build_mcp_stdio_server(agentify_server: AgentifyMCPServer) -> Server:
    """Build a low-level MCP Server wired to an AgentifyMCPServer adapter."""
    server = Server(
        "agentify-mcp-server",
        instructions=(
            "Agentify MCP server exposing the configured Agentify tools. "
            "Only use tools that are relevant to the current user request."
        ),
    )

    @server.list_tools()
    async def _list_tools() -> list[MCPTool]:
        return agentify_server.list_tools()

    @server.call_tool(validate_input=True)
    async def _call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
        return await agentify_server.call_tool(name, arguments)

    return server


async def run_stdio_server(agentify_server: AgentifyMCPServer) -> None:
    """Run the Agentify MCP adapter over MCP stdio transport."""
    server = build_mcp_stdio_server(agentify_server)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(NotificationOptions()),
        )


def generate_codex_mcp_config(
    *,
    name: str,
    registry: str,
    allow: Sequence[str] | None = None,
    command: str = "python",
) -> str:
    """Generate a Codex-compatible TOML MCP server block for Agentify tools."""
    args = ["-m", "agentify.mcp.server", "--registry", registry]
    if allow:
        args.extend(["--allow", ",".join(allow)])

    table_key = name if re.fullmatch(r"[A-Za-z0-9_-]+", name) else json.dumps(name)
    lines = [
        f"[mcp_servers.{table_key}]",
        f"command = {json.dumps(command)}",
        "args = [",
    ]
    lines.extend(f"  {json.dumps(arg)}," for arg in args)
    lines.append("]")
    if allow:
        lines.append("enabled_tools = [" + ", ".join(json.dumps(tool) for tool in allow) + "]")
    return "\n".join(lines) + "\n"


def build_server_from_registry(
    registry: str,
    allow: str | None = None,
    debug_log: str | Path | None = None,
) -> AgentifyMCPServer:
    """Load registry tools and return a scoped Agentify MCP adapter."""
    return AgentifyMCPServer(
        load_tool_registry(registry),
        allowlist=parse_allowlist(allow),
        debug_log=debug_log,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Agentify tools as an MCP stdio server.")
    parser.add_argument("--registry", required=True, help="Import path: module.path:function_name")
    parser.add_argument("--allow", help="Comma-separated tool allowlist")
    parser.add_argument("--debug-log", help="Path to append MCP server debug logs")
    args = parser.parse_args(argv)

    try:
        agentify_server = build_server_from_registry(args.registry, args.allow, args.debug_log)
    except Exception as exc:
        print(f"agentify-mcp-server: {exc}", file=sys.stderr)
        return 2

    agentify_server._log_debug("server started")

    try:
        asyncio.run(run_stdio_server(agentify_server))
    except KeyboardInterrupt:
        return 130
    return 0


def _tool_input_schema(tool: Tool) -> dict[str, Any]:
    params = tool.schema.get("parameters") or {"type": "object", "properties": {}}
    if not isinstance(params, dict):
        return {"type": "object", "properties": {}}
    if "type" not in params:
        return {"type": "object", **params}
    return params


def _validate_tool_arguments(tool: Tool, arguments: dict[str, Any]) -> None:
    try:
        validate(instance=arguments, schema=_tool_input_schema(tool))
    except ValidationError as exc:
        raise ValueError(
            f"Tool '{tool.name}' arguments failed schema validation: {exc.message}"
        ) from exc


async def _call_agentify_tool(tool: Tool, arguments: dict[str, Any]) -> Any:
    async_func = getattr(tool, "async_func", None)
    if async_func is not None and inspect.iscoroutinefunction(async_func):
        return await async_func(**arguments)

    if inspect.iscoroutinefunction(tool.func):
        return await tool.func(**arguments)

    return tool.func(**arguments)


def _serialize_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, (dict, list)):
        return json.dumps(result, ensure_ascii=False)
    return str(result)


def _text_result(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)])


def _error_result(message: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps({"error": message}, ensure_ascii=False))],
        isError=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
