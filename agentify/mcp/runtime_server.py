"""MCP stdio subprocess that proxies Codex tool calls to a runtime bridge."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from typing import Any

from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool as MCPTool


def build_runtime_proxy_server(host: str, port: int, token: str) -> Server:
    server = Server(
        "agentify-runtime-mcp-server",
        instructions=(
            "Agentify runtime MCP server exposing tools registered on the active "
            "Agentify Codex agent."
        ),
    )

    @server.list_tools()
    async def _list_tools() -> list[MCPTool]:
        response = await _send_request(host, port, token, {"action": "list_tools"})
        return [
            MCPTool(
                name=tool["name"],
                description=tool.get("description") or "",
                inputSchema=tool.get("inputSchema")
                or {"type": "object", "properties": {}},
            )
            for tool in response.get("tools", [])
        ]

    @server.call_tool(validate_input=True)
    async def _call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
        response = await _send_request(
            host,
            port,
            token,
            {"action": "call_tool", "name": name, "arguments": arguments},
        )
        content = response.get("content") or []
        return CallToolResult(
            content=[
                TextContent(
                    type=item.get("type", "text"),
                    text=str(item.get("text", "")),
                )
                for item in content
            ],
            isError=bool(response.get("isError", False)),
        )

    return server


async def run_runtime_proxy_server(host: str, port: int, token: str) -> None:
    server = build_runtime_proxy_server(host, port, token)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(NotificationOptions()),
        )


async def _send_request(
    host: str,
    port: int,
    token: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    reader, writer = await asyncio.open_connection(host, port)
    request = {"token": token, **payload}
    writer.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
    await writer.drain()
    line = await asyncio.wait_for(reader.readline(), timeout=30)
    writer.close()
    await writer.wait_closed()

    response = json.loads(line.decode("utf-8"))
    if not response.get("ok"):
        raise RuntimeError(
            str(response.get("error") or "Runtime MCP bridge request failed.")
        )
    return response


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a runtime Agentify MCP proxy server.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--token", required=True)
    args = parser.parse_args(argv)

    try:
        asyncio.run(run_runtime_proxy_server(args.host, args.port, args.token))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"agentify-runtime-mcp-server: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
