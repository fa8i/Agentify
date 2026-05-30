"""Adapter to convert MCP tools into Agentify Tool objects."""

import re
from typing import Any, Awaitable, Callable, Sequence

from mcp import ClientSession
from agentify.core.tool import Tool


def convert_mcp_tools_to_agentify(
    session: ClientSession,
    mcp_tools: Sequence[Any],
) -> list[Tool]:
    """Transforms MCP tools into Agentify-compatible Tool objects."""
    agentify_tools: list[Tool] = []

    for m_tool in mcp_tools:
        schema = {
            "name": m_tool.name,
            "description": m_tool.description or "",
            "parameters": m_tool.inputSchema or {
                "type": "object",
                "properties": {},
            },
        }

        wrapper = _create_tool_wrapper(session, m_tool.name)
        agentify_tools.append(Tool(schema=schema, func=wrapper))

    return agentify_tools


def _create_tool_wrapper(
    session: ClientSession,
    tool_name: str,
) -> Callable[..., Awaitable[str]]:
    """Creates an async wrapper function that calls the MCP server."""

    async def _mcp_tool_wrapper(**kwargs: Any) -> str:
        try:
            result = await session.call_tool(tool_name, arguments=kwargs)
        except Exception as exc:
            raise RuntimeError(f"Error calling MCP tool '{tool_name}'") from exc

        output_parts: list[str] = []

        for item in result.content or []:
            if item.type == "text":
                output_parts.append(item.text)
            elif item.type == "image":
                output_parts.append(f"[Image: {getattr(item, 'mimeType', 'unknown')}]")
            elif item.type == "resource":
                uri = getattr(getattr(item, "resource", None), "uri", "unknown")
                output_parts.append(f"[Resource: {uri}]")
            else:
                output_parts.append(f"[Unsupported MCP content type: {item.type}]")

        return "\n".join(output_parts)

    _mcp_tool_wrapper.__name__ = _safe_function_name(tool_name)
    _mcp_tool_wrapper.__doc__ = f"MCP Tool: {tool_name}"

    return _mcp_tool_wrapper


def _safe_function_name(name: str) -> str:
    """Converts an MCP tool name into a safer Python function name."""
    safe_name = re.sub(r"\W+", "_", name).strip("_")

    if not safe_name:
        return "mcp_tool"

    if safe_name[0].isdigit():
        return f"tool_{safe_name}"

    return safe_name