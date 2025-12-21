"""Adapter to convert MCP tools into Agentify Tool objects."""
from typing import Any, Callable, List
from mcp import ClientSession
from agentify.core.tool import Tool


async def convert_mcp_tools_to_agentify(
    session: ClientSession,
    mcp_tools: List[Any],
) -> List[Tool]:
    """Transforms MCP tools into Agentify-compatible Tool objects."""
    agentify_tools: List[Tool] = []

    for m_tool in mcp_tools:
        schema = {
            "name": m_tool.name,
            "description": m_tool.description or "",
            "parameters": m_tool.inputSchema,
        }
        wrapper = _create_tool_wrapper(session, m_tool.name)
        agentify_tools.append(Tool(schema=schema, func=wrapper))

    return agentify_tools


def _create_tool_wrapper(session: ClientSession, tool_name: str) -> Callable[..., Any]:
    """Creates an async wrapper function that calls the MCP server."""

    async def _mcp_tool_wrapper(**kwargs: Any) -> Any:
        result = await session.call_tool(tool_name, arguments=kwargs)

        output_parts = []
        if result.content:
            for item in result.content:
                if item.type == "text":
                    output_parts.append(item.text)
                elif item.type == "image":
                    output_parts.append(f"[Image: {item.mimeType}]")
                elif item.type == "resource":
                    output_parts.append(f"[Resource: {item.resource.uri}]")

        return "\n".join(output_parts)

    _mcp_tool_wrapper.__name__ = tool_name
    _mcp_tool_wrapper.__doc__ = f"MCP Tool: {tool_name}"
    return _mcp_tool_wrapper
