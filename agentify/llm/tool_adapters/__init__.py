from agentify.llm.tool_adapters.codex_mcp import CodexMCPToolAdapter

__all__ = [
    "CodexMCPToolAdapter",
    "native_tools_are_adaptable",
    "prepare_native_tool_params",
]


def native_tools_are_adaptable(*, async_client, has_tools: bool) -> bool:
    """Return whether a native backend can adapt Agentify tools internally."""
    return (
        bool(has_tools)
        and getattr(async_client, "is_native_thread_backend", False) is True
        and getattr(async_client, "supports_tools", True) is False
        and getattr(async_client, "supports_mcp_tools", False) is True
    )


def prepare_native_tool_params(
    *,
    async_client,
    agent_name,
    provider,
    tool_timeout,
    max_tool_iter,
    callbacks,
    execute_tool,
    add_memory,
    tools,
    addr,
):
    """Return provider-specific params for native backends that adapt tools."""
    if native_tools_are_adaptable(async_client=async_client, has_tools=bool(tools)):
        return CodexMCPToolAdapter(
            agent_name=agent_name,
            provider=provider,
            tool_timeout=tool_timeout,
            max_tool_iter=max_tool_iter,
            callbacks=callbacks,
            execute_tool=execute_tool,
            add_memory=add_memory,
        ).llm_params(tools, addr=addr)
    return {}
