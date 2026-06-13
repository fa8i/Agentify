"""Convenience constructors layered on top of :class:`BaseAgent`.

Fully backward compatible: the explicit ``BaseAgent`` / ``AgentConfig`` /
``MemoryService`` API is unchanged. ``Agent`` just wires sensible defaults so the
common case is a one-liner::

    from agentify import Agent

    agent = Agent("You are a helpful assistant.", model="gpt-4.1")
    print(agent.run("Hello!"))
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional

from agentify.core.agent import BaseAgent
from agentify.core.config import AgentConfig, ImageConfig
from agentify.core.tool import Tool
from agentify.llm.client import LLMClientFactory
from agentify.memory.interfaces import MemoryAddress
from agentify.memory.service import MemoryService


class Agent(BaseAgent):
    """``BaseAgent`` with batteries-included defaults.

    Only ``model`` is required. An in-process store and a default conversation
    address are created when ``memory``/``memory_address`` are omitted.
    ``provider`` defaults to ``"openai"``; pass it explicitly for others. Extra
    keyword arguments are forwarded to :class:`AgentConfig`.
    """

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        *,
        model: str,
        provider: str = "openai",
        name: str = "agent",
        tools: Optional[List[Tool]] = None,
        memory: Optional[MemoryService] = None,
        memory_address: Optional[MemoryAddress] = None,
        conversation_id: Optional[str] = None,
        temperature: float = 1.0,
        stream: bool = False,
        verbose: bool = False,
        image_config: Optional[ImageConfig] = None,
        pre_hooks: Optional[List[Callable]] = None,
        post_hooks: Optional[List[Callable]] = None,
        tool_pre_hooks: Optional[List[Callable]] = None,
        tool_post_hooks: Optional[List[Callable]] = None,
        client_factory: Optional[LLMClientFactory] = None,
        **config_kwargs: Any,
    ) -> None:
        if memory is None:
            # Lazy import keeps optional backends (redis/elastic) off the import path.
            from agentify.memory.stores import InMemoryStore

            memory = MemoryService(store=InMemoryStore(), log_enabled=verbose)

        if memory_address is None:
            memory_address = MemoryAddress(
                conversation_id=conversation_id or "default",
                agent_id=name,
            )

        config = AgentConfig(
            name=name,
            system_prompt=system_prompt,
            provider=provider,
            model_name=model,
            temperature=temperature,
            stream=stream,
            verbose=verbose,
            **config_kwargs,
        )

        super().__init__(
            config=config,
            memory=memory,
            memory_address=memory_address,
            client_factory=client_factory,
            tools=tools,
            image_config=image_config,
            pre_hooks=pre_hooks,
            post_hooks=post_hooks,
            tool_pre_hooks=tool_pre_hooks,
            tool_post_hooks=tool_post_hooks,
        )
