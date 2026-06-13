import asyncio
import contextvars
import json
import logging
import uuid
import base64
import inspect
from typing import Any, Dict, Generator, List, Optional, Union, Callable, AsyncGenerator

from agentify.core.runnable import Runnable

try:
    from openai import RateLimitError
except ModuleNotFoundError:  # pragma: no cover - optional provider dependency
    class RateLimitError(Exception):
        pass
from jsonschema import validate, ValidationError

from agentify.core.tool import Tool
from agentify.llm.client import LLMClientFactory, LLMClientType, AsyncLLMClientType
from agentify.memory.service import MemoryService
from agentify.memory.async_service import AsyncMemoryService
from agentify.memory.interfaces import MemoryAddress
from agentify.core.config import AgentConfig, ImageConfig
from agentify.core.callbacks import LoggingCallbackHandler
from agentify.core.multimodal import build_user_content
from agentify.core.sync_bridge import run_coro_blocking, stream_async_to_sync
from agentify.llm.tool_adapters import native_tools_are_adaptable, prepare_native_tool_params

logger = logging.getLogger(__name__)


class BaseAgent(Runnable):
    """Agent runtime with sync/async public API.

    `arun()` is the source of truth for execution. `run()` bridges to it for
    synchronous callers via `sync_bridge`.
    """

    def __init__(
        self,
        config: AgentConfig,
        memory: MemoryService,
        *,
        memory_address: Optional[MemoryAddress] = None,
        client_factory: Optional[LLMClientFactory] = None,
        tools: Optional[List[Tool]] = None,
        image_config: Optional[ImageConfig] = None,
        pre_hooks: Optional[List[Callable]] = None,
        post_hooks: Optional[List[Callable]] = None,
        tool_pre_hooks: Optional[List[Callable]] = None,
        tool_post_hooks: Optional[List[Callable]] = None,
    ) -> None:
        self.config = config
        self.memory = memory
        self.memory_address = memory_address
        self.image_config = image_config or ImageConfig()
        self.pre_hooks = pre_hooks or []
        self.post_hooks = post_hooks or []
        self.tool_pre_hooks = tool_pre_hooks or []
        self.tool_post_hooks = tool_post_hooks or []

        # Decouple callbacks from config to avoid mutation of shared config
        self.callbacks = list(self.config.callbacks) if self.config.callbacks else []
        if not self.callbacks and self.config.verbose:
            self.callbacks.append(LoggingCallbackHandler(logger))

        self._tools: Dict[str, Tool] = {t.name: t for t in tools or []}

        self._factory = client_factory or LLMClientFactory()
        self.client: LLMClientType = self._factory.create_client(
            provider=self.config.provider,
            config_override=self.config.client_config_override,
            timeout=self.config.timeout,
        )
        # Async client is created lazily on first arun() call
        self._async_client: Optional[AsyncLLMClientType] = None
        
        # Auto-wrap memory for async operations (transparent to user)
        self._async_memory = AsyncMemoryService.from_sync(memory)

    @property
    def tool_defs(self) -> List[Dict[str, Any]]:
        """Dynamically generate tool definitions for the LLM."""
        return [
            {"type": "function", "function": t.schema} for t in self._tools.values()
        ]

    @property
    def list_tools(self) -> List[str]:
        """Return the names of registered tools."""
        return list(self._tools.keys())

    def _build_user_content(
        self,
        user_input: str,
        *,
        image_path: Optional[str] = None,
        image_detail_override: Optional[str] = None,
    ) -> Optional[Union[str, List[Dict[str, Any]]]]:
        """Build user message content for text-only or multimodal inputs."""
        return build_user_content(
            user_input,
            image_config=self.image_config,
            image_path=image_path,
            image_detail_override=image_detail_override,
        )

    # Memory helpers

    def _addr_or_raise(self, addr: Optional[MemoryAddress]) -> MemoryAddress:
        """Ensure we have a MemoryAddress to operate on."""
        effective = addr or self.memory_address
        if effective is None:
            raise ValueError(
                "MemoryAddress required: pass it in constructor (memory_address=...) "
                "or in each call (addr=...)."
            )
        return effective

    def get_history(self, addr: MemoryAddress) -> List[Dict[str, Any]]:
        """Return current conversation history for this address."""
        return self.memory.get_history(addr)

    def add(
        self,
        role: str,
        content: Optional[Union[str, List[Dict[str, Any]]]] = None,
        *,
        addr: Optional[MemoryAddress] = None,
        **kwargs: Any,
    ) -> None:
        """Append a message to memory at the provided address."""
        a = self._addr_or_raise(addr)
        msg: Dict[str, Any] = {"role": role}
        if content is not None:
            msg["content"] = content
        msg.update(kwargs)
        self.memory.append_history(a, msg)

    def clear_memory(self, *, addr: Optional[MemoryAddress] = None) -> None:
        """Reset history for the provided address to the initial system prompt only."""
        a = self._addr_or_raise(addr)
        self.memory.reset_history(
            a, {"role": "system", "content": self.config.system_prompt}
        )

    # Async memory helpers (for async loop)
    
    async def _aensure_system_initialized(self, addr: MemoryAddress) -> None:
        """Async version: ensure system message is present and correct (non-blocking)."""
        history = await self._async_memory.get_history(addr)
        
        if not history:
            await self._async_memory.append_history(
                addr, {"role": "system", "content": self.config.system_prompt}
            )
            return

        is_first_correct = (
            history[0].get("role") == "system"
            and history[0].get("content") == self.config.system_prompt
        )
        has_duplicates = sum(
            1 for m in history 
            if m.get("role") == "system" and m.get("content") == self.config.system_prompt
        ) > 1

        if not is_first_correct or has_duplicates:
            new_history = [{"role": "system", "content": self.config.system_prompt}]
            for m in history:
                if m.get("role") == "system" and m.get("content") == self.config.system_prompt:
                    continue
                if m is history[0] and m.get("role") == "system":
                    continue
                new_history.append(m)
            
            await self._async_memory.replace_history(addr, new_history)

    async def _arollback_last_tool_turn(self, addr: MemoryAddress, tool_message_count: int) -> None:
        """Rollback assistant/tool messages from the last tool iteration.

        This is used when execution is interrupted (for example permission challenge)
        so the next retry does not inherit partial tool-call state.
        """
        if tool_message_count <= 0:
            return
        history = await self._async_memory.get_history(addr)
        remove_count = 1 + tool_message_count  # assistant tool_call message + tool messages
        if len(history) <= 1 or len(history) <= remove_count:
            return

        kept = history[:-remove_count]
        if not kept:
            return

        first = kept[0]
        if first.get("role") != "system":
            first = {"role": "system", "content": self.config.system_prompt}
            kept = [first] + kept

        await self._async_memory.replace_history(addr, kept)

    async def _aadd(
        self,
        role: str,
        content: Optional[Union[str, List[Dict[str, Any]]]] = None,
        *,
        addr: Optional[MemoryAddress] = None,
        **kwargs: Any,
    ) -> None:
        """Async version: append a message to memory (non-blocking)."""
        a = self._addr_or_raise(addr)
        msg: Dict[str, Any] = {"role": role}
        if content is not None:
            msg["content"] = content
        msg.update(kwargs)
        await self._async_memory.append_history(a, msg)

    def save_history(self, path: str, *, addr: Optional[MemoryAddress] = None) -> None:
        """Persist current history to a local JSON file."""
        a = self._addr_or_raise(addr)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.memory.get_history(a), f, ensure_ascii=False, indent=2)

    def load_history(self, path: str, *, addr: Optional[MemoryAddress] = None) -> None:
        """Load a previously exported JSON history into this address."""
        a = self._addr_or_raise(addr)
        with open(path, "r", encoding="utf-8") as f:
            raw: List[Dict[str, Any]] = json.load(f)

        if raw and raw[0].get("role") == "system":
            messages = raw
        else:
            messages = [{"role": "system", "content": self.config.system_prompt}] + raw

        self.memory.reset_history(a, messages[0])
        for m in messages[1:]:
            self.memory.append_history(a, m)

    # Hook Execution
    def _execute_hook(self, hook: Callable, **kwargs: Any) -> None:
        """Execute a hook injecting only the arguments it declares."""
        try:
            sig = inspect.signature(hook)
            # Filter kwargs to only those present in the hook's signature
            # If the hook accepts **kwargs, pass everything
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            
            if has_var_keyword:
                hook_kwargs = kwargs
            else:
                hook_kwargs = {
                    k: v for k, v in kwargs.items() if k in sig.parameters
                }
            
            hook(**hook_kwargs)
        except Exception as e:
            logger.error(f"Error executing hook '{hook.__name__}': {e}", exc_info=True)

    # Core Logic

    def _split_concatenated_json_objects(self, json_string: str) -> List[str]:
        """Attempt to split a string that may contain multiple concatenated JSON objects."""
        objects_str: List[str] = []
        decoder = json.JSONDecoder()
        s = json_string.strip()
        pos = 0

        if not s:
            return []
        try:
            json.loads(s)
            return [s]  # single valid JSON
        except json.JSONDecodeError:
            pass

        while pos < len(s):
            try:
                _, consumed = decoder.raw_decode(s[pos:])
                objects_str.append(s[pos : pos + consumed])
                pos += consumed
                while pos < len(s) and s[pos].isspace():
                    pos += 1
            except json.JSONDecodeError:
                if not objects_str:
                    logger.warning(f"Could not decode JSON from: '{json_string}'")
                    return [json_string]
                logger.warning(
                    f"Agent '{self.config.name}': could not decode more JSON at pos {pos} "
                    f"of '{s}'. Parsed objects: {len(objects_str)}."
                )
                break
        return objects_str if objects_str else [json_string]

    def _parse_tool_arguments(self, tool_name: str, args_value: Any) -> Dict[str, Any]:
        """Safely parse tool arguments from various formats."""
        if args_value is None:
            args_str = "{}"
        elif isinstance(args_value, str):
            args_str = args_value
        else:
            args_str = json.dumps(args_value)

        if not args_str.strip():
            args_str = "{}"

        try:
            return json.loads(args_str)
        except json.JSONDecodeError as exc:
            logger.warning(
                f"Invalid JSON arguments for '{tool_name}': {exc}. Received: '{args_str}'"
            )
            raise ValueError(f"Invalid JSON arguments: {exc}")

    def _validate_tool_arguments(self, tool: Tool, arguments: Dict[str, Any]) -> None:
        """Validate tool arguments against the tool's JSON schema."""
        if not isinstance(arguments, dict):
            raise ValueError(f"Tool '{tool.name}' arguments must be a JSON object.")

        params_schema = tool.schema.get("parameters") or {"type": "object"}
        if "type" not in params_schema:
            params_schema = {"type": "object", **params_schema}

        try:
            validate(instance=arguments, schema=params_schema)
        except ValidationError as exc:
            raise ValueError(
                f"Tool '{tool.name}' arguments failed schema validation: {exc.message}"
            ) from exc

    def _serialize_tool_result(self, result: Any) -> str:
        """Normalize tool results to a JSON string when possible."""
        if isinstance(result, bytes):
            try:
                return result.decode("utf-8")
            except UnicodeDecodeError:
                return base64.b64encode(result).decode("utf-8")

        if isinstance(result, (dict, list)):
            try:
                return json.dumps(result, ensure_ascii=False)
            except TypeError:
                return json.dumps({"result": str(result)}, ensure_ascii=False)

        return str(result)

    @staticmethod
    def _is_tool_call_consistency_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "tool_calls" in msg
            and "tool messages" in msg
            and "tool_call_id" in msg
        )

    def _process_sync_response(
        self, msg_object: Any
    ) -> tuple[Optional[str], List[Dict[str, Any]], Optional[str]]:
        """Process synchronous response, returning content, tool calls, and reasoning content."""
        content = getattr(msg_object, "content", None)
        
        # Handle reasoning content if present
        reasoning_content = getattr(msg_object, "reasoning_content", None) or None
        if reasoning_content:
            for cb in self.callbacks:
                cb.on_reasoning_step(reasoning_content)

        tool_calls = []

        if getattr(msg_object, "tool_calls", None):
            for i, tc in enumerate(msg_object.tool_calls):
                tc_id = (
                    tc.id
                    or f"ns_{self.config.provider[:3]}_tc_{i}_{uuid.uuid4().hex[:6]}"
                )
                tool_calls.append(
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                )

        return content, tool_calls, reasoning_content

    def _expand_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Handle cases where the model concatenates multiple JSON objects in one argument string."""
        expanded_tool_calls: List[Dict[str, Any]] = []

        for tc in tool_calls:
            tool_name = tc.get("function", {}).get("name", "unknown_tool")
            args_value = tc.get("function", {}).get("arguments")
            original_id = tc.get("id", f"gen_id_{uuid.uuid4().hex[:4]}")

            if args_value is None:
                args_str = ""
            elif isinstance(args_value, str):
                args_str = args_value
            else:
                args_str = json.dumps(args_value)

            if not args_str.strip():
                tc["function"]["arguments"] = "{}"
                expanded_tool_calls.append(tc)
                continue

            split_args_json = self._split_concatenated_json_objects(args_str)

            if len(split_args_json) > 1:
                for i, single_arg_json in enumerate(split_args_json):
                    expanded_tool_calls.append(
                        {
                            "id": f"{original_id}_part_{i}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": single_arg_json,
                            },
                        }
                    )
            elif len(split_args_json) == 1:
                tc["function"]["arguments"] = split_args_json[0]
                expanded_tool_calls.append(tc)
            else:
                # Fallback
                expanded_tool_calls.append(tc)

        return expanded_tool_calls

    # Public entrypoint

    def run(
        self,
        user_input: str,
        *,
        addr: Optional[MemoryAddress] = None,
        image_path: Optional[str] = None,
        image_detail_override: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[str, Generator[str, None, None]]:
        """Synchronous wrapper around `arun()`."""
        if self.config.stream:
            return stream_async_to_sync(
                lambda: self.arun(
                    user_input,
                    addr=addr,
                    image_path=image_path,
                    image_detail_override=image_detail_override,
                    **kwargs,
                ),
                api_name="run",
                async_api_name="arun",
            )

        return run_coro_blocking(
            self.arun(
                user_input,
                addr=addr,
                image_path=image_path,
                image_detail_override=image_detail_override,
                **kwargs,
            ),
            api_name="run",
            async_api_name="arun",
        )

    async def arun(
        self,
        user_input: str,
        *,
        addr: Optional[MemoryAddress] = None,
        image_path: Optional[str] = None,
        image_detail_override: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Asynchronous execution entrypoint (source of truth)."""
        # If addr is not provided, try to get it from kwargs (Protocol compatibility)
        if addr is None and "memory_address" in kwargs:
            addr = kwargs["memory_address"]

        a = self._addr_or_raise(addr)
        response_generator = self._aexecute_agent_loop(
            user_input,
            addr=a,
            image_path=image_path,
            image_detail_override=image_detail_override,
            input_role=str(kwargs.get("input_role", "user")),
        )

        if self.config.stream:
            return response_generator

        parts: List[str] = []
        async for chunk in response_generator:
            parts.append(chunk)
        return "".join(parts).strip()

    # -------------------------------------------------------------------------
    # Async methods
    # -------------------------------------------------------------------------

    def _get_async_client(self) -> AsyncLLMClientType:
        """Lazily create and return the async client."""
        if self._async_client is None:
            self._async_client = self._factory.create_async_client(
                provider=self.config.provider,
                config_override=self.config.client_config_override,
                timeout=self.config.timeout,
            )
        return self._async_client

    async def aclose(self) -> None:
        """Close async provider resources owned by this agent."""
        if self._async_client is None:
            return
        close = getattr(self._async_client, "close", None)
        if close is not None:
            await close()
        self._async_client = None

    def close(self) -> None:
        """Synchronously close async provider resources owned by this agent."""
        run_coro_blocking(self.aclose())

    async def _aget_llm_response(
        self, *, addr: MemoryAddress
    ) -> Union[Any, AsyncGenerator[Dict[str, Any], None]]:
        """Perform the async LLM call with retries and error handling."""
        async_client = self._get_async_client()
        tool_choice_param = "auto" if self._tools else None
        
        # Use async memory to avoid blocking the event loop
        messages = await self._async_memory.get_history(addr)
        
        common_params: Dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
        }

        if self.config.reasoning_effort:
            common_params["reasoning_effort"] = self.config.reasoning_effort

        if self.config.model_kwargs:
            for k, v in self.config.model_kwargs.items():
                if k not in common_params:
                    common_params[k] = v

        # Only add tools if they exist
        tools_payload = self.tool_defs
        is_native_backend = getattr(async_client, "is_native_thread_backend", False) is True
        native_tools_adapted = native_tools_are_adaptable(
            async_client=async_client,
            has_tools=bool(tools_payload),
        )

        if tools_payload and not native_tools_adapted:
            common_params["tools"] = tools_payload
            common_params["tool_choice"] = tool_choice_param

        if native_tools_adapted:
            common_params.update(
                prepare_native_tool_params(
                    async_client=async_client,
                    agent_name=self.config.name,
                    provider=self.config.provider,
                    tool_timeout=self.config.tool_timeout,
                    max_tool_iter=self.config.max_tool_iter,
                    callbacks=self.callbacks,
                    execute_tool=self._aexecute_tool,
                    add_memory=self._aadd,
                    tools=list(self._tools.values()),
                    addr=addr,
                )
            )

        if is_native_backend and tools_payload and not native_tools_adapted and getattr(async_client, "supports_tools", True) is False:
            raise NotImplementedError(
                "Agentify's classic tool loop is not supported by the native Codex provider. "
                "Use provider='openai' for OpenAI-style tool_calls, or expose tools to Codex "
                "through Agentify MCP stdio."
            )

        if is_native_backend and self.config.stream and getattr(async_client, "supports_streaming", True) is False:
            raise NotImplementedError(
                "Streaming is not supported by the native Codex provider. "
                "Agentify reconstructs a single final response from Codex turn events."
            )

        for cb in self.callbacks:
            cb.on_llm_start(self.config.model_name, common_params["messages"])

        for attempt in range(self.config.max_retries):
            try:
                if is_native_backend:
                    last_content = common_params["messages"][-1]["content"] if common_params["messages"] else ""
                    if isinstance(last_content, list):
                        prompt = " ".join([c.get("text", "") for c in last_content if c.get("type") == "text"])
                    else:
                        prompt = last_content
                    # Stable per-conversation key so native backends (Codex)
                    # can map sessions to provider threads deterministically.
                    key_str = getattr(addr, "key_str", None)
                    session_id_str = key_str() if callable(key_str) else str(addr)
                    
                    if self.config.stream:
                        return await async_client.run_native(
                            session_id=session_id_str,
                            prompt=prompt,
                            stream=True,
                            **common_params
                        )
                    response = await async_client.run_native(
                        session_id=session_id_str,
                        prompt=prompt,
                        stream=False,
                        **common_params
                    )
                else:
                    if self.config.stream:
                        return await async_client.chat.completions.create(
                            **common_params, stream=True
                        )
                    response = await async_client.chat.completions.create(
                        **common_params, stream=False
                    )

                for cb in self.callbacks:
                    cb.on_llm_end(response)

                if response.choices and len(response.choices) > 0:
                    return response.choices[0].message
                raise ValueError("API response did not contain valid 'choices'.")
            except Exception as e:
                # Unify error handling
                for cb in self.callbacks:
                    cb.on_error(e, f"_aget_llm_response attempt {attempt + 1}")

                if getattr(e, "non_retryable_provider_error", False):
                    raise

                if isinstance(e, RateLimitError):
                    if attempt == self.config.max_retries - 1:
                        logger.error("API Rate Limit reached after retries.")
                        raise
                    sleep_time = 2**attempt
                    logger.warning(f"Rate limit reached. Retrying in {sleep_time}s...")
                    await asyncio.sleep(sleep_time)
                    continue

                # For other transient errors (timeouts, connection issues), log warning instead of full error trace
                if attempt < self.config.max_retries - 1:
                    if self._is_tool_call_consistency_error(e):
                        logger.debug(
                            "Consistency error in _aget_llm_response (attempt %s/%s). Retrying.",
                            attempt + 1,
                            self.config.max_retries,
                        )
                    else:
                        logger.warning(
                            f"Transient error in _aget_llm_response (attempt {attempt + 1}/{self.config.max_retries}): {e}. Retrying..."
                        )
                    await asyncio.sleep(2**attempt)
                else:
                    # Final attempt failed, log full error
                    if self._is_tool_call_consistency_error(e):
                        logger.debug(
                            "Consistency error in _aget_llm_response "
                            "(attempt %s/%s): %s",
                            attempt + 1,
                            self.config.max_retries,
                            e,
                        )
                    else:
                        logger.error(
                            f"Error in _aget_llm_response (attempt {attempt + 1}/{self.config.max_retries}): {e}",
                            exc_info=True,
                        )
                    raise

        msg = f"LLM completions ({async_client.__class__.__name__}) failed after {self.config.max_retries} retries."
        logger.critical(msg)
        raise RuntimeError(msg)

    async def _aexecute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a single tool asynchronously and return its output as a string."""
        tool = self._tools.get(tool_name)

        for cb in self.callbacks:
            cb.on_tool_start(tool_name, arguments)

        if not tool:
            err_msg = json.dumps({"error": f"Tool '{tool_name}' is not registered."})
            for cb in self.callbacks:
                cb.on_tool_finish(tool_name, err_msg)
            return err_msg

        try:
            self._validate_tool_arguments(tool, arguments)
            for hook in self.tool_pre_hooks:
                self._execute_hook(
                    hook,
                    agent=self,
                    tool=tool,
                    tool_name=tool_name,
                    arguments=arguments,
                )
            # Check for async_func attribute (used by AgentTool, FlowTool, SpawnAgentTool)
            if hasattr(tool, "async_func") and asyncio.iscoroutinefunction(tool.async_func):
                result = await tool.async_func(**arguments)
            # Check if the tool function itself is async
            elif asyncio.iscoroutinefunction(tool.func):
                result = await tool.func(**arguments)
            else:
                # Run sync function in thread pool to avoid blocking
                current_ctx = contextvars.copy_context()
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: current_ctx.run(tool, **arguments)
                )
            result_str = self._serialize_tool_result(result)
            for hook in self.tool_post_hooks:
                self._execute_hook(
                    hook,
                    agent=self,
                    tool=tool,
                    tool_name=tool_name,
                    arguments=arguments,
                    result=result_str,
                )
            for cb in self.callbacks:
                cb.on_tool_finish(tool_name, result_str)
            return result_str
        except Exception as e:
            for hook in self.tool_post_hooks:
                self._execute_hook(
                    hook,
                    agent=self,
                    tool=tool,
                    tool_name=tool_name,
                    arguments=arguments,
                    error=e,
                )
            for cb in self.callbacks:
                cb.on_error(e, f"Tool execution: {tool_name}")
            logger.error(
                f"Unexpected error executing tool '{tool_name}': {e}", exc_info=True
            )
            return json.dumps(
                {"error": f"Unexpected error executing tool '{tool_name}': {e}"}
            )

    async def _aprocess_stream_response(
        self, response_stream: Any
    ) -> AsyncGenerator[str, None]:
        """
        Process async streaming response, yielding content chunks.
        Returns tool calls via StopAsyncIteration or a final return.
        """
        tool_call_assembler: Dict[int, Dict[str, Any]] = {}
        full_content = []
        full_reasoning = []

        async for chunk in response_stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.content:
                for cb in self.callbacks:
                    cb.on_llm_new_token(delta.content)
                full_content.append(delta.content)
                yield delta.content

            # Handle reasoning content if present
            if hasattr(delta, "reasoning_content"):
                if delta.reasoning_content:
                    for cb in self.callbacks:
                        cb.on_reasoning_step(delta.reasoning_content)
                    full_reasoning.append(delta.reasoning_content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_assembler:
                        tool_call_assembler[idx] = {
                            "id": None,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    call_data = tool_call_assembler[idx]
                    if tc_delta.id and not call_data["id"]:
                        call_data["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            call_data["function"]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            call_data["function"]["arguments"] += (
                                tc_delta.function.arguments
                            )

        # Call on_llm_end with the full accumulated content
        full_response_text = "".join(full_content)
        for cb in self.callbacks:
            cb.on_llm_end(full_response_text)

        # Store results in instance for retrieval after iteration
        self._last_stream_tool_calls = []
        for idx in sorted(tool_call_assembler.keys()):
            call_data = tool_call_assembler[idx]
            if not call_data.get("id"):
                call_data["id"] = (
                    f"s_{self.config.provider[:3]}_tc_{idx}_{uuid.uuid4().hex[:6]}"
                )
            if call_data.get("function", {}).get("name"):
                self._last_stream_tool_calls.append(call_data)
        
        self._last_stream_reasoning = "".join(full_reasoning) or None

    async def _aexecute_agent_loop(
        self,
        user_input: str,
        *,
        addr: MemoryAddress,
        image_path: Optional[str] = None,
        image_detail_override: Optional[str] = None,
        input_role: str = "user",
    ) -> AsyncGenerator[str, None]:
        """Async version of the agent loop with parallel tool execution."""
        await self._aensure_system_initialized(addr)

        for cb in self.callbacks:
            cb.on_agent_start(self.config.name, user_input)

        for hook in self.pre_hooks:
            self._execute_hook(hook, agent=self, user_input=user_input)

        user_content = self._build_user_content(
            user_input,
            image_path=image_path,
            image_detail_override=image_detail_override,
        )
        if user_content is not None:
            await self._aadd(role=input_role, content=user_content, addr=addr)

        accumulated_response: List[str] = []

        iteration_count = 0
        reached_max_iter = False
        while True:
            if self.config.max_tool_iter is not None and iteration_count >= self.config.max_tool_iter:
                reached_max_iter = True
                break
            iteration_count += 1

            response_or_stream = await self._aget_llm_response(addr=addr)

            current_turn_content_parts: List[str] = []
            assembled_tool_calls: List[Dict[str, Any]] = []
            full_reasoning_content: Optional[str] = None

            if self.config.stream:
                # Process async stream
                async for content_chunk in self._aprocess_stream_response(response_or_stream):
                    yield content_chunk
                    current_turn_content_parts.append(content_chunk)
                    accumulated_response.append(content_chunk)
                # Retrieve tool calls from stream processing
                assembled_tool_calls = getattr(self, "_last_stream_tool_calls", [])
                full_reasoning_content = getattr(self, "_last_stream_reasoning", None)
            else:
                content, assembled_tool_calls, full_reasoning_content = self._process_sync_response(
                    response_or_stream
                )
                if content:
                    yield content
                    current_turn_content_parts.append(content)
                    accumulated_response.append(content)

            # Expand tool calls (fix for some models)
            assembled_tool_calls = self._expand_tool_calls(assembled_tool_calls)
            full_turn_content = "".join(current_turn_content_parts)

            # Exit if no tool calls are present
            if not assembled_tool_calls:
                msg_kwargs = {}
                if full_reasoning_content:
                    msg_kwargs["metadata"] = {"reasoning_content": full_reasoning_content}
                    if self.config.stream:
                        yield "\x1eagentify_event:" + json.dumps(
                            {
                                "type": "reasoning_full",
                                "text": full_reasoning_content,
                            },
                            ensure_ascii=False,
                        ) + "\x1e"
                
                await self._aadd(role="assistant", content=full_turn_content, addr=addr, **msg_kwargs)
                for cb in self.callbacks:
                    cb.on_agent_finish(self.config.name, full_turn_content)
                break

            # Record assistant message with tool calls
            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            if full_turn_content:
                assistant_msg["content"] = full_turn_content
            assistant_msg["tool_calls"] = assembled_tool_calls
            if full_reasoning_content:
                assistant_msg["metadata"] = {"reasoning_content": full_reasoning_content}

            tool_names = [
                str(tc.get("function", {}).get("name", "")).strip()
                for tc in assembled_tool_calls
                if str(tc.get("function", {}).get("name", "")).strip()
            ]
            for cb in self.callbacks:
                cb_func = getattr(cb, "on_assistant_tool_intent", None)
                if callable(cb_func):
                    cb_func(self.config.name, full_turn_content, tool_names)
            if self.config.stream and tool_names:
                yield "\x1eagentify_event:" + json.dumps(
                    {
                        "type": "assistant_with_tools",
                        "content": full_turn_content or "",
                        "tools": ", ".join(tool_names),
                        "reasoning": full_reasoning_content or "",
                    },
                    ensure_ascii=False,
                ) + "\x1e"
            
            await self._aadd(addr=addr, **assistant_msg)

            # Execute tools IN PARALLEL using asyncio.gather
            async def execute_single_tool(tc: Dict[str, Any]) -> tuple[str, str, str, Optional[BaseException]]:
                tool_name = tc["function"]["name"]
                tool_call_id = tc["id"]
                args_str = tc["function"]["arguments"]
                try:
                    args = self._parse_tool_arguments(tool_name, args_str)
                    # Add timeout to prevent indefinite hangs
                    result_content = await asyncio.wait_for(
                        self._aexecute_tool(tool_name, args),
                        timeout=float(self.config.tool_timeout),
                    )
                except asyncio.TimeoutError:
                    result_content = json.dumps(
                        {
                            "error": (
                                f"Tool '{tool_name}' execution timed out after "
                                f"{self.config.tool_timeout} seconds."
                            )
                        }
                    )
                    return tool_call_id, tool_name, result_content, None
                except ValueError as e:
                    result_content = json.dumps({"error": str(e)})
                    return tool_call_id, tool_name, result_content, None
                except BaseException as e:
                    scope = getattr(e, "scope", None)
                    reason = getattr(e, "reason", str(e))
                    if isinstance(scope, str) and scope:
                        result_content = json.dumps(
                            {
                                "ok": False,
                                "error": reason,
                                "scope": scope,
                                "interrupted": True,
                            }
                        )
                        return tool_call_id, tool_name, result_content, e
                    raise
                return tool_call_id, tool_name, result_content, None

            tool_results = await asyncio.gather(
                *[execute_single_tool(tc) for tc in assembled_tool_calls]
            )

            # Add tool results to memory
            interrupt_signal: Optional[BaseException] = None
            for tool_call_id, tool_name, result_content, signal in tool_results:
                await self._aadd(
                    role="tool",
                    content=result_content,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    addr=addr,
                )
                if interrupt_signal is None and signal is not None:
                    interrupt_signal = signal
            if interrupt_signal is not None:
                await self._arollback_last_tool_turn(addr, len(tool_results))
                raise interrupt_signal
        if reached_max_iter:
            warn_msg = f"\n[WARNING] Agent '{self.config.name}' reached max iterations ({self.config.max_tool_iter}).\n"
            logger.warning(warn_msg.strip())
            for cb in self.callbacks:
                cb.on_agent_finish(self.config.name, warn_msg)
            yield warn_msg
            accumulated_response.append(warn_msg)

        full_response = "".join(accumulated_response)
        for hook in self.post_hooks:
            self._execute_hook(
                hook, agent=self, user_input=user_input, response=full_response
            )

    # Tool registry management

    def tool_exists(self, name: str) -> bool:
        """Check whether a tool is registered."""
        return name in self._tools

    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool. Returns True if removed, False if missing."""
        if name not in self._tools:
            return False
        self._tools.pop(name)
        return True

    def register_tool(self, tool: Tool) -> None:
        """Register (or replace) a tool."""
        if tool.name in self._tools:
            # Check if it's the same tool object
            if self._tools[tool.name] == tool:
                return

            logger.debug(
                f"Overwriting existing tool '{tool.name}' in agent '{self.config.name}'"
            )

        self._tools[tool.name] = tool
