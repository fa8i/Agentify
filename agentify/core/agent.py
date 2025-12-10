from __future__ import annotations
import asyncio
import json
import logging
import time
import uuid
import base64
from io import BytesIO
import inspect
from typing import Any, Dict, Generator, List, Optional, Union, Iterator, Callable, AsyncGenerator

from PIL import Image
from openai import RateLimitError

from agentify.core.tool import Tool
from agentify.llm.client import LLMClientFactory, LLMClientType, AsyncLLMClientType
from agentify.memory.service import MemoryService
from agentify.memory.interfaces import MemoryAddress
from agentify.core.config import AgentConfig, ImageConfig
from agentify.core.callbacks import LoggingCallbackHandler

logger = logging.getLogger(__name__)


class BaseAgent:
    """Core AI Agent class based on chat completions interface.

    BaseAgent is the primary abstraction for building AI agents in Agentify. It provides
    a unified interface for interacting with various LLM providers (OpenAI, Azure, DeepSeek,
    Gemini, etc.) that implement the OpenAI SDK-compatible chat completions format.

    The agent orchestrates the interaction between users, LLMs, and registered tools, managing
    conversation history, tool execution, and model responses. It supports both synchronous
    and asynchronous execution patterns, with automatic parallel tool execution in async mode.

    Attributes:
        config (AgentConfig): Configuration object defining the agent's behavior, including
            model selection, provider, temperature, retry policies, and streaming options.
        memory (MemoryService): Service for managing conversation history. Abstracts the
            underlying storage backend (InMemoryStore, RedisStore, etc.) and handles
            memory policies like TTL and message limits.
        memory_address (Optional[MemoryAddress]): Default conversation identifier (session_id,
            user_id, agent_id). If provided during initialization, it can be omitted from
            individual `run()` or `arun()` calls.
        image_config (ImageConfig): Configuration for multimodal image processing, controlling
            resolution, compression quality, and detail level for vision-capable models.
        pre_hooks (List[Callable]): Functions executed before the agent loop starts. Hooks
            receive arguments via dependency injection based on their signature. Available
            arguments: `agent` (BaseAgent), `user_input` (str).
        post_hooks (List[Callable]): Functions executed after the agent loop completes. Hooks
            receive arguments via dependency injection. Available arguments: `agent` (BaseAgent),
            `user_input` (str), `response` (str).
        callbacks (List[AgentCallbackHandler]): Event handlers for observability. Receive
            notifications for LLM calls, tool executions, errors, and reasoning steps.
            Default: LoggingCallbackHandler if none provided.
        client (LLMClientType): Synchronous LLM client (e.g., OpenAI, AzureOpenAI) initialized
            during construction based on `config.provider`.

    Notes:
        Supports both synchronous (run) and asynchronous (arun) execution.
        Use arun for non-blocking I/O and parallel tool execution.
        
        The async LLM client is created lazily on the first arun() call to avoid
        unnecessary connections in sync-only usage.
        
        In async mode, when the LLM requests multiple independent tools in a single
        turn, they are executed concurrently using asyncio.gather.
        
        Conversation history is isolated by MemoryAddress. Each unique combination
        of (conversation_id, user_id, agent_id) maintains a separate history.
        
        Transient errors (timeouts, rate limits) are automatically retried with
        exponential backoff. Only the final failure logs a full traceback.
        
        Tools can be created by subclassing Tool or using the @tool decorator for
        automatic schema generation from function signatures.
        
        For multimodal models, pass image_path to run() or arun(). The image is
        automatically encoded to base64 and resized according to image_config.
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
    ) -> None:
        self.config = config
        self.memory = memory
        self.memory_address = memory_address
        self.image_config = image_config or ImageConfig()
        self.pre_hooks = pre_hooks or []
        self.post_hooks = post_hooks or []

        # Decouple callbacks from config to avoid mutation of shared config
        self.callbacks = list(self.config.callbacks) if self.config.callbacks else []
        if not self.callbacks:
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

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Opens an image, resizes it, compresses it, and returns it as base64."""
        try:
            with Image.open(image_path) as img_pil:
                if img_pil.mode not in ("RGB", "L"):
                    img_pil = img_pil.convert("RGB")

                max_side = self.image_config.max_side_px
                img_pil.thumbnail((max_side, max_side))

                buf = BytesIO()
                img_pil.save(
                    buf,
                    format="JPEG",
                    quality=self.image_config.quality,
                    optimize=True,
                )
                return base64.b64encode(buf.getvalue()).decode("utf-8")

        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            raise
        except Exception as e:
            logger.error(f"Image processing error for {image_path}: {e}", exc_info=True)
            raise

    def _build_user_content(
        self,
        user_input: str,
        *,
        image_path: Optional[str] = None,
        image_detail_override: Optional[str] = None,
    ) -> Optional[Union[str, List[Dict[str, Any]]]]:
        """Build the `content` field of the user message supporting:
        - text only
        - image only
        - image + text (OpenAI-like multimodal list)
        """
        has_text = bool(user_input and user_input.strip())
        has_image = bool(image_path)

        if not has_text and not has_image:
            return None

        if not has_image:
            return user_input

        b64_image_data = self._encode_image_to_base64(image_path)  # type: ignore[arg-type]
        detail_level = image_detail_override or self.image_config.detail

        parts: List[Dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_image_data}",
                    "detail": detail_level,
                },
            }
        ]
        if has_text:
            parts.append({"type": "text", "text": user_input})

        return parts

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

    def _ensure_system_initialized(self, addr: MemoryAddress) -> None:
        """Ensure the system message is present exactly once at the beginning."""
        history = self.memory.get_history(addr)
        if not history or history[0].get("role") != "system":
            self.memory.append_history(
                addr, {"role": "system", "content": self.config.system_prompt}
            )

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

    def _get_llm_response(
        self, *, addr: MemoryAddress
    ) -> Union[Any, Generator[Dict[str, Any], None, None]]:
        """Perform the LLM call with retries and error handling."""
        tool_choice_param = "auto" if self._tools else None
        common_params: Dict[str, Any] = {
            "model": self.config.model_name,
            "messages": self.memory.get_history(addr),
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
        if tools_payload:
            common_params["tools"] = tools_payload
            common_params["tool_choice"] = tool_choice_param

        for cb in self.callbacks:
            cb.on_llm_start(self.config.model_name, common_params["messages"])

        for attempt in range(self.config.max_retries):
            try:
                if self.config.stream:
                    return self.client.chat.completions.create(
                        **common_params, stream=True
                    )
                response = self.client.chat.completions.create(
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
                    cb.on_error(e, f"_get_llm_response attempt {attempt + 1}")

                if isinstance(e, RateLimitError):
                    if attempt == self.config.max_retries - 1:
                        logger.error("API Rate Limit reached after retries.")
                        raise
                    sleep_time = 2**attempt
                    logger.warning(f"Rate limit reached. Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                    continue

                # For other transient errors (timeouts, connection issues), log warning instead of full error trace
                if attempt < self.config.max_retries - 1:
                    logger.warning(
                        f"Transient error in _get_llm_response (attempt {attempt + 1}/{self.config.max_retries}): {e}. Retrying..."
                    )
                    time.sleep(2**attempt)
                else:
                    # Final attempt failed, log full error
                    logger.error(
                        f"Error in _get_llm_response (attempt {attempt + 1}/{self.config.max_retries}): {e}",
                        exc_info=True,
                    )
                    raise

        msg = f"LLM completions ({self.client.__class__.__name__}) failed after {self.config.max_retries} retries."
        logger.critical(msg)
        raise RuntimeError(msg)

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

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a single tool and return its output as a string."""
        tool = self._tools.get(tool_name)

        for cb in self.callbacks:
            cb.on_tool_start(tool_name, arguments)

        if not tool:
            err_msg = json.dumps({"error": f"Tool '{tool_name}' is not registered."})
            for cb in self.callbacks:
                cb.on_tool_finish(tool_name, err_msg)
            return err_msg

        try:
            result = tool(**arguments)
            result_str = str(result)
            for cb in self.callbacks:
                cb.on_tool_finish(tool_name, result_str)
            return result_str
        except Exception as e:
            for cb in self.callbacks:
                cb.on_error(e, f"Tool execution: {tool_name}")
            logger.error(
                f"Unexpected error executing tool '{tool_name}': {e}", exc_info=True
            )
            return json.dumps(
                {"error": f"Unexpected error executing tool '{tool_name}': {e}"}
            )

    def _process_stream_response(
        self, response_stream: Iterator[Any]
    ) -> Generator[str, None, List[Dict[str, Any]]]:
        """
        Process streaming response, yielding content and collecting tool calls.
        Returns a tuple of (assembled tool calls, full reasoning content).
        """
        tool_call_assembler: Dict[int, Dict[str, Any]] = {}

        full_content = []
        full_reasoning = []

        for chunk in response_stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.content:
                for cb in self.callbacks:
                    cb.on_llm_new_token(delta.content)
                full_content.append(delta.content)
                yield delta.content

            # Handle reasoning content if present (e.g. DeepSeek-R1)
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
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
            # Construct a minimal mock response object or just pass the text if the handler supports it.
            # The protocol says 'response: Any'.
            cb.on_llm_end(full_response_text)

        assembled_tool_calls = []
        for idx in sorted(tool_call_assembler.keys()):
            call_data = tool_call_assembler[idx]
            if not call_data.get("id"):
                call_data["id"] = (
                    f"s_{self.config.provider[:3]}_tc_{idx}_{uuid.uuid4().hex[:6]}"
                )
            if call_data.get("function", {}).get("name"):
                assembled_tool_calls.append(call_data)

        return assembled_tool_calls, "".join(full_reasoning)

    def _process_sync_response(
        self, msg_object: Any
    ) -> tuple[Optional[str], List[Dict[str, Any]], Optional[str]]:
        """Process synchronous response, returning content, tool calls, and reasoning content."""
        content = getattr(msg_object, "content", None)
        
        # Handle reasoning content if present
        reasoning_content = getattr(msg_object, "reasoning_content", None)
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

    def _execute_agent_loop(
        self,
        user_input: str,
        *,
        addr: MemoryAddress,
        image_path: Optional[str] = None,
        image_detail_override: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Orchestrates the agent logic:
        1. Prepare context (system + user message).
        2. Loop:
           - Get LLM response (stream or sync).
           - Yield content if streaming.
           - If tool calls: execute and append results.
           - If no tool calls: break.
        """
        self._ensure_system_initialized(addr)

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
            self.add(role="user", content=user_content, addr=addr)

        accumulated_response: List[str] = []

        iteration_count = 0
        while True:
            if self.config.max_tool_iter is not None and iteration_count >= self.config.max_tool_iter:
                break
            iteration_count += 1

            response_or_stream = self._get_llm_response(addr=addr)

            current_turn_content_parts: List[str] = []
            assembled_tool_calls: List[Dict[str, Any]] = []
            full_reasoning_content: Optional[str] = None

            if self.config.stream:
                gen = self._process_stream_response(response_or_stream)  # type: ignore
                try:
                    while True:
                        content_chunk = next(gen)
                        yield content_chunk
                        current_turn_content_parts.append(content_chunk)
                        accumulated_response.append(content_chunk)
                except StopIteration as e:
                    assembled_tool_calls, full_reasoning_content = e.value
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

            # If no tool calls, we are done
            if not assembled_tool_calls:
                # Add reasoning to metadata if present
                msg_kwargs = {}
                if full_reasoning_content:
                    msg_kwargs["metadata"] = {"reasoning_content": full_reasoning_content}
                
                self.add(role="assistant", content=full_turn_content, addr=addr, **msg_kwargs)
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
            
            self.add(addr=addr, **assistant_msg)

            # Execute tools
            for tc in assembled_tool_calls:
                tool_name = tc["function"]["name"]
                tool_call_id = tc["id"]
                args_str = tc["function"]["arguments"]

                try:
                    args = self._parse_tool_arguments(tool_name, args_str)
                    result_content = self._execute_tool(tool_name, args)
                except ValueError as e:
                    result_content = json.dumps({"error": str(e)})

                self.add(
                    role="tool",
                    content=result_content,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    addr=addr,
                )
        else:
            warn_msg = f"\n[WARNING] Agent '{self.config.name}' reached max iterations ({self.config.max_tool_iter}).\n"
            logger.warning(warn_msg.strip())
            # Notify finish even on max iterations
            for cb in self.callbacks:
                cb.on_agent_finish(self.config.name, warn_msg)
            yield warn_msg
            accumulated_response.append(warn_msg)

        full_response = "".join(accumulated_response)
        for hook in self.post_hooks:
            self._execute_hook(
                hook, agent=self, user_input=user_input, response=full_response
            )

    # Public entrypoint

    def run(
        self,
        user_input: str,
        *,
        addr: Optional[MemoryAddress] = None,
        image_path: Optional[str] = None,
        image_detail_override: Optional[str] = None,
    ) -> Union[str, Generator[str, None, None]]:
        """Main entrypoint to interact with the agent.
        
        Args:
            user_input: The text input from the user.
            addr: The memory address for the conversation.
            image_path: Optional path to an image file.
            image_detail_override: Optional detail level for image processing.
            
        Returns:
            The agent's response as a string or a generator if streaming is enabled.
        """
        a = self._addr_or_raise(addr)
        response_generator = self._execute_agent_loop(
            user_input,
            addr=a,
            image_path=image_path,
            image_detail_override=image_detail_override,
        )

        if self.config.stream:
            return response_generator

        parts: List[str] = list(response_generator)
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

    async def _aget_llm_response(
        self, *, addr: MemoryAddress
    ) -> Union[Any, AsyncGenerator[Dict[str, Any], None]]:
        """Perform the async LLM call with retries and error handling."""
        async_client = self._get_async_client()
        tool_choice_param = "auto" if self._tools else None
        common_params: Dict[str, Any] = {
            "model": self.config.model_name,
            "messages": self.memory.get_history(addr),
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
        if tools_payload:
            common_params["tools"] = tools_payload
            common_params["tool_choice"] = tool_choice_param

        for cb in self.callbacks:
            cb.on_llm_start(self.config.model_name, common_params["messages"])

        for attempt in range(self.config.max_retries):
            try:
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
                    logger.warning(
                        f"Transient error in _aget_llm_response (attempt {attempt + 1}/{self.config.max_retries}): {e}. Retrying..."
                    )
                    await asyncio.sleep(2**attempt)
                else:
                    # Final attempt failed, log full error
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
            # Check for async_func attribute (used by AgentTool, FlowTool, SpawnAgentTool)
            if hasattr(tool, "async_func") and asyncio.iscoroutinefunction(tool.async_func):
                result = await tool.async_func(**arguments)
            # Check if the tool function itself is async
            elif asyncio.iscoroutinefunction(tool.func):
                result = await tool.func(**arguments)
            else:
                # Run sync function in thread pool to avoid blocking
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: tool(**arguments)
                )
            result_str = str(result)
            for cb in self.callbacks:
                cb.on_tool_finish(tool_name, result_str)
            return result_str
        except Exception as e:
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
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
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
        
        self._last_stream_reasoning = "".join(full_reasoning)

    async def _aexecute_agent_loop(
        self,
        user_input: str,
        *,
        addr: MemoryAddress,
        image_path: Optional[str] = None,
        image_detail_override: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Async version of the agent loop with parallel tool execution."""
        self._ensure_system_initialized(addr)

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
            self.add(role="user", content=user_content, addr=addr)

        accumulated_response: List[str] = []

        iteration_count = 0
        while True:
            if self.config.max_tool_iter is not None and iteration_count >= self.config.max_tool_iter:
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

            # If no tool calls, we are done
            if not assembled_tool_calls:
                msg_kwargs = {}
                if full_reasoning_content:
                    msg_kwargs["metadata"] = {"reasoning_content": full_reasoning_content}
                
                self.add(role="assistant", content=full_turn_content, addr=addr, **msg_kwargs)
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
            
            self.add(addr=addr, **assistant_msg)

            # Execute tools IN PARALLEL using asyncio.gather
            async def execute_single_tool(tc: Dict[str, Any]) -> tuple[str, str, str]:
                tool_name = tc["function"]["name"]
                tool_call_id = tc["id"]
                args_str = tc["function"]["arguments"]
                try:
                    args = self._parse_tool_arguments(tool_name, args_str)
                    result_content = await self._aexecute_tool(tool_name, args)
                except ValueError as e:
                    result_content = json.dumps({"error": str(e)})
                return tool_call_id, tool_name, result_content

            tool_results = await asyncio.gather(
                *[execute_single_tool(tc) for tc in assembled_tool_calls]
            )

            # Add tool results to memory
            for tool_call_id, tool_name, result_content in tool_results:
                self.add(
                    role="tool",
                    content=result_content,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    addr=addr,
                )
        else:
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

    async def arun(
        self,
        user_input: str,
        *,
        addr: Optional[MemoryAddress] = None,
        image_path: Optional[str] = None,
        image_detail_override: Optional[str] = None,
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Async entrypoint to interact with the agent.
        
        This is the async version of `run()`. It executes LLM calls
        and tool executions asynchronously, with parallel tool execution
        when multiple tools are called.
        
        Args:
            user_input: The text input from the user.
            addr: The memory address for the conversation.
            image_path: Optional path to an image file.
            image_detail_override: Optional detail level for image processing.
            
        Returns:
            The agent's response as a string or an async generator if streaming is enabled.
        """
        a = self._addr_or_raise(addr)
        response_generator = self._aexecute_agent_loop(
            user_input,
            addr=a,
            image_path=image_path,
            image_detail_override=image_detail_override,
        )

        if self.config.stream:
            return response_generator

        parts: List[str] = []
        async for chunk in response_generator:
            parts.append(chunk)
        return "".join(parts).strip()

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
