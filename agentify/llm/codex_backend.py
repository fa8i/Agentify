import logging
import asyncio
import os
import shutil
from collections.abc import Sequence
from typing import Any, Dict

from agentify.core.tool import Tool
from agentify.llm.codex_errors import raise_codex_errors
from agentify.llm.codex_inputs import build_codex_turn_input, extract_output_schema
from agentify.mcp.runtime_bridge import RuntimeMCPBridge

try:
    from openai_codex import AsyncCodex
except ImportError:
    AsyncCodex = None

logger = logging.getLogger(__name__)

class DummyMessage:
    def __init__(self, content: str):
        self.content = content
        self.tool_calls = None
        self.role = "assistant"

class DummyChoice:
    def __init__(self, content: str):
        self.message = DummyMessage(content)
        self.delta = DummyMessage(content)

class DummyResponse:
    def __init__(self, content: str):
        self.choices = [DummyChoice(content)]


class CodexThreadBackend:
    """Native Codex Thread Backend.

    By default, Agentify memory is the source of truth: every turn starts a
    fresh Codex thread and sends the Agentify-managed history in the prompt.
    Set ``memory_mode="codex_thread"`` to reuse Codex thread IDs per Agentify
    session instead.
    """
    is_native_thread_backend = True
    supports_tools = False
    supports_openai_tool_calls = False
    supports_mcp_tools = True
    supports_streaming = True

    def __init__(self, config: Dict[str, Any], timeout: int):
        if AsyncCodex is None:
            raise ImportError(
                "openai-codex is not installed. "
                "Please install it with `pip install agentify-core[codex]`."
            )
        self.config = config
        self.timeout = timeout
        self.mcp_tools_enabled = bool(config.get("mcp_tools_enabled", True))
        self.auto_mcp_tools = bool(config.get("auto_mcp_tools", True))
        self.memory_mode = str(config.get("memory_mode", "agentify"))
        if self.memory_mode not in {"agentify", "codex_thread"}:
            raise ValueError("Codex memory_mode must be 'agentify' or 'codex_thread'.")
        self._runtime_mcp_bridge: RuntimeMCPBridge | None = None

        # agentify_session_id → codex_thread_id (string, not the live object)
        self.thread_ids: Dict[str, str] = {}

        codex_path = _find_codex_binary()
        if codex_path is None:
            logger.warning(
                "No global 'codex' executable found in PATH. "
                "AsyncCodex might fail to start if openai-codex-cli-bin is missing."
            )
            self.codex = AsyncCodex()
        else:
            try:
                from openai_codex import CodexConfig
                self.codex = AsyncCodex(config=CodexConfig(codex_bin=codex_path))
            except ImportError:
                self.codex = AsyncCodex()

    async def close(self) -> None:
        """Close runtime resources owned by the Codex backend."""
        if self._runtime_mcp_bridge is not None:
            await self._runtime_mcp_bridge.close()
            self._runtime_mcp_bridge = None
        close = getattr(self.codex, "close", None)
        if close is not None:
            await close()

    # ------------------------------------------------------------------
    # Thread lifecycle helpers
    # ------------------------------------------------------------------

    async def _get_or_create_thread(
        self,
        session_id: str,
        model: str,
        *,
        thread_config: dict[str, Any] | None = None,
    ):
        """Return a live AsyncThread, creating or resuming as needed."""
        if self.memory_mode == "agentify":
            logger.debug("Starting ephemeral Codex thread for Agentify-managed memory")
            try:
                return await self.codex.thread_start(
                    model=model,
                    ephemeral=True,
                    config=thread_config,
                )
            except TypeError:
                return await self.codex.thread_start(model=model)

        thread_id = self.thread_ids.get(session_id)

        if thread_id is not None:
            # Resume an existing Codex thread by its persisted ID
            logger.debug("Resuming Codex thread %s for session %s", thread_id, session_id)
            thread = await self.codex.thread_resume(
                thread_id,
                model=model,
                config=thread_config,
            )
        else:
            # First interaction — start a brand-new thread
            logger.debug("Starting new Codex thread for session %s", session_id)
            thread = await self.codex.thread_start(model=model, config=thread_config)
            self.thread_ids[session_id] = thread.id
            logger.info(
                "Mapped session %s → codex thread %s", session_id, thread.id
            )

        return thread

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_native(
        self,
        session_id: str,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs,
    ) -> Any:
        """Send *only* the latest prompt to the Codex thread for this session."""
        prompt = build_codex_turn_input(
            prompt,
            kwargs.get("messages"),
            memory_mode=self.memory_mode,
        )
        thread_config = await self._build_thread_config(
            kwargs.get("agentify_tools"),
            tool_timeout=kwargs.get("agentify_tool_timeout"),
            tool_executor=kwargs.get("agentify_tool_executor"),
        )
        thread = await self._get_or_create_thread(
            session_id,
            model,
            thread_config=thread_config,
        )
        output_schema = extract_output_schema(kwargs)

        if stream:
            return self._stream_thread_turn_events(
                thread,
                prompt,
                output_schema=output_schema,
            )

        if hasattr(thread, "turn"):
            response_content = await self._run_thread_turn_from_events(
                thread,
                prompt,
                output_schema=output_schema,
            )
        else:
            if self.mcp_tools_enabled:
                raise RuntimeError(
                    "Codex MCP tools require event streaming via thread.turn(...).stream(). "
                    "The installed openai-codex SDK does not expose this API."
                )
            try:
                result = await thread.run(prompt)
            except AttributeError:
                logger.error(
                    "Codex Thread object does not have 'run' method. "
                    "SDK might have changed."
                )
                raise
            response_content = self._extract_response_content(result)

        return DummyResponse(response_content)

    async def _build_thread_config(
        self,
        tools: Any,
        *,
        tool_timeout: float | None = None,
        tool_executor: Any = None,
    ) -> dict[str, Any] | None:
        if not tools:
            return None
        if not self.auto_mcp_tools:
            raise NotImplementedError(
                "Agentify tools were passed to the native Codex provider, but "
                "auto_mcp_tools=False. Enable auto_mcp_tools or expose tools through "
                "Agentify MCP stdio manually."
            )
        if not self.mcp_tools_enabled:
            raise NotImplementedError(
                "Agentify tools for the native Codex provider require MCP tools. "
                "Set mcp_tools_enabled=True or use provider='openai' for classic tool_calls."
            )
        if not isinstance(tools, Sequence) or not all(
            isinstance(tool, Tool) for tool in tools
        ):
            raise TypeError("agentify_tools must be a sequence of Agentify Tool objects.")

        if self._runtime_mcp_bridge is None:
            self._runtime_mcp_bridge = RuntimeMCPBridge()
            await self._runtime_mcp_bridge.start()
        self._runtime_mcp_bridge.update_tools(list(tools))
        self._runtime_mcp_bridge.update_tool_timeout(tool_timeout)
        self._runtime_mcp_bridge.update_tool_executor(tool_executor)
        return self._runtime_mcp_bridge.codex_config()

    async def _run_thread_turn_from_events(
        self,
        thread: Any,
        prompt: Any,
        *,
        output_schema: dict[str, Any] | None = None,
    ) -> str:
        from agentify.llm.codex_events import CodexEventCollector

        collector = CodexEventCollector()
        turn = await self._start_turn(thread, prompt, output_schema=output_schema)
        stream = turn.stream()
        try:
            async for event in self._iter_turn_events(stream):
                collector.ingest(event)
                if collector.turn_completed:
                    break
        finally:
            await stream.aclose()

        result = collector.result()
        if result.errors:
            raise_codex_errors(result.errors)
        if not result.final_text:
            raise RuntimeError(
                "Codex turn completed without reconstructible text. "
                "No agent message delta, agent message item, or MCP tool result was exposed "
                "by the event stream."
            )
        return result.final_text

    async def _stream_thread_turn_events(
        self,
        thread: Any,
        prompt: Any,
        *,
        output_schema: dict[str, Any] | None = None,
    ):
        from agentify.llm.codex_events import CodexEventCollector

        if not hasattr(thread, "turn"):
            raise RuntimeError(
                "Streaming requires Codex event streaming via thread.turn(...).stream()."
        )

        collector = CodexEventCollector()
        turn = await self._start_turn(thread, prompt, output_schema=output_schema)
        stream = turn.stream()
        try:
            async for event in self._iter_turn_events(stream):
                delta = self._event_agent_message_delta(event)
                collector.ingest(event)
                if delta:
                    yield DummyResponse(delta)
                if collector.turn_completed:
                    break
        finally:
            await stream.aclose()

        result = collector.result()
        if result.errors:
            raise_codex_errors(result.errors)

    async def _start_turn(
        self,
        thread: Any,
        prompt: Any,
        *,
        output_schema: dict[str, Any] | None = None,
    ) -> Any:
        if output_schema is not None:
            return await thread.turn(prompt, output_schema=output_schema)
        return await thread.turn(prompt)

    async def _iter_turn_events(self, stream: Any):
        iterator = stream.__aiter__()
        while True:
            try:
                yield await asyncio.wait_for(
                    iterator.__anext__(),
                    timeout=float(self.timeout),
                )
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    f"Timed out waiting for Codex turn events after {self.timeout} seconds."
                ) from exc

    def _event_agent_message_delta(self, event: Any) -> str | None:
        method = (
            event.get("method")
            if isinstance(event, dict)
            else getattr(event, "method", "")
        )
        if method not in {"item/agentMessage/delta", "agentMessage/delta"}:
            return None
        payload = (
            event.get("payload")
            if isinstance(event, dict)
            else getattr(event, "payload", None)
        )
        delta = (
            payload.get("delta")
            if isinstance(payload, dict)
            else getattr(payload, "delta", None)
        )
        return str(delta) if delta else None

    def _extract_response_content(self, result: Any) -> str:
        """Extract useful text from Codex turn results.

        Some Codex turns that primarily use MCP tools can complete with
        ``final_response=None`` while still containing MCP output in turn items.
        """
        final_response = getattr(result, "final_response", None)
        if final_response:
            return str(final_response)

        items = getattr(result, "items", None) or []
        for item in reversed(items):
            root = getattr(item, "root", item)
            text = getattr(root, "text", None)
            if text:
                return str(text)

            mcp_result = getattr(root, "result", None)
            if mcp_result is not None:
                content_text = self._extract_mcp_content_text(getattr(mcp_result, "content", None))
                if content_text:
                    return content_text

        return str(result)

    def _extract_mcp_content_text(self, content: Any) -> str | None:
        if not content:
            return None

        parts = []
        for item in content:
            text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
                continue
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))

        return "\n".join(parts) if parts else None

    # Fallback Adapter interface
    class Chat:
        class Completions:
            def __init__(self, backend: 'CodexThreadBackend'):
                self.backend = backend
            
            async def create(self, **kwargs) -> Any:
                messages = kwargs.get("messages", [])
                prompt = _latest_message_text(messages)
                
                model = kwargs.get("model", "gpt-4")
                session_id = kwargs.get("session_id", "compat_session")
                stream = kwargs.get("stream", False)
                
                # Remove duplicate keys from kwargs before passing
                clean_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in ("session_id", "model", "stream")
                }
                return await self.backend.run_native(
                    session_id,
                    model,
                    prompt,
                    stream,
                    **clean_kwargs,
                )
                
    @property
    def chat(self):
        if not hasattr(self, "_chat"):
            self._chat = self.Chat()
            self._chat.completions = self.Chat.Completions(self)
        return self._chat


def _find_codex_binary() -> str | None:
    codex_path = shutil.which("codex")
    if codex_path:
        return codex_path

    for candidate in _candidate_codex_paths():
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _candidate_codex_paths() -> list[str]:
    paths = [
        os.path.expanduser("~/.npm-global/bin/codex"),
        os.path.expanduser(
            "~/.npm-global/lib/node_modules/@openai/codex/"
            "node_modules/@openai/codex-linux-x64/vendor/"
            "x86_64-unknown-linux-musl/codex/codex"
        ),
    ]

    vscode_ext_dir = os.path.expanduser("~/.vscode/extensions")
    if os.path.isdir(vscode_ext_dir):
        for entry in sorted(os.listdir(vscode_ext_dir), reverse=True):
            if entry.startswith("openai.chatgpt-"):
                paths.append(
                    os.path.join(
                        vscode_ext_dir,
                        entry,
                        "bin",
                        "linux-x86_64",
                        "codex",
                    )
                )
    return paths


def _latest_message_text(messages: list[dict[str, Any]]) -> str:
    last_content = messages[-1]["content"] if messages else ""
    if isinstance(last_content, list):
        return " ".join(
            str(item.get("text", ""))
            for item in last_content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(last_content)
