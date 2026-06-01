import logging
import asyncio
from typing import Any, Dict

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
    supports_streaming = False

    def __init__(self, config: Dict[str, Any], timeout: int):
        if AsyncCodex is None:
            raise ImportError(
                "openai-codex is not installed. "
                "Please install it with `pip install agentify-core[codex]`."
            )
        self.config = config
        self.timeout = timeout
        self.mcp_tools_enabled = bool(config.get("mcp_tools_enabled", True))
        self.memory_mode = str(config.get("memory_mode", "agentify"))
        if self.memory_mode not in {"agentify", "codex_thread"}:
            raise ValueError("Codex memory_mode must be 'agentify' or 'codex_thread'.")

        # agentify_session_id → codex_thread_id (string, not the live object)
        self.thread_ids: Dict[str, str] = {}

        import shutil
        import os

        codex_path = shutil.which("codex")
        if not codex_path:
            # Fallback for common global install paths
            common_paths = [
                os.path.expanduser("~/.npm-global/bin/codex"),
                os.path.expanduser(
                    "~/.npm-global/lib/node_modules/@openai/codex/"
                    "node_modules/@openai/codex-linux-x64/vendor/"
                    "x86_64-unknown-linux-musl/codex/codex"
                ),
            ]
            # Dynamically discover VSCode extension binaries
            vscode_ext_dir = os.path.expanduser("~/.vscode/extensions")
            if os.path.isdir(vscode_ext_dir):
                for entry in sorted(os.listdir(vscode_ext_dir), reverse=True):
                    if entry.startswith("openai.chatgpt-"):
                        candidate = os.path.join(
                            vscode_ext_dir, entry, "bin", "linux-x86_64", "codex"
                        )
                        common_paths.append(candidate)

            for p in common_paths:
                if os.path.exists(p) and os.access(p, os.X_OK):
                    codex_path = p
                    break

        if not codex_path:
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

    # ------------------------------------------------------------------
    # Thread lifecycle helpers
    # ------------------------------------------------------------------

    async def _get_or_create_thread(self, session_id: str, model: str):
        """Return a live AsyncThread, creating or resuming as needed."""
        if self.memory_mode == "agentify":
            logger.debug("Starting ephemeral Codex thread for Agentify-managed memory")
            try:
                return await self.codex.thread_start(model=model, ephemeral=True)
            except TypeError:
                return await self.codex.thread_start(model=model)

        thread_id = self.thread_ids.get(session_id)

        if thread_id is not None:
            # Resume an existing Codex thread by its persisted ID
            logger.debug("Resuming Codex thread %s for session %s", thread_id, session_id)
            thread = await self.codex.thread_resume(thread_id, model=model)
        else:
            # First interaction — start a brand-new thread
            logger.debug("Starting new Codex thread for session %s", session_id)
            thread = await self.codex.thread_start(model=model)
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
        if stream:
            raise NotImplementedError(
                "Streaming is not supported by the native Codex provider. "
                "Agentify reconstructs a single final response from Codex turn events."
            )

        prompt = self._build_prompt(prompt, kwargs.get("messages"))
        thread = await self._get_or_create_thread(session_id, model)

        if hasattr(thread, "turn"):
            response_content = await self._run_thread_turn_from_events(thread, prompt)
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

    def _build_prompt(self, fallback_prompt: str, messages: Any) -> str:
        """Build the Codex turn prompt from Agentify memory when configured."""
        if self.memory_mode != "agentify" or not isinstance(messages, list):
            return fallback_prompt

        parts = [
            "Use the following Agentify-managed conversation state as the source of truth. "
            "Do not rely on previous Codex thread state for memory."
        ]
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "message")).upper()
            content = self._stringify_message_content(message.get("content"))
            if content:
                parts.append(f"{role}: {content}")
        return "\n\n".join(parts)

    def _stringify_message_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and item.get("text"):
                        text_parts.append(str(item["text"]))
                    elif item.get("type") in {"image_url", "input_image"}:
                        text_parts.append(
                            "[image omitted: Codex provider does not support "
                            "Agentify image input]"
                        )
            return "\n".join(text_parts)
        if content is None:
            return ""
        return str(content)

    async def _run_thread_turn_from_events(self, thread: Any, prompt: str) -> str:
        from agentify.llm.codex_events import CodexEventCollector

        turn = await thread.turn(prompt)
        collector = CodexEventCollector()
        stream = turn.stream()
        try:
            await self._collect_turn_events(stream, collector)
        finally:
            await stream.aclose()

        result = collector.result()
        if result.errors:
            raise RuntimeError("; ".join(result.errors))
        if not result.final_text:
            raise RuntimeError(
                "Codex turn completed without reconstructible text. "
                "No agent message delta, agent message item, or MCP tool result was exposed "
                "by the event stream."
            )
        return result.final_text

    async def _collect_turn_events(self, stream: Any, collector: Any) -> None:
        iterator = stream.__aiter__()
        while True:
            try:
                event = await asyncio.wait_for(iterator.__anext__(), timeout=float(self.timeout))
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    f"Timed out waiting for Codex turn events after {self.timeout} seconds."
                ) from exc

            collector.ingest(event)
            if collector.turn_completed:
                break

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
                # If content is a list (multimodal), we extract text or serialize
                last_content = messages[-1]["content"] if messages else ""
                if isinstance(last_content, list):
                    prompt = " ".join([c.get("text", "") for c in last_content if c.get("type") == "text"])
                else:
                    prompt = last_content
                
                model = kwargs.get("model", "gpt-4")
                session_id = kwargs.get("session_id", "compat_session")
                stream = kwargs.get("stream", False)
                
                # Remove duplicate keys from kwargs before passing
                clean_kwargs = {k: v for k, v in kwargs.items() if k not in ("session_id", "model", "stream")}
                return await self.backend.run_native(session_id, model, prompt, stream, **clean_kwargs)
                
    @property
    def chat(self):
        if not hasattr(self, "_chat"):
            self._chat = self.Chat()
            self._chat.completions = self.Chat.Completions(self)
        return self._chat
