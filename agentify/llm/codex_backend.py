import logging
import asyncio
import inspect
import json
import os
import shutil
import time
from collections.abc import Sequence
from typing import Any, Dict

from agentify.core.tool import Tool
from agentify.llm.codex_errors import (
    CodexBackendError,
    CodexCLINotFoundError,
    CodexEmptyTurnError,
    CodexStreamTimeoutError,
    raise_codex_errors,
)
from agentify.llm.codex_inputs import (
    build_codex_turn_input,
    extract_output_schema,
    extract_system_prompt,
    prepend_system_prompt,
)
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

    Memory modes:

    - ``memory_mode="agentify"`` (default): Agentify memory is the source of
      truth. Every turn starts a fresh **ephemeral** Codex thread and resends
      the full Agentify-managed history in the prompt. This keeps Agentify
      memory portable across providers, but each turn pays for reprocessing
      the whole history and — when Agentify tools are attached — for a fresh
      MCP server startup, so latency grows with conversation length. Best for
      batch/one-shot usage or when Agentify memory must stay authoritative.
    - ``memory_mode="codex_thread"``: one persistent Codex thread per Agentify
      session (``thread_ids`` maps session → thread ID). Only the latest user
      message is sent per turn and the MCP server stays warm across turns.
      Recommended for interactive, multi-turn assistants. Agentify memory is
      still written (tool calls and responses are recorded) but Codex thread
      state is what the model actually sees.

    Tools are registered on the runtime MCP bridge per session, so concurrent
    turns for different sessions on the same backend keep isolated executors
    and memory bindings.

    Set ``thread_map_path`` in the config to persist the session → Codex
    thread mapping across process restarts (Codex threads themselves are
    persisted by the Codex CLI under ``~/.codex/``).
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

        # How the agent system prompt reaches a persistent Codex thread in
        # codex_thread mode. "developer" layers it on top of Codex's base
        # coding-agent harness (keeps native tool framing intact); "base"
        # replaces that harness for full persona control. Both are passed as
        # thread-level metadata on every thread_start/thread_resume, so they sit
        # in the preserved prefix and survive context compaction.
        self.instructions_mode = str(config.get("instructions_mode", "developer"))
        if self.instructions_mode not in {"base", "developer"}:
            raise ValueError("Codex instructions_mode must be 'base' or 'developer'.")
        self._instructions_kwarg = (
            "base_instructions" if self.instructions_mode == "base" else "developer_instructions"
        )
        self._runtime_mcp_bridge: RuntimeMCPBridge | None = None

        # agentify_session_id → codex_thread_id (string, not the live object)
        self.thread_ids: Dict[str, str] = {}
        # Sessions explicitly dropped (via drop_session) since load, so a
        # merge-on-save never resurrects them from a stale on-disk entry.
        self._dropped_sessions: set[str] = set()
        # Optional JSON file persisting the session → thread mapping across
        # process restarts (Codex threads themselves survive in ~/.codex/).
        self.thread_map_path: str | None = (
            str(config["thread_map_path"]) if config.get("thread_map_path") else None
        )
        self._thread_map_loaded = False

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

        # Whether the installed SDK accepts the chosen instructions kwarg on
        # thread_start/thread_resume. Older SDKs do not, in which case we fall
        # back to text-injecting the system prompt on the first turn.
        self._supports_instructions = self._detect_instructions_support()
        if self.memory_mode == "codex_thread" and not self._supports_instructions:
            logger.warning(
                "Installed openai-codex SDK does not accept '%s' on thread_start/resume; "
                "falling back to first-turn text injection of the system prompt (it may "
                "degrade under context compaction). Upgrade with `pip install -U openai-codex`.",
                self._instructions_kwarg,
            )

    def _detect_instructions_support(self) -> bool:
        def accepts(fn: Any) -> bool:
            try:
                params = list(inspect.signature(fn).parameters.values())
            except (ValueError, TypeError):  # pragma: no cover - defensive
                return False
            if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
                return True
            return self._instructions_kwarg in {p.name for p in params}

        return accepts(self.codex.thread_start) and accepts(self.codex.thread_resume)

    def _instr_kwargs(self, instructions: str | None) -> dict[str, str]:
        """Thread-level instruction kwargs to pass to thread_start/thread_resume."""
        if instructions and self._supports_instructions:
            return {self._instructions_kwarg: instructions}
        return {}

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
        instructions: str | None = None,
    ):
        """Return a live AsyncThread, creating or resuming as needed."""
        started_at = time.monotonic()
        try:
            thread = await self._acquire_thread(
                session_id, model, thread_config=thread_config, instructions=instructions
            )
        except FileNotFoundError as exc:
            raise CodexCLINotFoundError(
                "Codex CLI binary was not found or could not be started. "
                "Install it (e.g. `npm install -g @openai/codex`) or set "
                f"client_config_override={{'codex_bin': ...}}. Original error: {exc}"
            ) from exc
        logger.debug(
            "Codex thread ready in %.2fs (memory_mode=%s, tools=%s)",
            time.monotonic() - started_at,
            self.memory_mode,
            bool(thread_config),
        )
        return thread

    async def _acquire_thread(
        self,
        session_id: str,
        model: str,
        *,
        thread_config: dict[str, Any] | None = None,
        instructions: str | None = None,
    ):
        if self.memory_mode == "agentify":
            logger.debug("Starting ephemeral Codex thread for Agentify-managed memory")
            try:
                return await self.codex.thread_start(
                    model=model,
                    ephemeral=True,
                    config=thread_config,
                )
            except TypeError as exc:
                if thread_config is not None:
                    raise CodexBackendError(
                        "The installed openai-codex SDK does not accept "
                        "'ephemeral'/'config' thread parameters, which are required "
                        "to expose Agentify tools through MCP. Upgrade with "
                        "`pip install -U openai-codex`."
                    ) from exc
                logger.warning(
                    "openai-codex SDK rejected ephemeral thread parameters (%s); "
                    "falling back to a persistent thread.",
                    exc,
                )
                return await self.codex.thread_start(model=model)

        self._load_thread_map()
        thread_id = self.thread_ids.get(session_id)

        if thread_id is not None:
            # Resume an existing Codex thread by its persisted ID. Instructions
            # are re-passed every turn so the persona stays in the preserved
            # prefix even after Codex compacts older turns.
            logger.debug("Resuming Codex thread %s for session %s", thread_id, session_id)
            try:
                return await self.codex.thread_resume(
                    thread_id,
                    model=model,
                    config=thread_config,
                    **self._instr_kwargs(instructions),
                )
            except Exception as exc:
                if not _is_missing_thread_error(exc):
                    raise
                # The thread vanished from ~/.codex (deleted, archived, or a
                # different machine). Recover with a fresh thread instead of
                # failing the session forever.
                logger.warning(
                    "Codex thread %s for session %s could not be resumed (%s); "
                    "starting a new thread. Previous Codex-side context is lost.",
                    thread_id,
                    session_id,
                    exc,
                )
                self.thread_ids.pop(session_id, None)

        # First interaction (or recovery) — start a brand-new thread
        logger.debug("Starting new Codex thread for session %s", session_id)
        thread = await self.codex.thread_start(
            model=model,
            config=thread_config,
            **self._instr_kwargs(instructions),
        )
        self.thread_ids[session_id] = thread.id
        self._save_thread_map()
        logger.info("Mapped session %s → codex thread %s", session_id, thread.id)

        return thread

    # ------------------------------------------------------------------
    # Session → thread map persistence (codex_thread mode)
    # ------------------------------------------------------------------

    def _load_thread_map(self) -> None:
        if self._thread_map_loaded or not self.thread_map_path:
            return
        self._thread_map_loaded = True
        path = os.path.expanduser(self.thread_map_path)
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return
        except Exception as exc:
            logger.warning("Could not load Codex thread map from %s: %s", path, exc)
            return
        if isinstance(data, dict):
            for key, value in data.items():
                self.thread_ids.setdefault(str(key), str(value))

    def _save_thread_map(self) -> None:
        if not self.thread_map_path:
            return
        path = os.path.expanduser(self.thread_map_path)
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            # Merge with whatever is on disk so concurrent backends sharing the
            # same map file (e.g. a CLI and a Telegram bridge) never clobber each
            # other's sessions. This backend's entries win for keys it owns.
            merged: Dict[str, str] = {}
            try:
                with open(path, encoding="utf-8") as handle:
                    existing = json.load(handle)
                if isinstance(existing, dict):
                    merged.update({str(k): str(v) for k, v in existing.items()})
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.warning("Could not read Codex thread map before saving %s: %s", path, exc)
            # Drop keys this backend deliberately removed since it last loaded.
            for key in self._dropped_sessions:
                merged.pop(key, None)
            merged.update(self.thread_ids)
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(merged, handle, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except Exception as exc:
            logger.warning("Could not persist Codex thread map to %s: %s", path, exc)

    def drop_session(self, session_id: str) -> str | None:
        """Forget the Codex thread mapped to a session (e.g. on a memory reset).

        Removes the mapping from memory and the persisted map file so the next
        turn for that session starts a fresh Codex thread instead of resuming the
        old one. Returns the dropped Codex thread ID, if there was one. The Codex
        thread itself stays on disk under ``~/.codex/`` and is simply no longer
        referenced.
        """
        self._load_thread_map()
        thread_id = self.thread_ids.pop(session_id, None)
        self._dropped_sessions.add(session_id)
        self._save_thread_map()
        if thread_id is not None:
            logger.info("Dropped Codex thread mapping for session %s (was %s)", session_id, thread_id)
        return thread_id

    def get_thread_id(self, session_id: str) -> str | None:
        """Return the Codex thread ID mapped to a session, if any."""
        self._load_thread_map()
        return self.thread_ids.get(session_id)

    async def read_session_history(self, session_id: str, *, include_turns: bool = True):
        """Read the native Codex thread history for a session.

        Only meaningful in ``memory_mode="codex_thread"``. Returns the SDK
        ``ThreadReadResponse`` (``response.thread.items`` per turn when
        ``include_turns=True``), hydrated from the rollout files Codex keeps
        under ``~/.codex/``.
        """
        thread_id = self.get_thread_id(session_id)
        if thread_id is None:
            raise KeyError(f"No Codex thread is mapped for session '{session_id}'.")
        thread = await self.codex.thread_resume(thread_id)
        return await thread.read(include_turns=include_turns)

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
        # In codex_thread mode only the latest user message is sent per turn, so
        # the agent's system prompt must reach Codex out-of-band. We pass it as
        # thread-level instructions on every thread_start/thread_resume, which
        # keeps it in the compaction-preserved prefix. On SDKs that lack the
        # instructions kwarg we fall back to text-injecting it on the first turn.
        instructions: str | None = None
        seed_system_prompt: str | None = None
        if self.memory_mode == "codex_thread":
            self._load_thread_map()
            system_prompt = extract_system_prompt(kwargs.get("messages"))
            if self._supports_instructions:
                instructions = system_prompt
            elif system_prompt and session_id not in self.thread_ids:
                seed_system_prompt = system_prompt

        prompt = build_codex_turn_input(
            prompt,
            kwargs.get("messages"),
            memory_mode=self.memory_mode,
        )
        if seed_system_prompt:
            prompt = prepend_system_prompt(prompt, seed_system_prompt)
        thread_config = await self._build_thread_config(
            kwargs.get("agentify_tools"),
            session_id=session_id,
            tool_timeout=kwargs.get("agentify_tool_timeout"),
            tool_executor=kwargs.get("agentify_tool_executor"),
        )
        thread = await self._get_or_create_thread(
            session_id,
            model,
            thread_config=thread_config,
            instructions=instructions,
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
                session_id=session_id if thread_config is not None else None,
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
        session_id: str,
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
        # Per-session registration: concurrent turns for different sessions
        # each keep their own tools, timeout, and memory-bound executor.
        self._runtime_mcp_bridge.register_session(
            session_id,
            tools=list(tools),
            tool_timeout=tool_timeout,
            tool_executor=tool_executor,
        )
        return self._runtime_mcp_bridge.codex_config(session_id)

    async def _run_thread_turn_from_events(
        self,
        thread: Any,
        prompt: Any,
        *,
        output_schema: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> str:
        from agentify.llm.codex_events import CodexEventCollector

        collector = CodexEventCollector()
        started_at = time.monotonic()
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
        logger.debug(
            "Codex turn finished in %.2fs (events=%d, mcp_tool_calls=%d, "
            "transient_errors=%d)",
            time.monotonic() - started_at,
            result.event_count,
            len(result.mcp_tool_calls),
            len(result.warnings),
        )
        if result.errors:
            raise_codex_errors(result.errors)
        if not result.final_text:
            mcp_status = None
            if session_id is not None:
                mcp_status = await self._mcp_server_diagnostics()
            raise CodexEmptyTurnError(
                self._empty_turn_message(result, session_id=session_id, mcp_status=mcp_status)
            )
        if result.warnings:
            logger.info(
                "Codex turn recovered after transient errors: %s",
                "; ".join(result.warnings),
            )
        return result.final_text

    def _empty_turn_message(
        self,
        result: Any,
        *,
        session_id: str | None = None,
        mcp_status: str | None = None,
    ) -> str:
        """Build an actionable error for a turn that produced no text."""
        parts = [
            "Codex turn completed without reconstructible text. "
            "No agent message delta, agent message item, or MCP tool result was exposed "
            f"by the event stream ({result.event_count} events received)."
        ]
        if result.warnings:
            parts.append(f"Transient errors during the turn: {'; '.join(result.warnings)}")
        bridge = self._runtime_mcp_bridge
        if bridge is not None and session_id is not None:
            served = bridge.session_connection_count(session_id)
            if served == 0:
                parts.append(
                    "The Agentify runtime MCP server never connected: Codex likely "
                    "failed to start it (check the `codex` CLI version, MCP support, "
                    "and that the Python interpreter can import agentify)."
                )
            else:
                parts.append(
                    f"Runtime MCP bridge served {served} request(s) for this session, "
                    "so the MCP path was working."
                )
        if mcp_status:
            parts.append(mcp_status)
        return " ".join(parts)

    async def _mcp_server_diagnostics(self) -> str | None:
        """Best-effort: ask Codex which MCP servers and tools it can see."""
        try:
            from openai_codex.generated.v2_all import ListMcpServerStatusResponse

            client = getattr(self.codex, "_client", None)
            request = getattr(client, "request", None)
            if request is None:
                return None
            response = await asyncio.wait_for(
                request(
                    "mcpServerStatus/list",
                    {"limit": 50},
                    response_model=ListMcpServerStatusResponse,
                ),
                timeout=10,
            )
            servers = getattr(response, "data", None) or []
            if not servers:
                return "Codex reports no MCP servers configured."
            summaries = [
                f"{getattr(server, 'name', '?')} ({len(getattr(server, 'tools', None) or {})} tools)"
                for server in servers
            ]
            return "Codex MCP servers visible: " + ", ".join(summaries) + "."
        except Exception as exc:
            logger.debug("mcpServerStatus/list diagnostics unavailable: %s", exc)
            return None

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
        if result.warnings:
            logger.info(
                "Codex turn recovered after transient errors: %s",
                "; ".join(result.warnings),
            )

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
                raise CodexStreamTimeoutError(
                    f"Timed out waiting for Codex turn events after {self.timeout} seconds "
                    "(no event arrived within the per-event timeout; the Codex CLI may be "
                    "stalled or a tool may be blocking without progress events)."
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


def _is_missing_thread_error(exc: Exception) -> bool:
    """True when an exception indicates the Codex thread no longer exists."""
    message = str(exc).lower()
    return any(
        marker in message
        for marker in ("not found", "no such", "does not exist", "unknown thread", "missing")
    )


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
