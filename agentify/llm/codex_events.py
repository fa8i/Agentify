"""Experimental Codex event-stream diagnostics.

This module intentionally does not replace ``CodexThreadBackend``. It consumes
``openai-codex`` turn notifications directly to diagnose whether MCP tool output
is available before ``thread.run()`` collapses the stream into ``TurnResult``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from agentify.llm.codex_backend import CodexThreadBackend


@dataclass(slots=True)
class CodexEventResult:
    final_text: str
    turn_completed: bool
    mcp_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    mcp_tool_results: list[str] = field(default_factory=list)
    mcp_server_statuses: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    event_count: int = 0


class CodexEventCollector:
    """Collect useful state from Codex turn notifications."""

    def __init__(self) -> None:
        self.agent_message_parts: list[str] = []
        self.mcp_tool_calls: list[dict[str, Any]] = []
        self.mcp_tool_results: list[str] = []
        self.mcp_server_statuses: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.turn_completed = False
        self.event_count = 0

    def ingest(self, event: Any) -> None:
        self.event_count += 1
        method = _event_method(event)
        payload = getattr(event, "payload", None)

        if method in {"item/agentMessage/delta", "agentMessage/delta"}:
            delta = getattr(payload, "delta", None)
            if delta:
                self.agent_message_parts.append(str(delta))
            return

        if method == "item/completed":
            self._ingest_completed_item(_payload_value(payload, "item"))
            return

        if method == "turn/completed":
            self.turn_completed = True
            turn = getattr(payload, "turn", None)
            error = getattr(turn, "error", None)
            message = normalize_codex_error(error)
            if message:
                self.errors.append(message)
            return

        if method == "mcpServer/startupStatus/updated":
            self._ingest_mcp_server_status(payload)
            return

        if method == "error":
            message = getattr(payload, "message", None) or str(payload)
            self.errors.append(str(message))

    def result(self) -> CodexEventResult:
        final_text = "".join(self.agent_message_parts)
        if not final_text and self.mcp_tool_results:
            final_text = "\n".join(self.mcp_tool_results)
        return CodexEventResult(
            final_text=final_text,
            turn_completed=self.turn_completed,
            mcp_tool_calls=self.mcp_tool_calls,
            mcp_tool_results=self.mcp_tool_results,
            mcp_server_statuses=self.mcp_server_statuses,
            errors=self.errors,
            event_count=self.event_count,
        )

    def _ingest_completed_item(self, item: Any) -> None:
        root = getattr(item, "root", item)
        item_type = _value(root, "type")
        if item_type == "agentMessage":
            text = _value(root, "text")
            if text and str(text) not in "".join(self.agent_message_parts):
                self.agent_message_parts.append(str(text))
            return

        if item_type == "mcpToolCall":
            call = {
                "server": _value(root, "server"),
                "tool": _value(root, "tool"),
                "status": str(_value(root, "status") or ""),
            }
            self.mcp_tool_calls.append(call)
            result = _value(root, "result")
            text = extract_mcp_result_text(result)
            if text:
                self.mcp_tool_results.append(text)

    def _ingest_mcp_server_status(self, payload: Any) -> None:
        params = _value(payload, "params") or payload
        status = _value(params, "status")
        server = _value(params, "server") or _value(params, "server_name") or _value(params, "name")
        error = _value(params, "error")
        message = normalize_codex_error(error) or _value(params, "message")
        self.mcp_server_statuses.append(
            {"server": server, "status": str(status or ""), "message": message}
        )
        if status and str(status).lower() in {"failed", "error"}:
            self.errors.append(
                f"Codex MCP server failed to start: {server or 'unknown'}: {message or 'unknown error'}"
            )


class CodexEventBackend:
    """Experimental event-stream backend for manual diagnostics."""

    def __init__(self, *, timeout: int = 300) -> None:
        self._backend = CodexThreadBackend(config={}, timeout=timeout)

    async def run_turn_events(
        self,
        *,
        session_id: str,
        model: str,
        prompt: str,
        events_log: str | Path | None = None,
        timeout: float = 300,
    ) -> CodexEventResult:
        thread = await self._backend._get_or_create_thread(session_id, model)
        turn = await thread.turn(prompt)
        collector = CodexEventCollector()
        log_path = Path(events_log) if events_log else None

        stream = turn.stream()
        try:
            async with asyncio.timeout(timeout):
                async for event in stream:
                    if log_path is not None:
                        append_event_jsonl(log_path, event)
                    collector.ingest(event)
                    if collector.turn_completed:
                        break
        finally:
            await stream.aclose()

        if not collector.turn_completed:
            raise TimeoutError("Codex turn did not complete before timeout.")
        return collector.result()


def normalize_codex_error(error: Any) -> str | None:
    if error is None:
        return None

    message = _value(error, "message") or str(error)
    info = _value(error, "codex_error_info")
    info_text = str(info or "")

    if "usageLimitExceeded" in info_text or "usage limit" in message.lower():
        return f"Codex usage limit exceeded: {message}"
    if "not supported" in message.lower() and "model" in message.lower():
        return f"Codex model is not supported for this account or CLI version: {message}"
    if "mcp" in message.lower() and any(word in message.lower() for word in ("start", "spawn", "initialize")):
        return f"Codex MCP server failed to start: {message}"
    if "tool" in message.lower() and any(word in message.lower() for word in ("not found", "unknown")):
        return f"Codex MCP tool was not found: {message}"
    return str(message)


def append_event_jsonl(path: Path, event: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(serialize_event(event), ensure_ascii=False) + "\n")


def serialize_event(event: Any) -> dict[str, Any]:
    payload = getattr(event, "payload", None)
    return {
        "method": _event_method(event),
        "payload_type": payload.__class__.__name__ if payload is not None else None,
        "payload": _jsonable(payload),
    }


def extract_mcp_result_text(result: Any) -> str | None:
    if result is None:
        return None
    content = getattr(result, "content", None)
    if not content and isinstance(result, dict):
        content = result.get("content")
    if not content:
        return None

    parts: list[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if text:
            parts.append(str(text))
        elif isinstance(item, dict) and item.get("text"):
            parts.append(str(item["text"]))
    return "\n".join(parts) if parts else None


def _event_method(event: Any) -> str:
    if isinstance(event, dict):
        return str(event.get("method", ""))
    return str(getattr(event, "method", ""))


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _payload_value(payload: Any, key: str) -> Any:
    value = _value(payload, key)
    if value is not None:
        return value
    params = _value(payload, "params")
    if isinstance(params, dict):
        return params.get(key)
    return None


def _value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
