"""Manual diagnostics for Codex provider parity features.

This script intentionally performs real Codex calls. It is not part of CI.

Examples:

    .venv/bin/python scripts/manual_codex_feature_diagnostics.py --case all
    .venv/bin/python scripts/manual_codex_feature_diagnostics.py --case image --image IMAGEN.png
    .venv/bin/python scripts/manual_codex_feature_diagnostics.py --case streaming
    .venv/bin/python scripts/manual_codex_feature_diagnostics.py --case tool-limit

It logs:

- callbacks
- streaming chunks
- final responses
- Agentify memory history
- raw SQLite payload rows
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentify.core.agent import BaseAgent
from agentify.core.callbacks import AgentCallbackHandler
from agentify.core.config import AgentConfig
from agentify.core.tool import Tool
from agentify.memory.interfaces import MemoryAddress
from agentify.memory.service import MemoryService
from agentify.memory.stores.sqlite_store import SQLiteStore


class VerboseCallback(AgentCallbackHandler):
    def on_agent_start(self, agent_name: str, user_input: str) -> None:
        log("callback.agent_start", agent=agent_name, input=user_input)

    def on_agent_finish(self, agent_name: str, response: str) -> None:
        log("callback.agent_finish", agent=agent_name, response=response)

    def on_tool_start(self, tool_name: str, args: dict[str, Any]) -> None:
        log("callback.tool_start", tool=tool_name, args=args)

    def on_tool_finish(self, tool_name: str, output: str) -> None:
        log("callback.tool_finish", tool=tool_name, output=output)

    def on_llm_start(self, model_name: str, messages: list[dict[str, Any]]) -> None:
        log("callback.llm_start", model=model_name, message_count=len(messages))
        for index, message in enumerate(messages):
            log("callback.llm_message", index=index, message=summarize_message(message))

    def on_llm_new_token(self, token: str) -> None:
        log("callback.llm_token", token=token)

    def on_llm_end(self, response: Any) -> None:
        log("callback.llm_end", response_type=response.__class__.__name__)

    def on_reasoning_step(self, content: str) -> None:
        log("callback.reasoning", content=content)

    def on_error(self, error: Exception, context: str) -> None:
        log("callback.error", context=context, error_type=error.__class__.__name__, error=str(error))

    def on_assistant_tool_intent(self, agent_name: str, content: str, tools: list[str]) -> None:
        log("callback.assistant_tool_intent", agent=agent_name, content=content, tools=tools)


def log(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, ensure_ascii=False, default=str), flush=True)


def summarize_message(message: dict[str, Any]) -> dict[str, Any]:
    summary = dict(message)
    content = summary.get("content")
    if isinstance(content, list):
        compact = []
        for item in content:
            if item.get("type") == "image_url":
                image_url = item.get("image_url") or {}
                url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                compact.append({"type": "image_url", "url_prefix": url[:40], "length": len(url)})
            else:
                compact.append(item)
        summary["content"] = compact
    return summary


def make_memory(db_path: Path) -> MemoryService:
    return MemoryService(store=SQLiteStore(str(db_path)), log_enabled=True, max_log_length=1000)


def make_config(
    *,
    name: str,
    model: str,
    system_prompt: str,
    stream: bool = False,
    output_schema: dict[str, Any] | None = None,
    max_tool_iter: int | None = 10,
) -> AgentConfig:
    model_kwargs = {"output_schema": output_schema} if output_schema is not None else None
    return AgentConfig(
        name=name,
        system_prompt=system_prompt,
        provider="codex",
        model_name=model,
        stream=stream,
        verbose=False,
        timeout=180,
        tool_timeout=30,
        max_retries=1,
        max_tool_iter=max_tool_iter,
        model_kwargs=model_kwargs,
        callbacks=[VerboseCallback()],
    )


async def run_structured(model: str, db_path: Path) -> None:
    log("case.start", case="structured")
    memory = make_memory(db_path)
    addr = MemoryAddress(conversation_id="codex-diagnostics-structured", agent_id="structured")
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "items": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
        },
        "required": ["title", "items", "confidence"],
        "additionalProperties": False,
    }
    agent = BaseAgent(
        config=make_config(
            name="CodexStructuredDiagnostics",
            model=model,
            system_prompt="Return only data that conforms to the requested JSON schema.",
            output_schema=schema,
        ),
        memory=memory,
        memory_address=addr,
    )
    try:
        response = await agent.arun("Return a structured summary with exactly 3 items about MCP tools.")
        log("case.response", case="structured", response=response)
        validate_json_response(response, required_keys={"title", "items", "confidence"})
    finally:
        await agent.aclose()
        dump_memory(memory, addr)


async def run_image(model: str, image_path: Path, db_path: Path) -> None:
    log("case.start", case="image", image=str(image_path), image_exists=image_path.exists())
    memory = make_memory(db_path)
    addr = MemoryAddress(conversation_id="codex-diagnostics-image", agent_id="image")
    schema = {
        "type": "object",
        "properties": {
            "visible_text": {"type": "string"},
            "main_colors": {"type": "array", "items": {"type": "string"}},
            "short_description": {"type": "string"},
        },
        "required": ["visible_text", "main_colors", "short_description"],
        "additionalProperties": False,
    }
    agent = BaseAgent(
        config=make_config(
            name="CodexImageDiagnostics",
            model=model,
            system_prompt="Analyze the provided image and return JSON that matches the schema.",
            output_schema=schema,
        ),
        memory=memory,
        memory_address=addr,
    )
    try:
        response = await agent.arun(
            "Analyze this image. Mention any visible symbol or text.",
            image_path=str(image_path),
        )
        log("case.response", case="image", response=response)
        validate_json_response(response, required_keys={"visible_text", "main_colors", "short_description"})
    finally:
        await agent.aclose()
        dump_memory(memory, addr)


async def run_streaming(model: str, db_path: Path) -> None:
    log("case.start", case="streaming")
    memory = make_memory(db_path)
    addr = MemoryAddress(conversation_id="codex-diagnostics-streaming", agent_id="streaming")
    agent = BaseAgent(
        config=make_config(
            name="CodexStreamingDiagnostics",
            model=model,
            system_prompt="Answer concisely but stream enough text to observe chunks.",
            stream=True,
        ),
        memory=memory,
        memory_address=addr,
    )
    try:
        stream = await agent.arun("Write three short numbered bullets about why streaming is useful.")
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
            log("case.streaming_chunk", chunk=chunk)
        response = "".join(chunks)
        log("case.response", case="streaming", response=response, chunk_count=len(chunks))
    finally:
        await agent.aclose()
        dump_memory(memory, addr)


async def run_tool_limit(model: str, db_path: Path) -> None:
    log("case.start", case="tool-limit")
    memory = make_memory(db_path)
    addr = MemoryAddress(conversation_id="codex-diagnostics-tool-limit", agent_id="tool-limit")
    state = {"calls": 0}

    def count_tool(label: str) -> str:
        state["calls"] += 1
        return f"COUNT_TOOL_CALL_{state['calls']}: {label}"

    tool = Tool(
        schema={
            "name": "count_tool",
            "description": "Return a visible marker with the number of times this tool was called.",
            "parameters": {
                "type": "object",
                "properties": {"label": {"type": "string"}},
                "required": ["label"],
            },
        },
        func=count_tool,
    )
    agent = BaseAgent(
        config=make_config(
            name="CodexToolLimitDiagnostics",
            model=model,
            system_prompt=(
                "You must call count_tool exactly twice with labels 'first' and 'second', "
                "then explain what happened."
            ),
            max_tool_iter=1,
        ),
        memory=memory,
        memory_address=addr,
        tools=[tool],
    )
    try:
        response = await agent.arun(
            "Call count_tool twice. First use label first, then use label second."
        )
        log("case.response", case="tool-limit", response=response, actual_tool_calls=state["calls"])
    finally:
        await agent.aclose()
        dump_memory(memory, addr)


def validate_json_response(response: str, *, required_keys: set[str]) -> None:
    try:
        data = json.loads(response)
    except json.JSONDecodeError as exc:
        log("case.json_validation_failed", error=str(exc), response=response)
        return
    missing = sorted(required_keys - set(data)) if isinstance(data, dict) else sorted(required_keys)
    log("case.json_validation", ok=not missing, missing=missing, parsed=data)


def dump_memory(memory: MemoryService, addr: MemoryAddress) -> None:
    history = memory.get_history(addr)
    log("memory.history", address=addr.key_str(), message_count=len(history))
    for index, message in enumerate(history):
        log("memory.message", index=index, message=summarize_message(message))


def dump_sqlite(db_path: Path) -> None:
    if not db_path.exists():
        log("sqlite.missing", path=str(db_path))
        return
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, conversation_id, agent_id, payload FROM messages ORDER BY id ASC"
        ).fetchall()
    finally:
        conn.close()
    log("sqlite.rows", path=str(db_path), count=len(rows))
    for row_id, conversation_id, agent_id, payload_json in rows:
        payload = json.loads(payload_json)
        log(
            "sqlite.row",
            id=row_id,
            conversation_id=conversation_id,
            agent_id=agent_id,
            payload=summarize_message(payload),
        )


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run manual Codex feature diagnostics.")
    parser.add_argument("--case", choices=["all", "structured", "image", "streaming", "tool-limit"], default="all")
    parser.add_argument("--model", default="gpt-5.3-codex")
    parser.add_argument("--image", default="IMAGEN.png")
    parser.add_argument("--db", default="/tmp/opencode/agentify_codex_feature_diagnostics.db")
    parser.add_argument("--keep-db", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db)
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = ROOT / image_path
    if not args.keep_db and db_path.exists():
        db_path.unlink()

    log("diagnostics.start", case=args.case, model=args.model, db=str(db_path), image=str(image_path))

    cases = [args.case] if args.case != "all" else ["structured", "image", "streaming", "tool-limit"]
    failures = 0
    for case in cases:
        try:
            if case == "structured":
                await run_structured(args.model, db_path)
            elif case == "image":
                await run_image(args.model, image_path, db_path)
            elif case == "streaming":
                await run_streaming(args.model, db_path)
            elif case == "tool-limit":
                await run_tool_limit(args.model, db_path)
        except Exception as exc:
            failures += 1
            log("case.failed", case=case, error_type=exc.__class__.__name__, error=str(exc))

    dump_sqlite(db_path)
    log("diagnostics.finish", failures=failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
