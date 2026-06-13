"""Benchmark Codex memory modes: agentify vs codex_thread.

Runs the same multi-turn conversation in both modes against the real Codex
CLI (requires `codex login`) and prints per-turn wall-clock latency, so the
cost of ephemeral threads + full-history resend (and per-turn MCP restarts
when tools are attached) can be measured directly.

Usage:
    .venv/bin/python scripts/benchmark_codex_memory_modes.py --turns 4
    .venv/bin/python scripts/benchmark_codex_memory_modes.py --turns 4 --with-tool
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

from agentify.core.agent import BaseAgent
from agentify.core.config import AgentConfig
from agentify.core.tool import Tool
from agentify.memory.interfaces import MemoryAddress
from agentify.memory.service import MemoryService
from agentify.memory.stores.in_memory_store import InMemoryStore


def _echo_tool() -> Tool:
    return Tool(
        schema={
            "name": "echo_tool",
            "description": "Echo back the given text with a marker prefix.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
        func=lambda text: f"ECHO: {text}",
    )


def _turn_prompts(turns: int, with_tool: bool) -> list[str]:
    prompts = []
    for i in range(turns):
        if with_tool:
            prompts.append(
                f'Call echo_tool with text "ping-{i}" and reply with only its output.'
            )
        else:
            prompts.append(
                f"This is message number {i}. Reply with one short sentence that "
                f"mentions the number {i} and the very first number I sent you."
            )
    return prompts


async def _run_mode(model: str, memory_mode: str, turns: int, with_tool: bool) -> list[float]:
    addr = MemoryAddress(conversation_id=f"codex-bench-{memory_mode}-{int(time.time())}")
    agent = BaseAgent(
        config=AgentConfig(
            name="CodexBench",
            system_prompt="Answer concisely in one short sentence.",
            provider="codex",
            model_name=model,
            verbose=False,
            client_config_override={"memory_mode": memory_mode},
        ),
        memory=MemoryService(store=InMemoryStore(), log_enabled=False),
        memory_address=addr,
        tools=[_echo_tool()] if with_tool else [],
    )
    timings: list[float] = []
    try:
        for prompt in _turn_prompts(turns, with_tool):
            started = time.monotonic()
            answer = await agent.arun(prompt, addr=addr)
            elapsed = time.monotonic() - started
            timings.append(elapsed)
            print(f"  [{memory_mode}] turn {len(timings)}: {elapsed:6.2f}s  -> {answer[:70]!r}")
    finally:
        await agent.aclose()
    return timings


def _summary(label: str, timings: list[float]) -> str:
    return (
        f"{label:>14}: total {sum(timings):6.2f}s | mean {statistics.mean(timings):6.2f}s | "
        f"first {timings[0]:6.2f}s | last {timings[-1]:6.2f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--with-tool", action="store_true")
    args = parser.parse_args()

    print(f"Model: {args.model} | turns: {args.turns} | tool: {args.with_tool}\n")
    results = {}
    for mode in ("agentify", "codex_thread"):
        print(f"Running memory_mode={mode} ...")
        results[mode] = asyncio.run(_run_mode(args.model, mode, args.turns, args.with_tool))
        print()

    print("Summary:")
    for mode, timings in results.items():
        print(_summary(mode, timings))
    speedup = sum(results["agentify"]) / max(sum(results["codex_thread"]), 1e-9)
    print(f"\ncodex_thread total speedup vs agentify: {speedup:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
