from types import SimpleNamespace

import pytest

from agentify.llm.codex_events import (
    CodexEventCollector,
    extract_mcp_result_text,
    normalize_codex_error,
    serialize_event,
)


def _event(method, payload):
    return SimpleNamespace(method=method, payload=payload)


def test_agent_message_delta_reconstructs_final_text():
    collector = CodexEventCollector()

    collector.ingest(_event("item/agentMessage/delta", SimpleNamespace(delta="hello ")))
    collector.ingest(_event("item/agentMessage/delta", SimpleNamespace(delta="world")))
    collector.ingest(_event("turn/completed", SimpleNamespace(turn=SimpleNamespace(error=None))))

    result = collector.result()

    assert result.final_text == "hello world"
    assert result.turn_completed is True


def test_item_completed_agent_message_reconstructs_final_text():
    collector = CodexEventCollector()
    item = SimpleNamespace(root=SimpleNamespace(type="agentMessage", text="final"))

    collector.ingest(_event("item/completed", SimpleNamespace(item=item)))


    assert collector.result().final_text == "final"


def test_item_completed_mcp_result_is_captured():
    collector = CodexEventCollector()
    result = SimpleNamespace(content=[SimpleNamespace(text="ECHO_FROM_AGENTIFY: hola-agentify")])
    item = SimpleNamespace(
        root=SimpleNamespace(
            type="mcpToolCall",
            server="agentify-e2e",
            tool="echo_tool",
            status="completed",
            result=result,
        )
    )

    collector.ingest(_event("item/completed", SimpleNamespace(item=item)))

    parsed = collector.result()
    assert parsed.mcp_tool_calls[0]["tool"] == "echo_tool"
    assert parsed.mcp_tool_results == ["ECHO_FROM_AGENTIFY: hola-agentify"]
    assert parsed.final_text == "ECHO_FROM_AGENTIFY: hola-agentify"


def test_item_completed_unknown_notification_dict_mcp_result_is_captured():
    collector = CodexEventCollector()
    payload = SimpleNamespace(
        params={
            "item": {
                "type": "mcpToolCall",
                "server": "agentify-e2e",
                "tool": "echo_tool",
                "status": "completed",
                "result": {"content": [{"type": "text", "text": "ECHO_FROM_AGENTIFY: hola-agentify"}]},
            }
        }
    )

    collector.ingest(_event("item/completed", payload))

    parsed = collector.result()
    assert parsed.mcp_tool_calls[0]["tool"] == "echo_tool"
    assert parsed.mcp_tool_results == ["ECHO_FROM_AGENTIFY: hola-agentify"]


def test_error_event_with_will_retry_is_warning_not_error():
    collector = CodexEventCollector()
    payload = SimpleNamespace(
        error=SimpleNamespace(message="stream disconnected", codex_error_info=None),
        will_retry=True,
    )

    collector.ingest(_event("error", payload))

    result = collector.result()
    assert result.errors == []
    assert result.warnings == ["stream disconnected"]


def test_error_event_without_will_retry_is_fatal_and_normalized():
    collector = CodexEventCollector()
    payload = SimpleNamespace(
        error=SimpleNamespace(
            message="You've hit your usage limit.",
            codex_error_info="usageLimitExceeded",
        ),
        will_retry=False,
    )

    collector.ingest(_event("error", payload))

    result = collector.result()
    assert result.warnings == []
    assert result.errors[0].startswith("Codex usage limit exceeded")


def test_unauthorized_error_is_normalized_to_auth_message():
    error = SimpleNamespace(message="HTTP 401", codex_error_info="unauthorized")

    normalized = normalize_codex_error(error)

    assert "Codex auth is not available" in normalized
    assert "codex login" in normalized


def test_root_model_error_info_is_unwrapped():
    class FakeEnum:
        value = "usageLimitExceeded"

    error = SimpleNamespace(
        message="limit", codex_error_info=SimpleNamespace(root=FakeEnum())
    )

    assert normalize_codex_error(error).startswith("Codex usage limit exceeded")


def test_context_window_error_is_normalized():
    error = SimpleNamespace(message="too long", codex_error_info="contextWindowExceeded")

    assert normalize_codex_error(error).startswith("Codex context window exceeded")


def test_turn_completed_error_is_captured():
    collector = CodexEventCollector()

    collector.ingest(
        _event("turn/completed", SimpleNamespace(turn=SimpleNamespace(error=SimpleNamespace(message="boom"))))
    )

    assert collector.result().errors == ["boom"]


def test_usage_limit_error_is_normalized():
    error = SimpleNamespace(message="You've hit your usage limit.", codex_error_info="usageLimitExceeded")

    assert normalize_codex_error(error).startswith("Codex usage limit exceeded")


def test_unsupported_model_error_is_normalized():
    error = SimpleNamespace(message="The 'gpt-x' model is not supported with a ChatGPT account.")

    assert normalize_codex_error(error).startswith("Codex model is not supported")


def test_mcp_server_startup_failure_is_captured():
    collector = CodexEventCollector()
    payload = SimpleNamespace(
        params={"server": "agentify-e2e", "status": "failed", "message": "spawn failed"}
    )

    collector.ingest(_event("mcpServer/startupStatus/updated", payload))

    result = collector.result()
    assert result.mcp_server_statuses[0]["server"] == "agentify-e2e"
    assert result.errors[0].startswith("Codex MCP server failed to start")


def test_extract_mcp_result_text_supports_dict_content():
    assert extract_mcp_result_text({"content": [{"text": "ok"}]}) == "ok"


def test_serialize_event_includes_method_and_payload_type():
    event = _event("item/agentMessage/delta", SimpleNamespace(delta="hello"))

    serialized = serialize_event(event)

    assert serialized["method"] == "item/agentMessage/delta"
    assert serialized["payload_type"] == "SimpleNamespace"


@pytest.mark.asyncio
async def test_timeout_controlled_when_turn_does_not_complete(monkeypatch):
    from agentify.llm.codex_events import CodexEventBackend

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            import asyncio

            await asyncio.sleep(1)
            return _event("noop", SimpleNamespace())

        async def aclose(self):
            pass

    class Turn:
        def stream(self):
            return Stream()

    class Thread:
        async def turn(self, prompt):
            return Turn()

    async def fake_get_thread(session_id, model):
        return Thread()

    backend = CodexEventBackend.__new__(CodexEventBackend)
    backend._backend = SimpleNamespace(_get_or_create_thread=fake_get_thread)

    with pytest.raises(TimeoutError):
        await backend.run_turn_events(
            session_id="s",
            model="m",
            prompt="p",
            timeout=0.01,
        )
