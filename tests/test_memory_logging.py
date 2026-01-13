import logging

from agentify.memory.interfaces import MemoryAddress, Message
from agentify.memory.service import MemoryService


class InMemoryStore:
    def __init__(self):
        self.data = {}

    def append_message(self, addr: MemoryAddress, msg: Message) -> None:
        key = addr.key_str()
        self.data.setdefault(key, []).append(msg)

    def read_messages(self, addr: MemoryAddress, start: int = 0, end: int = -1):
        key = addr.key_str()
        msgs = self.data.get(key, [])
        return msgs[start:] if end == -1 else msgs[start:end]

    def replace_messages(self, addr: MemoryAddress, messages):
        self.data[addr.key_str()] = list(messages)

    def delete_conversation(self, addr: MemoryAddress) -> None:
        self.data.pop(addr.key_str(), None)

    def set_ttl(self, addr: MemoryAddress, seconds: int) -> None:
        return None

    def list_conversations(self, limit: int = 100, offset: int = 0):
        return []


def test_memory_service_redacts_logs(caplog):
    store = InMemoryStore()
    service = MemoryService(store, log_enabled=True, max_log_length=5000)
    addr = MemoryAddress(conversation_id="conv:test")

    message = {
        "role": "user",
        "content": {"api_key": "secret-value", "note": "ok"},
        "metadata": {
            "tool_calls": [
                {
                    "function": {
                        "name": "do_thing",
                        "arguments": {"token": "token123"},
                    }
                }
            ]
        },
    }

    with caplog.at_level(logging.INFO):
        service.append_history(addr, message)

    assert "secret-value" not in caplog.text
    assert "token123" not in caplog.text
    assert "******" in caplog.text
