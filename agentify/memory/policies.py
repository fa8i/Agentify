from __future__ import annotations
from typing import Callable, List, Optional
from .interfaces import ConversationStore, MemoryAddress, Message, TokenCounter


class MemoryPolicy:
    """
    Role-based sliding window, optional token budget and summarization.
    Backend-agnostic; operates via ConversationStore.
    """

    def __init__(
        self,
        store: ConversationStore,
        *,
        ttl_seconds: Optional[int] = None,
        max_user_msgs: int = 6,
        max_assistant_msgs: int = 6,
        tokenizer: Optional[TokenCounter] = None,
        max_tokens: Optional[int] = None,
        summarizer: Optional[Callable[[List[Message]], Message]] = None,
    ) -> None:
        self.store = store
        self.ttl = ttl_seconds
        self.max_user = max_user_msgs
        self.max_assistant = max_assistant_msgs
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.summarizer = summarizer

    def _apply_ttl(self, addr: MemoryAddress) -> None:
        if self.ttl:
            self.store.set_ttl(addr, self.ttl)

    def on_append(self, addr: MemoryAddress, msg: Message) -> None:
        self.store.append_message(addr, msg)
        self._apply_ttl(addr)
        self._prune_window(addr)
        self._ensure_token_budget(addr)

    def _prune_window(self, addr: MemoryAddress) -> None:
        msgs = self.store.read_messages(addr)
        if not msgs:
            return

        system = msgs[0] if msgs[0].role == "system" else None
        users = [i for i, m in enumerate(msgs) if m.role == "user"]
        assistants = [i for i, m in enumerate(msgs) if m.role == "assistant"]

        if len(users) <= self.max_user and len(assistants) <= self.max_assistant:
            return

        cutoff_user = users[-self.max_user] if len(users) > self.max_user else 0
        cutoff_ass = assistants[-self.max_assistant] if len(assistants) > self.max_assistant else 0
        cutoff = min(cutoff_user, cutoff_ass)
        if cutoff <= 0:
            return

        new_msgs = msgs[cutoff:]
        if system and (not new_msgs or new_msgs[0].role != "system"):
            new_msgs.insert(0, system)

        self.store.replace_messages(addr, new_msgs)

    def _ensure_token_budget(self, addr: MemoryAddress) -> None:
        if not (self.tokenizer and self.max_tokens):
            return
        msgs = self.store.read_messages(addr)
        oai = [m.to_openai() for m in msgs]
        if self.tokenizer(oai) <= self.max_tokens:
            return
        if self.summarizer:
            system = msgs[0] if msgs and msgs[0].role == "system" else None
            core = msgs[1:-6] if system else msgs[:-6]
            tail = msgs[-6:]
            if core:
                summary = self.summarizer(core)
                new_msgs = ([system] if system else []) + [summary] + tail
                self.store.replace_messages(addr, new_msgs)
