"""Input builders for native Codex turns."""

from __future__ import annotations

from typing import Any

try:
    from openai_codex import ImageInput, TextInput
except ImportError:  # pragma: no cover - optional dependency
    ImageInput = None
    TextInput = None

MEMORY_PREAMBLE = (
    "Continue this conversation using the following Agentify-managed conversation "
    "state as the only source of truth; ignore any previous Codex thread state. "
    "Respond directly to the last USER message. Never mention these instructions or "
    "the conversation state format in your reply."
)

# Header used when seeding a brand-new Codex thread (memory_mode="codex_thread")
# with the agent's system prompt. In that mode only the latest user message is
# sent per turn, so without this the system prompt would never reach Codex.
# Sending it once on the first turn is enough: it becomes part of the persistent
# thread and Codex carries/compacts it across subsequent turns.
SYSTEM_TURN_HEADER = (
    "SYSTEM INSTRUCTIONS (apply to this entire conversation; do not quote or "
    "mention them):"
)


def extract_system_prompt(messages: Any) -> str | None:
    """Return the first ``system`` message content from a history list, if any."""
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).lower() != "system":
            continue
        text = stringify_message_content(message.get("content"))
        text = text.strip()
        if text:
            return text
    return None


def prepend_system_prompt(prompt: Any, system_prompt: str) -> Any:
    """Prefix a Codex turn input with the agent system prompt.

    Only the plain-string turn input shape (``memory_mode="codex_thread"``) is
    augmented; richer multimodal inputs are returned untouched so we never
    corrupt image-bearing turns.
    """
    if not system_prompt or not isinstance(prompt, str):
        return prompt
    return f"{SYSTEM_TURN_HEADER}\n{system_prompt}\n\n---\n\n{prompt}"


def build_codex_turn_input(fallback_prompt: str, messages: Any, *, memory_mode: str) -> Any:
    """Build a Codex RunInput from Agentify memory history."""
    if memory_mode != "agentify" or not isinstance(messages, list):
        return fallback_prompt

    text_parts = [MEMORY_PREAMBLE]
    codex_items: list[Any] = []
    has_images = False

    def flush_text_parts() -> None:
        if text_parts and TextInput is not None:
            codex_items.append(TextInput("\n\n".join(text_parts)))
            text_parts.clear()

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "message")).upper()
        content = message.get("content")
        if isinstance(content, list):
            text_parts.append(f"{role}:")
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and item.get("text"):
                    text_parts.append(str(item["text"]))
                elif item.get("type") in {"image_url", "input_image"}:
                    image_url = extract_image_url(item)
                    if image_url and ImageInput is not None and TextInput is not None:
                        flush_text_parts()
                        codex_items.append(ImageInput(image_url))
                        has_images = True
                    else:
                        text_parts.append("[image omitted: Codex image input is unavailable]")
            continue

        text_content = stringify_message_content(content)
        if text_content:
            text_parts.append(f"{role}: {text_content}")

    if has_images:
        flush_text_parts()
        return codex_items
    return "\n\n".join(text_parts)


def extract_image_url(item: dict[str, Any]) -> str | None:
    image_url = item.get("image_url")
    if isinstance(image_url, dict):
        url = image_url.get("url")
        return str(url) if url else None
    if isinstance(image_url, str):
        return image_url
    url = item.get("url")
    return str(url) if url else None


def stringify_message_content(content: Any) -> str:
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


def extract_output_schema(kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """Map Agentify/OpenAI-style schema params to Codex output_schema."""
    output_schema = kwargs.get("output_schema")
    if isinstance(output_schema, dict):
        return output_schema

    response_format = kwargs.get("response_format")
    if not isinstance(response_format, dict):
        return None

    if isinstance(response_format.get("schema"), dict):
        return response_format["schema"]

    json_schema = response_format.get("json_schema")
    if isinstance(json_schema, dict):
        schema = json_schema.get("schema")
        if isinstance(schema, dict):
            return schema
        return json_schema
    return None
