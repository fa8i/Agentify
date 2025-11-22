from __future__ import annotations
import json
import logging
import time
from typing import Any, Dict, Generator, List, Optional, Union
import uuid
import base64
from io import BytesIO
from PIL import Image
from agentify.base_tool import Tool
from openai import RateLimitError
from agentify.client_builder import LLMClientFactory, LLMClientType

from agentify.memory.service import MemoryService
from agentify.memory.interfaces import MemoryAddress

logger = logging.getLogger(__name__)  # Production-ready logger


class BaseAgent:
    """Clase del núcleo de agentes de IA agnóstico a frameworks."""

    MAX_TOOL_ITER: int = 5
    RETRIES: int = 3

    def __init__(
        self,
        name: str,
        system_prompt: str,
        provider: str,
        model_name: str,
        *,
        memory: MemoryService,
        memory_address: Optional[MemoryAddress] = None,
        client_factory: LLMClientFactory = LLMClientFactory(),
        temperature: Optional[float] = 0.7,
        tools: Optional[List[Tool]] = None,
        client_config_override: Optional[Dict[str, Any]] = None,
        agent_timeout: Optional[int] = 60,
        stream: bool = False,
        image_processing_config: Optional[Dict[str, Any]] = {
            "max_side_px": 1024,
            "quality": 90,
            "detail": "auto",
        },
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt.strip()
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature

        self._tools: Dict[str, Tool] = {t.name: t for t in tools or []}
        self._tool_defs = [
            {"type": "function", "function": t.schema} for t in self._tools.values()
        ]

        self.stream: bool = stream

        self.client: LLMClientType = client_factory.create_client(
            provider=self.provider,
            config_override=client_config_override,
            timeout=agent_timeout,
        )

        self.memory: MemoryService = memory
        self.memory_address: Optional[MemoryAddress] = memory_address

        self.image_config: Dict[str, Any] = image_processing_config

    # Image processing helper

    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Opens an image, resizes it, compresses it, and returns it as base64.
        """
        try:
            with Image.open(image_path) as img_pil:
                if img_pil.mode not in ("RGB", "L"):
                    img_pil = img_pil.convert("RGB")

                max_side = self.image_config["max_side_px"]
                img_pil.thumbnail((max_side, max_side))

                buf = BytesIO()
                img_pil.save(
                    buf,
                    format="JPEG",
                    quality=self.image_config["quality"],
                    optimize=True,
                )
                return base64.b64encode(buf.getvalue()).decode("utf-8")

        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            raise
        except Exception as e:
            logger.error(f"Image processing error for {image_path}: {e}", exc_info=True)
            raise

    # Message building helper

    def _build_user_content(
        self,
        user_input: str,
        *,
        image_path: Optional[str] = None,
        image_detail_override: Optional[str] = None,
    ) -> Optional[Union[str, List[Dict[str, Any]]]]:
        """
        Build the `content` field of the user message supporting:
        - text only
        - image only
        - image + text (OpenAI-like multimodal list)

        Returns:
            - str -> text-only message
            - list[dict] -> multimodal content
            - None -> neither text nor image
        """
        has_text = bool(user_input and user_input.strip())
        has_image = bool(image_path)

        if not has_text and not has_image:
            return None

        if not has_image:
            return user_input

        b64_image_data = self._encode_image_to_base64(image_path)  # type: ignore[arg-type]
        detail_level = image_detail_override or self.image_config["detail"]

        parts: List[Dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_image_data}",
                    "detail": detail_level,
                },
            }
        ]
        if has_text:
            parts.append({"type": "text", "text": user_input})

        return parts

    # Memory helpers

    def _addr_or_raise(self, addr: Optional[MemoryAddress]) -> MemoryAddress:
        """
        Ensure we have a MemoryAddress to operate on (either per-call or default).
        """
        effective = addr or self.memory_address
        if effective is None:
            raise ValueError(
                "MemoryAddress requerido: pásalo en el constructor (memory_address=...) "
                "o en cada llamada (addr=...)."
            )
        return effective

    def _ensure_system_initialized(self, addr: MemoryAddress) -> None:
        """
        Ensure the system message is present exactly once at the beginning
        for the given MemoryAddress.
        """
        history = self.memory.get_history(addr)
        if not history or history[0].get("role") != "system":
            self.memory.append_history(
                addr, {"role": "system", "content": self.system_prompt}
            )

    # Public surface

    def get_history(self, addr: MemoryAddress) -> List[Dict[str, Any]]:
        """
        Return current conversation history (OpenAI message shape) for this address.
        """
        return self.memory.get_history(addr)

    @property
    def list_tools(self) -> List[str]:
        """Return the names of registered tools."""
        return list(self._tools.keys())

    def add(
        self,
        role: str,
        content: Optional[Union[str, List[Dict[str, Any]]]] = None,
        *,
        addr: Optional[MemoryAddress] = None,
        **kwargs: Any,
    ) -> None:
        """
        Append a message to memory at the provided address.

        Note:
            - kwargs may include tool_calls, tool_call_id, name (role 'tool'), etc.
            - content can be text or multimodal payload (list of parts).
        """
        a = self._addr_or_raise(addr)
        msg: Dict[str, Any] = {"role": role}
        if content is not None:
            msg["content"] = content
        msg.update(kwargs)
        self.memory.append_history(a, msg)

    def clear_memory(self, *, addr: Optional[MemoryAddress] = None) -> None:
        """
        Reset history for the provided address to the initial system prompt only.
        """
        a = self._addr_or_raise(addr)
        self.memory.reset_history(a, {"role": "system", "content": self.system_prompt})

    def save_history(self, path: str, *, addr: Optional[MemoryAddress] = None) -> None:
        """
        Persist current history to a local JSON file (export utility).
        """
        a = self._addr_or_raise(addr)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.memory.get_history(a), f, ensure_ascii=False, indent=2)

    def load_history(self, path: str, *, addr: Optional[MemoryAddress] = None) -> None:
        """
        Load a previously exported JSON history into this address (import utility).
        Replaces the entire history.
        """
        a = self._addr_or_raise(addr)
        with open(path, "r", encoding="utf-8") as f:
            raw: List[Dict[str, Any]] = json.load(f)

        if raw and raw[0].get("role") == "system":
            messages = raw
        else:
            messages = [{"role": "system", "content": self.system_prompt}] + raw

        self.memory.reset_history(a, messages[0])
        for m in messages[1:]:
            self.memory.append_history(a, m)

    def _completion(
        self, *, addr: MemoryAddress
    ) -> Union[Any, Generator[Dict[str, Any], None, None]]:
        """
        Perform the LLM call with retries and error handling.
        Always pulls messages from the external memory for the given address.
        """
        tool_choice_param = "auto" if self._tool_defs else None
        common_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": self.memory.get_history(addr),
            "temperature": self.temperature,
        }
        if self._tool_defs:
            common_params["tools"] = self._tool_defs
            common_params["tool_choice"] = tool_choice_param

        for attempt in range(self.RETRIES):
            try:
                if self.stream:
                    return self.client.chat.completions.create(
                        **common_params, stream=True
                    )
                response = self.client.chat.completions.create(
                    **common_params, stream=False
                )
                if response.choices and len(response.choices) > 0:
                    return response.choices[0].message
                raise ValueError(
                    "La respuesta de la API no contenía 'choices' válidos."
                )
            except RateLimitError:
                if attempt == self.RETRIES - 1:
                    logger.error(
                        "Límite de tasa de API alcanzado después de reintentos."
                    )
                    raise
                sleep_time = 2**attempt
                logger.warning(
                    f"Límite de tasa alcanzado. Reintentando en {sleep_time}s..."
                )
                time.sleep(sleep_time)
            except Exception as e:
                logger.error(
                    f"Error en _completion (intento {attempt + 1}/{self.RETRIES}): {e}",
                    exc_info=True,
                )
                if attempt == self.RETRIES - 1:
                    raise
                time.sleep(2**attempt)

        msg = f"LLM completions ({self.client.__class__.__name__}) failed after {self.RETRIES} retries."
        logger.critical(msg)
        raise RuntimeError(msg)

    def _split_concatenated_json_objects(self, json_string: str) -> List[str]:
        """
        Attempt to split a string that may contain multiple concatenated JSON objects.
        """
        objects_str: List[str] = []
        decoder = json.JSONDecoder()
        s = json_string.strip()
        pos = 0

        if not s:
            return []
        try:
            json.loads(s)
            return [s]  # single valid JSON
        except json.JSONDecodeError:
            pass

        while pos < len(s):
            try:
                _, consumed = decoder.raw_decode(s[pos:])
                objects_str.append(s[pos : pos + consumed])
                pos += consumed
                while pos < len(s) and s[pos].isspace():
                    pos += 1
            except json.JSONDecodeError:
                if not objects_str:
                    logger.warning(f"No se pudo decodificar JSON de: '{json_string}'")
                    return [json_string]
                logger.warning(
                    f"Agente '{self.name}': no se pudo decodificar más JSON en la posición {pos} "
                    f"de '{s}'. Objetos parseados: {len(objects_str)}."
                )
                break
        return objects_str if objects_str else [json_string]

    def _process_agent_logic(
        self,
        user_input: str,
        *,
        addr: MemoryAddress,
        image_path: Optional[str] = None,
        image_detail_override: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Internal generator that orchestrates the agent logic and the LLM interaction
        (tools, streaming/non-streaming, multi-part tool args handling).
        """
        # Ensure system prompt and build the user content (text/image).
        self._ensure_system_initialized(addr)
        user_content = self._build_user_content(
            user_input,
            image_path=image_path,
            image_detail_override=image_detail_override,
        )
        if user_content is not None:
            self.add(role="user", content=user_content, addr=addr)

        # Tool-calling loop
        for iteration_count in range(self.MAX_TOOL_ITER):
            response_or_stream = self._completion(addr=addr)

            current_turn_content_parts: List[str] = []
            assembled_tool_calls: List[Dict[str, Any]] = []

            if self.stream:
                # Streaming mode expects an iterator of deltas
                if not (
                    hasattr(response_or_stream, "__iter__")
                    and hasattr(response_or_stream, "__next__")
                ):
                    raise TypeError(
                        f"Se esperaba un iterador en modo streaming, se obtuvo {type(response_or_stream)}."
                    )
                tool_call_assembler: Dict[int, Dict[str, Any]] = {}
                for chunk in response_or_stream:  # type: ignore[union-attr]
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    if delta.content:
                        yield delta.content
                        current_turn_content_parts.append(delta.content)

                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_call_assembler:
                                tool_call_assembler[idx] = {
                                    "id": None,
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            call_data = tool_call_assembler[idx]
                            if tc_delta.id and not call_data["id"]:
                                call_data["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    call_data["function"]["name"] = (
                                        tc_delta.function.name
                                    )
                                if tc_delta.function.arguments:
                                    call_data["function"]["arguments"] += (
                                        tc_delta.function.arguments
                                    )

                for idx in sorted(tool_call_assembler.keys()):
                    call_data = tool_call_assembler[idx]
                    if not call_data.get("id"):
                        call_data["id"] = (
                            f"s_{self.provider[:3]}_tc_{iteration_count}_{idx}_{uuid.uuid4().hex[:6]}"
                        )
                    if call_data.get("function", {}).get("name"):
                        assembled_tool_calls.append(call_data)
            else:
                # Non-streaming: the SDK returns a Message-like object
                msg_object = response_or_stream
                if not (
                    hasattr(msg_object, "content") or hasattr(msg_object, "tool_calls")
                ):
                    raise TypeError(
                        f"Se esperaba un objeto Message en modo no-streaming, se obtuvo {type(msg_object)}"
                    )

                if getattr(msg_object, "content", None):
                    yield msg_object.content
                    current_turn_content_parts.append(msg_object.content)

                if getattr(msg_object, "tool_calls", None):
                    for i, tc in enumerate(msg_object.tool_calls):
                        tc_id = (
                            tc.id
                            or f"ns_{self.provider[:3]}_tc_{iteration_count}_{i}_{uuid.uuid4().hex[:6]}"
                        )
                        assembled_tool_calls.append(
                            {
                                "id": tc_id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments or "{}",
                                },
                            }
                        )

            # Expand tool calls in case the model concatenated multiple JSON args
            expanded_tool_calls: List[Dict[str, Any]] = []
            if assembled_tool_calls:
                for tc_original in assembled_tool_calls:
                    tool_name = tc_original.get("function", {}).get(
                        "name", "unknown_tool"
                    )
                    args_value = tc_original.get("function", {}).get("arguments")
                    if args_value is None:
                        args_str = ""
                    elif isinstance(args_value, str):
                        args_str = args_value
                    else:
                        args_str = json.dumps(args_value)
                    original_id = tc_original.get(
                        "id", f"gen_id_{uuid.uuid4().hex[:4]}"
                    )

                    if not args_str.strip():
                        tc_original["function"]["arguments"] = "{}"
                        expanded_tool_calls.append(tc_original)
                        continue

                    split_args_json = self._split_concatenated_json_objects(args_str)
                    if len(split_args_json) > 1:
                        for i, single_arg_json in enumerate(split_args_json):
                            expanded_tool_calls.append(
                                {
                                    "id": f"{original_id}_part_{i}",
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": single_arg_json,
                                    },
                                }
                            )
                    elif len(split_args_json) == 1:
                        tc_original["function"]["arguments"] = split_args_json[0]
                        expanded_tool_calls.append(tc_original)
                    elif not split_args_json and args_str.strip():
                        expanded_tool_calls.append(tc_original)

            assembled_tool_calls = expanded_tool_calls
            full_turn_content = "".join(current_turn_content_parts)

            # If no tool calls, finalize by appending assistant content and exit
            if not assembled_tool_calls:
                if full_turn_content:
                    self.add(role="assistant", content=full_turn_content, addr=addr)
                break

            # Append assistant message including tool_calls
            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            if full_turn_content:
                assistant_msg["content"] = full_turn_content
            assistant_msg["tool_calls"] = assembled_tool_calls
            self.add(addr=addr, **assistant_msg)

            # Execute tools and append their outputs
            for tc_to_run in assembled_tool_calls:
                tool_name = tc_to_run["function"]["name"]
                tool_call_id = tc_to_run["id"]
                tool = self._tools.get(tool_name)
                result_content: str

                if not tool:
                    result_content = json.dumps(
                        {"error": f"La herramienta '{tool_name}' no está registrada."}
                    )
                else:
                    try:
                        tool_args_val = tc_to_run["function"].get("arguments")
                        if tool_args_val is None:
                            tool_args_str = ""
                        elif isinstance(tool_args_val, str):
                            tool_args_str = tool_args_val
                        else:
                            tool_args_str = json.dumps(tool_args_val)

                        if not tool_args_str.strip():
                            tool_args_str = "{}"
                        parsed_args = json.loads(tool_args_str)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            f"Argumentos JSON inválidos para '{tool_name}': {exc}. Recibido: '{tool_args_str}'"
                        )
                        result_content = json.dumps(
                            {
                                "error": f"Argumentos JSON inválidos para '{tool_name}': {exc}. Recibido: '{tool_args_str}'"
                            }
                        )
                    except Exception as e:
                        logger.error(
                            f"Error inesperado preparando herramienta '{tool_name}': {e}",
                            exc_info=True,
                        )
                        result_content = json.dumps(
                            {
                                "error": f"Error inesperado preparando herramienta '{tool_name}': {e}"
                            }
                        )
                    else:
                        result_content = tool(**parsed_args)

                self.add(
                    role="tool",
                    content=result_content,
                    tool_call_id=tool_call_id,
                    name=tool_name,  # OpenAI expects 'name' for role 'tool'
                    addr=addr,
                )
        else:
            warn_msg = f"\n[ADVERTENCIA] Agente '{self.name}' alcanzó el máximo de {self.MAX_TOOL_ITER} iteraciones de herramientas.\n"
            logger.warning(warn_msg.strip())
            yield warn_msg

    # -------------------------
    # Public entrypoint
    # -------------------------

    def respond(
        self,
        user_input: str,
        *,
        addr: Optional[MemoryAddress] = None,
        image_path: Optional[str] = None,
        image_detail_override: Optional[str] = None,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Main entrypoint to interact with the agent.

        Supports:
        - Text only
        - Image only
        - Text + image

        Args:
        - user_input: user text (can be empty if there is only an image).
        - addr: MemoryAddress to use (if not passed, self.memory_address is used).
        - image_path: path to the image file on disk.
        - image_detail_override: optional override for the level of detail ('low', 'high', 'auto'), if supported by the model.
        """
        a = self._addr_or_raise(addr)
        response_generator = self._process_agent_logic(
            user_input,
            addr=a,
            image_path=image_path,
            image_detail_override=image_detail_override,
        )

        if self.stream:
            return response_generator

        parts: List[str] = list(response_generator)
        return "".join(parts).strip()

    # -------------------------
    # Tool registry management
    # -------------------------

    def tool_exists(self, name: str) -> bool:
        """Check whether a tool is registered."""
        return name in self._tools

    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool. Returns True if removed, False if missing."""
        if name not in self._tools:
            return False
        self._tools.pop(name)
        self._tool_defs = [d for d in self._tool_defs if d["function"]["name"] != name]
        return True

    def register_tool(self, tool: "Tool") -> None:
        """Register (or replace) a tool, keeping _tool_defs consistent."""
        if self.tool_exists(tool.name):
            self.unregister_tool(tool.name)
        self._tools[tool.name] = tool
        self._tool_defs.append({"type": "function", "function": tool.schema})
