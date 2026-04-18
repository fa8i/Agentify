import gradio as gr
from agentify.memory.stores.in_memory_store import InMemoryStore  # noqa: F401
from agentify.memory.stores.redis_store import RedisStore  # noqa: F401
from agentify.memory import MemoryPolicy, MemoryService, MemoryAddress
from agentify.core import BaseAgent, AgentConfig
from agentify.llm import LLMClientFactory
from agentify.extensions.prompts import assistant_prompt
from agentify.extensions.tools import (
    TimeTool,
    CalculatorTool,
    WeatherTool,
)

BUILTIN_TOOLS_REGISTRY = {
    "get_current_time": TimeTool(),
    "calculate_expression": CalculatorTool(),
    "get_weather": WeatherTool(),
}

PROVIDERS = ["azure", "openai", "deepseek", "gemini", "anthropic", "local"]

PROVIDER_MODELS = {
    "azure": ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5", "gpt-5-mini"],
    "openai": ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5", "gpt-5-mini"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
    "anthropic": [
        "claude-opus-4-1-20250805",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "cclaude-3-7-sonnet-20250219",
        "claude-3-5-haiku-20241022",
        "claude-3-haiku-20240307",
    ],
    "local": [
        "google/gemma-4-e4b", 
        "google/gemma-4-e2b", 
        "google/gemma-4-26b-a4b", 
        "qwen/qwen3.5-9b", 
        "qwen/qwen3.6-35b-a3b",
        "openai/gpt-oss-20b",
    ]
}

# store = RedisStore(url="redis://localhost:6379/0")
store = InMemoryStore()
policy = MemoryPolicy(store, ttl_seconds=None, max_user_msgs=6, max_assistant_msgs=6)
memory = MemoryService(store, policy)


def create_agent_instance(
    name_val,
    provider_val,
    model_val,
    temperature_val,
    timeout_val,
    stream_val,
    selected_tools_val,
    local_base_url_val=None,
):
    """Create a new agent instance with the specified parameters."""
    tools_val = selected_tools_val or []
    tools = [BUILTIN_TOOLS_REGISTRY[t] for t in tools_val if t in BUILTIN_TOOLS_REGISTRY]
    config = AgentConfig(
        name=name_val or "GradioAgent",
        system_prompt=assistant_prompt,
        provider=provider_val,
        model_name=model_val,
        temperature=temperature_val,
        timeout=timeout_val,
        stream=stream_val,
    )
    
    if provider_val == "local" and local_base_url_val:
        config.client_config_override = {"base_url": local_base_url_val}
    
    # Build a fresh MemoryAddress for this agent instance so the logs show the agent's name.
    agent_addr = MemoryAddress(
        api_version="",
        tenant_id="",
        user_id="",
        conversation_id="",
        agent_id=name_val or "agent",
    )
    return BaseAgent(
        config=config,
        tools=tools,
        client_factory=LLMClientFactory(),
        memory=memory,
        memory_address=agent_addr,
    )


def stream_response_to_chatbot(
    message, agent_instance, chat_history_list, image_path=None
):
    """Handle response streaming for the chatbot (optionally with an image)."""
    chat_history_list = chat_history_list or []

    if image_path:
        msg_user_img = gr.ChatMessage(
            role="user",
            content=gr.Image(
                value=image_path,
                show_download_button=False,
                show_fullscreen_button=False,
            ),
        )
        chat_history_list.append(msg_user_img)

    if message and message.strip():
        msg_user_text = gr.ChatMessage(
            role="user",
            content=message,
        )
        chat_history_list.append(msg_user_text)

    placeholder = gr.ChatMessage(
        role="assistant",
        content="⏳ Generating response…",
    )
    chat_history_list.append(placeholder)

    yield chat_history_list

    try:
        response_stream = agent_instance.run(message, image_path=image_path)
    except TypeError:
        response_stream = agent_instance.run(message)

    if isinstance(response_stream, str):
        placeholder.content = response_stream
        yield chat_history_list
    else:
        full_response = ""
        for chunk in response_stream:
            full_response += chunk
            placeholder.content = full_response
            yield chat_history_list


def clear_agent_memory(agent_instance):
    """Clear the agent's memory."""
    if agent_instance:
        agent_instance.clear_memory()


def update_model_dropdown(provider):
    """Update model dropdown options based on the selected provider."""
    models = PROVIDER_MODELS.get(provider, ["gpt-4.1-nano"])
    return gr.Dropdown(choices=models, value=models[0])


def build_interface():
    with gr.Blocks(fill_height=True, fill_width=True, title="Agentify Chatbot") as demo:
        # Agent state
        agent_state = gr.State()

        # Sidebar with settings
        with gr.Sidebar():
            gr.Markdown("## Model Settings:")

            name_input = gr.Textbox(
                label="Agent Name", placeholder="Enter agent name", value="Agentify"
            )

            # Model configuration
            provider_input = gr.Dropdown(
                label="Provider", choices=PROVIDERS, value="openai"
            )

            model_input = gr.Dropdown(
                label="Model", choices=PROVIDER_MODELS["openai"], value="gpt-4.1-mini"
            )

            temperature_slider = gr.Slider(
                minimum=0, maximum=1, value=0.5, step=0.05, label="Temperature"
            )

            local_base_url = gr.Textbox(
                label="Local Base URL",
                value="http://localhost:1234/v1",
                placeholder="http://localhost:1234/v1",
                visible=False,
            )

            gr.Markdown("## Tools & Advanced:")

            # Available tools
            tools_checkbox_group = gr.CheckboxGroup(
                label="Available Tools",
                choices=list(BUILTIN_TOOLS_REGISTRY.keys()),
                value=["get_current_time", "calculate_expression", "get_weather"],
            )

            # Advanced configuration
            timeout_number = gr.Number(
                label="Timeout (seconds)", value=60, minimum=1, maximum=300
            )

            stream_checkbox = gr.Checkbox(label="Stream Responses", value=True)

            # Control buttons
            rebuild_button = gr.Button("Create/Reset Agent", variant="primary")

            clear_button = gr.Button("Clear Conversation", variant="secondary")

            # Agent status
            gr.Markdown("## Agent Status:")
            agent_status = gr.Textbox(
                label="Status", value="Not Initialized", interactive=False
            )

            active_tools = gr.Textbox(
                label="Active Tools", value="None", interactive=False
            )

        # Main area
        gr.Markdown("<div style='text-align: center;'><h1>Agentify Chatbot</h1></div>")

        chatbot_display = gr.Chatbot(
            scale=1,
            group_consecutive_messages=True,
            height=500,
            type="messages",
            render_markdown=True,
        )

        message_input = gr.MultimodalTextbox(
            show_label=False,
            placeholder="Type your message here or upload an image...",
            submit_btn=True,
            interactive=True,
            file_types=["image"],
            file_count="single",
            max_plain_text_length=100000,
        )

        # --- Event Handlers ---

        def agent_creation_logic(
            name_val,
            provider_val,
            model_val,
            temp_val,
            time_val,
            stream_val,
            tools_list_val,
            local_base_url_val,
        ):
            """Create an agent and update its state."""
            try:
                agent = create_agent_instance(
                    name_val,
                    provider_val,
                    model_val,
                    temp_val,
                    time_val,
                    stream_val,
                    tools_list_val,
                    local_base_url_val,
                )
                status = f"Active - {provider_val}/{model_val}"
                tools_text = ", ".join(tools_list_val) if tools_list_val else "None"
                return agent, status, tools_text
            except Exception as e:
                return None, f"Error: {str(e)}", "None"

        def handle_rebuild_click(
            name_val,
            provider_val,
            model_val,
            temp_val,
            time_val,
            stream_val,
            tools_list_val,
            local_base_url_val,
        ):
            """Handle rebuild button click."""
            new_agent, status, tools_text = agent_creation_logic(
                name_val,
                provider_val,
                model_val,
                temp_val,
                time_val,
                stream_val,
                tools_list_val,
                local_base_url_val,
            )
            clear_agent_memory(new_agent)
            return new_agent, [], {"text": "", "files": []}, status, tools_text

        def handle_send_message(inputs, chat_hist, current_agent):
            """Handle message sending with optional image from MultimodalTextbox."""
            inputs = inputs or {}
            text = inputs.get("text", "") or ""
            files = inputs.get("files", []) or []

            image_path_for_agent = files[0] if files else None

            if not text.strip() and not image_path_for_agent:
                yield chat_hist or []
                return

            if current_agent is None:
                error_msg = "Error: Agent not initialized. Please click 'Create/Reset Agent' first."
                chat_history = chat_hist or []

                if image_path_for_agent:
                    chat_history.append(
                        gr.ChatMessage(
                            role="user",
                            content=gr.Image(
                                value=image_path_for_agent,
                                show_download_button=False,
                                show_fullscreen_button=False,
                            ),
                        )
                    )
                if text.strip():
                    chat_history.append(
                        gr.ChatMessage(
                            role="user",
                            content=text,
                        )
                    )

                chat_history.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=error_msg,
                    )
                )
                yield chat_history
                return

            yield from stream_response_to_chatbot(
                text,
                current_agent,
                chat_hist,
                image_path=image_path_for_agent,
            )

        def handle_clear_conversation(current_agent):
            """Handle clearing the conversation."""
            clear_agent_memory(current_agent)
            return [], {"text": "", "files": []}

        def update_provider_change(provider):
            """Update the model when provider changes."""
            return update_model_dropdown(provider)

        # --- Event Connections ---

        provider_input.change(
            update_provider_change, inputs=[provider_input], outputs=[model_input]
        )

        def toggle_local_url(provider):
            return gr.update(visible=(provider == "local"))

        provider_input.change(
            toggle_local_url, inputs=[provider_input], outputs=[local_base_url]
        )

        demo.load(
            agent_creation_logic,
            inputs=[
                name_input,
                provider_input,
                model_input,
                temperature_slider,
                timeout_number,
                stream_checkbox,
                tools_checkbox_group,
                local_base_url,
            ],
            outputs=[agent_state, agent_status, active_tools],
        )

        rebuild_button.click(
            handle_rebuild_click,
            inputs=[
                name_input,
                provider_input,
                model_input,
                temperature_slider,
                timeout_number,
                stream_checkbox,
                tools_checkbox_group,
                local_base_url,
            ],
            outputs=[
                agent_state,
                chatbot_display,
                message_input,
                agent_status,
                active_tools,
            ],
        )

        message_input.submit(
            handle_send_message,
            inputs=[message_input, chatbot_display, agent_state],
            outputs=[chatbot_display],
        ).then(lambda: {"text": "", "files": []}, outputs=[message_input])

        clear_button.click(
            handle_clear_conversation,
            inputs=[agent_state],
            outputs=[chatbot_display, message_input],
        )

    return demo


if __name__ == "__main__":
    interface = build_interface()
    interface.launch(share=False, inbrowser=True, show_error=True)
