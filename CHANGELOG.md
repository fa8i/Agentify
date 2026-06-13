# Changelog

All notable changes to this project will be documented in this file.

## [0.6.0] - 2026-06-13

### Added
- **Per-Session Runtime MCP Bridge**: Tools, timeouts, and memory-bound executors are now registered on the runtime MCP bridge per session (`register_session`), so concurrent turns for different Agentify sessions on the same Codex backend are fully isolated. The legacy single-session bridge API keeps working.
- **Durable Codex Thread Mapping**: New `client_config_override={"thread_map_path": ...}` persists the session → Codex thread ID mapping as a JSON file, so `memory_mode="codex_thread"` sessions survive process restarts (Codex threads themselves are persisted by the CLI under `~/.codex/`).
- **Codex Thread Recovery**: If a mapped Codex thread no longer exists, Agentify logs a warning and transparently starts a new thread for that session; transient resume errors are still raised so context is never silently discarded.
- **Native Thread Introspection**: `CodexThreadBackend.get_thread_id(session_id)` and `await CodexThreadBackend.read_session_history(session_id)` expose the native Codex thread state (`ThreadReadResponse`).
- **Typed Codex Errors**: New actionable exception hierarchy (`CodexAuthError`, `CodexCLINotFoundError`, `CodexModelNotSupportedError`, `CodexUsageLimitError`, `CodexMCPStartupError`, `CodexToolNotFoundError`, `CodexEmptyTurnError`, `CodexStreamTimeoutError`), all carrying retry metadata honored by `BaseAgent`.
- **MCP Startup Diagnostics**: When a Codex turn completes without reconstructible text, the error now reports whether the runtime MCP bridge ever received a connection and (best effort) which MCP servers/tools Codex can see via `mcpServerStatus/list`.
- **Real-Codex Test Suite & Benchmark**: Env-gated E2E tests against real ChatGPT OAuth (`AGENTIFY_CODEX_E2E=1`, plus `AGENTIFY_CODEX_E2E_SLOW=1` for a >30s tool regression test) and `scripts/benchmark_codex_memory_modes.py` to compare memory modes.

### Changed
- **Codex Memory Mode Guidance**: Documentation now explains how Codex thread memory works (storage under `~/.codex/`, persistence, auto-compaction) and recommends `memory_mode="codex_thread"` for interactive assistants (~1.5–1.7x faster per turn in benchmarks; the gap grows with history length).
- **Agentify Memory Preamble**: Hardened the prompt preamble used by `memory_mode="agentify"` so the model no longer leaks the memory instructions into its replies.
- **Stable Session Keys**: Native backends now receive `MemoryAddress.key_str()` as the session key instead of an unstable fallback.

### Fixed
- **Transient Codex Errors No Longer Abort Turns**: Codex `error` events are parsed from their real payload (`ErrorNotification.error` / `willRetry`); errors Codex retries internally are recorded as warnings instead of failing turns that complete successfully.
- **Slow Tools Over MCP**: The runtime MCP proxy socket timeout is now derived from `tool_timeout` (`--call-timeout`), so tools slower than 30 seconds return their structured result instead of killing the proxy connection. Validated against real Codex with a 35s tool.
- **MCP Tool Config Never Silently Dropped**: The `thread_start` compatibility fallback no longer discards MCP tool configuration on older SDKs; it raises an actionable upgrade error instead.
- **Error Classification**: `codex_error_info` root models are unwrapped; `unauthorized` (OAuth missing/expired → run `codex login`) and `contextWindowExceeded` are now classified; a missing `codex` binary raises an installation hint instead of a raw `FileNotFoundError`.

## [0.5.0] - 2026-06-02

### Added
- **Codex Provider Parity**: `provider="codex"` now behaves much closer to standard Agentify providers while using native Codex threads internally.
- **Agentify-managed Codex Memory**: Codex uses Agentify memory stores (`SQLite`, in-memory, Redis, Elasticsearch) as the source of truth by default with `memory_mode="agentify"`. Native Codex thread memory remains available via `memory_mode="codex_thread"`.
- **Runtime MCP Tool Adapter**: Normal `BaseAgent(tools=[...])` tools are automatically exposed to Codex through a runtime MCP bridge. Users no longer need to manually wrap tools for common Codex usage.
- **Persisted MCP Tool History**: Codex MCP tool calls are logged and stored in Agentify memory as assistant tool intents plus `tool` result messages with `metadata.source="codex_mcp"`.
- **Codex Multimodal Input**: `image_path=...` is converted to Codex SDK image input when supported, while preserving the Agentify multimodal memory message.
- **Codex Structured Output**: Added support for Codex `output_schema` through `model_kwargs={"output_schema": ...}` and OpenAI-style `response_format={"type": "json_schema", ...}`.
- **Codex Streaming Events**: `stream=True` now emits text chunks reconstructed from Codex `thread.turn(...).stream()` events.
- **Codex Tool Iteration Limits**: Runtime MCP tool calls now respect `AgentConfig.max_tool_iter` inside a Codex turn.
- **Provider Lifecycle Cleanup**: Added `BaseAgent.close()`, `BaseAgent.aclose()`, and `CodexThreadBackend.close()` to release provider resources such as runtime MCP bridges.
- **Codex Diagnostics Script**: Added `scripts/manual_codex_feature_diagnostics.py` for manual validation of structured output, image input, streaming, and MCP tool limits.
- **Typed Package Marker**: Added `agentify/py.typed` to match package-data configuration.

### Changed
- **Codex Tool Architecture**: Moved provider-specific Codex MCP adaptation out of `BaseAgent` into `agentify.llm.tool_adapters`.
- **Codex Backend Structure**: Split Codex input building and retry-aware error classification into dedicated modules (`codex_inputs.py`, `codex_errors.py`) for maintainability.
- **Package Extras**: `agentify-core[all]` now includes the optional Codex dependency.
- **Project Metadata**: Updated package license metadata to modern SPDX-style `license = "MIT"`.
- **Documentation**: README, PyPI README, API reference, and core concepts now document native Codex usage, login, runtime MCP tools, structured output, multimodal input, streaming, and lifecycle cleanup.

### Fixed
- **Codex Retry Behavior**: Non-retryable Codex errors such as usage limits, unsupported models, MCP startup failures, and unknown MCP tools now stop retries immediately.
- **Codex Tool Observability**: MCP tool usage is now visible in callbacks/logs and persisted memory history.
- **Codex Packaging Hygiene**: Removed reliance on private Codex SDK input imports and verified package build contents.

## [0.4.1] - 2026-05-30

### Added
- **Tool Hooks System**: New `tool_pre_hooks` and `tool_post_hooks` in `BaseAgent` for executing custom logic before and after tool execution.
- **SpawnAgentTool Enhancements**: Support for passing `tools`, `pre_hooks`, `post_hooks`, `tool_pre_hooks`, and `tool_post_hooks` to spawned sub-agents.
- **MCP Tool Name Validation**: New `_safe_function_name()` function to safely convert MCP tool names into valid Python identifiers.
- **Experimental Codex Native Provider with MCP-backed Tools**: Agentify now supports Codex as a native experimental provider using ChatGPT OAuth and Codex threads. Unlike OpenAI providers, Codex does not use `tool_calls`; Agentify exposes tools through an MCP stdio server and reconstructs Codex responses from thread event streams.

### Changed
- **MCP Adapter**: `convert_mcp_tools_to_agentify()` is now synchronous (no longer async) with improved error handling and null-coalescing for `inputSchema`.
- **Shell Safe Tool**: Removed restrictive allowlist; now accepts any shell command with configurable timeout (increased from 60s to 1800s max).
- **System Message Integrity**: Enhanced `_aensure_system_initialized()` to verify system message position and eliminate duplicates in conversation history.
- **Type Hints Modernization**: Updated type aliases to use modern `TypeAlias` syntax and PEP 604 union operator (`|`) instead of `Union`.
- **LLM Client Factory**: Added support for new provider "llama" alongside existing providers.

### Fixed
- **Event Loop Detection**: Simplified `has_running_loop()` to use `asyncio.get_running_loop()`, eliminating DeprecationWarning on Python 3.10+.
- **MCP Tests**: Fixed async/await mismatch in test cases for `convert_mcp_tools_to_agentify()`.
- **Tool Execution Rollback**: New `_arollback_last_tool_turn()` method to recover from interrupted tool execution sequences.

## [0.4.0] - 2026-04-18

### Added
- **Local Provider Support**: Official support for local LLM servers (LM Studio, Ollama, etc.) via the `"local"` provider.
  - Automatically configured for LM Studio's default port (`http://localhost:1234/v1`).
  - Supports custom tools, streaming, and vision on local models.
  - New configuration environment variables: `LOCAL_API_BASE` and `LOCAL_API_KEY`.
- Dual execution bridge via `run()` (sync) and `arun()` (async) across agents and multi-agent runtimes.
- `agentify/core/sync_bridge.py` with loop-safety checks, sync coroutine execution, and async-to-sync streaming bridge.
- Delegation recovery controls in `AgentConfig`:
  - `delegation_recovery_enabled`
  - `delegation_recovery_mode`
  - `delegation_max_retries`
  - `delegation_retry_backoff_ms`
- `tool_timeout` in `AgentConfig` to control tool execution timeout independently from model timeout.
- `agentify/core/multimodal.py` to centralize image encoding and multimodal content building.

### Changed
- **Lazy Client Loading**: Improved performance by making the synchronous client instantiation lazy in `BaseAgent`. The client is only created when first accessed, reducing overhead for async-first workflows.
- Refactored `BaseAgent` to use `arun()` as the execution source of truth; `run()` now bridges to async runtime.
- Increased default `timeout` from `60` to `300` seconds for long-running reasoning scenarios.
- Set default `tool_timeout` to `300` seconds.
- Improved delegated tool-call recovery for consistency errors to reduce user-facing failures in concurrent multi-agent flows.
- Reduced verbosity/noise for recoverable consistency errors in callback logging.
- Updated docs and examples to consistently document dual API usage (`run()`/`arun()`).

## [0.3.1] - 2026-03-28
### Removed
- Removed legacy internal sync execution paths in `BaseAgent` that were no longer part of the active runtime.
