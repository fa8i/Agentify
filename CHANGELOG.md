# Changelog

All notable changes to this project will be documented in this file.

## [0.4.1] - 2026-05-30

### Added
- **Tool Hooks System**: New `tool_pre_hooks` and `tool_post_hooks` in `BaseAgent` for executing custom logic before and after tool execution.
- **SpawnAgentTool Enhancements**: Support for passing `tools`, `pre_hooks`, `post_hooks`, `tool_pre_hooks`, and `tool_post_hooks` to spawned sub-agents.
- **MCP Tool Name Validation**: New `_safe_function_name()` function to safely convert MCP tool names into valid Python identifiers.

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
