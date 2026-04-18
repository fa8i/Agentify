# Tool Catalog (Phase 1)

## 1) `system_info_tool`

- Purpose: Provide machine context (OS, CPU, memory, disk).
- Risk: Low.
- Permission default: `allow_always`.
- Timeout: 5s.

## 2) `filesystem_tool`

- Purpose: Read/write/list files under sandboxed roots.
- Operations:
  - `list_dir(path)`
  - `read_file(path)`
  - `write_file(path, content)`
- Risk: Medium.
- Permission default:
  - read: `ask_every_time`
  - write: `ask_every_time`
- Guardrails:
  - path normalization and allowlist roots only.
  - max bytes for reads/writes.
- Timeout: 10s.

Implemented in Agentify extensions as:

- `list_files` (`ListDirTool`)
- `read_file` (`ReadFileTool`)
- `write_file` (`WriteFileTool`)

## 3) `shell_safe_tool`

- Purpose: Execute limited local commands for productivity.
- Allowed commands (initial):
  - `pwd`, `ls`, `whoami`, `date`, `uname -a`, `git status`
- Risk: High if unrestricted.
- Permission default: `ask_every_time`.
- Guardrails:
  - strict allowlist, no shell interpolation, no pipes/redirection in MVP.
  - hard timeout and output size cap.
- Timeout: 15s.

Implemented in Agentify extensions as:

- `run_safe_command` (`ShellSafeTool`)

## 6) `mcp_proxy_tool`

- Purpose: Invoke enabled MCP connectors from the assistant.
- Risk: Variable by connector.
- Permission default: `ask_every_time`.
- Guardrails:
  - disabled connectors return controlled errors.
  - connector invocation is explicit (`connector_id`, `action`, `payload`).

## 4) `notes_tool`

- Purpose: Create/search/update notes in local SQLite.
- Operations:
  - `create_note`, `update_note`, `search_notes`, `list_notes`
- Risk: Low.
- Permission default: `allow_always`.
- Timeout: 5s.

## 5) `reminder_tool`

- Purpose: Manage reminders and schedules.
- Operations:
  - `create_reminder`, `list_reminders`, `cancel_reminder`, `snooze_reminder`
- Risk: Low.
- Permission default: `allow_always`.
- Timeout: 5s.

Implemented in runtime toolset as:

- `reminder_create_tool`
- `reminder_list_tool`
- `reminder_cancel_tool`

## 6) `mcp_proxy_tool`

- Purpose: Call MCP-provided tools through a controlled adapter.
- Risk: Variable by connector.
- Permission default: `ask_every_time`.
- Guardrails:
  - connector-level enable/disable.
  - per-connector timeout.
  - denylist for unsafe MCP tool names.

## Common Tool Policies

- Tool output format must be deterministic JSON where possible.
- All tool calls are logged to `audit_events` with correlation ID.
- Failures return user-safe errors and retain technical details in logs.
