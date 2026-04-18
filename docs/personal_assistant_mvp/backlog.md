# Backlog (Phase 1)

## Epic A - Runtime Foundation

### A1. Create runtime service skeleton
- Deliverables:
  - FastAPI app scaffold.
  - health endpoint.
  - chat endpoint contract.
- Acceptance:
  - `GET /health` returns 200.
  - `POST /chat` validates payload and returns stub response.

### A2. Integrate Agentify runtime
- Deliverables:
  - Base agent initialization and session addressing.
  - stream and non-stream chat path.
- Acceptance:
  - Non-stream chat returns text.
  - Stream chat returns chunks.

### A3. Add timeout policy
- Deliverables:
  - global response timeout.
  - tool execution timeout policy.
- Acceptance:
  - Long task returns graceful timeout error.

## Epic B - Tooling and Permissions

### B1. Implement `system_info_tool` and `notes_tool`
- Acceptance:
  - Tools callable from assistant in real conversations.

### B2. Implement `filesystem_tool` with sandbox
- Acceptance:
  - Access denied outside allowlist root.
  - Read/write inside sandbox works.

### B3. Implement `shell_safe_tool`
- Acceptance:
  - Allowlisted commands execute.
  - Non-allowlisted commands are denied.

### B4. Permission middleware
- Acceptance:
  - Runtime can emit permission challenge.
  - Decision persistence works (`allow`, `deny`).

## Epic C - Reminders and Notifications

### C1. Reminder persistence and scheduler
- Acceptance:
  - reminder fires at scheduled time.
  - cancellation and snooze work.

### C2. Linux desktop notification delivery
- Acceptance:
  - reminder triggers user-visible notification.

Status: Implemented Linux notification path via `notify-send` (best effort).

## Epic D - MCP Baseline

### D1. MCP connector management
- Acceptance:
  - enable/disable connector at runtime.

Status: Implemented baseline endpoints and in-process registry in runtime service.

### D2. MCP proxy tool
- Acceptance:
  - assistant can call enabled connector tool.
  - disabled connector returns controlled error.

Status: Implemented `mcp_proxy_tool` with connector enable/disable enforcement.

## Epic E - UI Shell

### E1. Chat UI + streaming rendering
- Acceptance:
  - User sees incremental response chunks.

Status: CLI shell implemented (`services/agent-runtime/cli/assistant_cli.py`) as interim interface.

### E2. Permission prompt UI
- Acceptance:
  - Prompt appears for `ask_every_time` tools.

### E3. Basic settings panel
- Acceptance:
  - sandbox root and connector toggles are configurable.

## Epic F - Reliability and Security

### F1. Structured logging + audit trail
- Acceptance:
  - tool actions logged with correlation ID.

Status: Implemented baseline `audit_events` store and `GET /audit/events` endpoint.

### F2. Test suite (unit + integration)
- Acceptance:
  - Coverage for policy engine, tool guards, chat path.

Status: Implemented runtime API + permission guard tests in `services/agent-runtime/tests`.

### F3. Packaging baseline
- Acceptance:
  - App starts on Linux from packaged artifact.

Status: Implemented runtime packaging script and optional `systemd --user` service template.

---

## Recommended Build Order

1. A1 -> A2 -> E1
2. B1 -> B2 -> B4
3. C1 -> C2
4. D1 -> D2
5. F1 -> F2 -> F3
