# Phase 1 MVP Plan

## 1) Product Objective

Build a Linux desktop assistant that can reliably help with daily local tasks,
while keeping user control and security first.

The assistant must be usable from a desktop UI and powered by Agentify as the
runtime orchestration layer.

## 2) Target Users

- Individual developers and power users on Linux.
- Users who want a local-first personal assistant with extensibility via tools
  and MCP.

## 3) MVP Scope (In)

- Chat UI with streaming responses.
- Agent runtime service (Python + Agentify).
- Core tools:
  - system info
  - filesystem sandbox
  - safe shell commands
  - notes
  - reminders
- Basic MCP connector support (1-2 connectors).
- Permission model per tool call (`ask`, `allow`, `deny`).
- Structured local logging and action audit trail.

## 4) Out of Scope (Phase 1)

- Voice input/output (STT/TTS).
- Autonomous background agents with elevated privileges.
- Cloud sync and account system.
- Plugin marketplace.

## 5) Core Non-Functional Requirements

- Local-first by default.
- No destructive action without explicit permission.
- Timeouts for agent response and tool execution.
- Deterministic auditability of tool actions.
- Clear error UX (no raw stack traces in main UI).

## 6) Proposed Stack

- Desktop UI: Tauri + React + TypeScript.
- Runtime service: Python 3.12 + FastAPI + Agentify.
- Data: SQLite (app state, reminders, notes, permissions).
- Scheduler: APScheduler.
- Notifications: Linux desktop notifications (DBus/`notify-send`).

## 7) Runtime Contracts

### 7.1 Chat API

- `POST /chat`
  - input: `message`, `session_id`, `user_id`, optional `stream`
  - output: plain response or stream chunks

### 7.2 Permissions API

- `GET /permissions`
- `POST /permissions/decision`
  - decisions: `allow_once`, `allow_always`, `deny`

### 7.3 Reminders API

- `POST /reminders`
- `GET /reminders`
- `POST /reminders/{id}/cancel`

### 7.4 MCP API (minimum)

- `GET /mcp/connectors`
- `POST /mcp/connectors/{id}/enable`
- `POST /mcp/connectors/{id}/disable`

## 8) Definition of Done (Phase 1)

- User can open the desktop app and chat with the assistant.
- At least 5 tool workflows function end-to-end with permissions.
- Reminder scheduling and desktop notifications work.
- One MCP connector can be enabled and used from the assistant.
- Logs are queryable locally and include action correlation IDs.
- Security checks prevent unapproved sensitive actions.

## 9) Risks and Mitigations

- Risk: Tool misuse by prompt injection.
  - Mitigation: strict permission gate + path/command allowlists.
- Risk: Event loop and concurrency regressions.
  - Mitigation: use Agentify `arun()` source of truth and timeout guards.
- Risk: Poor UX from permission spam.
  - Mitigation: policy memory (`allow_always`) and grouped prompts.

## 10) Incremental Delivery Plan

1. Runtime skeleton + chat endpoint + UI shell.
2. Tooling layer with permission middleware.
3. Reminder scheduler + notification delivery.
4. MCP connector integration.
5. Hardening, tests, packaging.
