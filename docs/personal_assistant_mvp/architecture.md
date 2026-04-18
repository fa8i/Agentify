# Architecture (Phase 1)

## 1) High-Level Layout

```text
Desktop UI (Tauri/React)
        |
        | local HTTP/IPC
        v
Agent Runtime Service (FastAPI + Agentify)
        |
        +-- Tool Layer (safe filesystem, shell-safe, notes, reminders, MCP proxy)
        +-- Policy Layer (permission decisions)
        +-- Memory Layer (SQLite-backed state + Agentify memory)
        +-- Scheduler (APScheduler for reminders)
```

## 2) Component Boundaries

### 2.1 Desktop App

- Responsibilities:
  - Chat interaction and streaming rendering.
  - Permission prompt UI.
  - Settings and connector management views.
- Does not execute sensitive actions directly.

### 2.2 Agent Runtime

- Responsibilities:
  - Prompt assembly and agent execution.
  - Tool call orchestration and error shaping.
  - Async execution path (`arun`) with sync bridge where needed.

### 2.3 Tool Layer

- Each tool is explicit, typed, and permission-aware.
- All side effects pass through policy checks.

### 2.4 Policy Layer

- Decision model:
  - `ask_every_time`
  - `allow_always`
  - `deny_always`
- Decisions stored per user and tool scope.

### 2.5 Memory Layer

- Conversation memory for the agent.
- Structured app state (notes, reminders, permissions, connector state).

## 3) Data Model (Minimum)

- `sessions(id, user_id, created_at, updated_at)`
- `notes(id, user_id, title, body, tags, created_at, updated_at)`
- `reminders(id, user_id, text, due_at, status, created_at)`
- `permissions(id, user_id, tool_name, scope, decision, updated_at)`
- `audit_events(id, correlation_id, user_id, action, payload, result, ts)`

## 4) Execution Flow (Chat)

1. UI sends message to runtime with `session_id`.
2. Runtime executes `agent.arun(...)`.
3. If tool call requested:
   - policy layer checks permission
   - if `ask`: runtime returns permission challenge to UI
   - UI posts decision, runtime resumes
4. Tool executes with timeout and structured result.
5. Runtime streams response chunks back to UI.

## 5) Security Baseline

- Filesystem restricted to allowed roots.
- Shell tool restricted to command allowlist.
- Network operations explicit and logged.
- Secrets redacted in logs.
- No sudo/elevated operations in Phase 1.

## 6) Observability Baseline

- Correlation ID per request.
- Events: `agent_start`, `tool_start`, `tool_finish`, `permission_prompt`,
  `permission_decision`, `agent_finish`, `error`.
- Local JSON logs with rotation.
