# Personal Assistant Linux App (MVP)

This folder defines the implementation plan for a Linux desktop assistant app
powered by Agentify.

> **Note:** paths like `services/agent-runtime/...` referenced in `backlog.md`
> and `qa_checklist.md` belong to the separate assistant application
> repository, not to this library repo. This repo only provides the Agentify
> library (including the `provider="codex"` backend) that the assistant
> consumes. For interactive assistants on Codex, prefer
> `client_config_override={"memory_mode": "codex_thread"}` (see
> `docs/core_concepts.md`).

## Documents

- `phase1_mvp_plan.md`: Scope, goals, constraints, and delivery criteria.
- `architecture.md`: Runtime architecture and component boundaries.
- `tool_catalog.md`: Tool inventory, permissions, and safety rules.
- `backlog.md`: Execution backlog with acceptance criteria.

## MVP Intent

Deliver a production-oriented Phase 1 that is useful on day one:

- Local chat assistant with stable sync and async execution.
- Safe local tools (filesystem, shell-safe, notes, reminders).
- Basic MCP connectivity.
- Permission controls and action audit logs.
