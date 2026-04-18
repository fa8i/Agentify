# QA Checklist - Personal Assistant MVP

## Runtime Health

- [ ] `GET /health` returns 200.
- [ ] `mode` is `agentify` when provider key is present.
- [ ] `mode` falls back to `stub` when provider key is missing.

## Chat

- [ ] `POST /chat` non-stream returns `status=ok` and text response.
- [ ] `POST /chat` stream returns NDJSON chunks and final `done` event.

## Permissions

- [ ] Unknown protected action returns `status=permission_required`.
- [ ] `allow_once` via `permission_grants` enables one request.
- [ ] `allow_always` persists and is reflected in `GET /permissions`.
- [ ] `deny_always` blocks tool execution.
- [ ] Tool-level guard blocks execution even if preflight is skipped.

## Filesystem Tools

- [ ] `list_files` works inside sandbox.
- [ ] `read_file` works inside sandbox.
- [ ] `write_file` works inside sandbox.
- [ ] Access outside sandbox is denied.

## Shell Tool

- [ ] Allowed command executes (`pwd`, `date`, etc.).
- [ ] Non-allowlisted command is denied.
- [ ] Timeout handling works.

## Notes

- [ ] Note creation works.
- [ ] Note listing returns persisted records for user.

## Reminders

- [ ] Reminder create/list/cancel endpoints work.
- [ ] Due reminders transition to `triggered` status by scheduler.
- [ ] Notification attempt is emitted/logged for due reminders.

## Stability

- [ ] `python -m compileall services/agent-runtime/app` succeeds.
- [ ] Basic smoke tests pass repeatedly (no intermittent crashes).

## Packaging (F3)

- [ ] `services/agent-runtime/scripts/package_runtime.sh` creates tar.gz artifact.
- [ ] `services/agent-runtime/scripts/run_runtime.sh` starts service successfully.
- [ ] Optional `systemd --user` unit installs and starts (when enabled).
