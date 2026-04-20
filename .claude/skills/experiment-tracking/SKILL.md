---
name: experiment-tracking
description: Experiment manifest persistence, append-only event logging, and command history with redaction. Use when modifying how experiment state is stored, events are recorded, or commands are logged.
---

## Manifest (single source of truth)

File: `{experiment_dir}/manifest.json`

The manifest is the canonical record of an experiment's current state. It contains slug, title, profile, target, environment, **status** (plain string, e.g. `draft`, `deployed`, `promotable` — not an enum), current_phase, current_epoch, idea, hypothesis, constraints, baseline, candidate, dataset_snapshot, hyperparams, decision, and decision_reason.

### Atomic writes

All manifest writes use the tmp-then-replace pattern in `src/autopilot/tracking/manifest.py`:

1. Serialize to JSON (2-space indent, no key sorting, UTF-8)
2. Write to `manifest.json.tmp`
3. Atomic replace via `tmp.replace(path)`
4. On serialization failure: delete tmp, raise `TrackingError`

Never write manifest non-atomically. Never use `open().write()` directly.

### Key functions

- `load_manifest(experiment_dir) -> Manifest`
- `save_manifest(experiment_dir, manifest)`
- `update_manifest_status(experiment_dir, status)` -- validates transition first
- `update_manifest_phase(experiment_dir, phase, epoch)`

## Event log (append-only)

File: `{experiment_dir}/events.jsonl`

One JSON object per line, appended via `append_event()`. Never rewrite the file.

Event types: `experiment_created`, `phase_started`, `phase_completed`, `phase_failed`, `status_transition`, `candidate_captured`, `policy_evaluated`

Each event has: `timestamp` (ISO UTC), `event_type`, `phase`, `message`, `metadata` dict.

## Command log (with redaction)

File: `{experiment_dir}/commands.json`

Full JSON array, rewritten on each append (not append-only like events).

### Redaction

`DEFAULT_REDACT_PATTERNS` in `src/autopilot/tracking/commands.py`:
```
token, secret, password, key, auth, cookie
```

Any argument containing these substrings (case-insensitive) is replaced with `[REDACTED]` in `redacted_args`. The original `args` field is also stored but should never be logged externally.

### CommandRecord fields

`timestamp`, `command`, `args`, `redacted_args`, `exit_code`, `duration_seconds`, `phase`

## Key files

- `src/autopilot/tracking/manifest.py` -- load/save/update manifest with atomic writes
- `src/autopilot/tracking/events.py` -- append-only JSONL event log
- `src/autopilot/tracking/commands.py` -- command history with redaction
- `src/autopilot/core/models.py` -- `Manifest`, `Event`, `CommandRecord` dataclasses

## Gotchas

- Manifest writes must always go through `save_manifest()` for atomicity. Direct file writes will corrupt state on failure.
- Events are append-only. Never truncate, rewrite, or sort `events.jsonl`.
- Commands log rewrites the full array each time (not append-only). This is intentional -- it uses atomic tmp+replace.
- Never log unredacted command args to external systems. Use `redacted_args` for any user-visible output.
- JSON format: 2-space indent, `sort_keys=False`, UTF-8 encoding. This is consistent across all tracking files.
