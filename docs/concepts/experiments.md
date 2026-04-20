# Experiments

## Manifest

Each experiment has a directory under the workspace (typically `autopilot/experiments/<slug>/`) with a canonical `manifest.json`. The manifest (`Manifest` in code) holds:

- Identity: `slug`, `title`, `profile`, `target`, `environment`
- Lifecycle: `status`, `current_phase`, `current_epoch`
- Intent: `idea`, `hypothesis`, `constraints`, `baseline`, `candidate`
- Repro: `dataset_snapshot`, `hyperparams`
- Outputs: `artifact_refs`, promotion fields (`decision`, `decision_reason`)

Loads and saves go through tracking helpers; writes are atomic.

## Events

`events.jsonl` is append-only JSON Lines. Each line is an `Event`: UTC `timestamp`, `event_type`, optional `phase` and `message`, plus a `metadata` object. Use events for audit trails and timeline reconstruction, not for replacing the manifest.

## Commands

`commands.json` stores a JSON array of `CommandRecord` entries: `timestamp`, `command`, `args`, `redacted_args` (secrets scrubbed via pattern list), `exit_code`, `duration_seconds`, `phase`. Logged invocations should stay machine-readable and safe to commit.

## Artifacts

`ArtifactRef` rows in the manifest link to files with a `kind` (`report`, `trace`, `log`, `summary`, `scorecard`, `snapshot`, `config`). Keep heavy blobs out of git; store paths or external refs the engine can resolve from the workspace.

## Scorecard

A `Scorecard` aggregates numeric `metrics`, per-metric `gates` outcomes (`pass` / `fail` / `warn` / `skip` style strings), a boolean `passed`, and a short `summary`. Built from observations via scoring rules and gate evaluation.

## Status state machine

Experiment **status** values are plain strings (for example `draft`, `prepared`, `deployed`, `train_running`). The core library does not ship a single canonical transition graph for every workspace.

When your code needs to enforce allowed moves, build a **`transitions: dict[str, list[str]]`** that maps each status to its allowed next statuses, then call:

```python
from autopilot.core.state import validate_transition

validate_transition(current, requested, transitions)
```

This raises **`StateTransitionError`** if **`requested`** is not listed under **`transitions[current]`**. The same **`transitions`** dict can be passed into **`transition_status(..., transitions=...)`** and **`evaluate_experiment_policy(..., transitions=...)`** so policy-driven moves (for example to **`human_review`**) stay consistent with your graph.

Typical paths (draft → prepared → deployed → running states, failure and review branches, terminals such as **`archived`**) are **project-defined** — document them in your overlay and keep CLI / **`Module`** behavior aligned with the graph you pass into these APIs.

## Phases

Phase names are plain strings carried on the manifest (**`current_phase`**), in **`Datum.phase`**, in events, and in **`Trainer.run_phase(phase, ...)`**. The library does not define a required ordering or a global list of valid phases. Projects choose names (for example **`preflight`**, **`deploy`**, **`train`**) and implement **`Module.forward`** (or child modules) to interpret them.
