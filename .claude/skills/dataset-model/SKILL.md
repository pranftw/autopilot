---
name: dataset-model
description: Dataset registry, project-defined split names, and content-hashed snapshotting for experiment reproducibility. Use when working with dataset files, splits, validation, or dataset commands.
---

## Split model

Split names are **plain strings** chosen by the project (for example **`train`**, **`val`**, **`holdout`**). There is no core enum or fixed required set: `src/autopilot/core/splits.py` only documents that splits are project-defined. `DatasetEntry.split` is typed as `str`.

## Directory layout

Paths are project-aware. Under `src/autopilot/core/paths.py`, dataset files live under `datasets(workspace, project)`, which resolves to `root(workspace, project) / 'datasets'`:

- With a resolved project: `workspace/autopilot/projects/<project>/datasets/<split>/<filename>`
- With no project: `workspace/autopilot/datasets/<split>/<filename>` (same as `root` with `project=None`)

Use `paths.dataset_split(workspace, split, filename, project=...)` or equivalent helpers instead of hardcoding directory names.

## Dataset selection (no registry file)

There is no `registry.toml` and no `load_dataset_registry()`. Split filenames come from the project `Module`: the dataset CLI reads `module.dataset_profile_config`, a dict that must include `datasets['splits']` mapping each split name to a filename (for example `{'train': 'items.jsonl', 'val': 'items.jsonl'}`).

## Validation flow

`validate_dataset(workspace, profile_config)` in `src/autopilot/core/datasets.py`:

1. Read `profile_config['datasets']['splits']` (defaults to `{}` if missing)
2. For each `(split_name, filename)` pair: resolve the path via `resolve_split_path()` / `paths.dataset_split` (using the string **`split_name`** as the directory segment)
3. Validate each file exists, count lines, compute SHA-256 hash (truncated to 16 hex chars)
4. Return a list of `DatasetEntry` objects

Only splits listed under `datasets['splits']` are validated. There is no library-level **`REQUIRED_SPLITS`** or **`validate_split_name`**; if the project needs a specific set of splits present, enforce that in project code or CLI.

## AI eval datasets (separate abstraction)

Core `DatasetEntry` / snapshots track experiment filesystem layout and hashing. The core data layer provides `ListDataset` in `src/autopilot/data/dataset.py`: generic list-backed datasets with JSONL load/save. The AI module uses `StratifiedSplitter` in `src/autopilot/ai/data.py` for train/val/test-style partitions inside eval workflows. Do not conflate JSONL on disk for experiments with `ListDataset` in memory unless you are explicitly loading the same files.

## Key types

- **DatasetEntry** -- `name`, `split: str`, `path`, `format` (default `'jsonl'`), `rows` (line count), `content_hash` (16-char SHA-256)
- **DatasetSnapshot** -- `created_at` (ISO timestamp), `entries: list[DatasetEntry]`. Immutable once created.

Both have `to_dict()` / `from_dict()` for JSON serialization.

## Key files

- `src/autopilot/core/datasets.py` -- validation, snapshotting, hashing, path resolution
- `src/autopilot/core/splits.py` -- project-defined split semantics (no required-split validators in core)
- `src/autopilot/core/models.py` -- `DatasetEntry`, `DatasetSnapshot`
- `src/autopilot/core/paths.py` -- `datasets`, `dataset_split`, `root`
- `src/autopilot/data/dataset.py` -- `ListDataset`, `StreamingDataset`
- `src/autopilot/ai/data.py` -- `StratifiedSplitter`, `SlotPlanner`
- `src/autopilot/cli/commands/dataset.py` -- CLI commands (list, show, validate, split, materialize, seed)

## Gotchas

- Content hash is SHA-256 truncated to 16 hex chars, not the full digest.
- Line counting treats each `\n`-delimited line as one row. Empty trailing lines are stripped before counting.
- Split file format is inferred from the file extension (`.jsonl`, `.csv`, etc.), defaulting to `'jsonl'`.
- `validate_dataset()` only checks splits present in `profile_config['datasets']['splits']`; omitting splits does not by itself fail validation.
