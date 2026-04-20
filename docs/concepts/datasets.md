# Datasets

## Core data layer

Generic dataset types live in `autopilot.data.dataset`:

- **`ListDataset[T]`** (`autopilot.data.dataset`) — a map-style dataset backed by an in-memory list. Load from JSON Lines with `ListDataset.from_jsonl(path, item_type)`, write back with `to_jsonl`, and take subsets by index.
- **`StreamingDataset[T]`** (`autopilot.data.dataset`) — lazily reads Pydantic model instances line-by-line from a JSONL file.

## AI module data model

AI-specific composition lives in `autopilot.ai.data`:

Supporting helpers:

- **`StratifiedSplitter`** — given a `ratios` dict whose keys are split name strings (for example `'train'`, `'val'`, `'test'`), a stratification key function, and a seed, produces a mapping from each split name to a `ListDataset` with items tagged to that split.
- **`SlotPlanner`** — builds weighted slot dicts from `VarDef` metadata for generators that need structured placeholders before items exist.

Projects construct these objects in code and pass them into generators, trainers, or CLI layers; there is no central string-to-class lookup in the library.

## Splits

Split names are **plain strings**. Conventional names and their roles:

| Split | Purpose |
| --- | --- |
| `train` | Primary training / iteration data |
| `train_fast` | Optional smaller subset for fast inner loops |
| `val` | Candidate selection and gating |
| `test` | Holdout; use sparingly |

A complete formal workflow typically uses `train`, `val`, and `test`. `train_fast` is optional. `StratifiedSplitter` and your own loaders decide which names appear; invalid or empty ratios should fail fast at construction or split time.

## Validation and files

Place or generate JSON Lines (or other formats your `DataItem` models parse) where your project expects them. `ListDataset.from_jsonl` validates each line against your Pydantic item type. Additional filesystem checks belong in project-specific  implementations or commands.

## Snapshotting

Experiment reproducibility still relies on recording **what** data was used. Build a `DatasetSnapshot` (see `autopilot.core.models`) with `DatasetEntry` rows (paths, format, row counts, content hashes) and store the resulting structure on the manifest under `dataset_snapshot` so runs remain tied to the exact files that were validated. The AI module produces the datasets; the tracking layer persists the snapshot dict on the experiment record.

## Seeding

The engine does not invent benchmark data. Generate or copy split files, run `StratifiedSplitter` with explicit ratios and a fixed seed when you need deterministic train/val/test partitions, and keep slot generation seeded via `SlotPlanner` for repeatable eval construction.
