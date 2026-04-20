---
name: framework-architecture
description: Core architecture including type system, config resolution, Module tree, Trainer/loop integration, and orchestration services. Use when understanding overall system structure or modifying core behavior.
---

## Module layout

```
autopilot.cli          -- CLI entry: Command/CLI (command.py), commands, context, output, main
autopilot.core         -- models, services, normalization, metric, loss, optimizer, module, parameter, trainer, loops, callbacks, store, store_callbacks, graph
autopilot.data         -- Dataset / IterableDataset / ConcatDataset, DataLoader, DataModule (torch.utils.data-shaped)
autopilot.tracking     -- manifest, events, commands persistence
autopilot.policy       -- policy protocols, gate evaluation, result helpers
autopilot.ai           -- generators, judges, runtime, PathParameter, FileStore, Agent, ClaudeCodeAgent, JudgeLoss, AgentOptimizer
```

The legacy `autopilot.adapters` package exists but new code should compose from **`Module`** subclasses and project code.

No `__init__.py` files (implicit namespace packages). Import from the exact module:
`from autopilot.core.models import Manifest` -- not from package re-exports.

**CLI entry:** stock binary uses **`AutoPilotCLI()()`** (`cli/main.py`). Projects subclass **`CLI`** / **`AutoPilotCLI`** with **`project='...'`**, assign **`Command`** trees and **`module` / `generator` / `judge`** in **`__init__`**, then **`MyCLI()()`** from **`cli.py`**.

## Core enums

**GateResult**: `pass`, `fail`, `warn`, `skip`

Split names and workflow strings are generally **plain `str` values** on models and manifests unless a project introduces its own enums. Status graphs, when used, are **project-defined** (metadata, overlay helpers, or command layers — not a single global core registry).

## Core dataclasses

All in `src/autopilot/core/models.py`, all have `to_dict()` / `from_dict()`:

- **Manifest** -- experiment state (slug, title, current_epoch, idea, hypothesis, hyperparams, decision, decision_reason, metadata)
- **Event** -- timestamp, event_type, phase, message, metadata
- **CommandRecord** -- command with redacted_args, exit_code, duration
- **Datum** -- normalized execution result (phase, split, epoch, metrics, metadata, success, error_message, nested items, eval fields)
- **Result** -- metrics dict, gates dict, passed bool, summary string
- **DatasetEntry** / **DatasetSnapshot** -- dataset file metadata with content hash
- **HyperparamSet** -- version, values, schema, locked

## Module tree (`core/module.py`)

**`Module`** follows the **`nn.Module`** pattern:

- Internal registries: **`_modules`**, **`_parameters`** (dicts keyed by attribute name).
- **`__setattr__`** registers **`Parameter` -> `_parameters`** and child **`Module` -> `_modules`** only. **`Metric(Module)`** and **`Loss(Module)`** are normal child modules (no separate **`_metrics`** registry).
- **`forward(*args, **kwargs) -> Datum`** -- override for computation; invoked via **`__call__`** (with optional autograd graph recording when grad mode is on).
- Tree API: **`children()`**, **`named_children()`**, **`modules()`**, **`named_modules()`**, **`parameters()`**, **`named_parameters()`**.
- **`train()`** / **`eval()`** toggle **`training`** and recurse to children (like **`nn.Module.train`** / **`eval`**).
- **`apply(fn)`**, **`state_dict()`**, **`load_state_dict()`** for checkpoint-style state.

**`AutoPilotModule(Module)`** extends **`Module`** like **`LightningModule`**: adds **`training_step`**, **`validation_step`**, **`test_step`**, **`configure_optimizers`**, optional **`trainer`** reference (set by **`Trainer.fit()`**), and lifecycle hooks such as **`setup`**, **`teardown`**, **`on_train_start`**, **`on_train_end`**, **`on_validation_start`**, **`on_validation_end`**.

## Parameter (`core/parameter.py`)

**`Parameter(Datum)`** declares mutable scope for an optimizer: **`requires_grad`**, optional **`grad`**, inherits **`Datum`** fields. Assigned on a **`Module`**, it is collected by **`module.parameters()`** and included in **`state_dict()`**.

**`PathParameter`** (`src/autopilot/ai/parameter.py`) specializes **`Parameter`** for filesystem scope: **`source`**, **`pattern`**, **`matched_files()`** for globbed paths under **`source`**.

## Experiment lifecycle

There is no central typed phase-state machine in core. **`Experiment`** (see **`core/experiment.py`**) owns manifest + injected **`Logger`** / **`Checkpoint`**. Status graphs and workflow transitions, when needed, are **project-defined** (metadata, overlay services, or command layers).

## Loss, optimizer, agent

- **`Loss(Module)`** (`core/loss.py`) -- **`forward(data, targets)`**, **`backward()`**, **`reset()`**; optional **`_loss_parameters`** list scopes gradient targets.
- **`Optimizer`** (`core/optimizer.py`) -- **not** a **`Module`**; **`step()`** / **`zero_grad()`**.
- **`Agent`** / **`AgentResult`** (`ai/agent.py`) -- Pydantic result surface for agent backends.
- **`ClaudeCodeAgent`** (`ai/coding.py`) -- MVP subprocess integration with the Agent SDK CLI.

## Loop

Abstract base class in **`src/autopilot/core/loops.py`**: **`Loop`**. Defines **`run(trainer, config)`** where **`LoopConfig`** carries **`max_epochs`**, **`dry_run`**, **`ctx`**, optional **`train_loader` / `val_loader`**, optional **`loss`**, **`optimizer`**, a **`metrics`** map, and **`accumulate_grad_batches`**.

Builtin implementation: **`EpochLoop`** -- drives epochs, dispatches generic **`on_epoch_*`** plus Lightning-style train/validation hooks via **`Trainer._dispatch`**. **`_run_epoch()`** iterates the train loader, calls **`training_step`**, applies **`loss(data, batch)`**, updates metrics, and only when **`_should_step(batch_idx, is_last, accumulate)`** is true runs **`loss.backward()`**, **`optimizer.step()`**, **`optimizer.zero_grad()`**, and **`loss.reset()`**. After **`metrics.compute()`**, an optional Trainer **`Policy`** evaluates a **`Result`**; **`GateResult.FAIL`** stops the epoch loop and may trigger **`Store.checkout`** when a **`Store`** is attached. Handles `IterableDataset` (no `__len__`) safely by materializing batches first -- accumulation works without known total.

**`Trainer.fit()`** builds **`LoopConfig`**, calls **`on_fit_start`**, **`on_loop_start`**, **`self._loop.run(self, config)`**, then **`on_loop_end`**, **`on_fit_end`**.

## Services layer

`src/autopilot/core/services.py` currently exposes **`evaluate_experiment_policy(experiment_dir, policy)`** for loading persisted **`Result`** data, running **`policy.forward` / `policy.explain`**, and appending a **`policy_evaluated`** event. Other experiment directory setup and manifest mutations live in **`Experiment`**, commands, or project overlays as needed.

## Config resolution

`src/autopilot/core/config.py`: JSON helpers (**`load_json`**, **`json.load`**), experiment path resolution, **`merge_overrides()`** for shallow CLI overrides. Workspace layout is wired in Python (project **`CLI`** subclass, **`Module`**, **`Trainer`**). No environment variables anywhere in `src/autopilot/`.

## Callbacks (`core/callbacks.py`)

**`Callback`** defines composable hooks. **Lightning-style** names include **`on_fit_start`**, **`on_fit_end`**, **`on_train_epoch_start`**, **`on_train_epoch_end`**, **`on_validation_epoch_start`**, **`on_validation_epoch_end`**, **`on_test_epoch_start`**, **`on_test_epoch_end`**, plus batch and optimizer hooks (**`on_train_batch_start`**, **`on_train_batch_end`**, **`on_before_backward`**, **`on_after_backward`**, **`on_before_optimizer_step`**, **`on_before_zero_grad`**). Framework hooks include **`on_epoch_start`**, **`on_epoch_end`**, **`on_loop_start`**, **`on_loop_end`**, plus **`state_dict` / `load_state_dict`** for callback checkpointing.

## Crash recovery

`RunStateCallback` writes `run_state.json` to the experiment dir with `status: "running"` on each `on_epoch_end` and `status: "completed"` with `stop_reason` on `on_loop_end`. If a process crashes, `run_state.json` still shows `status: "running"`. The `status` command detects this and reports `stop_reason: "crash"`.

`ExperimentSummaryData` now carries `stop_reason` and `last_good_epoch` fields, persisted to `summary.json` by `write_experiment_summary()`.

## Diagnostics

`DiagnoseCallback` produces per-epoch diagnostic artifacts:
- `trace_diagnoses.jsonl` -- one record per failure category with sample errors
- `node_heatmap.json` -- per-node `{total, failed, error_rate}` breakdown

`trace inspect --node <id>` filters `data.jsonl` by item_id/metadata.node and optionally pulls related memory records at `--depth > 1`.

## Key files

- `src/autopilot/core/models.py` -- shared types (including `GateResult`, `Datum`, `Result`)
- `src/autopilot/core/metric.py` -- **`Metric(Module)`**, **`CompositeMetric`**
- `src/autopilot/core/loss.py` -- **`Loss(Module)`**
- `src/autopilot/core/optimizer.py` -- **`Optimizer`**
- `src/autopilot/core/module.py` -- **`Module`**, **`AutoPilotModule`**
- `src/autopilot/core/parameter.py` -- **`Parameter`**
- `src/autopilot/ai/parameter.py` -- **`PathParameter`**
- `src/autopilot/core/config.py` -- path resolution, merge helpers, optional JSON loads
- `src/autopilot/core/services.py` -- **`evaluate_experiment_policy`**
- `src/autopilot/core/trainer.py` -- **`Trainer`**: **`fit(module, ...)`**, callback dispatch, optional **`policy`** / **`store`**
- `src/autopilot/core/loops.py` -- **`Loop`**, **`EpochLoop`**, **`LoopConfig`**
- `src/autopilot/core/callbacks.py` -- **`Callback`**
- `src/autopilot/core/store_callbacks.py` -- **`StoreCheckpoint`**, **`StorePromoter`**
- `src/autopilot/core/stage_callbacks.py` -- **`EpochRecorderCallback`**, **`RegressionCallback`**, **`DiagnoseCallback`**, **`RunStateCallback`**, **`MemoryCallback`**
- `src/autopilot/data/dataset.py` / `dataloader.py` / `datamodule.py` -- data primitives
- `src/autopilot/ai/agent.py` / `coding.py` / `loss.py` / `optimizer.py` -- agent + training helpers
- `src/autopilot/core/errors.py` -- exception hierarchy

## Gotchas

- No `os.environ`, `os.getenv`, or `load_dotenv` anywhere in `src/autopilot/`. This is a hard invariant.
- No `if TYPE_CHECKING:` blocks. All imports are unconditional.
- No relative imports. All imports are absolute from `autopilot.*`.
- Command handlers stay thin; they must not duplicate manifest persistence or policy scoring unless that is intentional project logic.
