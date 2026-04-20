---
name: optimization-loop
description: Optimization pipeline: Trainer.fit, EpochLoop, gradient accumulation, and callback hooks. Use when modifying how experiments run or extending loop behavior.
---

## Trainer

**`Trainer`** (`src/autopilot/core/trainer.py`) owns the **`Loop`**, **`Callback`** list, optional **`Logger`**, optional **`Policy`**, optional **`Store`**, **`dry_run`**, and **`accumulate_grad_batches`**. The experiment **`Module`** is **not** stored via **`Trainer(..., module=...)`** -- callers pass it to **`fit()`** only.

- **`fit(module, train_dataloaders=None, val_dataloaders=None, datamodule=None, max_epochs=10, ctx=None)`** -- attaches **`module`**, sets **`_trainer`** on **`AutoPilotModule`** subclasses, resolves train/val loaders from **`datamodule`** when loaders are omitted, gathers **`configure_optimizers()`** when applicable, finds the first **`Loss`** via **`module.modules()`**, collects **`Metric`** instances from **`module.named_modules()`** (excluding types that are also **`Loss`**), builds **`LoopConfig`**, then dispatches **`on_fit_start`**, **`on_loop_start`**, **`self._loop.run(self, config)`**, **`on_loop_end`**, **`on_fit_end`**, and runs teardown hooks.

Core **`Trainer`** does not expose **`run()`** or **`run_phase()`**; sequencing for legacy optimize-style CLIs is handled in command/project layers, not as generic Trainer methods.

## Phase-to-module mapping

**`Module.forward`** and **`AutoPilotModule.training_step` / `validation_step` / `test_step`** are project-defined. The stock **`EpochLoop`** calls **`training_step(batch)`** on train batches and **`validation_step(batch)`** (or plain **`__call__`**) during validation when an **`AutoPilotModule`** is used.

## EpochLoop (`core/loops.py`)

**`EpochLoop._should_step(batch_idx, is_last_batch, accumulate)`** returns **`((batch_idx + 1) % accumulate == 0) or is_last_batch`**. It gates **`loss.backward()`**, **`optimizer.step()`**, **`optimizer.zero_grad()`**, and **`loss.reset()`** so multiple micro-batches can accumulate before an optimizer step when **`accumulate_grad_batches > 1`**.

**`_run_epoch()`** flow (train path):

1. Fire **`AutoPilotModule.on_train_start`**, set **`module.train()`**, dispatch **`on_train_epoch_start`**.
2. For each train batch: **`on_train_batch_start`**, forward via **`training_step`** (or **`module(batch)`** for plain modules), **`loss_fn(data, batch)`** when configured, **`metric.update(data)`** for each discovered metric, **`on_train_batch_end`**.
3. When **`_should_step`** is true and a loss is configured: **`on_before_backward`**, **`loss.backward()`**, **`on_after_backward`**.
4. When stepping and an optimizer exists: **`on_before_optimizer_step`**, **`optimizer.step()`**, **`on_before_zero_grad`**, **`optimizer.zero_grad()`**, then **`loss.reset()`** if a loss exists.
5. Aggregate **`metrics.compute()`** into a single **`metric_values`** dict.
6. If **`trainer.policy`** is set, build **`Result(metrics=metric_values)`**, call **`policy(result)`**; on **`GateResult.FAIL`**, optionally **`store.checkout(best_epoch)`** and return **`stopped: True`**.
7. Otherwise update **`trainer._best_epoch`**, run optional validation loop + hooks, **`metric.reset()`**, **`on_train_epoch_end`**, **`AutoPilotModule.on_train_end`**.

## Policy evaluation

**`evaluate_experiment_policy()`** in **`services.py`** loads a persisted **`Result`** and runs **`policy.forward(result)`** / **`policy.explain(result)`** for offline review flows.

During **`Trainer.fit()`**, an attached **`Policy`** is evaluated each training epoch after metrics are computed, before validation, using the in-memory **`Result`** constructed from **`metric_values`**.

## Callback hooks (batch-level)

Trainer forwards Lightning-style batch and optimizer hooks to callbacks: **`on_train_batch_start`**, **`on_train_batch_end`**, **`on_before_backward`**, **`on_after_backward`**, **`on_before_optimizer_step`**, **`on_before_zero_grad`**, plus epoch/fit hooks documented in **`core/callbacks.py`**.

## Stop reason persistence

`ExperimentSummaryData` carries `stop_reason` (str | None) and `last_good_epoch` (int). `build_experiment_summary()` extracts these from the loop result dict.

`RunStateCallback` writes `run_state.json` atomically per epoch (`status: "running"`) and on loop end (`status: "completed"` with `stop_reason`). Used for crash detection: if `run_state.json` shows `status: "running"`, the process died mid-loop.

## IterableDataset support

`EpochLoop._run_epoch()` materializes train batches via `list(enumerate(loader))` to handle `IterableDataset` (no `__len__`). `_should_step` determines accumulation flush using the actual batch count. `DataLoader` accepts `length_hint` kwarg for `IterableDataset` length estimation in `__len__`.

## Default stage callbacks in optimize loop

`optimize loop` wires: `EpochRecorderCallback`, `RegressionCallback`, `DiagnoseCallback`, `MemoryCallback`, `RunStateCallback`, `CostTracker`.

## Key files

- `src/autopilot/core/trainer.py` -- **`Trainer`**
- `src/autopilot/core/loops.py` -- **`Loop`**, **`EpochLoop`**, **`LoopConfig`**, **`_should_step`**
- `src/autopilot/core/callbacks.py` -- **`Callback`**
- `src/autopilot/core/stage_callbacks.py` -- stage-specific callbacks
- `src/autopilot/cli/commands/optimize.py` -- **`OptimizeCommand`** and related handlers (project-facing sequencing)

## Gotchas

- **`Trainer.module`** is **`None`** until **`fit()`** runs; prefer capturing the module from caller scope when wiring overlays.
- Gradient accumulation changes how often **`loss.reset()`** runs relative to **`metric.update()`**; metrics still compute once per epoch after the full train pass.
- **`fit`** and bespoke optimize commands are different entrypoints; keep phase strings and loader wiring consistent with your **`Module`** implementation.
- `IterableDataset` data is materialized per epoch. For very large streaming datasets, consider batching at the dataset level or using `from_jsonl_streaming()`.
