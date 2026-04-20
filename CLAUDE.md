# AutoPilot — agent notes

## Framework boundaries

AutoPilot is a reusable library. Project-specific targets, configs, and overlays live in the consuming repo (workspace under `autopilot/`, datasets). Do not fold one-off product logic into `src/autopilot` unless it belongs in the shared library.

## Extension model

- **Module**: subclass `Module`, override `forward(batch)`, assign child `Module`s, `Parameter`s, `Metric`s, and `Loss`es as attributes. Like `nn.Module`. `Module.__setattr__` auto-registers **`Parameter` -> `_parameters`** and child **`Module` -> `_modules`** (including **`Metric(Module)`** and **`Loss(Module)`**); there is **no** separate `_metrics` dict. Tree traversal matches PyTorch: `children()`, `modules()`, `parameters()`, `named_*()`, `train()`, `eval()`, `apply()`, `state_dict()`.
- **AutoPilotModule**: subclass `AutoPilotModule`, override step methods (`training_step`, `validation_step`, `test_step`, `configure_optimizers`) and lifecycle hooks (`setup`, `teardown`, `on_train_start`, `on_train_end`, `on_validation_start`, `on_validation_end`). Like `LightningModule`.
- **Parameter**: subclass `Parameter` (`Parameter(Datum)`), with `requires_grad` and `grad`. Auto-registered into `_parameters` by `Module.__setattr__`. **`PathParameter`** (`ai/parameter.py`) extends it for file-system scope.
- **Store**: subclass `Store` in `core/store.py`, override all methods for custom backends. **`FileStore(Store)`** (`ai/store.py`) is the built-in content-addressed implementation. Idempotent `__init__(path, slug, parameters)` follows the `Experiment` pattern: loads existing slug or initializes epoch_0 baseline. Operations: `snapshot`, `checkout`, `diff`, `branch`, `merge`, `log`, `status`, `promote`. One instance per slug; cross-slug operations reference foreign slugs by name.
- **Graph**: `Graph`, `Node`, `AccumulateGrad` in `core/graph.py`; `no_grad()` / `enable_grad()` and `RemovableHandle` mirror PyTorch-style graph capture and autograd hooks.
- **Experiment**: subclass `Experiment` in `core/experiment.py`; idempotent `__init__` (loads existing manifest or creates). `promote(reason)` and `reject(reason)` require a **`reason`** argument. `advance_epoch()`, `finalize(status)`. Accepts injected **`Logger`** / **`Checkpoint`** (callers wire `JSONLogger`/`JSONCheckpoint`). Sits **above** **`Trainer`** -- overlays orchestrate **`Experiment`** and **`Trainer`** separately; **`Trainer`** does not accept an **`Experiment`**.
- **Policies**: subclass `Policy`, override `forward(self, result: Result)` / `explain(self, result: Result)`, pass to Trainer.
- **Loss**: subclass `Loss(Module)` in `core/loss.py`, override `forward(data, targets)` / `backward()` / `reset()`, assign on a parent `Module` like a child module. Optional `_loss_parameters` scopes which `Parameter`s receive gradients.
- **Optimizer**: subclass `Optimizer` in `core/optimizer.py` (**not** a `Module`); `step()` / `zero_grad()`. `AgentOptimizer` in `ai/optimizer.py` composes an `Agent` with context (workspace, epoch, metrics, file paths). After `agent.forward()`, clears `param.grad` on `PathParameter`s to signal application. `update_context()` refreshes epoch/metrics between epochs.
- **Metrics**: subclass `Metric(Module)` in `core/metric.py`, override `update()`/`compute()`. `CompositeMetric` composes via `+`.
- **Agent**: subclass `Agent` in `ai/agent.py`; `forward(...) -> AgentResult` (Pydantic). `ClaudeCodeAgent` in `ai/coding.py` is the MVP CLI-backed implementation.
- **Memory**: subclass `Memory` in `core/memory.py`, override `learn()`, `recall()`, `trends()`, `context()`, `block_strategy()`, `is_strategy_blocked()`, `blocked_strategies()`, `state_dict()`, `load_state_dict()`. **`FileMemory(Memory)`** (`core/memory.py`) is the built-in file-backed implementation using `knowledge_base.jsonl` + `strategy_blocklist.json`. Supports `strategy` filter in `recall()` and `_apply_filters()`. `MemoryCallback` auto-captures strategy, validation metrics, and regression outcome.
- **Data**: `Dataset` / `IterableDataset` / `ConcatDataset` / `ListDataset` / `StreamingDataset`, `DataLoader`, `DataModule` under `autopilot.data` (see `data/dataset.py`, `data/dataloader.py`, `data/datamodule.py`). `ListDataset` is a generic list-backed map-style dataset with `from_jsonl`/`to_jsonl`/`subset`. `StreamingDataset` lazily reads JSONL files line-by-line. AI-specific composition (`StratifiedSplitter`, `SlotPlanner`) lives in `ai/data.py`. `DataLoader` supports `length_hint` kwarg for `IterableDataset` length estimation. `EpochLoop` safely handles `IterableDataset` without requiring `__len__`.
- **Gates**: subclass `Gate`, override `forward()`. Built-ins: `MinGate`, `MaxGate`, `RangeGate`, `CustomGate`.
- **Callbacks**: subclass `Callback`, override hook methods, pass list to Trainer. Lightning-style: `on_fit_start`, `on_fit_end`, `on_train_epoch_start`, `on_train_epoch_end`, `on_validation_epoch_start`, `on_validation_epoch_end`, `on_test_epoch_start`, `on_test_epoch_end`. Framework loop hooks: `on_epoch_start`, `on_epoch_end`, `on_loop_start`, `on_loop_end`. Plus `state_dict` / `load_state_dict` for callback checkpointing.
- **Stage Callbacks**: `EpochRecorderCallback`, `JudgeValidationCallback`, `RegressionCallback`, `MemoryCallback`, `DiagnoseCallback`, `RunStateCallback` in `core/stage_callbacks.py`. Configurable artifact filenames via constructor params. `DiagnoseCallback` produces `trace_diagnoses.jsonl` and `node_heatmap.json` per epoch. `RunStateCallback` persists `run_state.json` for crash detection and stop-reason forensics.
- **Loop**: subclass `Loop`, override `run()`/`_run_epoch()`, pass instance to Trainer.
- **EpochOrchestrator**: `EpochOrchestrator(EpochLoop)` in `core/orchestrator.py`. Overrides only `run()`, delegates to `super()._run_epoch()`. Adds plateau detection, regression rollback (sole rollback owner), and stop conditions. Configured via `OrchestratorConfig`.
- **Logger**: subclass `Logger` in `core/logger.py` (Lightning-style `name`, `version`, `log_metrics`, `log_hyperparams`, `log`, `finalize`); **`JSONLogger`** appends JSONL under the experiment directory.
- **Checkpoint**: subclass `Checkpoint` in `core/checkpoint.py` (Lightning **`CheckpointIO`**-style manifest persistence); **`JSONCheckpoint`** uses `tracking/manifest.py`.
- **Manifest**: `Manifest` in `core/models.py` is a typed record with generic fields only: `slug`, `title`, `current_epoch`, `idea`, `hypothesis`, `hyperparams`, `decision`, `decision_reason`, `metadata`. Removed library columns such as workflow status enums, profile, target, environment, constraints, baseline, candidate, and dataset snapshot -- put project-specific data in `metadata` or overlay code.
- **CLI**: subclass `CLI` (or `AutoPilotCLI`), assign `Command` instances as attributes in `__init__`; `__call__` runs the parser and dispatch.
- **Command**: subclass `Command`, override `forward()` for leaf handlers; nest child `Command`s or use `@subcommand` / `@argument` for grouped subcommands.
- **No registries for training components.** Components are objects, not string keys. (`CLI._project_registry` exists for project dispatch.)
- **Layering**: **`Experiment`** owns manifest + default logger/checkpoint wiring above the training loop. **`Trainer`** runs **`fit`** with the loop, callbacks, optional **`Logger`**, optional **`Policy`**, optional **`Store`**, and **`accumulate_grad_batches`**. The root module is passed only to **`fit(module, ...)`**. There is no central **`state.py`** transition API or **`Trainer.run()`**.
- **`Trainer.optimizer`**: public read-only property, set during `fit()` from `configure_optimizers()`.
- **`Trainer.regression_detected`**: public bool attribute, reset to `False` each epoch by `_run_epoch()`. Set by `RegressionCallback`. Read by `EpochOrchestrator` for rollback decisions.
- **`--expose` mechanism**: global CLI flag; `ExposeCollector` in `CLIContext` tracks command execution for JSON audit trail.
- **Workflows**: expressed via code composition and CLI commands -- pure Python, no config files required.
- **AI Generators**: subclass `DataGenerator`, override `create_slots()`/`define_steps()`/`assemble_item()`/`stratify_key()`, pass instance directly. Use **`run()`** for sync (**`asyncio.run`** over **`async_run()`**) or **`async_run()`** for async.
- **AI Judges**: subclass `Judge`, override `define_steps()`/`assemble_result()`/`build_summary()`, pass instance directly. Same **`run()`** / **`async_run()`** pattern as generators.

> Phase 3: **`Module`** is the execution building block. **`Graph`** records autograd-style computation. **`AutoPilotModule`** adds Lightning-style steps and hooks. **`Parameter`** is a first-class registered leaf. **`Experiment`**, **`Logger`**, and **`Checkpoint`** model lifecycle and I/O above the loop.
>
> Phase 4: **`Store`** (abstract base in `core/store.py`) and **`FileStore`** (content-addressed implementation in `ai/store.py`) provide code versioning. SHA-256 objects, 2-char prefix sharding, JSON snapshot manifests, atomic writes, lock files.
>
> Phases 5--8 (store callbacks + CLI, **`Loss`/`Optimizer`**, **`Agent`/`ClaudeCodeAgent`**, **`autopilot.data`** + **`Trainer.fit`** integration with policy/store/accumulation) are implemented; see VISION.md for roadmap beyond Phase 9.
>
> Phase 9--10: **`Memory`** / **`FileMemory`**, stage models and artifact I/O (`stage_models`, `stage_io`), stage callbacks, **`EpochOrchestrator`** (plateau/regression/stop), **`CostTracker`**, regression and proposal helpers, experiment summary, **`tracking/io`** primitives, extended CLI commands, and the **`--expose`** audit trail (`ExposeCollector`).
>
> Post-MVP agent workflow gaps: **`RunStateCallback`** (crash detection via `run_state.json`), **`DiagnoseCallback`** (`trace_diagnoses.jsonl`, `node_heatmap.json`), enriched **`status`** command (regression, stop_reason, best_baseline), real **`propose verify`** (metric comparison) and **`propose revert`** (store checkout), **`trace inspect`** (node filtering + memory cross-reference), **`strategy`** filter in `FileMemory`, **`AgentOptimizer`** context/paths/result handling, **`IterableDataset`** loop compatibility, and **`project init`** skeleton generation.

## Configuration invariants

- **No environment variable access in autopilot.** Zero `os.environ`, `os.getenv`, or `load_dotenv` calls anywhere in `src/autopilot/`. All config flows through function args, constructor kwargs, or explicit dicts passed by callers.
- **Explicit configuration surfaces:** constructor kwargs on library objects, function parameters, and orchestrator-provided dicts (workspace, dry_run, candidates, hyperparams, execution options). No implicit global reads or silent fallbacks inside `src/autopilot/`.
- **Project overlay uses `dotenv_values()` exclusively** for envvar access. Never `os.environ` or `os.getenv`. Values from `dotenv_values()` are passed explicitly into project and library entrypoints.
- **Manifest is minimal:** the shared **`Manifest`** schema is intentionally generic; extended semantics and workflow keys live in **`metadata`** or project-specific modules, not in removed legacy manifest fields.
- **Structured data everywhere:** all persisted artifacts use typed dataclass fields (`MemoryRecord`, `ExposeRecord`, `RuleGradient`, etc.); no free-form strings as primary carriers of queryable information; `content`/`description` fields are optional human-readable supplements alongside structured fields.
- **Callbacks observe, loops control:** callbacks set flags (e.g., `trainer.regression_detected`); only the loop/orchestrator calls `store.checkout()` for rollback.
- **Optimizer blocklist via methods:** `block_strategy()` / `unblock_strategy()` / `is_strategy_blocked()` / `blocked_strategies` (frozenset property), not public mutable sets.
- **DRY I/O:** `tracking/io.py` is the sole primitive layer for JSON/JSONL operations; all writers/readers delegate to it.
- **CLI types stay in CLI:** `ExposeRecord` and other CLI-only types live under `cli/`, not `core/`.

## Artifacts and safety

Never commit: `.env`, API tokens or secrets, raw execution logs, or large generated outputs. Prefer references in manifests and redacted command logs where the library already supports it.

## When to use subagents

Use bounded subagents for: style extraction from a codebase, session or transcript mining, README or doc pattern research, and git-boundary or history analysis. Keep scope narrow and return structured findings.

## Code conventions

### Style baseline

Google Python Style Guide is the baseline for: absolute imports, package/module naming, explicit exception design, import-safe modules and `main()`, docstrings and comments, issue-linked TODOs, typed public APIs.

### Project overrides from Google PyGuide

- **2-space indentation** everywhere (overrides Google's 4-space rule). This is an explicit repo policy; ruff is configured to enforce it.
- **Single quotes** for all strings. Double quotes only when avoiding escape sequences. Ruff enforces via `flake8-quotes` with `inline-quotes = 'single'` and `quote-style = 'single'` in format.
- **No `if TYPE_CHECKING:` blocks.** All imports are unconditional.

### Import rules

- All imports at the top of the file, no exceptions. No inner/deferred imports in source code. If there is a circular import, resolve it through better module organization, not by deferring the import.
- `from` imports first, then `import` statements, no blank lines between them.
- No relative imports. Ruff enforces via `flake8-tidy-imports` with `ban-relative-imports = "all"`.
- No dynamic imports anywhere. No `importlib`, no dotted-path runtime loading, no plugin auto-discovery by import string.
- Ruff isort config: `from-first = true`, `no-sections = true`, `lines-between-types = 0`.
- No `# noqa` comments anywhere in `src/autopilot/`. Fix the underlying lint violation instead of suppressing it.

### Component wiring

- Components are Python objects instantiated directly, not looked up by string key.
- The project overlay's `cli.py` defines a `CLI` subclass: `__init__` assigns `module`, `generator`, `judge`, and any `Command` instances; entry is `MyCLI()()`.
- Core layer: compose **`Module`** / **`AutoPilotModule`**, optional **`Graph`** usage, **`Experiment`** for experiment-directory lifecycle, and `core` / `tracking` helpers.
- Trainer layer: construct **`Trainer(callbacks=..., loop=..., dry_run=..., logger=..., policy=..., store=..., accumulate_grad_batches=1)`** (no root **`module=`**). Pass **`Module`** to **`Trainer.fit(module, train_dataloaders=..., val_dataloaders=..., datamodule=..., max_epochs=..., ctx=...)`**; that path dispatches `on_fit_start` / `on_fit_end`, loader-driven epochs, optional loss/optimizer steps, and loop callbacks. **`Trainer.run()`** and **`Trainer.run_phase()`** are not part of the API.

### DRY rules

There must be one canonical implementation for each of these concerns:
- Config loading and resolution
- Trainer construction and component wiring
- Trainer **`fit`** and loop lifecycle (loop run, callback dispatch; policy evaluation only where explicitly wired)
- Callback dispatch
- Graph recording and backward (`core/graph.py`)
- Experiment lifecycle (`core/experiment.py`: idempotent open, `promote` / `reject`, `advance_epoch`, `finalize`)
- Logger and checkpoint protocols (`core/logger.py`, `core/checkpoint.py`)
- Manifest schema (`core/models.py`) and atomic load/save (`tracking/manifest.py`)
- Event append/write
- Train/val/test result normalization
- Result computation
- Policy evaluation when persisting events (`evaluate_experiment_policy()` in `services.py`)
- DRY I/O: `tracking/io.py` provides `atomic_write_json`, `append_jsonl`, `read_jsonl`, `read_json`; new code should delegate to these
- Memory persistence
- Stage artifact I/O
- Regression comparison
- Proposal I/O

Command handlers orchestrate only; they must not duplicate backend logic, manifest logic, or summary logic.

### Linting

Ruff is the sole linter/formatter. Key config in `pyproject.toml`:
- `indent-width = 2`
- `line-length = 100`
- Lint rules: `E`, `E402`, `F`, `I`, `Q`, `W191`, `TID`
- `ban-relative-imports = "all"`
- `inline-quotes = 'single'`
- isort: `from-first = true`, `no-sections = true`, `lines-between-types = 0`
- Format: `quote-style = 'single'`, `indent-style = 'space'`

### Prohibited anti-patterns

- No `getattr(args, 'x', default)` on declared argparse arguments. Use `args.x` directly. Argparse guarantees the attribute exists when the argument is declared.
- No fake fallback objects (e.g., constructing a `Datum` with `success=False` when a precondition fails). Fail early and explicitly with `ctx.output.error()` and `return`.
- No `getattr(args, 'x', default) or other_default` double-fallback chains.
- No inline file-content strings for templates or scaffolding. Use real files in `templates/`.
- No module-level uppercase constants that cache a function call (e.g., `_FOO = some_func()`). Call the function at the use site; `paths.py` exists so every path is available in one place.
- No module-level uppercase constants that cache a function call (e.g., `_FOO = some_func()`). Call the function at the use site; `paths.py` exists so every path is available in one place.
- No `# noqa` comments. Fix the underlying lint violation.
- No inner/deferred imports. All imports at the top of the file.

### Module layout

Match the existing structure: `autopilot.cli`, `autopilot.core`, `autopilot.data`, `autopilot.tracking`, `autopilot.policy`, `autopilot.ai`. No `__init__.py` files; use implicit namespace packages. Policy implementations live directly in `autopilot.policy`. Import directly from the exact module.

Key files:
- `core/paths.py` -- centralized path computation. Every directory name string appears exactly once. Callers never join paths.
- `core/module.py` -- `Module` and `AutoPilotModule`.
- `core/parameter.py` -- `Parameter(Datum)` base class.
- `core/graph.py` -- `Graph`, `Node`, `AccumulateGrad`, `no_grad` / `enable_grad`, `RemovableHandle`.
- `core/experiment.py` -- `Experiment` lifecycle above `Trainer`.
- `core/logger.py` -- `Logger`, `JSONLogger` (Lightning-style logging surface).
- `core/checkpoint.py` -- `Checkpoint`, `JSONCheckpoint` (Lightning **`CheckpointIO`**-style persistence).
- `core/cost_tracker.py` -- `CostTracker(Callback)` for per-epoch timing.
- `core/store.py` -- `Store` ABC and supporting dataclasses (`FileEntry`, `SnapshotManifest`, `DiffResult`, `MergeResult`, `StatusResult`, `SnapshotEntry`).
- `core/metric.py` -- `Metric(Module)` and `CompositeMetric` base classes (like torchmetrics).
- `core/memory.py` -- `Memory` base class and `FileMemory(Memory)`.
- `core/loss.py` -- `Loss(Module)` base class.
- `core/optimizer.py` -- `Optimizer` base class (not a Module).
- `core/orchestrator.py` -- `EpochOrchestrator(EpochLoop)`, `OrchestratorConfig`.
- `core/proposal.py` -- `record_proposal()`, `read_proposals()`, `record_verdict()`, `read_verdict()`.
- `core/regression.py` -- `compare_metrics()`, `is_regression()`, `read_best_baseline()`, `write_best_baseline()`.
- `core/store_callbacks.py` -- `StoreCheckpoint`, `StorePromoter` callbacks.
- `core/stage_callbacks.py` -- `EpochRecorderCallback`, `JudgeValidationCallback`, `RegressionCallback`, `MemoryCallback`, `DiagnoseCallback`, `RunStateCallback`.
- `core/models.py` -- `Manifest`, `Result`, `Event`, and shared typed records.
- `core/services.py` -- `evaluate_experiment_policy()` and related helpers (policy on stored results + optional event append).
- `core/normalization.py` -- generic split summary helpers (**no hardcoded `SUMMARY_FILENAMES`**).
- `core/splits.py` -- intentionally minimal / empty of shared constants such as `REQUIRED_SPLITS` (projects own split conventions).
- `core/stage_models.py` -- Stage data models: `MemoryRecord`, `TrendResult`, `MemoryContext`, `BlockedStrategy`, `EpochMetrics`, `ChangeProposal`, `ProposalVerdict`, `JudgeValidation`, `RegressionAnalysis`, `CostEntry`, `ExperimentSummaryData`.
- `core/stage_io.py` -- Thin composition of `paths.py` + `tracking/io.py` for epoch/experiment artifacts.
- `core/loops.py` -- loop implementations and epoch orchestration hooks.
- `core/trainer.py` -- `Trainer` composes loop, callbacks, optional `logger`, optional `policy`, optional `store`, `accumulate_grad_batches`; **`fit(module, ...)`** only (no `run` / `_execute`).
- `core/summary.py` -- `build_experiment_summary()`, `write_experiment_summary()` functions.
- `tracking/manifest.py` -- `load_manifest` / `save_manifest` for `Manifest`.
- `tracking/io.py` -- Canonical I/O primitives: `atomic_write_json`, `append_jsonl`, `read_jsonl`, `read_json`.
- `cli/command.py` -- `Command`, `CLI`, `Argument` / `Flag`, `@subcommand`, argparse registration.
- `cli/commands/project_cmd.py` -- project management (list, init, doctor).
- `cli/commands/store_cmd.py` -- `StoreCommand` (Store protocol CLI).
- `cli/commands/status_cmd.py` -- `StatusCommand` (experiment overview).
- `cli/commands/judge_cmd.py` -- `JudgeCommand` (run/distribution).
- `cli/commands/diagnose_cmd.py` -- `DiagnoseCommand` (run/heatmap).
- `cli/commands/trace_cmd.py` -- `TraceCommand` (collect/inspect).
- `cli/commands/propose_cmd.py` -- `ProposeCommand` (create/verify/revert/list).
- `cli/commands/memory_cmd.py` -- `MemoryCommand` (query/record/trends/context).
- `cli/commands/agent_cmd.py` -- `AgentCommand` (run/list/session).
- `cli/main.py` -- `AutoPilotCLI`, `main()` (`AutoPilotCLI()()`).
- `cli/expose.py` -- `ExposeRecord`, `ExposeCollector`, `inject_expose`, `expose_command`.

- `data/dataset.py` -- `Dataset`, `IterableDataset`, `ConcatDataset`, `ListDataset`, `StreamingDataset`.
- `data/dataloader.py` -- `DataLoader`.
- `data/datamodule.py` -- `DataModule`.

AI module: `autopilot.ai`. Files: `models.py`, `data.py`, `steps.py`, `generator.py`, `judge.py`, `checkpoints.py`, `runtime.py`, `parameter.py` (`PathParameter`), `store.py` (`FileStore(Store)`), `agent.py` (`Agent`, `AgentResult`), `coding.py` (`ClaudeCodeAgent`), `loss.py` (`JudgeLoss`), `optimizer.py` (`AgentOptimizer`). All async-first, Pydantic models for data transfer (`StepMeta` uses `@dataclass`). Step decorators (`@llm_step`, `@python_step`, `@back_step`, `@stratify_by`) attach metadata to methods; `collect_steps()` gathers them. **`DataGenerator`** and **`Judge`** expose **`run()`** (sync via `asyncio.run`) and **`async_run()`**.

### Multi-project workspace layout

```
workspace/
  autopilot/
    pyproject.toml
    projects/
      <name>/
        cli.py           # defines CLI subclass with __init__
        trainer.py       # builds Trainer with the project Module
        ai/
        experiments/
        datasets/
        records/
```

The primary project interface is a `CLI` subclass registered with `project='...'` on `__init_subclass__`:

```python
from autopilot.cli.main import AutoPilotCLI


class MyCLI(AutoPilotCLI, project='my-project'):
  def __init__(self):
    super().__init__()
    self.module = my_module
    self.generator = MyGenerator()
    self.judge = MyJudge()
```

Invoke with `MyCLI()()` (or `AutoPilotCLI()()` for the stock CLI). Project dispatch and optional `cli.py` execution live inside `CLI.run()`.

Project resolution order: `--project` / `-p` flag > CWD detection under `autopilot/projects/`.

## Commands

Always drive behavior through the `autopilot` CLI (`uv run autopilot ...`). Avoid ad-hoc shell fragments that bypass manifest updates, events, or workspace conventions.
