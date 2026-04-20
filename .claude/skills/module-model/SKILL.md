---
name: module-model
description: Module and AutoPilotModule tree (nn.Module-style), Parameter and PathParameter, and ctx/params runtime dicts for experiment execution. Use when implementing a new backend graph, wiring Trainer, or debugging module config flow.
---

## Module (`core/module.py`)

**`Module`** is the base composable unit (like **`torch.nn.Module`**):

- **`_modules`**, **`_parameters`** -- internal dicts. **`Module.__setattr__`** registers **`Parameter` -> `_parameters`** and child **`Module` -> `_modules`** only. **`Metric`** and **`Loss`** are implemented as **`Module`** subclasses, so they live in **`_modules`** (there is **no** `_metrics` dict).
- **`forward(*args, **kwargs) -> Datum`** -- override; invoked through **`__call__`**.
- Iteration helpers: **`children`**, **`named_children`**, **`modules`**, **`named_modules`**, **`parameters`**, **`named_parameters`**.
- **`train(mode=True)`** / **`eval()`** -- toggles **`training`** and recurses.
- **`apply(fn)`**, **`state_dict()`**, **`load_state_dict()`**.

## AutoPilotModule

**`AutoPilotModule(Module)`** adds **`LightningModule`‑style** entrypoints: **`training_step`**, **`validation_step`**, **`test_step`**, **`configure_optimizers`**, a **`trainer`** reference set by **`Trainer.fit()`**, and lifecycle hooks (**`setup`**, **`teardown`**, **`on_train_start`**, **`on_validation_end`**, etc.). Use **`Module`** for minimal graphs; use **`AutoPilotModule`** when the trainer should call step methods and optimizers explicitly.

## Parameter (`core/parameter.py`)

**`Parameter(Datum)`** marks data the optimizer may change. Fields include inherited **`Datum`** payload plus **`requires_grad`** and **`grad`**. Registered on a **`Module`** like **`nn.Parameter`** on **`nn.Module`**.

**`PathParameter`** (`src/autopilot/ai/parameter.py`) -- filesystem scope: **`source`**, **`pattern`**, **`matched_files()`** for globs under **`source`**.

## Two runtime dicts (distinct from ctor kwargs)

1. **`ctx: dict`** -- orchestration context: **`workspace`**, **`dry_run`**, **`hyperparams`**, **`candidate`**, and other fields built by the project CLI / **`run()`** path.
2. **`params: dict`** -- per-call inputs: **`split`**, **`epoch`**, **`limit`**, command strings, timeouts, and similar.

Constructor kwargs on your **`Module`** subclasses hold static wiring (clients, nested modules, file paths). Do not merge the three sources or read **`os.environ`** inside **`src/autopilot/`**.

## Wiring (no string lookup)

Build a tree of **`Module`** instances in the project overlay and pass the root to **`Trainer.fit(module, ...)`**. **`forward`** takes runtime data only (no string phase dispatch in core). No registry keys.

## Adding a new submodule

1. Subclass **`Module`** (or **`AutoPilotModule`**) in project code under your workspace layout.
2. Implement **`forward`** (and optional **`preflight(ctx) -> list[str]`** on children for **`optimize preflight`**).
3. Assign instances as attributes on the parent module so registration runs.
4. Construct the root module when building **`Trainer`** / **`CLI`**.

## Loss and metrics on the tree

- **`Metric(Module)`** and **`Loss(Module)`** use the same **`Module`** registration path as any other submodule.
- Assign them as attributes on your experiment module so they appear in **`module.modules()`**; **`Trainer.fit`** discovers **`Loss`** / **`Metric`** instances from that walk.

## Key files

- `src/autopilot/core/module.py` -- **`Module`**, **`AutoPilotModule`**
- `src/autopilot/core/parameter.py` -- **`Parameter`**
- `src/autopilot/ai/parameter.py` -- **`PathParameter`**
- `src/autopilot/core/models.py` -- **`Datum`**, **`Result`**
- `src/autopilot/core/metric.py` -- **`Metric(Module)`**, **`CompositeMetric`**
- `src/autopilot/core/loss.py` -- **`Loss(Module)`**
- `src/autopilot/core/trainer.py` -- **`Trainer.fit`**

## Gotchas

- Library code must not access environment variables; pass values through kwargs, **`ctx`**, or **`params`** from the overlay.
- Return **`Datum`** from **`forward`**; use **`success=False`** and **`error_message`** on failure instead of raising for expected backend errors unless your contract says otherwise.
- **`preflight`** returns a **list of error strings**; empty means success.
