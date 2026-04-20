# AutoPilot Design Philosophy

AutoPilot's architecture blends the simple explicit core of PyTorch with the
higher-level compositions of PyTorch Lightning. Users can work at either
layer depending on their needs -- core for maximum control, trainer for
convenience.

## Core Principles

### Principle 1: Usability over performance

AutoPilot's primary goal is usability. We maintain flexibility for agents and
researchers building on top of our abstractions. We avoid restriction-first
regimes without a clear-eyed view of the tradeoffs.

In concrete terms: we don't impose rigid schemas, locked-down config
formats, or opaque abstractions to gain marginal performance. If a design
choice makes the system harder to understand, debug, or extend, it needs
a compelling justification.

### Principle 2: Simple over easy

Explicit is better than implicit. Simple is better than complex.

The core exposes simple building blocks -- modules, policies, metrics --
as plain Python objects. You instantiate them, call their methods,
see exactly what's happening. No string-key lookups, no hidden registries,
no magic wiring. Simple solutions are immediately understandable and
debuggable.

This doesn't mean higher-level "easy" APIs aren't valuable. The Trainer
layer provides exactly that. But the simple core exists underneath, and
users can always drop down to it when they need to leave the beaten path.
Not automating at the start allows us to reach better automation faster.

### Principle 3: Python first

AutoPilot is not a config format with Python bindings. Components are Python
objects used naturally. You can write a custom module in Python, compose
it with existing components, and pass it directly to the Trainer -- like you
would use NumPy, scikit-learn, or PyTorch itself.

### Principle 4: Code skeleton, agent intelligence

The optimization loop skeleton -- Loop.run, step methods, callbacks, policy
gates -- is deterministic code. At epoch boundaries, `optimizer.step()` applies
changes. The code decides WHEN things happen; the agent decides WHAT to change.
Like PyTorch: the training loop controls flow, `model(x)` computes output,
`optimizer.step()` updates weights.

### Principle 5: Progressive disclosure

Three layers, each building on the one below:

- **Core**: maximum control. Use modules, policies, metrics directly.
  Write your own loop.
- **Trainer**: convenience. Constructor injection, automated orchestration,
  callbacks for cross-cutting concerns.
- **Optimizer + Agent** (planned -- see VISION.md): intelligent automation.
  The agent provides creative reasoning within the code-driven loop.

### Principle 6: Workflows in code, not config

No TOML, YAML, or JSON config files for workflow structure. Phases, loops,
and orchestration are Python code. Configuration flows through constructor
kwargs, function args, and Module attributes -- never through config files
that impose structure from outside.

### Principle 7: Agent-friendly data surface

All tracking state -- manifests, events, metric records, traces, experiment
histories -- is stored in agent-friendly MD and JSON files. The framework
writes this data deterministically; the agent reads and reasons over it.
This gives the agent total visibility into everything: approaches explored,
experiments performed, decisions backed by evidence. Orchestration is code,
data is agent-readable files.

## Two Layers

### Core layer

The PyTorch-style layer. Everything is explicit:

    datum = module(phase, ctx, params)          # returns Datum
    result = metric.to_result(datum.metrics)   # Metric -> Result
    gate_result = policy(result, phase)        # Policy -> GateResult

No Trainer, no callbacks, no indirection. Maximum control. Use this when
you need to understand exactly what's happening or build something the
Trainer doesn't support yet.

> The vision (see VISION.md) evolves this to: `data = module(batch)`,
> `loss(data, targets)`, `metrics.update(data)`, `loss.backward()`,
> `optimizer.step()` -- matching PyTorch's batch-iteration pattern.

### Trainer layer

The Lightning-style layer. Composes core components:

    trainer = Trainer(callbacks=[...])
    trainer.fit(module, experiment_dir, max_epochs=5,
                phases=['deploy', 'train', 'validate'])

Constructor injection -- all components are passed as objects, not looked up
by string key. Callback system for cross-cutting concerns. Automated
orchestration for the optimization loop.

> The vision (see VISION.md) evolves this to:
> `trainer.fit(module, train_loader, val_loader)` with batch-iteration,
> matching Lightning's pattern.

### Vision: Optimizer + Agent layer (planned -- see VISION.md)

The third layer -- planned, not yet implemented. The Agent provides
intelligent intervention at hook points within the code-driven loop:

    class MyModule(Module):
        def __init__(self):
            super().__init__()
            self.backend = MyBackendModule(...)
            self.loss = JudgeLoss(judge=MyJudge())
            self.metrics = AccuracyMetric()

        def forward(self, batch):
            return self.backend(batch)

The Optimizer protocol defines how optimization happens (`step()` reads
accumulated gradients from Loss); the Agent provides the intelligence behind
it (`forward(prompt, context)`). Like PyTorch's `Optimizer`/`Adam` split,
but with an LLM agent doing the creative reasoning instead of gradient math.

## Module / Trainer Ownership

Following Lightning's ownership model:

**Module owns** (domain/experiment concerns):
- `forward(phase, ctx, params)` -- the primary computation (current signature)
- Child modules as attributes (auto-registered via `__setattr__`)
- Policy and metric as attributes

> The vision (see VISION.md Section 12) evolves Module to:
> `forward(batch)` only, sub-modules as attributes (like `nn.Module`
> children), parameters as attributes, `self.loss`, `self.metrics`.
> A separate `AutoPilotModule(Module)` adds step methods and
> `configure_optimizers()`, matching `LightningModule(nn.Module)`.

**Trainer owns** (orchestration):
- `policy` + gates (like Lightning's `EarlyStopping` callback)
- `callbacks` for cross-cutting concerns
- `loop` and `max_epochs`
- DataLoader iteration and which step method to call

Step methods take **only data** -- no phase strings, no ctx dicts, no
infrastructure params. Deploy/infrastructure is a lifecycle hook or
callback, not a step method. This mirrors how PyTorch's `forward()` takes
only tensors; train/eval mode is module state, not a forward argument.

## Naming Conventions

Following PyTorch's naming philosophy:

- The base class IS the name: `Module` not `BaseModule`, `Policy` not
  `BasePolicy`. Same as PyTorch's `Module`, `Optimizer`, `Dataset`.
- Implementations are descriptive: `QualityFirstPolicy`,
  `AgentOptimizer`, `JudgeLoss`, `CodingAgent`.
  Semantic names that describe what they do.
- Protocols carry a `Protocol` suffix: `PolicyProtocol`.
  These exist for structural typing alongside the base classes.
- Files match their primary class: `module.py` contains `Module`,
  `policy.py` contains `Policy`, `loss.py` contains `Loss`.

## Extension Model

To define an experiment module: subclass `Module`, override
`forward(phase, ctx, params)`, assign adapters as attributes.

To add a new policy: subclass `Policy`, override `forward()` and
`explain()`, pass to `Trainer`.

To add a metric: subclass `Metric`, override `update(datum)` and
`compute()`. `CompositeMetric` composes multiple via `+`.

To add a new loop: subclass `Loop`, override `run()`.

To add cross-cutting behavior: subclass `Callback`, override the relevant
hooks, pass the instance to `Trainer`.

To add a CLI command: subclass `Command`, override `forward()`. Subclass
`CLI` for project-level entry points.

To add an eval generator: subclass `DataGenerator`, override
`create_slots()`, `define_steps()`, `assemble_item()`, `stratify_key()`.

To add a judge: subclass `Judge`, override `define_steps()`,
`assemble_result()`, `build_summary()`.

No registration. No config files required. Pure Python objects.

> Everything is Module (like `nn.Module`), with additional abstractions:
> Loss, Optimizer, Agent, Parameter, Store.

## Anti-Patterns

AutoPilot explicitly avoids these patterns:

### No registries or magic variables

Components are Python objects passed directly. Never:
- String-key registries (`register('my_adapter', MyAdapter)`)
- Magic variable names (`COMPONENTS = ...`, `__adapter__ = ...`)
- Config files that force specific Python structure
- Decorator-based registration (`@register_adapter`)

### No rigid config protocols

Projects customize by writing Python scripts that import the framework
and call it -- the same way you use PyTorch or Lightning. The framework
provides `CLI` / `AutoPilotCLI`; the project subclasses and passes its objects
from `__init__`:

```python
from autopilot.cli.main import AutoPilotCLI


class MyCLI(AutoPilotCLI, project='my-project'):
  def __init__(self):
    super().__init__()
    self.module = my_module
    self.generator = MyGenerator()
    self.judge = MyJudge()


MyCLI()()
```

No forced file names, no required exports, no schema to conform to.
Just Python objects and a small CLI subclass.

### No phase strings in forward (target)

> Note: the current `Module.forward(phase, ctx, params)` signature
> predates this principle. VISION.md Section 12 Phase 3 migrates to
> `forward(batch)` with no phase strings.

Step methods should take data, not orchestration context:
- `forward(batch, context={'workspace': ...})` -- infrastructure config
  belongs on `__init__` or module state
- `forward(batch, mode='train')` -- train/eval is module state
  (`self.training`), not a forward argument

## Built-in and Extensible

Following PyTorch's pattern of shipping useful defaults alongside extensible base
classes:

- `SlotPlanner` and `VarDef` are built-in components for variable/distribution-based
  slot generation. Most projects use them directly; those needing something different
  override `create_slots()` entirely.
- `QualityFirstPolicy` and `QualityFirstMetric` are built-in policy and metric
  implementations. `MinGate`, `MaxGate`, `RangeGate`, `CustomGate` are built-in gates.
- `CheckpointIO` provides a default append-only JSONL backend. Override for cloud
  storage or databases. Like Lightning's `CheckpointIO`.
- `Checkpointable` protocol: anything with `state_dict()`/`load_state_dict()` can
  be persisted through checkpoints. Like PyTorch's Module state dict pattern.

> Vision-level built-ins (planned, see VISION.md): `JudgeLoss`,
> `AgentOptimizer`, `PathParameter`, `FileStore`, `StoreCheckpoint`,
> `StorePromoter`, `CodingAgent`.

## Step-Based Workflows

LLM operations use explicit step-based workflows instead of tool-call-driven
agent loops:

- `LLMStep`: produces structured data via pydantic-ai with typed output.
  Tools can be attached when genuinely needed, but workflow control stays with code.
- `PythonStep`: regular Python function for execution, validation, API calls.
  Fully deterministic.
- `BackStep`: conditional loopback with iteration limits. Formalizes retry logic.

Code controls the entire flow. LLM steps only fill in structured content. This
gives reproducible, debuggable pipelines versus non-deterministic tool-call loops.

These step-based workflows power `DataGenerator` and `Judge` -- structured
pipelines where the step order is deterministic.

> The planned `Agent` abstraction (see VISION.md) is complementary:
> autonomous tool-loop agents where the LLM decides what to do.

## Base vs Overlay

Clear separation between what lives in the base library vs project overlays:

- **Base library**: protocols, base classes, data abstractions (`Datum`,
  `Dataset`), checkpoint infrastructure, CLI commands, runtime utilities.
- **Overlay (project)**: concrete implementations (subclass DataGenerator/Judge),
  Pydantic output models for LLM steps, system prompts, domain-specific validation
  functions, taxonomy definitions.

The test: "Would this exist in a translation eval? A RAG eval?" If yes, it belongs
in base. If it's project-specific, it belongs in the overlay.

## Influences

- PyTorch design principles: usability over performance, simple over easy,
  Python first (https://pytorch.org/docs/stable/community/design.html)
- PyTorch Lightning: Trainer/Callback/Module separation, constructor
  injection, hook-based lifecycle, Fabric as the simple layer,
  `configure_optimizers()` on Module, `EarlyStopping` as Callback
- PyTorch `Tensor`: inspiration for `Datum` as the universal data object
- PyTorch `nn.CrossEntropyLoss` / `loss.backward()`: inspiration for
  planned `Loss` / `JudgeLoss` with forward/backward split
- PyTorch `Optimizer` / `Adam`: inspiration for planned `Optimizer` / `AgentOptimizer`
- `torchmetrics`: inspiration for `Metric` design
- TypeScript coder agent (`~/Documents/coder`): influence for the
  planned Agent / CodingAgent abstraction
- The Zen of Python: explicit is better than implicit, simple is better
  than complex, flat is better than nested
