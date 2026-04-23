# AutoPilot Design Philosophy

AutoPilot's architecture blends the simple explicit core of PyTorch with the
higher-level compositions of PyTorch Lightning. Users can work at either
layer depending on their needs -- core for maximum control, trainer for
convenience.

## Core Principles

### Principle 1: Usability over performance

AutoPilot's primary goal is usability. We maintain flexibility for developers and
autonomous agents building on top of our abstractions. We avoid restriction-first
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

### Principle: isinstance on core classes only

`isinstance` checks against core framework classes (`Parameter`, `Module`,
`AutoPilotModule`, `Gradient`, `Datum`) are fine -- that's how PyTorch
works (`isinstance(value, Parameter)` in `Module.__setattr__`,
`isinstance(param, Tensor)` in `Optimizer`). What's banned is `isinstance`
against concrete leaf types (`PathParameter`, `TextGradient`,
`ClaudeCodeAgent`). These break extensibility because a new subclass won't
be recognized. Instead, base classes expose methods that subclasses
override -- `Parameter.render()`, `Parameter.snapshot()`,
`Gradient.render()`, `Gradient.accumulate()`. Adding a new Parameter,
Gradient, or Loss subclass requires zero framework changes. Built-in
concrete types exist for DRY, not for special-casing.

### Principle: Store decoupled from parameter types

The Store interacts with parameters exclusively through `snapshot()`
(capture managed content) and `restore()` (restore from snapshot).
It never imports concrete parameter types, never probes for domain-
specific attributes. A `PathParameter` snapshots file contents; a
`PromptParameter` snapshots prompt text; the store doesn't know or
care. Note: `snapshot()`/`restore()` is the content versioning API,
separate from `state_dict()`/`load_state_dict()` which is the uniform
checkpointing API across Module, Optimizer, Callback, etc.

### Principle: Public extension methods

All customization hooks are public methods -- never underscore-prefixed.
A user customizing any component overrides a public method with a clear
contract, the same way they override `forward()` on a Module. Examples:
`AgentCollator.build_prompt()`, `AgentOptimizer.build_context()`,
`Parameter.render()`, `Gradient.render()`. This ensures a familiar,
consistent DX across the framework without mandating a universal method name.

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
- **Optimizer + Agent**: intelligent automation. LLM-backed optimization within
  the code-driven loop. `AgentOptimizer` composes an `Agent` with context;
  `JudgeLoss` bridges judge feedback into typed gradients. Deterministic
  optimizers (like `RuleOptimizer`) skip the LLM entirely.

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

## Three Layers

### Core layer

The PyTorch-style layer. Everything is explicit:

    data = module(batch)                       # returns Datum
    loss(data, targets)                        # Loss accumulates
    loss.backward()                            # assigns param.grad
    optimizer.step()                           # applies changes
    metric.update(data)                        # Metric tracks state
    result = compute_result(observation, gates) # Result with gates

No Trainer, no callbacks, no indirection. Maximum control. Use this when
you need to understand exactly what's happening or build something the
Trainer doesn't support yet.

### Trainer layer

The Lightning-style layer. Composes core components:

    trainer = Trainer(callbacks=[...], policy=my_policy, experiment=my_experiment)
    trainer.fit(
      module,
      train_dataloaders=train_loader,
      val_dataloaders=val_loader,
      max_epochs=10,
    )

Constructor injection -- all components are passed as objects, not looked up
by string key. Callback system for cross-cutting concerns. Automated
orchestration for the optimization loop.

### Optimizer + Agent layer

The third layer. The Agent provides intelligent intervention at hook points
within the code-driven loop:

    from autopilot.ai.gradient import ConcatCollator
    from autopilot.ai.loss import JudgeLoss
    from autopilot.core.module import Module


    class MyModule(Module):
      def __init__(self):
        super().__init__()
        self.backend = MyBackendModule(...)
        self.loss = JudgeLoss(judge=MyJudge(), collator=ConcatCollator())
        self.metric = MyMetric()  # project-defined Metric

      def forward(self, batch):
        return self.backend(batch)

The Optimizer protocol defines how optimization happens (`step()` reads
accumulated gradients from Loss). `Agent` is one backend: `AgentOptimizer`
composes an `Agent` and calls `forward(prompt, context=...)` for LLM-driven
edits. But optimizers can also be fully deterministic (like `RuleOptimizer`)
with zero LLM calls. Like PyTorch's `Optimizer`/`Adam` split -- the base
class defines the protocol, concrete implementations provide the strategy.

## Module / Trainer Ownership

Following Lightning's ownership model:

**Module owns** (domain/experiment concerns):
- `forward(batch)` -- the primary computation
- Child modules as attributes (auto-registered via `__setattr__`)
- Parameters, Loss, Metrics as attributes
- `AutoPilotModule(Module)` adds step methods (`training_step`,
  `validation_step`) and `configure_optimizers()`

**Trainer owns** (orchestration):
- `policy` + gates (like Lightning's `EarlyStopping` callback)
- `callbacks` for cross-cutting concerns
- `loop` (defaults to `EpochLoop`; `EpochOrchestrator` adds `should_rollback` / plateau / richer stop reasons)
- optional `experiment`, `logger`, `dry_run`, `accumulate_grad_batches`
- `fit()` drives DataLoader iteration, `max_epochs`, and which step method to call; `max_epochs` is not a `Trainer` field

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
`forward(batch)`, assign child modules and parameters as attributes.

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

### No phase strings in forward

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
- `ConcatCollator` and `AgentCollator` are built-in gradient collators. Users
  subclass `GradientCollator` for custom collation logic.
- `CheckpointIO` (`ai/checkpoints.py`) provides a default append-only JSONL
  backend for generator/judge runs. `JSONCheckpoint` (`core/checkpoint.py`)
  handles experiment manifest persistence. Like Lightning's `CheckpointIO`.
- `Checkpointable` protocol: anything with `state_dict()`/`load_state_dict()` can
  be persisted through checkpoints. Like PyTorch's Module state dict pattern.
- `JudgeLoss`, `AgentOptimizer`, `PathParameter`, `FileStore`, `StoreCheckpointCallback`,
  `StorePromoterCallback`, `ClaudeCodeAgent` are built-in implementations for the
  agent-driven optimization loop.

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

Each step type overrides `execute()` for polymorphic dispatch; the workflow
engine calls `await step.execute()` uniformly. `StepLoopback` is a sentinel
returned by `BackStep.execute()` to signal loopback.

These step-based workflows power `DataGenerator` and `Judge` -- structured
pipelines where the step order is deterministic. The `Agent` abstraction is
complementary: autonomous tool-loop agents where the LLM decides what to do.

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
  `Loss` / `JudgeLoss` with forward/backward split
- PyTorch `Optimizer` / `Adam`: inspiration for `Optimizer` / `AgentOptimizer`
- `torchmetrics`: inspiration for `Metric` design
- TypeScript coder agent (`~/Documents/coder`): influence for the
  `Agent` / `ClaudeCodeAgent` abstraction
- The Zen of Python: explicit is better than implicit, simple is better
  than complex, flat is better than nested
