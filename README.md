# AutoPilot

What if you could optimize any software system the same way you train a neural network?

AutoPilot is a PyTorch/Lightning-inspired framework for **generalized optimization**. It brings the rigor and developer experience of deep learning to non-differentiable systems. Structured feedback replaces numerical gradients. State mutations (like code edits or config updates) replace weight updates. The same `forward -> loss -> backward -> optimizer.step()` loop that trains neural networks now optimizes prompts, heuristics, rule engines, agents, and configurations -- deterministically, with memory, rollback, and policy gating.

## The problem

Building complex, non-differentiable systems -- like AI agents, RAG pipelines, fraud detection heuristics, or rule-based engines -- is a manual, informal process today. You tweak a prompt or a regex rule, run the system, look at the output, decide if it got better, and repeat.

This process lacks the structured feedback loop that made deep learning iteration so fast:

- **No Memory**: There is no automatic log of what was already tried. You often re-try the same failed strategy multiple times.
- **No Structured Feedback**: Evaluation is often "looks right to me." There is no quantitative tracking of metrics across held-out validation sets.
- **No Automatic Rollback**: When a change makes things worse, you undo it by hand. If you're not sure, you guess.
- **No Scalability**: One person manually iterating is slow. There is no way to run this overnight, no way to hand it to an autonomous system, and no way to reproduce what happened three experiments ago.

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) proved the loop works. Give an agent code, let it modify and evaluate, keep improvements, discard regressions, repeat. It ran 100 experiments overnight on a single file. But the entire orchestration lives in a markdown prompt. When to keep, when to discard, how to log results, when to revert are all natural language instructions the agent has to interpret correctly every time.

AutoPilot solves this by formalizing the iteration loop into the same structural abstractions that powered the deep learning revolution.

## The core idea: PyTorch for everything else

Optimizing any iterative system follows the same structure as training a neural network. AutoPilot formalizes this mapping into a real, typed interface.

In deep learning, you pass data through a model (**forward pass**). A loss function scores the output. Backpropagation computes **gradients** that explain how parameters should change. An **optimizer** reads those gradients and updates the **weights**. You repeat this in epochs, validate on held-out data, and checkpoint good states.

AutoPilot applies this exact structure to general software optimization:

- **Module** is your system (agent, rule engine, pipeline), exactly like `nn.Module`.
- **Loss** wraps an evaluator (Judge, profiler, or test suite) that produces structured feedback (**gradients**).
- **Parameters** mark what can be edited (prompts, JSON configs, source files via `PathParameter`).
- **Optimizer** applies changes based on gradients -- this can be an AI coding agent or a deterministic algorithm.
- **Backward** propagates structured feedback through the computation graph.
- **Step** triggers the update to the underlying parameters.

The difference is in what flows through the loop. Gradients can be text, JSON, or any arbitrary Python object. Weight updates can be code edits, file rewrites, or config tweaks. But the structure, the separation of concerns, and the lifecycle are identical to the PyTorch experience you already know.

## What you can optimize

AutoPilot is built for extreme extensibility. As long as you can define a forward pass and a way to score the result, you can optimize it:

- **Prompt & AI Pipelines**: Tune system prompts, RAG chunking parameters, or multi-agent routing logic based on LLM-judged evaluations.
- **Heuristic & Rule Engines**: Evolve fraud detection thresholds, spam filters, or trading algorithms where loss is based on precision/recall metrics.
- **Configuration Tuning**: Optimize database settings, cache eviction policies, or compiler flags using performance profiling reports as structured gradients.
- **Simulation & Game Balancing**: Adjust unit stats, physics parameters, or generation seeds based on win-rate or equilibrium metrics.
- **Code Performance**: Refactor SQL queries or tight loops using `EXPLAIN ANALYZE` plans and profiler outputs as structured feedback for a coding optimizer.

## The ML analogy

AutoPilot isn't just a borrowed analogy; it's a structural equivalent that transfers everything ML practitioners know about training loops directly to software engineering:

| ML workflow | AutoPilot workflow |
| --- | --- |
| Training data | Eval dataset (test cases with ground truth) |
| Forward pass (`model(x)`) | Run the system on eval items (`module(batch)`) |
| Loss computation | Evaluator scores outputs, accumulates structured feedback |
| Backward (`loss.backward()`) | Feedback flows back to fill `param.grad` with "gradients" |
| Optimizer step (`optimizer.step()`) | Optimizer reads gradients and applies state mutations |
| Validation | Run on held-out split to check for regressions |
| Epoch | One full cycle: run all items -> judge -> gradient -> update -> redeploy |
| Overfitting | System tuned for train set quirks, failing on val/test |
| Checkpoint | Store snapshots code/config at each epoch, enabling rollback |

## How to use AutoPilot

1. **Model your system** as a `Module` with `forward(batch)`. Declare what can change as `Parameter` attributes -- files via `PathParameter`, or custom subclasses for configs, prompts, thresholds.

2. **Define a Loss** that accumulates per-batch feedback in `forward()` and fills `param.grad` with a structured `Gradient` in `backward()`. This isn't just a number -- it tells the optimizer WHERE something failed and WHAT to fix.

3. **Choose an Optimizer**: deterministic (like `RuleOptimizer` -- reads gradients, applies heuristic fixes with zero LLM calls) or LLM-backed (`AgentOptimizer` with `ClaudeCodeAgent` -- reads gradients, edits code and prompts).

4. **Run the loop** -- either a manual PyTorch-style `for epoch` loop, or `Trainer.fit()` which handles batching, validation, callbacks, and gradient accumulation automatically.

5. **Wire experiment lifecycle** for production: `Experiment` manages the manifest and optional `Store` for content-addressed snapshots. `StoreCheckpointCallback` auto-snapshots each epoch. `Policy` gates progression and triggers rollback on regression. `Memory` blocks failed strategies so the optimizer doesn't repeat mistakes.

Two entry points: **library** (import and compose in Python) and **CLI** (`uv run autopilot ...`) for workspace operations -- experiments, store history, memory queries, status, proposals.

## Why not just a for loop?

A hand-rolled `for epoch: run(); eval(); if bad: revert()` works for one-off tweaking. It breaks down when you need:

- **Structured feedback** that tells the optimizer WHERE and WHAT to fix -- `Loss.backward()` produces typed `Gradient` on each `Parameter`, not just "accuracy dropped"
- **Gradient accumulation** across batches with correct step boundaries -- `accumulate_grad_batches` on `Trainer`, automatic `_should_step` logic in `EpochLoop`
- **Train/val split discipline** with separate metric phases -- `EpochLoop` switches `module.eval()`, runs `validation_step`, calls `experiment.on_validation_complete` after val
- **Policy gating with automatic rollback** to the correct epoch via content-addressed snapshots -- `Policy` returns pass/fail; `EpochOrchestrator` calls `experiment.rollback(best_epoch)`
- **Persistent memory** of what was tried, what failed, and which strategies are blocked -- `FileMemory` with `learn()`, `recall()`, `block_strategy()`
- **Reproducible experiment records** with manifests, events, and artifacts -- `Experiment` with `JSONLogger`, `JSONCheckpoint`, epoch directories
- **The same Module** working in both a manual loop and an automated Trainer -- progressive disclosure from explicit to orchestrated

AutoPilot standardizes all of this into a composable protocol with the same separation of Module / Loss / Optimizer / Trainer that made PyTorch productive for ML.

## Two layers: PyTorch core + Lightning automation

Like PyTorch + Lightning, AutoPilot offers two orchestration layers:

**Manual loop (PyTorch-style)** -- full control, plain Python objects:

```python
from autopilot.ai.coding import ClaudeCodeAgent
from autopilot.ai.gradient import ConcatCollator
from autopilot.ai.loss import JudgeLoss
from autopilot.ai.optimizer import AgentOptimizer
from autopilot.core.module import Module

module = MyModule()
loss = JudgeLoss(judge=MyJudge(), collator=ConcatCollator())
optimizer = AgentOptimizer(agent=ClaudeCodeAgent(), parameters=module.parameters())

module.train()
for epoch in range(5):
  for batch in train_loader:
    data = module(batch)
    loss(data, batch)
  loss.backward()       # structured feedback fills param.grad
  optimizer.step()      # optimizer applies improvements (e.g. edits code)
  optimizer.zero_grad()
```

**Automated loop (Lightning-style)** -- define steps, let Trainer handle the rest:

```python
from autopilot.ai.coding import ClaudeCodeAgent
from autopilot.ai.optimizer import AgentOptimizer
from autopilot.core.module import AutoPilotModule
from autopilot.core.trainer import Trainer

class MyModule(AutoPilotModule):
  def training_step(self, batch):
    return self.forward(batch)

  def configure_optimizers(self):
    return AgentOptimizer(agent=ClaudeCodeAgent(), parameters=self.parameters())

trainer = Trainer(callbacks=[...], policy=my_policy, experiment=my_experiment)
trainer.fit(module, train_dataloaders=loader, max_epochs=10)
```

## Component mapping

| PyTorch / Lightning | AutoPilot |
| --- | --- |
| `nn.Module` | `Module` |
| `LightningModule` | `AutoPilotModule` |
| Lightning `Trainer` | `Trainer` |
| `nn.CrossEntropyLoss` | `Loss` / `JudgeLoss` |
| `optim.Adam` | `Optimizer` / `AgentOptimizer` |
| `nn.Parameter` | `Parameter` / `PathParameter` |
| `Tensor` | `Datum` / `Gradient` (can be any object) |
| `torchmetrics.Metric` | `Metric` |
| `EarlyStopping` | `Policy` + `Gate` |
| `ModelCheckpoint` | `Store` + `StoreCheckpointCallback` |
| Autograd engine | `Graph` / `Node` (propagates arbitrary objects) |
| `Dataset` / `DataLoader` | `ListDataset` / `DataLoader` |
| Lightning `Callback` | `Callback` |
| Lightning `FitLoop` | `Loop` / `EpochLoop` |
| No equivalent | `Memory` (persistent cross-epoch learning) |
| No equivalent | `DataGenerator` (structured dataset creation) |
| No equivalent | `Judge` (structured output scoring) |

## Examples

See [examples/](examples/) for runnable, self-contained projects:

- **[textmatch](examples/textmatch/)** -- **Deterministic Rule Optimization.** Optimizes regex rules using a deterministic `RuleOptimizer` and zero LLM calls. Shows the power of the framework without AI.
- **[protim](examples/protim/)** -- **Agent-Driven Prompt Optimization.** Optimizes a prompt file using `AgentOptimizer` and Claude Code.

Each example is its own uv package. Clone, `cd examples/<name>`, `uv sync`, `uv run python run.py`.

## Quick start

```bash
uv sync && uv run autopilot --help
```

## Key features

- **Uniform, Typed Interface**: Compose systems the same way you compose PyTorch components. No string registries, no YAML configs. Instantiate objects, pass them in, call methods.
- **Structured Feedback**: `backward()` fills `param.grad` with actionable feedback, not just opaque scores. The optimizer reads `param.grad.render()` and `param.render()` to make targeted fixes.
- **Real Code/State Versioning**: `FileStore` uses SHA-256 content addressing, snapshot manifests, and atomic writes. `store.checkout(epoch)` restores any previous state.
- **Persistent Memory**: `FileMemory` records what was tried, what failed, and which strategies are blocked across epochs. `MemoryCallback` captures this automatically.
- **Policy Gating**: Use `MinGate`, `MaxGate`, `RangeGate`, and `CustomGate` to enforce quality bars and automate early stopping with rollback.
- **Experiment Lifecycle**: `Experiment` manages store, lifecycle hooks (`on_epoch_complete`, `on_validation_complete`, `on_loop_complete`), rollback, and best-epoch tracking above the training loop.
- **Production Infrastructure**: Built-in CLI for experiments, project health, dataset management, diagnostics, and audit trails via `--expose`.

## Key commands

| Command | Role |
| --- | --- |
| `optimize` | Drive the optimization loop |
| `ai` | Dataset generation and judging |
| `experiment` | Create, list, and manage experiment slugs and manifests |
| `project` | Create, list, and check project health |
| `store` | Content-addressed code versioning |
| `status` | Experiment overview (epoch, metrics, stop reason) |
| `memory` | Query, record, trends, and context |
| `diagnose` | Trace diagnostics and node heatmaps |
| `propose` | Create, verify, revert, and list proposals |
| `promote` | Promotion decisions and workflow |

Run `uv run autopilot <command> --help` for subcommands and flags.

## Package layout

```
src/autopilot/
  core/         # Module, Trainer, Loss, Optimizer, Parameter, Gradient, Graph, Metric, Memory, Store, Experiment
  data/         # Dataset, ListDataset, StreamingDataset, DataLoader, DataModule
  ai/           # DataGenerator, Judge, Agent, AgentOptimizer, JudgeLoss, TextGradient, GradientCollator, step workflows
  cli/          # argparse CLI, commands, context, output
  tracking/     # manifest, events, command history
  policy/       # Policy, Gate base classes
```

## Multi-project workspaces

AutoPilot supports multiple projects in one workspace under `autopilot/projects/<name>/`:

```
workspace/
  autopilot/
    pyproject.toml
    projects/
      my-project/
        cli.py
        trainer.py
        ai/
        experiments/
        datasets/
```

Each project has a `cli.py` that subclasses `AutoPilotCLI` and wires components in `__init__`:

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

## Documentation

Comprehensive documentation lives in source docstrings. See [PHILOSOPHY.md](PHILOSOPHY.md) for design
principles. CLI command details are in the `cli-conventions` skill and source docstrings.
