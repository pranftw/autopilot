# AutoPilot

What if you could train an AI agent the same way you train a neural network?

AutoPilot is a PyTorch/Lightning-inspired framework for agent optimization. Text feedback replaces numerical gradients. Code edits replace weight updates. The same `forward -> loss -> backward -> optimizer.step()` loop that trains neural networks now trains AI agents -- deterministically, with memory, rollback, and policy gating.

## The problem

Building agents today is manual. You tweak a prompt, run the agent, look at the output, decide if it got better, and repeat. There is no structured feedback loop. There is no memory of what was already tried. There is no automatic rollback when a change makes things worse.

This affects everyone building agents, from solo developers to large teams. You make a change and run it. If it looks worse, you undo it by hand. If you are not sure, you guess. After a few dozen iterations you have lost track of what you tried at iteration five. Someone else on the team tries the same thing again next week.

Evaluation is informal. "Looks right to me" is the bar. There is no held-out validation set, no quantitative metric tracking, no policy that blocks a bad change from shipping. The agent either seems to work or it doesn't, and the criteria live in someone's head.

Scaling this is not possible. One person manually iterating can improve an agent slowly. But there is no way to run this overnight, no way to hand it to an autonomous system, no way to reproduce what happened three experiments ago.

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) proved the loop works. Give an agent code, let it modify and evaluate, keep improvements, discard regressions, repeat. It ran 100 experiments overnight on a single file. But the entire orchestration lives in a markdown prompt. When to keep, when to discard, how to log results, when to revert are all natural language instructions the agent has to interpret correctly every time.

## The core idea

Optimizing an agent follows the same structure as training a neural network. Not loosely, not as a borrowed analogy. The workflow maps one-to-one if you formalize what each step actually does.

In ML, data flows through a model (forward pass). A loss function scores the output. Backpropagation computes gradients that explain how each parameter should change. An optimizer reads those gradients and updates the weights. You repeat this in epochs, validate on held-out data, checkpoint good states, and stop when quality plateaus.

Agent optimization has the same shape. Eval cases flow through the agent (forward pass). A Judge scores each output and produces structured feedback explaining what went wrong (text gradients). A coding agent reads those gradients and edits the source files (optimizer step). You repeat in epochs, validate on a held-out split, snapshot the code, and roll back if quality drops.

The difference is in what flows through the loop. Gradients are text instead of numbers. Weight updates are code edits instead of tensor arithmetic. The optimizer is a coding agent instead of Adam. But the structure, the flow of information, the separation of concerns, and the lifecycle are identical.

AutoPilot formalizes this mapping into a real typed interface. `Module` is your agent, with `forward(batch)` and auto-registered child modules and parameters, exactly like `nn.Module`. `Loss` wraps a Judge as a loss function. `Parameter` and `PathParameter` mark what the optimizer can edit. `Optimizer.step()` triggers a coding agent that reads `param.grad` and modifies code. `Metric` tracks quantitative progress with `update()` and `compute()`, composable via `+`. `Store` is content-addressed code versioning with snapshot, checkout, diff, branch, and merge. `Policy` with `Gate` hierarchies is your early stopping. `Callback` hooks into every point in the loop. `DataLoader` batches eval items from `Dataset` and `DataModule`.

Every ML workflow concept has a concrete counterpart in AutoPilot. Epochs are real epoch boundaries enforced by the loop. Batches are real batched iteration over eval datasets. Train/val/test splits are real held-out data that catches overfitting. Checkpointing is real content-addressed snapshots. Early stopping is a real policy gate that triggers rollback. These are not borrowed terms. They describe real structural equivalents that behave the same way and benefit from the same infrastructure.

This means everything ML practitioners already know about training loops transfers directly. You think in epochs. You worry about overfitting to your train set. You validate on held-out data. You checkpoint and rollback. You track metrics across runs. The vocabulary is the same because the structure is the same.

## What this gives you

- **A uniform, typed interface.** Every component is a Python object with a known protocol. Module, Loss, Optimizer, Metric, Policy, Store, Memory, Callback. You compose them the same way you compose PyTorch components. No string registries, no YAML configs, no magic wiring. Instantiate objects, pass them in, call methods.

- **The full ML toolkit applied to agent optimization.** Train/val/test splits prevent overfitting. Metrics track progress quantitatively across epochs. Policy gates with MinGate, MaxGate, RangeGate, and CustomGate enforce quality bars. Store checkpoints enable rollback to any previous epoch. Memory tracks what was tried, what worked, and which strategies are blocked.

- **Structured feedback, not opaque scores.** A Judge does not just produce a number. `JudgeLoss` accumulates structured feedback per batch, and `backward()` fills `param.grad` with text that explains what went wrong and why. The optimizer reads these text gradients to make targeted fixes. This is `loss.backward()` for agents.

- **Real code versioning.** `FileStore` uses SHA-256 content addressing, snapshot manifests, and atomic writes. `store.checkout(epoch)` restores code to any previous state. Diff shows what changed between epochs. Branch and merge support parallel experimentation. `StoreCheckpoint` and `StorePromoter` callbacks automate snapshotting and promotion.

- **Memory that persists across epochs.** `FileMemory` records what was tried, whether it improved metrics, and which strategies caused regressions. `MemoryCallback` captures this automatically. The optimizer's strategy blocklist prevents re-trying failed approaches. No ML framework needs this because weight-space optimization is stateless across steps. Agent optimization is not.

- **Deterministic orchestration.** The loop runs the same way every time. `EpochOrchestrator` handles plateau detection, regression rollback, and stop conditions. `RegressionCallback` compares validation metrics to the best baseline and flags regressions. `RunStateCallback` tracks run state for crash detection. Epoch boundaries, evaluation, gating, and rollback are all code.

- **Composability across use cases.** The textmatch example optimizes regex rules with a deterministic `RuleOptimizer` and zero LLM calls. The protim example optimizes a prompt file with `AgentOptimizer` and Claude Code. Same Module, same Loss interface, same Trainer, same Store, different optimizers. You can optimize prompts, code, configs, pipelines, or anything else that can be evaluated and improved through text feedback.

- **Two layers of control.** Write the loop yourself with explicit `loss.backward()` and `optimizer.step()` calls for full control (PyTorch-style). Or define `training_step` and `configure_optimizers` on an `AutoPilotModule` and let `Trainer.fit()` handle the rest (Lightning-style). Same components, same protocols, different levels of automation.

- **Production infrastructure included.** `CostTracker` measures wall-clock time per epoch. `DiagnoseCallback` produces trace diagnostics and node heatmaps. Proposal and verdict tracking manages hypotheses across experiments. A full CLI covers every operation: `optimize loop` for automated training, `store` for version control, `memory` for inspection, `diagnose` and `trace` for debugging, `experiment` and `project` for lifecycle management. The `--expose` flag produces a JSON audit trail of every command.

## Two layers: PyTorch core + Lightning automation

Like PyTorch + Lightning, AutoPilot offers two orchestration layers:

**Manual loop (PyTorch-style)** -- full control, plain Python objects:

```python
from autopilot.ai.coding import ClaudeCodeAgent
from autopilot.ai.loss import JudgeLoss
from autopilot.ai.optimizer import AgentOptimizer
from autopilot.core.module import Module

module = MyModule()
loss = JudgeLoss(judge=MyJudge())
optimizer = AgentOptimizer(agent=ClaudeCodeAgent(), parameters=module.parameters())

module.train()
for epoch in range(5):
    for batch in train_loader:
        data = module(batch)
        loss(data, batch)
    loss.backward()       # text gradients fill param.grad
    optimizer.step()      # coding agent applies improvements
    optimizer.zero_grad()
```

**Automated loop (Lightning-style)** -- define steps, let Trainer handle the rest:

```python
from autopilot.core.module import AutoPilotModule
from autopilot.core.trainer import Trainer

class MyModule(AutoPilotModule):
    def training_step(self, batch):
        return self.forward(batch)

    def configure_optimizers(self):
        return AgentOptimizer(agent=ClaudeCodeAgent(), parameters=self.parameters())

trainer = Trainer(callbacks=[...], policy=my_policy, store=my_store)
trainer.fit(module, train_dataloaders=loader, max_epochs=10)
```

## The ML analogy

In ML, you train a model by passing data through it, computing a loss, backpropagating gradients, and updating weights. AutoPilot applies the same structure to agent optimization -- text feedback replaces numerical gradients, code edits replace weight updates:

| ML workflow | AutoPilot workflow |
| --- | --- |
| Training data (MNIST images) | Eval dataset (test cases with ground truth) |
| Forward pass (`model(x)`) | Run the agent on eval items (`module(batch)`) |
| Loss computation (`criterion(output, target)`) | Judge scores outputs, accumulates structured feedback |
| Backward (`loss.backward()` fills `param.grad`) | Judge feedback becomes text gradients on parameters |
| Optimizer step (`optimizer.step()` updates weights) | Coding agent reads gradients, edits source files |
| Validation (check generalization) | Run on val split, check for regressions |
| Epoch | Full cycle: run all items -> judge -> gradient -> code edit -> redeploy |
| Overfitting | Agent tuned for train set quirks, failing on val/test |
| Early stopping | Policy gate fails -> stop and rollback |
| Checkpoint | Store snapshots code at each epoch, enables rollback |
| Learning rate | How aggressive: prompt tweak vs full architectural restructure |

The key insight: `Datum` is to AutoPilot what `Tensor` is to PyTorch -- the universal data object that flows through the entire loop. Input, output, loss, gradients are all structured around `Datum`.

## Component mapping

| PyTorch / Lightning | AutoPilot |
| --- | --- |
| `nn.Module` | `Module` |
| `LightningModule` | `AutoPilotModule` |
| Lightning `Trainer` | `Trainer` |
| `nn.CrossEntropyLoss` | `Loss` / `JudgeLoss` |
| `optim.Adam` | `Optimizer` / `AgentOptimizer` |
| `nn.Parameter` | `Parameter` / `PathParameter` |
| `Tensor` | `Datum` |
| `torchmetrics.Metric` | `Metric` |
| `EarlyStopping` | `Policy` + `Gate` |
| `ModelCheckpoint` | `Store` + `StoreCheckpoint` |
| Autograd engine | `Graph` / `Node` |
| `Dataset` / `DataLoader` | `ListDataset` / `DataLoader` |
| Lightning `Callback` | `Callback` |
| Lightning `FitLoop` | `Loop` / `EpochLoop` |
| No equivalent | `Memory` (persistent cross-epoch learning) |
| No equivalent | `DataGenerator` (structured dataset creation) |
| No equivalent | `Judge` (structured output scoring) |

## Examples

See [examples/](examples/) for runnable, self-contained projects:

- **[textmatch](examples/textmatch/)** -- regex-rule optimization with Module, Loss, Optimizer, Trainer, Policy, Store. No LLM required.
- **[protim](examples/protim/)** -- agent-optimized prompt with ClaudeCodeAgent for both inference and code editing. Requires `claude` CLI.

Each example is its own uv package. Clone, `cd examples/<name>`, `uv sync`, `uv run python run.py`.

## Quick start

```bash
uv sync && uv run autopilot --help
```

Install the `autopilot` package, then use the `autopilot` CLI from the project root.

## Key commands

| Command | Role |
| --- | --- |
| `optimize` | Drive the optimization loop |
| `ai` | Dataset generation and judging |
| `experiment` | Create, list, and manage experiment slugs and manifests |
| `project` | Create, list, and check project health |
| `workspace` | Init layout, health checks, tree under `autopilot/` |
| `dataset` | Registry, splits, validation |
| `policy` | Policies, scoring, and gate inspection |
| `store` | Content-addressed code versioning |
| `report` | Summaries and reporting |
| `promote` | Promotion decisions and workflow |

Run `uv run autopilot <command> --help` for subcommands and flags.

## Package layout

```
src/autopilot/
  core/         # Module, Trainer, Loss, Optimizer, Parameter, Graph, Metric, Memory, Store, Experiment
  data/         # Dataset, ListDataset, StreamingDataset, DataLoader, DataModule
  ai/           # DataGenerator, Judge, Agent, AgentOptimizer, JudgeLoss, step workflows
  cli/          # argparse CLI, commands, context, output
  tracking/     # manifest, events, command history
  policy/       # Policy, Gate base classes
```

## AI module

The `autopilot.ai` module provides dataset generation, judging, and agent-driven optimization:

- **DataGenerator**: slot planning, step-based LLM workflows, checkpointing, stratified splitting
- **Judge**: per-item step workflows, checkpointing, summary aggregation
- **Steps**: `LLMStep` (structured output), `PythonStep` (deterministic code), `BackStep` (conditional retry)
- **Agent / AgentOptimizer**: wraps a coding agent (e.g. Claude Code) as an optimizer that reads text gradients and applies code changes
- **JudgeLoss**: wraps a Judge as a Loss -- accumulates feedback per batch, `backward()` fills `param.grad` with structured text
- **Data**: `ListDataset`, `StreamingDataset`, `StratifiedSplitter`, `SlotPlanner`
- **Runtime**: `RPMLimiter`, `ParallelRunner` for rate-limited concurrent execution

Subclass `DataGenerator` and `Judge` in project overlays. Pass instances directly -- no registries.

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

Use `--project` / `-p` to select a project, or let AutoPilot auto-detect from CWD under `autopilot/projects/`:

```bash
autopilot ai generate run -p my-project --config gen.json
autopilot project list
autopilot project init new-project
autopilot project doctor my-project
```

## Concepts and guides

- [Experiment model](docs/concepts/experiments.md)
- [Dataset model](docs/concepts/datasets.md)
- [Policies](docs/concepts/policies.md)
- [CLI reference](docs/cli/README.md)
