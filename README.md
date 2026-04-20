# AutoPilot

What if you could optimize any software system the same way you train a neural network?

AutoPilot is a PyTorch/Lightning-inspired framework for **generalized optimization**. It brings the rigor and developer experience of deep learning to non-differentiable systems. Structured feedback replaces numerical gradients. State mutations (like code edits or config updates) replace weight updates. The same `forward -> loss -> backward -> optimizer.step()` loop that trains neural networks now optimizes prompts, heuristics, rule engines, agents, and configurations—deterministically, with memory, rollback, and policy gating.

## The problem

Building complex, non-differentiable systems—like AI agents, RAG pipelines, fraud detection heuristics, or rule-based engines—is a manual, informal process today. You tweak a prompt or a regex rule, run the system, look at the output, decide if it got better, and repeat. 

This process lacks the structured feedback loop that made deep learning iteration so fast:
- **No Memory**: There is no automatic log of what was already tried. You often re-try the same failed strategy multiple times.
- **No Structured Feedback**: Evaluation is often "looks right to me." There is no quantitative tracking of metrics across held-out validation sets.
- **No Automatic Rollback**: When a change makes things worse, you undo it by hand. If you're not sure, you guess.
- **No Scalability**: One person manually iterating is slow. There is no way to run this overnight, no way to hand it to an autonomous system, and no way to reproduce what happened three experiments ago.

AutoPilot solves this by formalizing the iteration loop into the same structural abstractions that powered the deep learning revolution.

## The core idea: PyTorch for everything else

Optimizing any iterative system follows the same structure as training a neural network. AutoPilot formalizes this mapping into a real, typed interface.

In deep learning, you pass data through a model (**forward pass**). A loss function scores the output. Backpropagation computes **gradients** that explain how parameters should change. An **optimizer** reads those gradients and updates the **weights**. You repeat this in epochs, validate on held-out data, and checkpoint good states.

AutoPilot applies this exact structure to general software optimization:
- **Module** is your system (agent, rule engine, pipeline), exactly like `nn.Module`.
- **Loss** wraps an evaluator (Judge, profiler, or test suite) that produces structured feedback (**gradients**).
- **Parameters** mark what can be edited (prompts, JSON configs, source files via `PathParameter`).
- **Optimizer** applies changes based on gradients—this can be an AI coding agent or a deterministic algorithm.
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
  loss.backward()       # structured feedback fills param.grad
  optimizer.step()      # optimizer applies improvements (e.g. edits code)
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
| `ModelCheckpoint` | `Store` + `StoreCheckpoint` |
| Autograd engine | `Graph` / `Node` (propagates arbitrary objects) |
| `Dataset` / `DataLoader` | `ListDataset` / `DataLoader` |

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

- **Uniform, Typed Interface**: Compose systems the same way you compose PyTorch components.
- **Structured Feedback**: `backward()` fills `param.grad` with actionable feedback, not just opaque scores.
- **Real Code/State Versioning**: `FileStore` uses content addressing and snapshots for atomic rollbacks.
- **Persistent Memory**: `FileMemory` records what was tried and what failed across epochs to block regression-prone strategies.
- **Policy Gating**: Use `MinGate`, `MaxGate`, and `RangeGate` to enforce quality bars and automate early stopping.
- **Production Infrastructure**: Built-in CLI for experiments, project health, dataset management, and audit trails via `--expose`.

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

## Concepts and guides

- [Experiment model](docs/concepts/experiments.md)
- [Dataset model](docs/concepts/datasets.md)
- [Policies](docs/concepts/policies.md)
- [CLI reference](docs/cli/README.md)
