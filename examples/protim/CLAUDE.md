# protim -- Agent Operations Guide

You are an external coding agent working on the **protim** project, powered by the
AutoPilot framework. This document is your operational manual. Every decision you
make must be traceable, every action must carry a reason, and every experiment must
be tracked. AutoPilot gives you total visibility and control -- use it.

## Mental Model

AutoPilot is PyTorch/Lightning for non-differentiable optimization. The core idea:
any iterative improvement process (code, prompts, configs, infrastructure) can be
expressed as the ML training loop: `forward -> loss -> backward -> optimizer.step()`.

- **Module** = your system (code, prompts, configs)
- **Parameter** = what you optimize (files via PathParameter, values via ScalarParameter)
- **Loss** = how you measure quality (JudgeLoss for LLM-judged, programmatic for metrics)
- **Gradient** = feedback on what to improve (TextGradient from judges, NumericGradient from metrics)
- **Optimizer** = how you apply changes (AgentOptimizer for LLM-driven, custom for rules)
- **Trainer** = orchestration (epoch loop, callbacks, policy gates, checkpoints)
- **DataModule** = structured evaluation data (train/val/test splits, samplers)
- **Store** = git for parameters (content-addressed snapshots, branches, tags, reflog)
- **Forest/Tree/Node** = experiment tracking (hypotheses as trees, experiments as nodes)
- **IsolatedEnvironment** = each experiment gets its own worktree (no cross-contamination)

## Getting Started: How to Wire the Loop

When you arrive at a new project, your first job is to figure out the automated
optimization loop. The end goal is **full automation**: `Trainer.fit()` handles
everything -- evaluation, feedback, changes, quality gates, versioning.

### Step 1: Explore and understand the project

```bash
autopilot workspace status --json
autopilot tree list --json
autopilot query --all-trees --json
autopilot query --deployed --all-trees --json
```

Read the codebase. Understand what the project does, what files are the
"parameters" (the things you'd change to improve quality), and how quality is
currently measured (tests, benchmarks, user feedback, metrics).

### Step 2: Identify the loop components

Every component is fully extensible. You do NOT need built-in datasets or losses.
The project's own infrastructure IS your evaluation. Ask yourself:

- **What are the parameters?** Which files/values, when changed, improve the system?
  These become `PathParameter` (for files) or `ScalarParameter` (for numeric knobs).
- **What is "forward"?** How do you run the system against an input and get output?
  This becomes `Module.forward()` / `training_step()`.
- **What is the evaluation data?** You do NOT need to create separate JSONL files.
  The project's own test suite, benchmarks, contract tests, or workload traces ARE
  the dataset. Write a custom `Dataset` that wraps whatever evaluation already
  exists (e.g., `pytest --collect-only` to discover test items). Adding a new test
  = adding a row to the dataset. This is test-driven development via AutoPilot.
- **What is "loss"?** How do you measure if the output is good or bad? The project's
  own quality tools (pytest, mypy, linters, benchmarks) ARE the loss function. Write
  a custom `Loss` that runs them and turns failures into gradients.
- **What produces feedback?** What tells you specifically WHAT to change?
  This becomes your `Gradient` (TextGradient for semantic, NumericGradient for numeric).
  Test failure output with stack traces = rich TextGradient content.
- **What makes changes?** An LLM agent editing files, or a programmatic rule engine?
  This becomes your `Optimizer` (AgentOptimizer or custom subclass).
- **What are the quality gates?** What must never regress? Write custom gates for
  each quality dimension: `MinGate('test_pass_rate', 1.0)`, `MaxGate('lint_errors', 0)`,
  `CustomGate(fn=lambda v: ...)` for arbitrary predicates.
- **What is the deployment lifecycle?** Staging, production, rollback?
  Wire these into the loop via callbacks, deployment labels, and store tags.

### Step 3: Try a first pass manually via CLI

Before wiring the full loop, do a quick manual pass to validate your understanding:

```bash
# run the system against one example
autopilot execute -c 'from protim.module import ...; result = m.forward(batch); print(result)'

# check what parameters exist
autopilot execute -c 'from protim.module import ...; print(list(m.parameters()))'
```

### Step 4: Wire the automated loop

Write the Module, Loss, Optimizer, DataModule, and `run_trainer.py`:

```python
# run_trainer.py -- the fully automated loop
trainer = Trainer(
  config=config,
  store=store,
  experiment=experiment,
  callbacks=[StoreCheckpointCallback(), CostTrackerCallback()],
  policy=QualityFirstPolicy(gates=[MinGate('pass_rate', 0.95)]),
)
trainer.fit(module, datamodule=dm, max_epochs=10)
```

Once `trainer.fit()` runs, everything is automated: evaluation, gradient
computation, optimizer changes, policy gate checks, store snapshots, context
logging. No manual intervention needed.

### Step 5: Wire in the full lifecycle

The dream is full automation of the entire lifecycle:
- **New feature request** -> create experiment, write code, run evals
- **Improvement detected** -> deploy to staging (store tag + deployment label)
- **Staging passes** -> promote to production
- **Regression detected** -> rollback to last known good
- **Cycle continues** -> new experiments, evolving datasets, continuous improvement

Wire staging/deployment checks into callbacks or post-training scripts:

```bash
# after trainer.fit completes with good metrics:
autopilot experiment deploy <slug> --as staging --context 'metrics passed all gates'

# run staging validation (separate eval pass)
autopilot execute run_staging_tests.py

# if staging passes:
autopilot experiment deploy <slug> --as production --replace \
  --context 'staging passed, promoting to production'
```

## The Outer Loop: Agent as Orchestrator

`Trainer.fit()` handles the INNER loop (per-epoch optimization). YOU are the
OUTER loop. Your job is to:

1. **Decide what to try next** based on experiment history and gradients
2. **Create experiments** with clear hypotheses
3. **Run `Trainer.fit()`** for each experiment
4. **Analyze results** across experiments and trees
5. **Decide: continue, branch, deploy, or rollback**
6. **Handle lifecycle events**: client requirement changes, dataset evolution,
   production incidents, budget constraints

```bash
# orient: what do I know?
autopilot query --all-trees --json
autopilot recommend --metric <primary_metric> --json

# decide: what should I try?
autopilot tree create <hypothesis-name> --context 'why this approach'
autopilot experiment add --id <slug> --hypothesis '...' --context '...'

# execute: run the automated loop
autopilot execute run_trainer.py --max-epochs 5

# analyze: did it work?
autopilot experiment show <slug> --json
autopilot experiment compare <baseline> <candidate> \
  --higher-metric accuracy --lower-metric latency_ms --json

# decide: what next?
autopilot recommend --metric <primary_metric> --json
```

## Always Use --json

Every command supports `--json` for structured output. Always use it. Parse the
JSON to make decisions programmatically. Never rely on text output for automation.

## Always Use --context

Every mutating command requires `--context 'reason'`. This is not optional -- it
builds the decision journal that lets you (or the next agent) understand WHY
something was done. Write context as if explaining to a colleague who has zero
prior knowledge.

Good: `--context 'deploying exp-reranker-v3: accuracy improved 5% over baseline with acceptable latency trade (180ms vs 150ms, within 300ms SLA)'`

Bad: `--context 'deploying'`

## Key Commands Reference

### Workspace & Trees

| Command | Purpose | Context Required |
|---------|---------|-----------------|
| `workspace status --json` | Full workspace overview (description, trees, health, recent activity) | No |
| `workspace doctor --json` | Health check (forest, store, directories) | No |
| `tree create <name> --context '...'` | New exploration direction | Yes |
| `tree list --json` | All trees with descriptions | No |
| `tree switch <name> --context '...'` | Change active tree | Yes |

### Experiments

| Command | Purpose | Context Required |
|---------|---------|-----------------|
| `experiment add --id <slug> --hypothesis '...' --context '...'` | Start new experiment | Yes |
| `experiment complete <slug> --metrics 'JSON' --context '...'` | Record results | Yes |
| `experiment fail <slug> --error '...' --context '...'` | Record failure | Yes |
| `experiment show <slug> --json` | Full details (parent, tree, metrics, deployed_as) | No |
| `experiment show <slug> --context-log --json` | Decision journal for this experiment | No |
| `experiment compare <a> <b> --higher-metric X --lower-metric Y --json` | Direction-aware comparison | No |
| `experiment impact <slug> --json` | Dependencies, children, dependents | No |
| `experiment deploy <slug> --as <label> --context '...'` | Mark as deployed | Yes |
| `experiment deploy <slug> --as <label> --replace --context '...'` | Rotate deployment | Yes |
| `experiment undeploy <label> --context '...'` | Remove deployment label | Yes |
| `experiment metadata set <slug> <key> <value> --context '...'` | Record config | Yes |
| `experiment metadata show <slug> --json` | View all metadata | No |
| `experiment notes write <slug> --body '...' --context '...'` | Add detailed notes | Yes |

### Query & Recommend

| Command | Purpose | Context Required |
|---------|---------|-----------------|
| `query --json` | All experiments in active tree | No |
| `query --all-trees --json` | All experiments across all trees | No |
| `query --sort <metric> --json` | Sorted by metric (descending) | No |
| `query --sort <metric> --asc --json` | Sorted ascending (lowest first) | No |
| `query --best <metric> --all-trees --json` | Single best experiment | No |
| `query --metric-gt X:N --metric-lt Y:M --best Z --json` | Constrained best | No |
| `query --deployed --all-trees --json` | What's currently deployed | No |
| `query --context-contains 'text' --json` | Search notes and context logs | No |
| `recommend --metric <name> --json` | Recommendation (deploy/rollback/continue/branch/investigate) | No |

### Store (Parameter Versioning)

| Command | Purpose | Context Required |
|---------|---------|-----------------|
| `store snapshot --context '...'` | Version current parameter state | Yes |
| `store checkout --epoch N --context '...'` | Restore parameter state | Yes |
| `store stash --context '...'` | Temporarily shelve changes | Yes |
| `store stash-pop --context '...'` | Restore shelved changes | Yes |
| `store doctor --json` | Store health check | No |
| `store doctor --repair --context '...'` | Fix repairable issues | Yes |
| `store tag create <name> --epoch N --context '...'` | Tag a release | Yes |
| `store tag verify <name> --json` | Verify tag integrity | No |
| `store recover --reflog-entry N --context '...'` | Restore from reflog | Yes |
| `debug store reflog --json` | Full store operation history | No |

### Analysis & Reports

| Command | Purpose | Context Required |
|---------|---------|-----------------|
| `report trend <metric> --json` | Metric trajectory in active tree | No |
| `policy check --json` | Do current metrics pass all gates? | No |
| `policy explain --json` | Gate details with hints | No |
| `debug executions list --json` | Full command history with context | No |

## Decision Patterns

### "Should I deploy this experiment?"

```bash
# 1. check policy gates
autopilot policy check --json
# if passed=true, gates are satisfied

# 2. compare against current production
autopilot experiment compare <deployed> <candidate> \
  --higher-metric accuracy --lower-metric latency_ms --json
# check verdict: improved/regressed/inconclusive

# 3. get recommendation
autopilot recommend --metric accuracy --json
# follow the action: deploy/continue/investigate

# 4. if deploying
autopilot experiment deploy <candidate> --as production --replace \
  --context 'accuracy improved 5% (0.85->0.90), latency within SLA (180ms < 300ms)'
```

### "Something broke in production -- what happened?"

```bash
# 1. what's deployed?
autopilot query --deployed --all-trees --json

# 2. when was it deployed and why?
autopilot experiment show <deployed-id> --context-log --json

# 3. what changed between the previous and current deployment?
autopilot experiment compare <previous-deployed> <current-deployed> \
  --higher-metric accuracy --lower-metric latency_ms --json

# 4. check store for parameter changes
autopilot debug store reflog --json

# 5. roll back if needed
autopilot experiment deploy <previous-deployed> --as production --replace \
  --context 'rolling back: regression detected in <metric>'
```

### "A new tree or continue the current one?"

```bash
# check current tree's trend
autopilot report trend accuracy --json
# if direction=plateau -> new tree (different approach)
# if direction=improving -> continue (more of the same)
# if direction=degrading -> investigate (something went wrong)

# check recommendation
autopilot recommend --metric accuracy --json
# if action=branch -> create new tree
```

### "The client changed requirements mid-flight"

```bash
# 1. record the change
autopilot experiment metadata set <current> requirements_version v2 \
  --context 'client added latency SLA: must be under 200ms'

# 2. re-evaluate all experiments against new constraints
autopilot query --metric-gt accuracy:0.85 --metric-lt latency_ms:200 \
  --best accuracy --all-trees --json

# 3. if nothing meets constraints, branch
autopilot tree create latency-focused \
  --context 'client added 200ms latency SLA, existing approaches too slow'

# 4. if something already meets constraints, deploy it
autopilot experiment deploy <id> --as production --replace \
  --context 'meets new SLA: accuracy 0.87, latency 180ms'
```

## Recording Metadata

Always record experiment configuration as metadata. This is how you (or the next
agent) can reproduce results:

```bash
autopilot experiment metadata set <slug> model gpt-4o --context 'recording config'
autopilot experiment metadata set <slug> temperature 0.7 --context 'recording config'
autopilot experiment metadata set <slug> prompt_version v3 --context 'recording config'
autopilot experiment metadata set <slug> chunk_size 512 --context 'recording config'
```

## Error Recovery

| Situation | Recovery |
|-----------|----------|
| Wrong experiment deployed | `experiment deploy <correct> --as <label> --replace --context 'correcting deployment'` |
| Wrong epoch checked out | `store recover --reflog-entry <N> --context 'reverting checkout'` |
| Parameter files corrupted | `store checkout --epoch <last-good> --context 'restoring from known-good state'` |
| Store corruption detected | `store doctor --repair --context 'repairing store'` |
| Need to undo WIP changes | `store stash --context 'shelving WIP'` then `store checkout --epoch <N>` |

## What NOT to Do

- Do not edit `.autopilot/` files directly. Use CLI commands.
- Do not skip `--context` on mutating commands. The decision journal is critical.
- Do not compare experiments by eyeballing metrics. Use `experiment compare --json`.
- Do not deploy without checking policy gates first.
- Do not create experiments without a hypothesis. The hypothesis explains what you expect.
- Do not ignore `inconclusive` verdicts. Investigate the tradeoffs before deciding.
- Do not assume metrics direction. Use `--higher-metric` / `--lower-metric` explicitly.
- Do not run parallel CLI commands that mutate the same workspace. Serialize mutations.

## Real-World Situations

### New feature request from client

```bash
# 1. record the requirement
autopilot experiment metadata set <current> requirements_version v2 \
  --context 'client wants order tracking in addition to refunds'

# 2. evolve the dataset (add new test scenarios for the feature)
# edit datasets/train.jsonl, datasets/val.jsonl

# 3. create experiment for the feature
autopilot tree create order-tracking \
  --context 'client requested order tracking capability'
autopilot experiment add --id order-v1 \
  --hypothesis 'adding order tracking tools and updating prompt' \
  --context 'new client requirement, expanding scope'

# 4. run the automated loop with evolved dataset
autopilot execute run_trainer.py --max-epochs 5

# 5. compare against baseline
autopilot experiment compare <baseline> order-v1 \
  --higher-metric task_success_rate --json
```

### Dataset evolution

As the project evolves, new test scenarios surface. The dataset grows and changes.
AutoPilot tracks this via `DatasetFingerprint`:

```bash
# check if dataset has drifted since last experiment
autopilot execute -c '
from autopilot.ai.fingerprint import DatasetFingerprint
fp = DatasetFingerprint.from_file("datasets/train.jsonl")
print(fp.hexdigest)
'

# record fingerprint as experiment metadata
autopilot experiment metadata set <slug> dataset_fingerprint <hex> \
  --context 'recording dataset version for reproducibility'
```

### Something broke in production

```bash
# 1. what's deployed?
autopilot query --deployed --all-trees --json

# 2. what happened? trace the full decision history
autopilot experiment show <deployed-id> --context-log --json

# 3. what changed between previous and current deployment?
autopilot experiment compare <previous> <current> \
  --higher-metric accuracy --lower-metric latency_ms --json

# 4. check parameter change history
autopilot debug store reflog --json

# 5. rollback
autopilot experiment deploy <previous> --as production --replace \
  --context 'rolling back: regression in <metric>, root cause: <explanation>'

# 6. fix: create new experiment from rolled-back state
autopilot tree create hotfix-<issue> --context 'fixing regression: <details>'
```

### Budget constraints

```bash
# check current spending
autopilot debug cost --json

# query experiments within budget
autopilot query --all-trees --metric-lt cost_usd:50 --best accuracy --json
```

## Python API: Wiring the Full Loop

The Python API is how you wire Module/Loss/Optimizer/DataModule for `Trainer.fit()`:

```python
from pathlib import Path
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.callbacks.checkpoint import CheckpointCallback
from autopilot.core.callbacks.store import StoreCheckpointCallback
from autopilot.core.profiler import SimpleProfiler
from autopilot.ai.experiment import AutoPilotExperiment
from autopilot.ai.store import FileStore
from autopilot.ai.environment import IsolatedEnvironment
from autopilot.policy.policy import QualityFirstPolicy
from autopilot.policy.gate import MinGate

# wire everything
config = AutoPilotConfig(workspace=Path('.'))
store = FileStore(config)
experiment = AutoPilotExperiment(experiment_id='my-experiment')
env = IsolatedEnvironment(ignore_patterns=['.autopilot', '__pycache__'])

trainer = Trainer(
  config=config,
  store=store,
  experiment=experiment,
  callbacks=[
    StoreCheckpointCallback(),
    CheckpointCallback(directory=Path('checkpoints')),
  ],
  policy=QualityFirstPolicy(gates=[MinGate('pass_rate', 0.95)]),
  profiler=SimpleProfiler(),
)

# this runs the ENTIRE automated loop
trainer.fit(module, datamodule=datamodule, max_epochs=10)
```

Key API patterns:

- `Trainer.fit(module, max_epochs=N)` -- max_epochs is on fit(), not Trainer constructor
- `Metric.compute()` returns `dict[str, float]`, not a scalar
- `Optimizer.step()` takes no arguments (params accessed via self)
- `Result.passed` is a computed property from `gates`, not a constructor argument
- `select(datum, index)` -- datum first, index second
- `TextGradient(text='...')` -- `text=` parameter, not `direction=`
- `ContextLog.append(entry)` -- accepts pre-built ContextEntry directly
- `CheckpointCallback(directory=...)` -- not `dirpath`
- `Experiment(experiment_id='...')` -- not `id`
- `Forest.create_tree(name)` -- not `add_tree(tree)`
- `Tree(name, store=store)` -- store is required
- `PathParameter(source='path/as/string')` -- source must be str, not Path

## Exit Codes

- 0: success
- 1: operation failed (gate failure, unhealthy state, command error)
- 2: argument error (wrong flags, missing required args)

Always check exit codes in automated flows.
