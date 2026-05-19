# Agent Harness Optimization

End-to-end agent harness optimization example using AutoPilot. A multi-turn,
tool-using retail customer service agent optimized via gradient-free
optimization with policy gates, parameter versioning, and experiment tracking.

## Architecture

```
scenarios/*.jsonl
      |
      v
HarnessDataModule (train/val/test splits)
      |                                             DatasetFingerprint
      v                                             (experiment.dataset_meta stamp)
HarnessModule (forward pass = run conversation)
  |-- system_prompt.md  (PathParameter)
  |-- policies.md       (PathParameter)
  |-- retail_tools.py   (PathParameter -- Python code!)
  |-- HarnessAgent (Pydantic AI + OpenRouter)
  |-- UserSimulator (scripted responses)
  |-- RetailDB (in-memory JSON database)
      |
      v
ConversationEvaluator -> EvaluationResult
      |
      +-- use_judge=True (default):
      |     JudgeLoss + AgentCollator + TextGradient
      |       |-- HarnessJudge (PythonStep: deterministic scoring)
      |       |                (LLMStep: HarnessVerdict critique)
      |       |-- PydanticAgent -> AgentCollator (Gemma-4 collator)
      |       v
      |     Per-parameter TextGradient (direction, attribution, severity, evidence)
      |
      +-- use_judge=False:
            HarnessLoss -> HarnessGradient (structured failure buckets)
      |
      v
HarnessMetrics (12 metrics: 9 quality + 3 cost tracking)
      |
      v
EpochOrchestrator (plateau detection + auto-rollback)
      |
      v
QualityFirstPolicy (gates: min thresholds)
      |
      v
FileStore (content-addressed parameter snapshots)

Offline evaluation (ai judge):
  HarnessCLI.judge = HarnessJudge
    autopilot -p harness ai judge run --input <path>
```

## Setup

```bash
cd examples/harness
uv sync --extra dev
```

This installs runtime dependencies (autopilot, pydantic-ai) plus dev tools
(pytest, ruff, tau2 converter). Set `OPENROUTER_API_KEY` in your environment
for inference.

## Quick Start

### Standalone run (single forward pass)

```bash
uv run python run.py
```

### Parameterized trainer (multi-epoch)

```bash
uv run python run_trainer.py --max-epochs 3
uv run python run_trainer.py --max-epochs 3 --no-judge   # heuristic loss
uv run python run_trainer.py --max-epochs 3 --use-judge   # explicit judge mode
```

### Via CLI

```bash
# initialize workspace (first time)
autopilot -p harness --workspace . workspace init --context 'bootstrap harness workspace'

# run optimization (judge mode by default)
autopilot -p harness --workspace . optimize loop --max-epochs 5 --json --context 'initial optimization run'

# heuristic loss mode
autopilot -p harness --workspace . optimize loop --max-epochs 5 --no-judge --json --context 'heuristic loss baseline'

# offline judge evaluation (read-only, no --context required)
autopilot -p harness ai judge run --input harness/scenarios/val.jsonl

# check results (read-only, no --context required)
autopilot -p harness --workspace . --experiment <slug> --epoch <N> propose verify --proposal-id <id> --json
```

## How Optimization Works

1. **Epoch**: Run all training scenarios through the agent, collect metrics.
2. **Loss**: When `use_judge=True` (default), `JudgeLoss` + `AgentCollator` produce
   per-parameter `TextGradient` with direction, attribution, severity, and evidence.
   When `use_judge=False`, `HarnessLoss` categorizes failures into structured buckets.
3. **Gradient**: Rendered feedback for the optimizer agent (semantic, not numeric).
4. **Optimizer step**: ClaudeCodeAgent rewrites prompt/tool files based on gradients.
5. **Policy gates**: Check if metrics meet thresholds. If not, rollback to last
   accepted epoch.
6. **Store snapshot**: Version parameter files for the accepted epoch.

`EpochOrchestrator` monitors `task_success_rate`, detects plateaus
(window=3, threshold=0.01), and auto-rolls back on metric regression.

Repeat for `max_epochs`. Each epoch's metrics are persisted to
`epoch_N_metrics.json` for traceability.

## Context / Decision Journal

Every experiment accumulates a **decision journal** (`experiment.context_log`) that
records *why* actions were taken -- not just what metrics resulted.

Context entries come from three sources:

1. **Programmatic emission** via `trainer.emit_context()` -- callbacks like
   `OptimizerContextCallback` emit entries when significant state transitions
   occur (e.g. validation metric improves over prior best).

2. **Script-level decisions** via `experiment.add_context()` -- `run_trainer.py`
   records initial configuration choices (max epochs, judge mode, model) before
   the training loop starts.

3. **CLI `--context` flag** -- every mutating autopilot command requires
   `--context '<reason>'`. This flows to `experiment.add_context(source='user')`
   on the experiment and to `ExecutionRecord.context` in the execution log.

### Inspecting the context log

```bash
autopilot -p harness --experiment <slug> experiment show --context-log
autopilot -p harness --experiment <slug> experiment show --context-log --context-source harness --limit 5
```

### What ends up in the journal

| Source | When | Example reason |
|--------|------|----------------|
| `'examples.harness.run_trainer'` | Before `trainer.fit()` | "Chose max_epochs=5 with env=dev, use_judge=True" |
| `'harness'` | Val metric improves | "harness optimization decision: val improved vs prior best" |
| `'trainer'` | Experiment completes/fails | "experiment completed successfully" |
| `'policy'` | Policy gate accepts/rejects | "epoch 2 accepted by policy gate" |
| `'user'` | CLI `--context` on mutating commands | Agent-supplied reason string |

## Framework Primitives

| AutoPilot Concept | Harness Role |
|-------------------|--------------|
| `JudgeLoss` | Semantic loss from judge evaluation (use_judge=True) |
| `TextGradient` | Per-parameter attributed feedback (direction, severity, evidence) |
| `AgentCollator` | Batch critique via `PydanticAgent` (Gemma-4); produces `TextGradient` |
| `HarnessJudge` (`JudgeAgent`) | Hybrid `PythonStep` + `LLMStep` for `ai judge` pipeline |
| `PythonStep` / `LLMStep` | Deterministic scoring + structured LLM critique steps |
| `HarnessVerdict` | Structured LLM output (score, critique, dimension_scores, recommendations) |
| `EpochOrchestrator` | Plateau detection, auto-rollback, enriched stop reasons |
| `DatasetFingerprint` | Scenario content hashing for drift detection and reproducibility |
| `FileStore` | Content-addressed parameter versioning and merge workflow |
| `store merge-*` | Three-step merge: analysis -> preview -> resolve -> apply |
| `propose verify` | `MetricsComparator` against baseline/candidate experiments |
| `experiment compare` | Side-by-side metric comparison across experiments |
| `debug cost` | Cost attribution from `CostTrackerCallback` |
| `QualityFirstPolicy` | Multi-gate acceptance with min-threshold checks |
| `PathParameter` | Optimizable file content (prompts, policies, tool code) |
| `AgentOptimizer` | ClaudeCodeAgent-driven parameter editing |

## Dev vs Prod

Control via `HARNESS_ENV` environment variable:

| Setting | `HARNESS_ENV=dev` (default) | `HARNESS_ENV=prod` |
|---------|-------|------|
| Max epochs | 5 | 10 |
| Max turns | 15 | 10 |
| task_success_rate gate | >= 0.3 | >= 0.7 |
| tool_recall gate | >= 0.4 | >= 0.8 |
| tool_precision gate | -- | >= 0.7 |
| policy_compliance gate | -- | >= 0.8 |

**Promotion workflow**: Optimize in dev (relaxed gates) -> verify improvement ->
switch to `HARNESS_ENV=prod` -> run with strict gates -> stabilize.

## File Structure

```
examples/harness/
  pyproject.toml              # project config, autopilot path-sourced
  README.md                   # this file
  AGENT_GUIDE.md              # agent-facing workflow docs
  AUTOPILOT_LEARNINGS.md      # dogfooding insights
  run.py                      # standalone run script
  run_trainer.py              # parameterized trainer script
  scripts/
    convert_tau_data.py       # tau-bench -> JSONL converter
    run_feature_request.py    # feature request pipeline
  harness/
    __init__.py
    module.py                 # HarnessModule (AutoPilotModule, use_judge toggle)
    trainer.py                # build_trainer (EpochOrchestrator, fingerprint)
    cli.py                    # HarnessCLI + HarnessOptimizeCommand (--use-judge)
    judge.py                  # HarnessJudge + HarnessVerdict (hybrid eval)
    agents.py                 # PydanticAgent (AgentCollator bridge)
    environments.py           # dev/prod presets
    callbacks.py              # MetricsWriter, OptimizerContext, Deploy
    agent.py                  # HarnessAgent (Pydantic AI)
    tool_loader.py            # dynamic tool exec loading
    database.py               # RetailDB (in-memory)
    simulator.py              # UserSimulator (scripted)
    evaluator.py              # ConversationEvaluator
    loss.py                   # HarnessLoss + HarnessGradient (use_judge=False)
    metrics.py                # 12 metrics in MetricCollection
    data.py                   # HarnessDataset + HarnessDataModule
    prompts/
      system_prompt.md        # agent system prompt (optimized)
      policies.md             # business policy rules (optimized)
    tools/
      retail_tools.py         # tool code (optimized Python)
    scenarios/
      train.jsonl             # training scenarios (from tau-bench)
      val.jsonl               # validation scenarios
      test.jsonl              # test scenarios
    db/
      retail.json             # mock retail database
  tests/
    __init__.py
    conftest.py               # shared fixtures
    test_data.py              # dataset/datamodule tests
    test_database.py          # RetailDB tests
    test_tool_loader.py       # tool loading tests
    test_agent.py             # agent runtime tests
    test_simulator.py         # simulator tests
    test_evaluator.py         # evaluator tests
    test_loss.py              # loss/gradient tests
    test_metrics.py           # metrics tests
    test_module.py            # module integration tests (dual-mode)
    test_trainer.py           # trainer wiring tests (orchestrator, fingerprint)
    test_cli.py               # CLI registration + judge/optimize flag tests
    test_judge.py             # HarnessJudge + HarnessVerdict tests
    test_callbacks.py         # callback behavior tests
    test_environments.py      # environment preset tests
    test_integration.py       # end-to-end pipeline integration tests
```
