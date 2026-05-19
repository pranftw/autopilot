# textmatch

Regex-rule text classifier optimized through the AutoPilot loop. No LLM required -- runs entirely offline.

## What this demonstrates

- `Module` / `AutoPilotModule` with `forward()`, `training_step()`, `validation_step()`
- `PathParameter` declaring mutable files (the rules JSON)
- `Loss` accumulating errors per batch, `backward()` producing structured `RuleGradient`
- `Optimizer` reading `param.grad` and editing rules on disk
- `Metric` tracking accuracy per epoch
- `DataModule` / `DataLoader` / `Dataset` wiring
- `Trainer.fit()` with `Policy`, `AutoPilotExperiment`, `FileStore`, and `StoreCheckpointCallback`; `Trainer` takes `experiment=` and `store=`

## How it works

A support ticket classifier matches text against regex rules and assigns categories (billing, technical, account, etc.). The optimization loop:

1. **Forward**: classify each eval item against current rules
2. **Loss**: accumulate failures (no match, wrong category)
3. **Backward**: structure failures into a `RuleGradient` on the `PathParameter`
4. **Optimizer step**: read the gradient, add missing patterns, refine wrong rules
5. **Validation**: check accuracy on val split

## Files

| File | What it does |
| --- | --- |
| `textmatch/module.py` | `TextMatchModule`, `TextMatchLoss`, `AccuracyMetric`, `RuleGradient` |
| `textmatch/optimizer.py` | `RuleOptimizer` -- reads gradients, edits `rules.json` |
| `textmatch/data.py` | `TextMatchDataset`, `TextMatchDataModule` |
| `textmatch/trainer.py` | `AccuracyPolicy`, `build_trainer()` -- `FileStore` + `AutoPilotExperiment`, `Trainer(..., experiment=..., store=..., policy=...)`. `next_slug()` generates sequential experiment IDs |
| `textmatch/judge.py` | `RuleJudge` -- failure categorization |
| `textmatch/cli.py` | `TextMatchCLI` -- AutoPilotCLI subclass with wired module + datamodule |
| `run.py` | Manual PyTorch-style loop |
| `run_trainer.py` | Lightning-style `Trainer.fit()` |
| `datasets/` | Train/val/test JSONL (5 items each) |
| `rules/rules.json` | Seed rules (3 categories) |

## Before you run

This example uses a path dependency on the `autopilot` package (see `pyproject.toml`).
You must run from within the autopilot repository checkout.

**Workspace scaffold for `-p textmatch`:** The `autopilot -p textmatch` project CLI
requires the AutoPilot workspace layout. Initialize from the repo root using
`workspace init` (and optionally `project init` for project-specific scaffolding):

```bash
autopilot workspace init --context 'bootstrap textmatch workspace'
```

After initialization, run commands from the `examples/textmatch` directory
(matching the working directory assumption in `run_trainer.py`):

```bash
cd examples/textmatch
autopilot -p textmatch optimize train --context 'initial training run'
```

Without the workspace scaffold, `-p textmatch` commands will fail with a
configuration error.

## Run

```bash
cd examples/textmatch
uv sync
uv run python run.py            # manual loop (defaults)
uv run python run_trainer.py    # Trainer.fit() (defaults)
```

## CLI flags

### run_trainer.py

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--rules-dir` | PATH | `./rules` | Rules directory |
| `--datasets-dir` | PATH | `./datasets` | Datasets directory |
| `--store-path` | PATH | `./.store` | FileStore path |
| `--max-epochs` | int | 5 | Number of training epochs |
| `--threshold` | float | 0.30 | Accuracy policy threshold |
| `--accumulate-grad-batches` | int | 100 | Gradient accumulation batches |
| `--experiment` | SLUG | auto | Experiment slug |
| `--json` | flag | off | Emit structured JSON output |

### run.py

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--rules-dir` | PATH | `./rules` | Rules directory |
| `--datasets-dir` | PATH | `./datasets` | Datasets directory |
| `--max-epochs` | int | 5 | Number of training epochs |
| `--json` | flag | off | Emit structured JSON output |

## Agent usage

### Mutating commands require `--context`

All mutating autopilot commands require `--context '<reason>'`. Read-only commands
(`query`, `debug`, `tree list`) are exempt. The context flows to the experiment's
decision journal and the execution log.

### Parameterized scripts

```bash
uv run python run_trainer.py --max-epochs 2 --threshold 0.5 --json
uv run python run.py --max-epochs 3 --json
```

### Inline execution with autopilot execute

```bash
autopilot execute -c "
from textmatch.module import TextMatchModule
m = TextMatchModule('rules')
print(len(list(m.parameters())))
" --context 'inspect module parameters'
```

### File mode

```bash
autopilot execute run_trainer.py --max-epochs 2 --context 'quick 2-epoch test run'
```

### Module mode

```bash
autopilot execute -m textmatch.module --context 'validate module import'
```

### Stdin pipe (avoids escaping)

```bash
echo 'from textmatch.module import TextMatchModule
m = TextMatchModule("rules")
for p in m.parameters():
    print(p.source, p.pattern)' | autopilot execute --context 'list rule file paths'
```

### Escaping tips

- For `-c` mode, prefer single-quoted code for `$`, `{}`, or f-strings: `autopilot execute -c 'print(f"x={42}")'`
- Use stdin pipe for complex multi-line code -- avoids all escaping issues
- Do NOT use `--` between code and extra args
- Autopilot global flags (`--experiment`, `--json`) are consumed by autopilot, not forwarded
- Global flags go before the `execute` subcommand's own arguments
