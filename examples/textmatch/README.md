# textmatch

Regex-rule text classifier optimized through the AutoPilot loop. No LLM required -- runs entirely offline.

## What this demonstrates

- `Module` / `AutoPilotModule` with `forward()`, `training_step()`, `validation_step()`
- `PathParameter` declaring mutable files (the rules JSON)
- `Loss` accumulating errors per batch, `backward()` producing structured `RuleGradient`
- `Optimizer` reading `param.grad` and editing rules on disk
- `Metric` tracking accuracy per epoch
- `DataModule` / `DataLoader` / `Dataset` wiring
- `Trainer.fit()` with `Policy`, `Experiment` (owns `FileStore` via `store=`), and `StoreCheckpointCallback`; `Trainer` takes `experiment=`, not `store=`

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
| `textmatch/trainer.py` | `AccuracyPolicy`, `build_trainer()` -- `FileStore` on `Experiment`, `Trainer(..., experiment=..., policy=...)` |
| `textmatch/judge.py` | `RuleJudge` -- failure categorization |
| `textmatch/cli.py` | `TextMatchCLI` -- AutoPilotCLI subclass |
| `run.py` | Manual PyTorch-style loop |
| `run_trainer.py` | Lightning-style `Trainer.fit()` |
| `datasets/` | Train/val/test JSONL (5 items each) |
| `rules/rules.json` | Seed rules (3 categories) |

## Run

```bash
cd examples/textmatch
uv sync
uv run python run.py            # manual loop
uv run python run_trainer.py    # Trainer.fit()
```
