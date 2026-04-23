# protim

System prompt optimized through the AutoPilot loop with an LLM coding agent (Claude Code). Requires the `claude` CLI.

## What this demonstrates

- `ClaudeCodeAgent` used for both inference and optimization (two instances, different tool access)
- `PathParameter` declaring the prompt file as a mutable parameter
- `Loss` producing text gradients from QA failures
- `AgentOptimizer` passing gradients to a coding agent that edits the prompt
- The complete forward -> loss -> backward -> optimizer.step() loop with an LLM-backed optimizer

## How it works

A QA assistant answers factual questions using a system prompt. The optimization loop:

1. **Forward**: inference agent (no tools) answers each question using `prompts/system.txt`
2. **Loss**: accumulate wrong answers with expected vs actual
3. **Backward**: structure failures into a text gradient on the `PathParameter`
4. **Optimizer step**: optimizer agent (file tools) reads the gradient and edits `system.txt`

Two `ClaudeCodeAgent` instances:
- **Inference** (`allowed_tools=[]`): pure reasoning, no file access. Answers questions.
- **Optimizer** (`allowed_tools=['Edit', 'Write', 'Read']`): reads gradient feedback, edits the prompt file.

## Files

| File | What it does |
| --- | --- |
| `protim/module.py` | `PromptModule`, `PromptLoss`, `QAAccuracyMetric` |
| `protim/trainer.py` | `AccuracyPolicy`, `build_trainer()` -- `FileStore` on `Experiment`, `Trainer(..., experiment=..., policy=...)` |
| `protim/data.py` | `QADataset`, `QADataModule` |
| `run.py` | Manual loop: forward -> loss -> backward -> agent step |
| `run_trainer.py` | Lightning-style `Trainer.fit()` |
| `datasets/train.jsonl` | 8 factual QA items |
| `prompts/system.txt` | Seed system prompt (intentionally minimal) |

## Prerequisites

Install the [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code):

```bash
npm install -g @anthropic-ai/claude-code
```

## Run

```bash
cd examples/protim
uv sync
uv run python run.py
```

The script copies `prompts/` to `_work/prompts/` so the original seed prompt is preserved. Each epoch prints accuracy and the updated prompt.
