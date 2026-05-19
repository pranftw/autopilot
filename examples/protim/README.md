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

Two `ClaudeCodeAgent` instances (both use `model='haiku'` for cost efficiency):
- **Inference** (`allowed_tools=[]`): pure reasoning, no file access. Answers questions.
- **Optimizer** (`allowed_tools=['Edit', 'Write', 'Read']`): reads gradient feedback, edits the prompt file.

## Files

| File | What it does |
| --- | --- |
| `protim/module.py` | `PromptModule`, `PromptLoss`, `QAAccuracyMetric` |
| `protim/trainer.py` | `AccuracyPolicy`, `build_trainer()` -- `FileStore` + `AutoPilotExperiment`, `Trainer(..., experiment=..., store=..., policy=...)` |
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

Each epoch prints accuracy and the updated prompt. The prompt is edited in place at `prompts/system.txt`.

## Agent usage

### Mutating commands require `--context`

All mutating autopilot commands require `--context '<reason>'`. Read-only commands
(`query`, `debug`, `tree list`) are exempt. The context flows to the experiment's
decision journal and the execution log.

### Parameterized scripts

```bash
uv run python run.py
uv run python run_trainer.py --max-epochs 3
```

### Inline execution with autopilot execute

```bash
autopilot execute -c "
from protim.module import PromptModule
m = PromptModule('prompts')
print(len(list(m.parameters())))
" --context 'inspect prompt module parameters'
```

### File mode

```bash
autopilot execute run.py --context 'run manual optimization loop'
autopilot execute run_trainer.py --max-epochs 2 --context 'quick 2-epoch trainer test'
```

### Module mode

```bash
autopilot execute -m protim.module --context 'validate module import'
```

### Stdin pipe (avoids escaping)

```bash
echo 'from protim.module import PromptModule
m = PromptModule("prompts")
for name, p in m.named_parameters():
    print(name)' | autopilot execute --context 'list prompt parameter names'
```

### Escaping tips

- For `-c` mode, prefer single-quoted code for `$`, `{}`, or f-strings: `autopilot execute -c 'print(f"x={42}")'`
- Use stdin pipe for complex multi-line code -- avoids all escaping issues
- Do NOT use `--` between code and extra args
- Autopilot global flags (`--experiment`, `--json`) are consumed by autopilot, not forwarded
- Global flags go before the `execute` subcommand's own arguments
