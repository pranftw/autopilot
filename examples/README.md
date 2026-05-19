# AutoPilot Examples

Self-contained examples demonstrating the AutoPilot optimization framework.

| Example | What it demonstrates | Requirements |
| --- | --- | --- |
| [harness](harness/) | End-to-end agent harness optimization with JudgeLoss, TextGradient, AgentOptimizer, and policy gates | `OPENROUTER_API_KEY` |
| [textmatch](textmatch/) | Regex-rule optimization with Module, Loss, Optimizer, `Trainer`, Policy, `Experiment` (with `FileStore`) | None (runs offline) |
| [protim](protim/) | LLM-optimized prompt using ClaudeCodeAgent for inference and optimization | `claude` CLI |
| [multi_module](multi_module/) | Pipeline of multiple Modules orchestrated by a single AutoPilotModule and Trainer | None (runs offline) |

## Mutating commands require `--context`

All mutating autopilot CLI commands require a `--context '<reason>'` flag. This is
enforced at the CLI layer -- omitting it will produce an error. Read-only commands
(`query`, `debug`, `tree list`, `experiment show`, `experiment compare`, etc.) are
exempt.

The `--context` value is persisted in two places:

1. **Execution log** (`executions.jsonl`) via `ExecutionRecord.context` -- for
   command-level audit trails.
2. **Experiment decision journal** (`experiment.context_log`) via
   `experiment.add_context(source='user')` -- for experiment-scoped decision
   history.

Together, these form a **decision journal**: a queryable, append-only record of
why each action was taken. The journal also accumulates entries from internal
components (policy gates, optimizer steps, trainer lifecycle) via
`trainer.emit_context()`.

Inspect the journal:

```bash
autopilot -p <project> --experiment <slug> experiment show --context-log
```

See per-example READMEs for `--context` usage on specific commands.

## Getting started

Each example is its own uv package. To run one:

```bash
cd examples/<name>
uv sync
uv run python run.py
```

The `autopilot` library is referenced as an editable dependency, so changes to `src/autopilot/` are immediately reflected.
