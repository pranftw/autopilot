# AutoPilot Examples

Self-contained examples demonstrating the AutoPilot optimization framework.

| Example | What it demonstrates | Requirements |
| --- | --- | --- |
| [textmatch](textmatch/) | Regex-rule optimization with Module, Loss, Optimizer, Trainer, Policy, Store | None (runs offline) |
| [protim](protim/) | Agent-optimized prompt with ClaudeCodeAgent for inference and optimization | `claude` CLI |

## Getting started

Each example is its own uv package. To run one:

```bash
cd examples/<name>
uv sync
uv run python run.py
```

The `autopilot` library is referenced as an editable dependency, so changes to `src/autopilot/` are immediately reflected.
