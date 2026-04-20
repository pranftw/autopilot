# CLI (`autopilot`)

Entry point: `autopilot.cli.main:main`, program name `autopilot`. Implementation uses `Command` subclasses (`forward()`, nested commands, `@subcommand`) composed on `AutoPilotCLI`; `main()` calls `AutoPilotCLI()()`.

## Top-level commands

| Command | Summary |
| --- | --- |
| `workspace` | `init`, `doctor`, `tree` |
| `project` | Create, list, and check project health |
| `dataset` | Registry and split operations |
| `experiment` | Experiment lifecycle |
| `optimize` | Optimization driver |
| `debug` | Debug workflows |
| `policy` | Policy and scoring inspection |
| `ai` | AI eval generation, judging, datasets |
| `report` | Reporting |
| `promote` | Promotion workflow |
| `agent` *(planned)* | Agent operations (run, list, session) — vision-level; not in the stock CLI yet |

Omitting a subcommand where required exits with help or error per `argparse` rules. Running `autopilot` with no command prints top-level help and exits 0.

## Global flags

Leaf parsers and inline `@subcommand` parsers pick up shared flags via `make_subparser()` (invoked from `Command.register()` in `cli/command.py`), which calls `add_global_flags()` in `cli/shared.py`:

| Flag | Meaning |
| --- | --- |
| `--config PATH` | Path to project config override |
| `-p`, `--project NAME` | Project name (auto-detected when cwd is under `autopilot/projects/<name>`) |
| `--env ENV` | Environment (e.g. staging, production) |
| `--workspace PATH` | Workspace root (default: current directory) |
| `--experiment SLUG` | Active experiment |
| `--dataset NAME` | Dataset name |
| `--split NAME` | Dataset split: `train`, `val`, `test` |
| `--epoch N` | Epoch number |
| `--hyperparams PATH` | Hyperparameters JSON file |
| `--dry-run` | Show what would happen without executing |
| `--verbose` | More logging; on errors, print traceback |
| `--no-color` | Disable ANSI color |
| `--json` | Machine-readable output (see below) |

## `--help`

Standard `argparse` help on the root parser and on each subcommand. Prefer `uv run autopilot <cmd> --help` when developing.

## `--json` behavior

When `--json` is set, `Output` buffers structured messages. `info`, `success`, `warn`, and `error` append objects with `level` and `message`. `data` and `table` append typed records. `result` prints a final JSON object on stdout; shape is equivalent to:

```python
{
  'ok': True,
  'result': {'key': 'value'},
  'messages': [{'level': 'info', 'message': '...'}],
}
```

Parse stdout as JSON for scripting; stderr still carries errors in human form unless commands only use `Output`. Use `flush_json` only where a command explicitly ends with buffered rows and no `result` envelope.
