---
name: cli-conventions
description: CLI command patterns, global flags, project bootstrap, and workspace layout for the autopilot CLI. Use when adding commands, subcommands, modifying global flags, creating new workspaces, diagnosing workspace issues, or understanding the project overlay structure.
---

## Entry point

`autopilot.cli.main:main`, program name `autopilot`. Implementation uses `Command` subclasses (`forward()`, nested commands, `@subcommand`) composed on `AutoPilotCLI`; `main()` calls `AutoPilotCLI()()`.

## Top-level commands

| Command | Summary |
| --- | --- |
| `workspace` | `init`, `doctor`, `tree` |
| `project` | Create, list, and check project health |
| `dataset` | List, show, seed, split |
| `experiment` | Experiment lifecycle |
| `optimize` | Optimization driver |
| `debug` | Debug workflows |
| `policy` | Policy and scoring inspection |
| `ai` | AI eval generation and judging (`ai generate`, `ai judge` with run/resume/summarize/distribution) |
| `report` | Reporting |
| `promote` | Promotion workflow |
| `store` | Content-addressed code versioning |
| `status` | Experiment overview (epoch, metrics, stop reason) |
| `diagnose` | Trace diagnostics and node heatmaps |
| `trace` | Collect and inspect computation traces |
| `propose` | Create, verify, revert, and list proposals |
| `memory` | Query, record, trends, and context |
| `agent` | Agent operations (run, list, session) |

Omitting a subcommand where required exits with help or error per `argparse` rules. Running `autopilot` with no command prints top-level help and exits 0.

## Global flags

Leaf parsers and inline `@subcommand` parsers pick up shared flags via `make_subparser()` (invoked from `Command.register()` in `cli/command.py`), which calls `add_global_flags()` in `cli/resolvers.py`:

| Flag | Meaning |
| --- | --- |
| `-p`, `--project NAME` | Project name (auto-detected when cwd is under `autopilot/projects/<name>`) |
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
| `--expose` | Enable JSON audit trail via `ExposeCollector` |

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

## Standard workspace layout

```
workspace/
  autopilot/
    pyproject.toml
    plugins/
    projects/
      <name>/
        cli.py                # defines CLI subclass; entry MyCLI()()
        trainer.py            # builds Trainer with project Module
        ai/
        experiments/
          <slug>/
            manifest.json
            events.jsonl
            commands.json
        records/
          promotions/
          notes/
        datasets/
```

No **`workspace.toml`**, **`project.toml`**, or workflow TOML files -- project wiring lives in Python (`cli.py`, `trainer.py`).

## Project CLI registration

Subclass **`AutoPilotCLI`** (or **`CLI`**) with **`project='<name>'`** on the class statement. **`CLI.__init_subclass__`** stores **`project -> CLI class`** in **`CLI._project_registry`**. When **`autopilot -p <name>`** runs, **`CLI.run()`** may execute **`projects/<name>/cli.py`** via `runpy`, then instantiates the registered class if present.

## Workspace commands

`src/autopilot/cli/commands/workspace.py`:

- **`autopilot workspace init`** -- creates the **`autopilot/`** tree (projects, experiments, datasets, records, plugins, etc.); idempotent.
- **`autopilot workspace doctor`** -- validates expected directories under **`autopilot/`**.
- **`autopilot workspace tree`** -- ASCII tree of **`autopilot/`** (bounded depth).

## Project commands

`src/autopilot/cli/commands/project.py`:

- **`autopilot project list`** -- lists project directories under **`autopilot/projects/`**.
- **`autopilot project init <name>`** -- scaffolds a project directory with `cli.py`, `module.py`, and `data.py` from templates (use `--bare` to skip template files).
- **`autopilot project doctor <name>`** -- health checks for a project (e.g. **`cli.py`**, data dirs).

## Project resolution

Order: **`--project` / `-p`** > current working directory under **`autopilot/projects/<name>/`**. Optional registered **`CLI`** subclass for that **`project`** name handles dispatch after **`cli.py`** import side effects.

## CLIContext

`src/autopilot/cli/context.py` -- paths via **`paths.*`**. **`generator`**, **`judge`**, **`module`** come from the active **`CLI`** instance when **`CLI._run_direct()`** builds context.

## Centralized paths

`src/autopilot/core/paths.py` -- single definitions for **`autopilot_dir`**, **`projects_dir`**, **`root(workspace, project)`**, experiments, datasets, records, **`project_cli`**, etc.

## Key files

- `src/autopilot/cli/commands/workspace.py`
- `src/autopilot/cli/commands/project.py`
- `src/autopilot/core/paths.py`
- `src/autopilot/core/config.py`
- `src/autopilot/cli/context.py`
- `src/autopilot/cli/command.py` -- **`CLI`**, project registry
- `src/autopilot/cli/main.py` -- **`AutoPilotCLI`**, **`main()`**
- `src/autopilot/cli/resolvers.py` -- **`make_subparser`**, **`add_global_flags`**
- `src/autopilot/cli/output.py` -- **`Output`** with `result()`, `info()`, `table()`
- `src/autopilot/cli/expose.py` -- **`ExposeRecord`**, **`ExposeCollector`**

## Gotchas

- Overlays live under **`autopilot/projects/<name>/`**, not in **`src/autopilot/`**.
- **`project init`** creates **`cli.py`**, **`module.py`**, **`data.py`** from templates; it does **not** create **`trainer.py`**. Use **`--bare`** to skip template files.
- Paths resolve with **`Path.resolve()`**; **`--workspace`** defaults to **`.`**.
- Runtime and infrastructure settings belong in Python (constructors, **`Trainer`**, **`Module`**) and explicit CLI flags. Do not assume TOML workspace or workflow layers.
