---
name: project-bootstrap
description: Workspace initialization, health checks, and standard directory layout for AutoPilot projects. Use when creating new workspaces, diagnosing workspace issues, or understanding the project overlay structure.
---

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

No **`workspace.toml`**, **`project.toml`**, or workflow TOML files — project wiring lives in Python (`cli.py`, `trainer.py`).

## Project CLI registration

Subclass **`AutoPilotCLI`** (or **`CLI`**) with **`project='<name>'`** on the class statement. **`CLI.__init_subclass__`** stores **`project -> CLI class`** in **`CLI._project_registry`**. When **`autopilot -p <name>`** runs, **`CLI.run()`** may execute **`projects/<name>/cli.py`** via `runpy`, then instantiates the registered class if present.

## Workspace commands

`src/autopilot/cli/commands/workspace.py`:

- **`autopilot workspace init`** — creates the **`autopilot/`** tree (projects, experiments, datasets, records, plugins, etc.); idempotent.
- **`autopilot workspace doctor`** — validates expected directories under **`autopilot/`**.
- **`autopilot workspace tree`** — ASCII tree of **`autopilot/`** (bounded depth).

## Project commands

`src/autopilot/cli/commands/project_cmd.py`:

- **`autopilot project list`** — lists project directories under **`autopilot/projects/`**.
- **`autopilot project init <name>`** — scaffolds a project directory (does not write **`cli.py`** / **`trainer.py`** — those are project code).
- **`autopilot project doctor <name>`** — health checks for a project (e.g. **`cli.py`**, data dirs).

## Project resolution

Order: **`--project` / `-p`** > current working directory under **`autopilot/projects/<name>/`**. Optional registered **`CLI`** subclass for that **`project`** name handles dispatch after **`cli.py`** import side effects.

## Config

Runtime and infrastructure settings belong in Python (constructors, **`Trainer`**, **`Module`**) and explicit CLI flags (e.g. **`--config`** JSON path). Do not assume TOML workspace or workflow layers.

## CLIContext

`src/autopilot/cli/context.py` — paths via **`paths.*`**. **`generator`**, **`judge`**, **`module`** come from the active **`CLI`** instance when **`CLI._run_direct()`** builds context.

## Centralized paths

`src/autopilot/core/paths.py` — single definitions for **`autopilot_dir`**, **`projects_dir`**, **`root(workspace, project)`**, experiments, datasets, records, **`project_cli`**, etc.

## Key files

- `src/autopilot/cli/commands/workspace.py`
- `src/autopilot/cli/commands/project_cmd.py`
- `src/autopilot/core/paths.py`
- `src/autopilot/core/config.py`
- `src/autopilot/cli/context.py`
- `src/autopilot/cli/command.py` — **`CLI`**, project registry
- `src/autopilot/cli/main.py` — **`AutoPilotCLI`**, **`main()`**

## Gotchas

- Overlays live under **`autopilot/projects/<name>/`**, not in **`src/autopilot/`**.
- **`project init`** does not create **`cli.py`** / **`trainer.py`**.
- Paths resolve with **`Path.resolve()`**; **`--workspace`** defaults to **`.`**.
