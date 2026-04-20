---
name: cli-conventions
description: CLI command patterns for the autopilot CLI using Command, CLI, and argparse. Use when adding commands, subcommands, or modifying global flags.
---

## Architecture

- **`Command`** (`cli/command.py`): recursive node. Assign child `Command` instances as attributes — `__setattr__` registers them. Leaf commands override **`forward(ctx, args)`**. Container commands nest children and/or use **`@subcommand`** / **`@argument`** on methods for inline handlers.
- **`CLI`** (`cli/command.py`): top-level orchestrator. Subclass **`AutoPilotCLI`** for a full stock CLI, or **`CLI`** for a minimal tree. **`__init__`** wires `self.module`, `self.generator`, `self.judge`, and attaches **`Command`** instances. **`__call__`** → **`run()`**: pre-parse `--project` / `--workspace`, optional project **`cli.py`** via `runpy`, then dispatch.
- **Project binding**: `class MyCLI(AutoPilotCLI, project='my-name'):` registers the class in **`CLI._project_registry`** via **`__init_subclass__`**. Entry: **`MyCLI()()`**.
- **Registration**: **`Command.register(subparsers)`** walks children and attaches parsers; **`make_subparser()`** (`cli/shared.py`) adds global flags. No `register(subparsers)` free functions in command modules.

## Adding a command

1. Create `src/autopilot/cli/commands/<name>.py`.
2. Subclass **`Command`**, set **`name`** / **`help`** if the auto-derived name is wrong.
3. Either override **`forward()`** (leaf) or in **`__init__`** assign child commands / define **`@subcommand`** methods.
4. Instantiate the command on **`AutoPilotCLI.__init__`** (or on a project **`CLI`** subclass).

Minimal leaf example:

```python
from autopilot.cli.command import Command
from autopilot.cli.context import CLIContext
import argparse


class WidgetCommand(Command):
  name = 'widget'
  help = 'Do something useful'

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    ctx.output.result({'ok': True})
```

Declarative flags use **`Argument`** / **`Flag`** class attributes; **`collect_arguments()`** adds them in **`register()`**.

## CLIContext

`src/autopilot/cli/context.py`. Resolved state for handlers:

- `workspace`, `project`, paths via **`paths.*`**
- `generator`, `judge`, `module` — set from the active **`CLI`** instance in **`CLI._run_direct()`**
- `trainer` — constructed when **`module`** is set
- `dry_run`, `verbose`, **`output`**

## Output

All user-facing text through **`ctx.output`**: **`info`**, **`warn`**, **`error`**, **`success`**, **`result`**, **`table`**, **`data`**.

## Global flags

`src/autopilot/cli/shared.py`: **`add_global_flags()`** on the root parser; **`make_subparser()`** repeats them on subparsers used by **`Command.register()`**.

## Entry points

- **Library / console script**: **`AutoPilotCLI()()`** or **`main()`** in **`cli/main.py`**.
- **Project overlay**: subclass **`AutoPilotCLI`** (or **`CLI`**) with **`project='...'`**, wire **`module` / `generator` / `judge`** in **`__init__`**, end **`cli.py`** with **`MyCLI()()`**.

## Key files

- `src/autopilot/cli/command.py` — **`Command`**, **`CLI`**, **`Argument`**, **`Flag`**, **`@subcommand`**, **`@argument`**
- `src/autopilot/cli/main.py` — **`AutoPilotCLI`**, **`main()`**
- `src/autopilot/cli/shared.py` — **`make_subparser()`**, **`add_global_flags()`**
- `src/autopilot/cli/context.py` — **`CLIContext`**, **`build_context()`**
- `src/autopilot/cli/output.py` — **`Output`**

## Notable command output shapes

**`status`** returns:
```json
{
  "slug": "...", "epoch": N, "trained_epochs": N, "decision": "...",
  "stop_reason": "plateau|crash|null", "last_good_epoch": N,
  "last_metrics": {...}, "best_baseline": {...},
  "regression": {"epoch": N, "verdict": "...", "regressed_metrics": [...]},
  "memory": {"total_records": N, "blocked_strategies": [...]}
}
```

**`propose verify`** uses `--check-epoch` (not `--epoch` to avoid global flag conflict). Returns `{verdict, items_tested, ...}`. `propose revert` uses `--restore-epoch`.

**`trace inspect --node <id>`** returns `{matches: [{batch_idx, item_id, success, feedback, error_message}], count}`. With `--depth > 1`, includes `memory_records`.

**`project init`** generates `cli.py`, `module.py`, `data.py` skeleton files. Use `--bare` to skip.

## Gotchas

- Handlers orchestrate only; delegate to **`core.services`** and adapters.
- Prefer **`uv run autopilot <command>`** over one-off scripts that skip manifests/events.
- Exceptions in **`forward()`** / subcommand handlers are caught in **`CLI.dispatch()`**, surfaced via **`ctx.output`**, exit **1**.
- Avoid `--epoch` as a subcommand-level argument name -- it conflicts with the global `--epoch` flag. Use names like `--check-epoch` or `--restore-epoch`.
