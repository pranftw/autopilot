"""Workspace management: initialize layout, health checks, and directory tree.

The doctor command checks workspace structure only. Auth and provider
checks are project-specific and belong in preflight or project plugins.
"""

from autopilot.cli.command import Command, subcommand
from autopilot.cli.context import CLIContext
from pathlib import Path
import argparse
import autopilot.core.paths as paths


class WorkspaceCommand(Command):
  name = 'workspace'
  help = 'Workspace management'

  @subcommand('init', help='Initialize workspace')
  def init(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    ctx.output.info('Initializing workspace...')
    ws = ctx.workspace
    autopilot = ctx.autopilot_dir
    autopilot.mkdir(parents=True, exist_ok=True)
    (autopilot / 'experiments').mkdir(exist_ok=True)
    (autopilot / 'records').mkdir(exist_ok=True)
    (autopilot / 'records' / 'promotions').mkdir(exist_ok=True)
    (autopilot / 'records' / 'notes').mkdir(exist_ok=True)
    (autopilot / 'datasets').mkdir(exist_ok=True)
    (autopilot / 'plugins').mkdir(exist_ok=True)
    paths.projects_dir(ws).mkdir(parents=True, exist_ok=True)

    ctx.output.result({'workspace': str(ws), 'status': 'initialized'})

  @subcommand('doctor', help='Check workspace health')
  def doctor(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    ctx.output.info('Checking workspace health...')
    ws = ctx.workspace

    checks = {
      'workspace_exists': ws.exists(),
      'autopilot_exists': ctx.autopilot_dir.exists(),
      'experiments_dir': ctx.experiments_dir.exists(),
      'records_dir': ctx.records_dir.exists(),
      'datasets_dir': ctx.datasets_dir.exists(),
      'projects_dir': paths.projects_dir(ws).is_dir(),
    }

    all_ok = all(checks.values())
    issues = [k for k, v in checks.items() if not v]

    if issues:
      for issue in issues:
        ctx.output.warn(f'missing: {issue}')

    ctx.output.result(
      {
        'workspace': str(ws),
        'healthy': all_ok,
        'checks': checks,
        'issues': issues,
      },
      ok=all_ok,
    )

  @subcommand('tree', help='Show autopilot directory tree')
  def tree(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    ctx.output.info('Workspace tree (autopilot):')
    base = ctx.autopilot_dir
    if not base.exists():
      ctx.output.warn('autopilot directory does not exist; run workspace init')
      ctx.output.result({'tree': [], 'root': str(base)})
      return
    tree = [str(base)] + _tree_lines(base)
    for line in tree:
      ctx.output.info(line)
    ctx.output.result({'root': str(base), 'lines': len(tree)})


def _tree_lines(root: Path, prefix: str = '', max_depth: int = 6) -> list[str]:
  lines: list[str] = []
  if max_depth <= 0 or not root.exists():
    return lines
  entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
  for i, path in enumerate(entries):
    connector = '└── ' if i == len(entries) - 1 else '├── '
    lines.append(f'{prefix}{connector}{path.name}')
    if path.is_dir():
      extension = '    ' if i == len(entries) - 1 else '│   '
      lines.extend(_tree_lines(path, prefix + extension, max_depth - 1))
  return lines
