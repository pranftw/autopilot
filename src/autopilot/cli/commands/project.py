"""Project management: create, list, and check project health."""

from autopilot.cli.command import Argument, Command, argument, subcommand
from autopilot.cli.context import CLIContext
from autopilot.core.config import list_projects
from pathlib import Path
import argparse
import autopilot.core.paths as paths


def _read_template(filename: str, **kwargs: str) -> str:
  text = (paths.project_templates_dir() / filename).read_text(encoding='utf-8')
  return text.format(**kwargs) if kwargs else text


def _write_if_missing(path: Path, content: str) -> bool:
  if path.exists():
    return False
  path.write_text(content, encoding='utf-8')
  return True


class ProjectInit(Command):
  name = 'init'
  help = 'Initialize a new project'
  project_name = Argument('name', help='project name')
  bare_flag = Argument('--bare', action='store_true', default=False, help='skip skeleton files')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Create a project directory with standard layout and optional skeleton files."""
    ws = ctx.workspace
    name = args.name
    bare = args.bare
    project_dir = paths.root(ws, name)

    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / 'ai').mkdir(exist_ok=True)
    paths.experiments(ws, name).mkdir(exist_ok=True)
    paths.datasets(ws, name).mkdir(exist_ok=True)
    records_dir = paths.records(ws, name)
    records_dir.mkdir(exist_ok=True)
    (records_dir / 'promotions').mkdir(exist_ok=True)
    (records_dir / 'notes').mkdir(exist_ok=True)

    files_created: list[str] = []
    if not bare:
      if _write_if_missing(project_dir / 'cli.py', _read_template('cli.py', name=name)):
        files_created.append('cli.py')
      if _write_if_missing(project_dir / 'module.py', _read_template('module.py')):
        files_created.append('module.py')
      if _write_if_missing(project_dir / 'data.py', _read_template('data.py')):
        files_created.append('data.py')

    ctx.output.result(
      {
        'project': name,
        'status': 'initialized',
        'path': str(project_dir),
        'files_created': files_created,
      }
    )


class ProjectCommand(Command):
  name = 'project'
  help = 'Project management'

  def __init__(self) -> None:
    super().__init__()
    self.init = ProjectInit()

  @subcommand('list', help='List all projects')
  def list(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """List all discovered projects in the workspace."""
    ws = ctx.workspace
    projects = list_projects(ws)

    rows = [{'name': name} for name in projects]

    if rows:
      ctx.output.table(rows, ['name'])

    ctx.output.result(
      {
        'projects': projects,
      }
    )

  @argument('name', help='project name')
  @subcommand('doctor', help='Check project health')
  def doctor(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Check project health: required dirs and skeleton files."""
    ws = ctx.workspace
    name = args.name
    project_dir = paths.root(ws, name)

    checks: dict[str, bool] = {}
    issues: list[str] = []

    checks['project_dir'] = project_dir.is_dir()
    checks['cli_py'] = paths.project_cli(ws, name).is_file()
    checks['experiments_dir'] = paths.experiments(ws, name).is_dir()
    checks['datasets_dir'] = paths.datasets(ws, name).is_dir()
    checks['records_dir'] = paths.records(ws, name).is_dir()

    for key, ok in checks.items():
      if not ok:
        issues.append(key)
        ctx.output.warn(f'missing: {key}')

    healthy = len(issues) == 0
    ctx.output.result(
      {
        'project': name,
        'healthy': healthy,
        'checks': checks,
        'issues': issues,
      },
      ok=healthy,
    )
