"""Project management: create, list, and check project health.

Template resolution order for ``project init``:

1. Workspace path (``AutoPilotConfig.templates_path / 'project' / filename``)
2. Package bundled defaults (``importlib.resources.files('autopilot') / 'templates' / ...``)

Workspace templates override bundled defaults when both exist.
"""

from autopilot.cli.command import Command
from autopilot.cli.context import CLIContext
from autopilot.cli.primitives import Argument, argument, subcommand
from autopilot.core.config import AutoPilotConfig
from pathlib import Path
import argparse
import importlib.resources


def _read_template(config: AutoPilotConfig, filename: str, **kwargs: str) -> str:
  """Read a project template, falling back to package bundled defaults.

  Lookup order:
    1. Workspace ``templates/project/<filename>``
    2. Package ``autopilot/templates/project/<filename>``

  Args:
    config: Config with ``templates_path`` for workspace lookup.
    filename: Template filename (e.g. ``cli.py``).
    **kwargs: Format substitutions applied to the template text.

  Returns:
    Template text with substitutions applied.

  Raises:
    FileNotFoundError: When the template is missing from both locations.
  """
  workspace_path = config.templates_path / 'project' / filename
  if workspace_path.exists():
    text = workspace_path.read_text(encoding='utf-8')
    return text.format(**kwargs) if kwargs else text

  pkg_base = importlib.resources.files('autopilot')
  pkg_ref = pkg_base.joinpath('templates').joinpath('project').joinpath(filename)
  try:
    text = pkg_ref.read_text(encoding='utf-8')
  except (FileNotFoundError, TypeError) as exc:
    msg = (
      f'template {filename!r} not found in workspace ({workspace_path}) '
      f'or package bundle; create it or install the full autopilot package'
    )
    raise FileNotFoundError(msg) from exc
  return text.format(**kwargs) if kwargs else text


def _write_if_missing(path: Path, content: str) -> bool:
  if path.exists():
    return False
  path.write_text(content, encoding='utf-8')
  return True


class ProjectInit(Command):
  """Creates a project directory with standard layout and skeleton files."""

  name = 'init'
  help = 'Initialize a new project'
  project_name = Argument('name', help='project name')
  bare_flag = Argument('--bare', action='store_true', default=False, help='skip skeleton files')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Create a project directory with standard layout and optional skeleton files."""
    name = args.name
    bare = args.bare
    config = AutoPilotConfig(workspace=ctx.workspace, project=name)
    project_dir = config.root

    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / 'ai').mkdir(exist_ok=True)
    config.experiments_path.mkdir(exist_ok=True)
    config.datasets_path.mkdir(exist_ok=True)
    records_dir = config.records_path
    records_dir.mkdir(exist_ok=True)
    (records_dir / 'promotions').mkdir(exist_ok=True)
    (records_dir / 'notes').mkdir(exist_ok=True)

    files_created: list[str] = []
    if not bare:
      for tpl_name, tpl_kwargs in [
        ('cli.py', {'name': name}),
        ('module.py', {}),
        ('data.py', {}),
        ('CLAUDE.md', {'name': name}),
      ]:
        try:
          content = _read_template(ctx.config, tpl_name, **tpl_kwargs)
        except FileNotFoundError:
          continue
        if _write_if_missing(project_dir / tpl_name, content):
          files_created.append(tpl_name)
      try:
        cfg_content = _read_template(ctx.config, 'config.py', name=name)
        if _write_if_missing(project_dir / 'config.py', cfg_content):
          files_created.append('config.py')
      except FileNotFoundError:
        pass

    ctx.output.result(
      {
        'project': name,
        'status': 'initialized',
        'path': str(project_dir),
        'files_created': files_created,
      }
    )


class ProjectCommand(Command):
  """``autopilot project`` group: init, list, and doctor health checks."""

  name = 'project'
  help = 'Project management'

  def __init__(self) -> None:
    """Wire project subcommands (``init`` plus inline ``list`` / ``doctor``)."""
    super().__init__()
    self.init = ProjectInit()

  @subcommand('list', help_text='List all projects')
  def list(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """List all discovered projects in the workspace."""
    pdir = ctx.config.projects_path
    if not pdir.exists():
      projects: list[str] = []
    else:
      projects = sorted(child.name for child in pdir.iterdir() if child.is_dir())

    rows = [{'name': name} for name in projects]
    if rows:
      ctx.output.table(rows, ['name'])

    ctx.output.result({'projects': projects})

  @argument('name', help='project name')
  @subcommand('doctor', help_text='Check project health')
  def doctor(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Check project health: required dirs and skeleton files."""
    name = args.name
    config = AutoPilotConfig(workspace=ctx.workspace, project=name)

    checks: dict[str, bool] = {}
    issues: list[str] = []

    checks['project_dir'] = config.root.is_dir()
    checks['cli_py'] = config.cli_file.is_file()
    checks['experiments_dir'] = config.experiments_path.is_dir()
    checks['datasets_dir'] = config.datasets_path.is_dir()
    checks['records_dir'] = config.records_path.is_dir()

    for key, ok in checks.items():
      if not ok:
        issues.append(key)
        ctx.output.warn(f'missing: {key}')

    healthy = not issues
    ctx.output.result(
      {
        'project': name,
        'healthy': healthy,
        'checks': checks,
        'issues': issues,
      },
      ok=healthy,
    )
