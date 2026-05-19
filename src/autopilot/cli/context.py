"""CLI execution context.

Carries workspace paths, config, trainer wiring, and output handler through
the command tree. The config field (Config) owns all path layout.
All path resolution goes through ctx.config.
"""

from autopilot.cli.expose import ExposeCollector
from autopilot.cli.output import Output
from autopilot.core.config import AutoPilotConfig
from autopilot.core.trainer.trainer import Trainer
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn
import argparse
import sys


@dataclass
class CLIContext:
  """Shared context available to all command handlers.

  All layout resolution goes through ctx.config (e.g. ctx.config.experiments_path,
  ctx.config.experiment_path(slug=...), ctx.config.store_path).

  The ``context`` field carries the ``--context`` flag value. Mutating commands
  require it (enforcement in ``CLI.dispatch`` via ``CLI.requires_context()``);
  read-only commands in the CLI's exempt set may omit it.
  """

  workspace: Path = field(default_factory=Path.cwd)
  project: str | None = None
  config: AutoPilotConfig = field(default_factory=lambda: AutoPilotConfig(workspace=Path.cwd()))
  experiment: str | None = None
  dataset: str | None = None
  split: str | None = None
  epoch: int | None = None
  hyperparams_file: str | None = None
  dry_run: bool = False
  verbose: bool = False
  output: Output = field(default_factory=Output)
  trainer: Trainer | None = None
  generator: Any | None = None
  judge: Any | None = None
  module: Any | None = None
  datamodule: Any | None = None
  expose: bool = False
  expose_collector: ExposeCollector | None = None
  context: str | None = None
  wait_timeout_ms: int | None = None
  retry_max: int = 0

  @property
  def autopilot_dir(self) -> Path:
    """Root ``.autopilot`` directory for this workspace."""
    return self.config.autopilot_path

  @property
  def experiments_dir(self) -> Path:
    """Directory containing experiment run folders."""
    return self.config.experiments_path

  @property
  def records_dir(self) -> Path:
    """Directory for promotion records and notes."""
    return self.config.records_path

  @property
  def datasets_dir(self) -> Path:
    """Directory for registered datasets."""
    return self.config.datasets_path

  def experiment_path(self, slug: str | None = None) -> Path:
    """Resolve the on-disk path for an experiment slug.

    Args:
      slug: Experiment id; defaults to ``self.experiment``.

    Returns:
      Absolute experiment directory under the workspace layout.

    Raises:
      ValueError: If neither ``slug`` nor ``self.experiment`` is set.
    """
    target = slug or self.experiment
    if not target:
      msg = 'no experiment specified'
      raise ValueError(msg)
    return self.config.experiment_path(slug=target)

  def fail(self, message: str, exit_code: int = 1, error_code: str | None = None) -> NoReturn:
    """Emit error message and exit with given code.

    Args:
      message: Human-readable error message.
      exit_code: Process exit code.
      error_code: Machine-stable error classification for JSON envelopes.
        Passed through to ``Output.flush_error``. When ``None``, the
        default ``'handler_error'`` is applied by ``flush_error``.
    """
    self.output.error(message)
    if self.output.use_json:
      self.output.flush_error(message, error_code=error_code)
    sys.exit(exit_code)


def resolve_project(workspace: Path, explicit: str | None) -> str | None:
  """Resolve project name from flag or cwd under the projects directory.

  Args:
    workspace: Workspace root used for layout resolution.
    explicit: Project from ``-p`` / ``--project``, if any.

  Returns:
    Detected project slug, or ``None`` if not under a project layout.
  """
  if explicit:
    return explicit
  config = AutoPilotConfig(workspace=workspace)
  pdir = config.projects_path
  if pdir.exists():
    try:
      rel = Path.cwd().relative_to(pdir)
      if rel.parts:
        return rel.parts[0]
    except ValueError:
      pass
  return None


def build_context(args: argparse.Namespace) -> CLIContext:
  """Build a CLIContext from parsed arguments.

  Args:
    args: Root parser namespace including workspace and global flags.

  Returns:
    Populated ``CLIContext`` for command handlers.
  """
  workspace = Path(args.workspace).resolve()
  project = resolve_project(workspace, args.project)
  config = AutoPilotConfig(workspace=workspace, project=project)
  expose = args.expose
  collector = ExposeCollector() if expose else None
  return CLIContext(
    workspace=workspace,
    project=project,
    config=config,
    experiment=args.experiment,
    dataset=args.dataset,
    split=args.split,
    epoch=args.epoch,
    hyperparams_file=args.hyperparams,
    dry_run=args.dry_run,
    verbose=args.verbose,
    output=Output(
      use_json=args.use_json,
      no_color=args.no_color,
      expose_collector=collector,
    ),
    expose=expose,
    expose_collector=collector,
    context=args.context,
    wait_timeout_ms=args.wait,
    retry_max=args.retry,
  )
