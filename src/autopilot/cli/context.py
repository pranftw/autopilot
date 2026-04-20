"""CLI execution context.

Carries workspace paths, trainer wiring, and output handler through the command tree.
"""

from autopilot.cli.expose import ExposeCollector
from autopilot.cli.output import Output
from autopilot.core.trainer import Trainer
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import argparse
import autopilot.core.paths as paths


@dataclass
class CLIContext:
  """Shared context available to all command handlers."""

  workspace: Path = field(default_factory=Path.cwd)
  project: str | None = None
  config_path: str = ''
  environment: str = ''
  experiment: str = ''
  dataset: str = ''
  split: str = ''
  epoch: int = 0
  hyperparams_file: str = ''
  dry_run: bool = False
  verbose: bool = False
  output: Output = field(default_factory=Output)
  trainer: Trainer | None = None
  generator: Any = None
  judge: Any = None
  module: Any = None
  datamodule: Any = None
  expose: bool = False
  expose_collector: ExposeCollector | None = None

  @property
  def autopilot_dir(self) -> Path:
    return paths.autopilot_dir(self.workspace)

  @property
  def experiments_dir(self) -> Path:
    return paths.experiments(self.workspace, self.project)

  @property
  def records_dir(self) -> Path:
    return paths.records(self.workspace, self.project)

  @property
  def datasets_dir(self) -> Path:
    return paths.datasets(self.workspace, self.project)

  def experiment_dir(self, slug: str | None = None) -> Path:
    target = slug or self.experiment
    if not target:
      raise ValueError('no experiment specified')
    return paths.experiment(self.workspace, target, self.project)


def _resolve_project(workspace: Path, explicit: str) -> str | None:
  """Priority: explicit flag > CWD under projects dir."""
  if explicit:
    return explicit
  pdir = paths.projects_dir(workspace)
  if pdir.exists():
    try:
      rel = Path.cwd().relative_to(pdir)
      if rel.parts:
        return rel.parts[0]
    except ValueError:
      pass
  return None


def build_context(args: argparse.Namespace) -> CLIContext:
  """Build a CLIContext from parsed arguments."""
  workspace = Path(args.workspace).resolve()
  project = _resolve_project(workspace, args.project)
  expose = args.expose
  collector = ExposeCollector() if expose else None
  return CLIContext(
    workspace=workspace,
    project=project,
    config_path=args.config,
    environment=args.env,
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
  )
