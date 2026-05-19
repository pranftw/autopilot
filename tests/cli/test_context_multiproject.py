"""Tests for CLIContext project paths and config field."""

from autopilot.cli.context import CLIContext
from autopilot.core.config import AutoPilotConfig
from pathlib import Path


def _ctx(workspace: Path, project: str | None = None, experiment: str | None = None) -> CLIContext:
  config = AutoPilotConfig(workspace=workspace, project=project)
  return CLIContext(
    workspace=workspace,
    project=project,
    config=config,
    experiment=experiment,
  )


class TestCLIContextProjectPaths:
  """Legacy property methods (kept for backward compat with old commands)."""

  def test_experiments_dir_with_project(self) -> None:
    ctx = _ctx(Path('/ws'), project='p1')
    assert ctx.experiments_dir == Path('/ws/.autopilot/projects/p1/experiments')

  def test_experiments_dir_without_project(self) -> None:
    ctx = _ctx(Path('/ws'))
    assert ctx.experiments_dir == Path('/ws/.autopilot/experiments')

  def test_datasets_dir_with_project(self) -> None:
    ctx = _ctx(Path('/ws'), project='p1')
    assert ctx.datasets_dir == Path('/ws/.autopilot/projects/p1/datasets')

  def test_records_dir_with_project(self) -> None:
    ctx = _ctx(Path('/ws'), project='p1')
    assert ctx.records_dir == Path('/ws/.autopilot/projects/p1/records')

  def test_module_field_default_none(self) -> None:
    ctx = _ctx(Path('/ws'))
    assert ctx.module is None

  def test_autopilot_dir_always_workspace_level(self) -> None:
    ctx = _ctx(Path('/ws'), project='p1')
    assert ctx.autopilot_dir == Path('/ws/.autopilot')

  def test_experiment_path_with_project(self) -> None:
    ctx = _ctx(Path('/ws'), project='p1', experiment='exp-001')
    assert ctx.experiment_path() == Path('/ws/.autopilot/projects/p1/experiments/exp-001')

  def test_generator_field_default_none(self) -> None:
    ctx = _ctx(Path('/ws'))
    assert ctx.generator is None

  def test_judge_field_default_none(self) -> None:
    ctx = _ctx(Path('/ws'))
    assert ctx.judge is None

  def test_project_field_default_none(self) -> None:
    ctx = _ctx(Path('/ws'))
    assert ctx.project is None


class TestCLIContextConfig:
  """New config field for Plan 6+ commands."""

  def test_config_is_set(self) -> None:
    config = AutoPilotConfig(workspace=Path('/ws'))
    ctx = CLIContext(workspace=Path('/ws'), config=config)
    assert ctx.config is config

  def test_config_has_correct_workspace(self) -> None:
    config = AutoPilotConfig(workspace=Path('/ws'))
    ctx = CLIContext(workspace=Path('/ws'), config=config)
    assert ctx.config.workspace == Path('/ws')

  def test_config_with_project(self) -> None:
    config = AutoPilotConfig(workspace=Path('/ws'), project='p1')
    ctx = CLIContext(workspace=Path('/ws'), project='p1', config=config)
    assert ctx.config.project == 'p1'

  def test_config_experiments_path(self) -> None:
    config = AutoPilotConfig(workspace=Path('/ws'))
    ctx = CLIContext(workspace=Path('/ws'), config=config)
    assert ctx.config.experiments_path == Path('/ws/.autopilot/experiments')

  def test_config_store_path(self) -> None:
    config = AutoPilotConfig(workspace=Path('/ws'))
    ctx = CLIContext(workspace=Path('/ws'), config=config)
    assert ctx.config.store_path == Path('/ws/.autopilot/store')

  def test_default_config_exists(self) -> None:
    ctx = CLIContext(workspace=Path('/ws'))
    assert ctx.config is not None
