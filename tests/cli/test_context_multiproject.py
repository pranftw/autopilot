from autopilot.cli.context import CLIContext
from pathlib import Path


class TestCLIContextProjectPaths:
  def test_experiments_dir_with_project(self) -> None:
    ctx = CLIContext(workspace=Path('/ws'), project='p1')
    assert ctx.experiments_dir == Path('/ws/autopilot/projects/p1/experiments')

  def test_experiments_dir_without_project(self) -> None:
    ctx = CLIContext(workspace=Path('/ws'))
    assert ctx.experiments_dir == Path('/ws/autopilot/experiments')

  def test_datasets_dir_with_project(self) -> None:
    ctx = CLIContext(workspace=Path('/ws'), project='p1')
    assert ctx.datasets_dir == Path('/ws/autopilot/projects/p1/datasets')

  def test_records_dir_with_project(self) -> None:
    ctx = CLIContext(workspace=Path('/ws'), project='p1')
    assert ctx.records_dir == Path('/ws/autopilot/projects/p1/records')

  def test_module_field_default_none(self) -> None:
    ctx = CLIContext(workspace=Path('/ws'))
    assert ctx.module is None

  def test_autopilot_dir_always_workspace_level(self) -> None:
    ctx = CLIContext(workspace=Path('/ws'), project='p1')
    assert ctx.autopilot_dir == Path('/ws/autopilot')

  def test_experiment_dir_with_project(self) -> None:
    ctx = CLIContext(workspace=Path('/ws'), project='p1', experiment='exp-001')
    assert ctx.experiment_dir() == Path('/ws/autopilot/projects/p1/experiments/exp-001')

  def test_generator_field_default_none(self) -> None:
    ctx = CLIContext(workspace=Path('/ws'))
    assert ctx.generator is None

  def test_judge_field_default_none(self) -> None:
    ctx = CLIContext(workspace=Path('/ws'))
    assert ctx.judge is None

  def test_project_field_default_none(self) -> None:
    ctx = CLIContext(workspace=Path('/ws'))
    assert ctx.project is None
