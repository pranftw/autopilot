from autopilot.core.paths import (
  autopilot_dir,
  datasets,
  experiment,
  experiments,
  project_cli,
  projects_dir,
  records,
  root,
  store,
)
from pathlib import Path


class TestRoot:
  def test_without_project_returns_autopilot_dir(self, tmp_path: Path) -> None:
    assert root(tmp_path) == tmp_path / 'autopilot'

  def test_with_project_returns_project_subdir(self, tmp_path: Path) -> None:
    assert root(tmp_path, 'my-project') == tmp_path / 'autopilot' / 'projects' / 'my-project'


class TestProjectScoped:
  def test_experiments(self, tmp_path: Path) -> None:
    assert experiments(tmp_path, 'p1') == tmp_path / 'autopilot' / 'projects' / 'p1' / 'experiments'

  def test_experiments_without_project(self, tmp_path: Path) -> None:
    assert experiments(tmp_path) == tmp_path / 'autopilot' / 'experiments'

  def test_experiment_slug(self, tmp_path: Path) -> None:
    assert experiment(tmp_path, 'exp-001', 'p1') == (
      tmp_path / 'autopilot' / 'projects' / 'p1' / 'experiments' / 'exp-001'
    )

  def test_datasets(self, tmp_path: Path) -> None:
    assert datasets(tmp_path, 'p1') == tmp_path / 'autopilot' / 'projects' / 'p1' / 'datasets'

  def test_records(self, tmp_path: Path) -> None:
    assert records(tmp_path, 'p1') == tmp_path / 'autopilot' / 'projects' / 'p1' / 'records'

  def test_project_cli(self, tmp_path: Path) -> None:
    assert project_cli(tmp_path, 'p1') == tmp_path / 'autopilot' / 'projects' / 'p1' / 'cli.py'


class TestWorkspaceScoped:
  def test_projects_dir(self, tmp_path: Path) -> None:
    assert projects_dir(tmp_path) == tmp_path / 'autopilot' / 'projects'

  def test_autopilot_dir(self, tmp_path: Path) -> None:
    assert autopilot_dir(tmp_path) == tmp_path / 'autopilot'


class TestStore:
  def test_store_without_project(self, tmp_path: Path) -> None:
    assert store(tmp_path) == tmp_path / 'autopilot' / '.store'

  def test_store_with_project(self, tmp_path: Path) -> None:
    assert store(tmp_path, 'my-project') == (
      tmp_path / 'autopilot' / 'projects' / 'my-project' / '.store'
    )
