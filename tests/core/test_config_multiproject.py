from autopilot.core.config import list_projects, merge_overrides, resolve_experiment_dir
from pathlib import Path


class TestResolveExperimentDir:
  def test_project_aware(self, tmp_path: Path) -> None:
    result = resolve_experiment_dir(tmp_path, 'exp-001', 'p1')
    assert result == tmp_path / 'autopilot' / 'projects' / 'p1' / 'experiments' / 'exp-001'

  def test_without_project(self, tmp_path: Path) -> None:
    result = resolve_experiment_dir(tmp_path, 'exp-001')
    assert result == tmp_path / 'autopilot' / 'experiments' / 'exp-001'


class TestListProjects:
  def test_discovers_project_directories(self, tmp_path: Path) -> None:
    for name in ('alpha', 'beta'):
      (tmp_path / 'autopilot' / 'projects' / name).mkdir(parents=True)
    assert list_projects(tmp_path) == ['alpha', 'beta']

  def test_ignores_files_in_projects_dir(self, tmp_path: Path) -> None:
    pdir = tmp_path / 'autopilot' / 'projects'
    pdir.mkdir(parents=True)
    (pdir / 'not-a-dir').write_text('x', encoding='utf-8')
    (pdir / 'real').mkdir()
    assert list_projects(tmp_path) == ['real']

  def test_empty_when_no_projects_dir(self, tmp_path: Path) -> None:
    assert list_projects(tmp_path) == []

  def test_sorted_alphabetically(self, tmp_path: Path) -> None:
    for name in ('zeta', 'alpha', 'mu'):
      (tmp_path / 'autopilot' / 'projects' / name).mkdir(parents=True)
    assert list_projects(tmp_path) == ['alpha', 'mu', 'zeta']


class TestMergeOverrides:
  def test_shallow_merge(self) -> None:
    base = {'a': 1, 'b': 2}
    merged = merge_overrides(base, {'b': 3, 'c': 4})
    assert merged == {'a': 1, 'b': 3, 'c': 4}
    assert base == {'a': 1, 'b': 2}
