"""Tests for IsolatedEnvironment symlink safety (BUG-004) and PathParameter thread model.

Covers:
- Worktree parameter writes that overlap core_files symlinks do not mutate the
  canonical project file (BUG-004 regression).
- Written parameter files are regular files, not symlinks.
- Non-overlapping core_file symlinks are preserved.
- PathParameter bind/unbind docstring contains frozen thread-safety sentence.
- Two sequential IsolatedEnvironment activations do not cross-contaminate.
"""

from autopilot.ai.environment import IsolatedEnvironment
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.module.module import Module
from pathlib import Path
import pytest

_TEST_IGNORE_PATTERNS: tuple[str, ...] = (
  '.autopilot/',
  '__pycache__/',
  '.pytest_cache/',
  '*.pyc',
  '.git/',
  '.ruff_cache/',
  '*.egg-info/',
)

_TEST_SYMLINK_AS_UNIT: tuple[str, ...] = ('.venv', 'node_modules')


class SimpleModule(Module):
  """Module with a single PathParameter for testing."""

  def __init__(self, source: str, pattern: str = '**/*') -> None:
    super().__init__()
    self.param = PathParameter(source=source, pattern=pattern)


def _make_project(root: Path, files: dict[str, str]) -> None:
  """Create a project directory with the given file contents."""
  root.mkdir(parents=True, exist_ok=True)
  for name, content in files.items():
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding='utf-8')


def _setup_store_with_snapshot(
  config: AutoPilotConfig,
  module: Module,
  experiment_id: str,
) -> FileStore:
  """Create a FileStore, register params, and snapshot at epoch 0."""
  named = dict(module.named_parameters())
  store = FileStore(config)
  if named:
    store.register_parameters(named)
  store.snapshot(experiment_id, 0)
  return store


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
  """Project with a param file that overlaps a core_file."""
  root = tmp_path / 'project'
  _make_project(
    root,
    {
      'pyproject.toml': '[project]\nname = "test"',
      'README.md': '# My Project',
      'config.toml': 'key = "original_value"',
      'entry.py': 'print("hello")',
    },
  )
  return root


@pytest.fixture
def config(project_root: Path) -> AutoPilotConfig:
  """AutoPilotConfig rooted at the project fixture."""
  cfg = AutoPilotConfig(workspace=project_root.parent)
  cfg.root = project_root
  return cfg


# -- 4.1 BUG-004 and symlink behavior --


class TestSymlinkSafety:
  """BUG-004: parameter writes must not follow symlinks into the source tree."""

  def test_worktree_write_to_overlapping_path_preserves_original(
    self, config: AutoPilotConfig, project_root: Path
  ) -> None:
    """Overlapping core symlink does not change original; worktree file is regular."""
    core_files = ('config.toml',)
    module = SimpleModule(source=str(project_root / 'config.toml'))
    store = _setup_store_with_snapshot(config, module, 'exp-safety')
    experiment = Experiment(experiment_id='exp-safety')

    original_content = (project_root / 'config.toml').read_text(encoding='utf-8')

    iso = IsolatedEnvironment(config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, core_files)
    wt = iso.setup(experiment, store, module)
    try:
      wt_file = wt / 'config.toml'
      assert not wt_file.is_symlink(), 'worktree param file should be regular, not a symlink'
      assert wt_file.exists()
      assert wt_file.read_text(encoding='utf-8') == original_content
      assert (project_root / 'config.toml').read_text(encoding='utf-8') == original_content
    finally:
      iso.teardown(experiment)

  def test_worktree_written_file_is_not_symlink(
    self, config: AutoPilotConfig, project_root: Path
  ) -> None:
    """After parameter write, wt_target.is_symlink() is False."""
    core_files = ('config.toml',)
    module = SimpleModule(source=str(project_root / 'config.toml'))
    store = _setup_store_with_snapshot(config, module, 'exp-nosym')
    experiment = Experiment(experiment_id='exp-nosym')

    iso = IsolatedEnvironment(config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, core_files)
    wt = iso.setup(experiment, store, module)
    try:
      wt_file = wt / 'config.toml'
      assert not wt_file.is_symlink()
      assert wt_file.is_file()
    finally:
      iso.teardown(experiment)

  def test_worktree_write_does_not_mutate_source(
    self, config: AutoPilotConfig, project_root: Path
  ) -> None:
    """Original project file content matches pre-write content after worktree setup."""
    core_files = ('config.toml',)
    original_content = (project_root / 'config.toml').read_text(encoding='utf-8')

    module = SimpleModule(source=str(project_root / 'config.toml'))
    store = _setup_store_with_snapshot(config, module, 'exp-safe')
    experiment = Experiment(experiment_id='exp-safe')

    iso = IsolatedEnvironment(config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, core_files)
    wt = iso.setup(experiment, store, module)
    try:
      wt_file = wt / 'config.toml'
      wt_file.write_text('key = "modified_in_worktree"', encoding='utf-8')
      assert (project_root / 'config.toml').read_text(encoding='utf-8') == original_content
    finally:
      iso.teardown(experiment)

  def test_non_overlapping_symlink_preserved(
    self, config: AutoPilotConfig, project_root: Path
  ) -> None:
    """Core file not overlapping any parameter remains a symlink."""
    core_files = ('pyproject.toml', 'config.toml')
    module = SimpleModule(source=str(project_root / 'config.toml'))
    store = _setup_store_with_snapshot(config, module, 'exp-nonoverlap')
    experiment = Experiment(experiment_id='exp-nonoverlap')

    iso = IsolatedEnvironment(config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, core_files)
    wt = iso.setup(experiment, store, module)
    try:
      wt_pyproject = wt / 'pyproject.toml'
      assert wt_pyproject.is_symlink(), 'non-overlapping core file should remain a symlink'
      assert wt_pyproject.read_text(encoding='utf-8') == '[project]\nname = "test"'
    finally:
      iso.teardown(experiment)


# -- 4.2 Documentation and isolation --


class TestDocumentation:
  """PathParameter bind/unbind docstring contains the frozen thread-safety sentence."""

  def test_bind_unbind_docstring_exists(self) -> None:
    """Chosen docstring surfaces contain the full frozen sentence."""
    frozen_words = (
      'Not thread-safe. Concurrent use across Trainer instances sharing '
      'the same Module is unsupported. Trainer owns bind/unbind lifecycle.'
    )
    bind_doc = PathParameter.bind.__doc__
    unbind_doc = PathParameter.unbind.__doc__
    assert bind_doc is not None, 'bind should have a docstring'
    assert unbind_doc is not None, 'unbind should have a docstring'
    normalized_bind = ' '.join(bind_doc.split())
    normalized_unbind = ' '.join(unbind_doc.split())
    assert frozen_words in normalized_bind, f'bind docstring missing frozen sentence: {bind_doc}'
    assert frozen_words in normalized_unbind, (
      f'unbind docstring missing frozen sentence: {unbind_doc}'
    )


class TestCrossContamination:
  """Two sequential IsolatedEnvironment activations do not cross-contaminate."""

  def test_two_isolated_environments_no_cross_contamination(
    self, config: AutoPilotConfig, project_root: Path
  ) -> None:
    """Parameter write in env A leaves env B and source canonical paths unchanged."""
    core_files = ('config.toml',)
    original_content = (project_root / 'config.toml').read_text(encoding='utf-8')

    module_a = SimpleModule(source=str(project_root / 'config.toml'))
    store = _setup_store_with_snapshot(config, module_a, 'env-a')

    module_b = SimpleModule(source=str(project_root / 'config.toml'))
    store.snapshot('env-b', 0)

    exp_a = Experiment(experiment_id='env-a')
    exp_b = Experiment(experiment_id='env-b')

    iso_a = IsolatedEnvironment(config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, core_files)
    wt_a = iso_a.setup(exp_a, store, module_a)

    iso_b = IsolatedEnvironment(config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, core_files)
    wt_b = iso_b.setup(exp_b, store, module_b)

    try:
      (wt_a / 'config.toml').write_text('key = "env_a_modified"', encoding='utf-8')

      assert (wt_b / 'config.toml').read_text(encoding='utf-8') == original_content
      assert (project_root / 'config.toml').read_text(encoding='utf-8') == original_content
    finally:
      iso_a.teardown(exp_a)
      iso_b.teardown(exp_b)
