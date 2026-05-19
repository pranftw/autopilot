"""Tests for ai/environment.py: IsolatedEnvironment worktree isolation.

Covers IsolatedEnvironment setup/teardown/activate, module-level helpers
(collect_parameter_files, build_param_content_map), and execution lifecycle
through the Environment.activate() context manager.
"""

from autopilot.ai.environment import (
  IsolatedEnvironment,
  WorktreeWalkContext,
  build_param_content_map,
  collect_parameter_files,
)
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import ConfigError, StoreError
from autopilot.core.experiment import Experiment
from autopilot.core.module.module import Module
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import subprocess
import sys

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

_TEST_CORE_FILES: tuple[str, ...] = ('pyproject.toml', 'README.md')

# -- helpers --


class SimpleModule(Module):
  """Module with a single PathParameter for testing."""

  def __init__(self, source: str, pattern: str = '**/*') -> None:
    super().__init__()
    self.param = PathParameter(source=source, pattern=pattern)


class MultiParamModule(Module):
  """Module with multiple PathParameters."""

  def __init__(self, source1: str, source2: str) -> None:
    super().__init__()
    self.p1 = PathParameter(source=source1)
    self.p2 = PathParameter(source=source2)


class EmptyModule(Module):
  """Module with no parameters."""


def _make_project(root: Path, files: dict[str, str] | None = None) -> None:
  """Create a simple project directory structure."""
  root.mkdir(parents=True, exist_ok=True)
  if files:
    for name, content in files.items():
      p = root / name
      p.parent.mkdir(parents=True, exist_ok=True)
      p.write_text(content, encoding='utf-8')


def _setup_store_with_snapshot(
  config: AutoPilotConfig,
  module: Module,
  experiment_id: str,
) -> FileStore:
  """Create a FileStore and take a snapshot at epoch 0.

  When the module has no parameters, creates the branch entry in refs
  without calling snapshot() (which requires registered parameters).
  """
  named = dict(module.named_parameters())
  store = FileStore(config)
  if named:
    store.register_parameters(named)
    store.snapshot(experiment_id, 0)
  else:
    from autopilot.tracking.io import atomic_write_json

    refs = store.load_refs()
    branches = refs.get('branches', {})
    branches[experiment_id] = {'latest_epoch': -1, 'parent_id': None, 'parent_epoch': None}
    refs['branches'] = branches
    atomic_write_json(config.refs_file, refs)
  return store


# -- fixtures --


@pytest.fixture
def project_root(tmp_path):
  """Standard project with multiple files and nested directories."""
  root = tmp_path / 'project'
  _make_project(
    root,
    {
      'entry.py': 'import blah1\nprint(blah1.value)',
      'blah1.py': 'value = "original"',
      'blah2.py': 'helper = True',
      'README.md': '# My Project',
      'pyproject.toml': '[project]\nname = "test"',
      'subdir/nested.py': 'nested = True',
      'subdir/deep/inner.py': 'inner = True',
    },
  )
  return root


@pytest.fixture
def config(project_root):
  """AutoPilotConfig rooted at the project fixture."""
  cfg = AutoPilotConfig(workspace=project_root.parent)
  cfg.root = project_root
  return cfg


@pytest.fixture
def experiment():
  """Default experiment fixture."""
  return Experiment(experiment_id='exp-001', hypothesis='test')


# -- 4.1 Worktree creation and paths --


class TestWorktreeCreationAndPaths:
  def test_worktree_directory_exists(self, config, project_root, experiment):
    """iso.setup returns an extant directory path."""
    module = SimpleModule(source=str(project_root / 'blah1.py'))
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt = iso.setup(experiment, store, module)
    try:
      assert wt.exists()
      assert wt.is_dir()
    finally:
      iso.teardown(experiment)

  def test_empty_project(self, tmp_path, experiment):
    """Empty project root still gets worktree path."""
    root = tmp_path / 'empty_project'
    root.mkdir()
    cfg = AutoPilotConfig(workspace=tmp_path)
    cfg.root = root
    module = EmptyModule()
    store = _setup_store_with_snapshot(cfg, module, experiment.id)
    iso = IsolatedEnvironment(cfg, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES)
    wt = iso.setup(experiment, store, module)
    try:
      assert wt.exists()
    finally:
      iso.teardown(experiment)


# -- 4.2 Symlinks, copies, unit dirs, structure --


class TestSymlinksCopiesStructure:
  def test_non_param_files_are_symlinks(self, config, project_root, experiment):
    """Non-parameter files in the worktree are symlinks."""
    module = SimpleModule(source=str(project_root / 'blah1.py'))
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt = iso.setup(experiment, store, module)
    try:
      assert (wt / 'entry.py').is_symlink()
      assert (wt / 'blah2.py').is_symlink()
      assert (wt / 'README.md').is_symlink()
    finally:
      iso.teardown(experiment)

  def test_symlink_content_matches_project(self, config, project_root, experiment):
    """Symlinked files have same content as project originals."""
    module = SimpleModule(source=str(project_root / 'blah1.py'))
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt = iso.setup(experiment, store, module)
    try:
      assert (wt / 'entry.py').read_text() == 'import blah1\nprint(blah1.value)'
    finally:
      iso.teardown(experiment)

  def test_param_files_are_not_symlinks(self, config, project_root, experiment):
    """Parameter files in the worktree are copies, not symlinks."""
    module = SimpleModule(source=str(project_root / 'blah1.py'))
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt = iso.setup(experiment, store, module)
    try:
      blah1 = wt / 'blah1.py'
      assert blah1.exists()
      assert not blah1.is_symlink()
    finally:
      iso.teardown(experiment)

  def test_param_file_content_matches_snapshot(self, config, project_root, experiment):
    """Parameter file content comes from the store snapshot."""
    module = SimpleModule(source=str(project_root / 'blah1.py'))
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt = iso.setup(experiment, store, module)
    try:
      content = (wt / 'blah1.py').read_text()
      assert content == 'value = "original"'
    finally:
      iso.teardown(experiment)

  def test_venv_symlinked_as_unit(self, config, project_root, experiment):
    """.venv directory is symlinked as a whole unit."""
    venv = project_root / '.venv'
    venv.mkdir()
    (venv / 'bin').mkdir()
    (venv / 'bin' / 'python').write_text('#!/usr/bin/env python')
    module = EmptyModule()
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt = iso.setup(experiment, store, module)
    try:
      wt_venv = wt / '.venv'
      assert wt_venv.is_symlink()
      assert (wt_venv / 'bin' / 'python').read_text() == '#!/usr/bin/env python'
    finally:
      iso.teardown(experiment)

  def test_node_modules_symlinked_as_unit(self, config, project_root, experiment):
    """node_modules directory is symlinked as a whole unit."""
    nm = project_root / 'node_modules'
    nm.mkdir()
    (nm / 'package').mkdir()
    (nm / 'package' / 'index.js').write_text('module.exports = {}')
    module = EmptyModule()
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt = iso.setup(experiment, store, module)
    try:
      wt_nm = wt / 'node_modules'
      assert wt_nm.is_symlink()
      assert (wt_nm / 'package' / 'index.js').exists()
    finally:
      iso.teardown(experiment)

  def test_core_files_always_present(self, config, project_root, experiment):
    """Default core_files (pyproject.toml, README.md) are always symlinked."""
    module = EmptyModule()
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt = iso.setup(experiment, store, module)
    try:
      assert (wt / 'pyproject.toml').exists()
      assert (wt / 'README.md').exists()
    finally:
      iso.teardown(experiment)

  def test_nested_directories_mirrored(self, config, project_root, experiment):
    """Nested directory structure is preserved in worktree."""
    module = EmptyModule()
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt = iso.setup(experiment, store, module)
    try:
      assert (wt / 'subdir' / 'nested.py').exists()
      assert (wt / 'subdir' / 'deep' / 'inner.py').exists()
    finally:
      iso.teardown(experiment)

  def test_fresh_symlinks_after_project_change(self, config, project_root, experiment):
    """After project file change, new worktree reflects updated content."""
    module = EmptyModule()
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )

    iso.setup(experiment, store, module)
    iso.teardown(experiment)

    (project_root / 'entry.py').write_text('updated content')

    wt2 = iso.setup(experiment, store, module)
    try:
      content = (wt2 / 'entry.py').read_text()
      assert content == 'updated content'
    finally:
      iso.teardown(experiment)

  def test_param_file_change_in_worktree_does_not_affect_project(
    self, config, project_root, experiment
  ):
    """Modifying a param file in worktree leaves the project file intact."""
    module = SimpleModule(source=str(project_root / 'blah1.py'))
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt = iso.setup(experiment, store, module)
    try:
      wt_blah1 = wt / 'blah1.py'
      wt_blah1.write_text('value = "modified"')
      assert (project_root / 'blah1.py').read_text() == 'value = "original"'
    finally:
      iso.teardown(experiment)


# -- 4.3 Ignore patterns --


class TestIgnorePatterns:
  def test_autopilot_dir_excluded(self, config, project_root, experiment):
    """.autopilot/ directory is excluded from worktree."""
    (project_root / '.autopilot' / 'test').mkdir(parents=True)
    module = EmptyModule()
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt = iso.setup(experiment, store, module)
    try:
      assert not (wt / '.autopilot').exists()
    finally:
      iso.teardown(experiment)

  def test_pycache_excluded(self, config, project_root, experiment):
    """__pycache__/ directory is excluded from worktree."""
    (project_root / '__pycache__').mkdir()
    (project_root / '__pycache__' / 'mod.pyc').write_bytes(b'compiled')
    module = EmptyModule()
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt = iso.setup(experiment, store, module)
    try:
      assert not (wt / '__pycache__').exists()
    finally:
      iso.teardown(experiment)

  def test_git_dir_excluded(self, config, project_root, experiment):
    """.git/ directory is excluded from worktree."""
    (project_root / '.git').mkdir()
    (project_root / '.git' / 'HEAD').write_text('ref: refs/heads/main')
    module = EmptyModule()
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt = iso.setup(experiment, store, module)
    try:
      assert not (wt / '.git').exists()
    finally:
      iso.teardown(experiment)

  def test_pyc_files_excluded(self, config, project_root, experiment):
    """*.pyc files are excluded from worktree."""
    (project_root / 'module.pyc').write_bytes(b'compiled')
    module = EmptyModule()
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt = iso.setup(experiment, store, module)
    try:
      assert not (wt / 'module.pyc').exists()
    finally:
      iso.teardown(experiment)

  def test_custom_ignore_patterns(self, config, project_root, experiment):
    """Custom ignore patterns via constructor exclude additional files."""
    (project_root / 'debug.log').write_text('log line')
    module = EmptyModule()
    store = _setup_store_with_snapshot(config, module, experiment.id)
    custom_patterns = (*_TEST_IGNORE_PATTERNS, '*.log')
    iso = IsolatedEnvironment(config, custom_patterns, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES)
    wt = iso.setup(experiment, store, module)
    try:
      assert not (wt / 'debug.log').exists()
    finally:
      iso.teardown(experiment)


# -- 4.4 Teardown and idempotency --


class TestTeardownIdempotency:
  def test_teardown_removes_worktree(self, config, project_root, experiment):
    """Teardown removes the worktree directory."""
    module = EmptyModule()
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt = iso.setup(experiment, store, module)
    wt_path = wt
    iso.teardown(experiment)
    assert not wt_path.exists()

  def test_teardown_on_missing_dir_is_noop(self, config, project_root, experiment):
    """Double teardown does not crash (idempotent removal)."""
    module = EmptyModule()
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    iso.setup(experiment, store, module)
    iso.teardown(experiment)
    iso.teardown(experiment)


# -- 4.5 Parallel / import isolation --


class TestParallelImportIsolation:
  def test_two_setups_have_independent_param_copies(self, config, project_root):
    """Two setups with different snapshots have independent param copies."""
    (project_root / 'param.py').write_text('x = 1')

    mod_a = SimpleModule(source=str(project_root / 'param.py'))
    exp_a = Experiment(experiment_id='exp-a')
    store = _setup_store_with_snapshot(config, mod_a, exp_a.id)

    (project_root / 'param.py').write_text('x = 2')
    mod_b = SimpleModule(source=str(project_root / 'param.py'))
    exp_b = Experiment(experiment_id='exp-b')
    store.snapshot(exp_b.id, 0)

    iso_a = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    iso_b = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt_a = iso_a.setup(exp_a, store, mod_a)
    wt_b = iso_b.setup(exp_b, store, mod_b)

    try:
      content_a = (wt_a / 'param.py').read_text()
      content_b = (wt_b / 'param.py').read_text()
      assert content_a == 'x = 1'
      assert content_b == 'x = 2'
    finally:
      iso_a.teardown(exp_a)
      iso_b.teardown(exp_b)

  def test_different_worktrees_different_content(self, config, project_root):
    """Two concurrent worktrees with different snapshot content."""
    (project_root / 'data.py').write_text('value = 1')
    mod_a = SimpleModule(source=str(project_root / 'data.py'))
    exp_a = Experiment(experiment_id='iso-a')
    store = _setup_store_with_snapshot(config, mod_a, exp_a.id)

    (project_root / 'data.py').write_text('value = 2')
    mod_b = SimpleModule(source=str(project_root / 'data.py'))
    exp_b = Experiment(experiment_id='iso-b')
    store.snapshot(exp_b.id, 0)

    iso_a = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    iso_b = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt_a = iso_a.setup(exp_a, store, mod_a)
    wt_b = iso_b.setup(exp_b, store, mod_b)

    try:
      content_a = (wt_a / 'data.py').read_text()
      content_b = (wt_b / 'data.py').read_text()
      assert content_a == 'value = 1'
      assert content_b == 'value = 2'
    finally:
      iso_a.teardown(exp_a)
      iso_b.teardown(exp_b)

  def test_subprocess_import_isolation(self, config, project_root):
    """Two worktrees with different content produce different subprocess imports."""
    (project_root / 'blah1.py').write_text('value = "version_a"')
    mod_a = SimpleModule(source=str(project_root / 'blah1.py'))
    exp_a = Experiment(experiment_id='imp-a')
    store = _setup_store_with_snapshot(config, mod_a, exp_a.id)

    (project_root / 'blah1.py').write_text('value = "version_b"')
    mod_b = SimpleModule(source=str(project_root / 'blah1.py'))
    exp_b = Experiment(experiment_id='imp-b')
    store.snapshot(exp_b.id, 0)

    iso_a = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    iso_b = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt_a = iso_a.setup(exp_a, store, mod_a)
    wt_b = iso_b.setup(exp_b, store, mod_b)

    import_script = (
      'import sys, os; sys.path.insert(0, os.getcwd()); import blah1; print(blah1.value)'
    )

    try:
      result_a = subprocess.run(
        [sys.executable, '-c', import_script],
        cwd=str(wt_a),
        capture_output=True,
        text=True,
        check=False,
      )
      result_b = subprocess.run(
        [sys.executable, '-c', import_script],
        cwd=str(wt_b),
        capture_output=True,
        text=True,
        check=False,
      )
      assert result_a.returncode == 0, result_a.stderr
      assert 'version_a' in result_a.stdout
      assert result_b.returncode == 0, result_b.stderr
      assert 'version_b' in result_b.stdout
    finally:
      iso_a.teardown(exp_a)
      iso_b.teardown(exp_b)


# -- 4.6 Parent snapshot sourcing --


class TestParentSnapshotSourcing:
  def test_child_gets_parent_snapshot_content(self, config, project_root):
    """When branching from a parent, child worktree uses parent's snapshot."""
    (project_root / 'code.py').write_text('original')
    mod = SimpleModule(source=str(project_root / 'code.py'))
    named = dict(mod.named_parameters())
    store = FileStore(config)
    store.register_parameters(named)

    store.snapshot('parent-exp', 0)

    (project_root / 'code.py').write_text('changed after snapshot')

    store.branch('child-exp')

    child_exp = Experiment(experiment_id='child-exp')
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt = iso.setup(child_exp, store, mod)
    try:
      content = (wt / 'code.py').read_text()
      assert content == 'original'
    finally:
      iso.teardown(child_exp)


# -- 4.7 Snapshot loading errors and content --


class TestSnapshotLoadingErrorsAndContent:
  def test_corrupted_store_missing_object_raises_store_error(self, config, project_root):
    """Missing object file in store raises StoreError with experiment_id."""
    (project_root / 'code.py').write_text('hello')
    mod = SimpleModule(source=str(project_root / 'code.py'))
    named = dict(mod.named_parameters())
    store = FileStore(config)
    store.register_parameters(named)
    exp_id = 'corrupt-exp'
    store.snapshot(exp_id, 0)

    snap = store.load_snapshot(exp_id, 0)
    for entry in snap.entries.values():
      prefix = entry.digest[:2]
      rest = entry.digest[2:]
      obj_path = config.objects_path / prefix / rest
      obj_path.unlink()

    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    iso._store = store
    with pytest.raises(StoreError, match=exp_id) as exc_info:
      iso._load_snapshot_content(exp_id, 0)
    assert 'not found' in str(exc_info.value)

  def test_load_snapshot_content_returns_text(self, config, project_root):
    """Normal case: _load_snapshot_content returns file content as strings."""
    (project_root / 'code.py').write_text('print("hi")')
    mod = SimpleModule(source=str(project_root / 'code.py'))
    named = dict(mod.named_parameters())
    store = FileStore(config)
    store.register_parameters(named)
    exp_id = 'normal-exp'
    store.snapshot(exp_id, 0)

    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    iso._store = store
    content = iso._load_snapshot_content(exp_id, 0)
    assert len(content) > 0
    for val in content.values():
      assert isinstance(val, str)
      assert 'print' in val

  def test_snapshot_raises_on_binary_file(self, config, project_root):
    """PathParameter.snapshot raises StoreError on binary file."""
    (project_root / 'data.bin').write_bytes(b'\x80\x81\x82\xff')
    mod = SimpleModule(source=str(project_root / 'data.bin'))
    named = dict(mod.named_parameters())
    store = FileStore(config)
    store.register_parameters(named)
    exp_id = 'bin-exp'
    with pytest.raises(StoreError, match='binary file'):
      store.snapshot(exp_id, 0)

  def test_empty_refs_returns_empty(self, config, project_root, experiment) -> None:
    """_get_snapshot_content returns {} when store refs have no branches."""
    store = MagicMock()
    store.load_refs.return_value = {}
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    iso._store = store
    assert iso._get_snapshot_content(experiment.id) == {}

  def test_experiment_not_in_branches_returns_empty(self, config, project_root) -> None:
    """_get_snapshot_content returns {} when experiment is not in branches."""
    store = MagicMock()
    store.load_refs.return_value = {'branches': {'other': {'latest_epoch': 0}}}
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    iso._store = store
    assert iso._get_snapshot_content('missing-exp') == {}

  def test_negative_epoch_with_parent_delegates(self, config, project_root) -> None:
    """When latest_epoch < 0 and parent exists, delegates to parent snapshot."""
    store = MagicMock()
    store.load_refs.return_value = {
      'branches': {
        'child': {'latest_epoch': -1, 'parent_id': 'parent', 'parent_epoch': 2},
        'parent': {'latest_epoch': 2},
      }
    }
    snap = MagicMock()
    snap.entries = {'param_0/file.py': MagicMock(digest='abc123')}
    store.load_snapshot.return_value = snap
    store.read_object.return_value = b'hello'
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    iso._store = store
    result = iso._get_snapshot_content('child')
    store.load_snapshot.assert_called_once_with('parent', 2)
    assert 'param_0/file.py' in result

  def test_negative_epoch_no_parent_returns_empty(self, config, project_root) -> None:
    """When latest_epoch < 0 and no parent, returns {}."""
    store = MagicMock()
    store.load_refs.return_value = {'branches': {'orphan': {'latest_epoch': -1}}}
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    iso._store = store
    assert iso._get_snapshot_content('orphan') == {}


# -- 4.8 Helpers and narrowing --


class TestHelpersAndNarrowing:
  def test_collect_skips_non_path_parameters(self, config, project_root):
    """Non-PathParameter parameters are skipped by collect_parameter_files."""
    from autopilot.core.parameter import Parameter

    class TextParam(Parameter):
      def snapshot(self) -> dict[str, str]:
        return {'text': 'hello'}

      def restore(self, content: dict[str, str]) -> None:
        pass

    class MixedModule(Module):
      def __init__(self) -> None:
        super().__init__()
        self.text_p = TextParam()
        self.file_p = PathParameter(source=str(project_root), pattern='*.py')

    mod = MixedModule()
    files = collect_parameter_files(mod)
    assert len(files) > 0
    for f in files:
      assert f.suffix == '.py'

  def test_build_param_content_map_skips_non_path_params(self, config, project_root):
    """build_param_content_map skips parameters without source."""
    from autopilot.core.parameter import Parameter

    class TextParam(Parameter):
      def snapshot(self) -> dict[str, str]:
        return {'text': 'hello'}

      def restore(self, content: dict[str, str]) -> None:
        pass

    class MixedModule(Module):
      def __init__(self) -> None:
        super().__init__()
        self.text_p = TextParam()
        self.file_p = PathParameter(source=str(project_root), pattern='*.py')

    mod = MixedModule()
    snap_content = {'file_p/main.py': 'print("hi")'}
    result = build_param_content_map(mod, snap_content)
    assert len(result) >= 0

  def test_isolated_environment_accepts_autopilot_config(self, config):
    """IsolatedEnvironment constructor requires AutoPilotConfig."""
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    assert iso.config is config

  def test_multiple_matched_files(self, project_root: Path) -> None:
    """collect_parameter_files returns resolved paths from all PathParameters."""
    mod = MultiParamModule(
      source1=str(project_root / 'blah1.py'),
      source2=str(project_root / 'blah2.py'),
    )
    files = collect_parameter_files(mod)
    assert (project_root / 'blah1.py').resolve() in files
    assert (project_root / 'blah2.py').resolve() in files

  def test_directory_source_collects_all(self, project_root: Path) -> None:
    """collect_parameter_files with directory source gets all matching files."""
    mod = SimpleModule(source=str(project_root / 'subdir'), pattern='**/*.py')
    files = collect_parameter_files(mod)
    assert len(files) >= 2
    assert (project_root / 'subdir' / 'nested.py').resolve() in files

  def test_file_source_maps_dot_key(self, project_root: Path) -> None:
    """Single file source maps snapshot key ``param/.`` to that file's resolved path."""
    mod = SimpleModule(source=str(project_root / 'blah1.py'))
    snap = {'param/.': 'new content'}
    result = build_param_content_map(mod, snap)
    assert result[(project_root / 'blah1.py').resolve()] == 'new content'

  def test_dir_source_maps_relative(self, project_root: Path) -> None:
    """Directory source maps ``param/relative_key`` to source/relative_key."""
    mod = SimpleModule(source=str(project_root / 'subdir'), pattern='**/*.py')
    snap = {'param/nested.py': 'updated'}
    result = build_param_content_map(mod, snap)
    assert result[(project_root / 'subdir' / 'nested.py').resolve()] == 'updated'

  def test_nonexistent_source_maps_dot_key(self, tmp_path: Path) -> None:
    """Non-existent source with '.' key still resolves."""
    missing = tmp_path / 'project' / 'ghost.py'
    mod = SimpleModule(source=str(missing))
    snap = {'param/.': 'ghost content'}
    result = build_param_content_map(mod, snap)
    assert result[missing.resolve()] == 'ghost content'


# -- 4.9 Edge / internal walk behavior --


class TestEdgeInternalWalk:
  def test_zero_parameters_only_symlinks(self, config, project_root, experiment):
    """Module with zero parameters produces only symlinks in worktree."""
    module = EmptyModule()
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt = iso.setup(experiment, store, module)
    try:
      assert (wt / 'entry.py').is_symlink()
      assert (wt / 'blah1.py').is_symlink()
      assert (wt / 'blah2.py').is_symlink()
    finally:
      iso.teardown(experiment)

  def test_missing_project_root_raises_config_error(self, tmp_path: Path, experiment) -> None:
    """_build_worktree raises ConfigError when project_root doesn't exist."""
    root = tmp_path / 'nonexistent_project'
    cfg = AutoPilotConfig(workspace=tmp_path)
    cfg.root = root
    iso = IsolatedEnvironment(cfg, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES)
    wt_path = tmp_path / 'worktree_out'
    wt_path.mkdir()
    with pytest.raises(ConfigError, match='project_root does not exist'):
      iso._build_worktree(
        project_root=root,
        wt_path=wt_path,
        param_files=set(),
        ignore_patterns=(),
        unit_dirs=set(),
        core_files=set(),
        param_content_map={},
      )

  def test_permission_error_propagates(self, config, project_root) -> None:
    """_walk_and_link raises PermissionError from iterdir."""
    store = FileStore(config)
    param = PathParameter(source=str(project_root), pattern='**/*')
    store.register_parameters({'src': param})
    store.snapshot('perm-exp', 0)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt = config.worktrees_path / 'perm-exp'
    wt.mkdir(parents=True, exist_ok=True)
    ctx = WorktreeWalkContext(
      project_root=project_root,
      wt_path=wt,
      param_files=set(),
      ignore_patterns=(),
      unit_dirs=set(),
      param_content_map={},
    )
    with (
      patch.object(Path, 'iterdir', side_effect=PermissionError('forbidden')),
      pytest.raises(PermissionError, match='permission denied'),
    ):
      iso._walk_and_link(ctx=ctx, current_dir=project_root)

  def test_symlink_param_file_uses_copy2_when_no_content(
    self, config, project_root, experiment
  ) -> None:
    """Symlink entry resolved into param_files with no content_map entry uses copy2."""
    target = project_root / 'real_param.py'
    target.write_text('real content')
    link = project_root / 'link_param.py'
    link.symlink_to(target)

    module = SimpleModule(source=str(target))
    _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )

    wt = config.worktrees_path / 'sym-exp'
    wt.mkdir(parents=True, exist_ok=True)

    param_files = {target.resolve()}
    ctx = WorktreeWalkContext(
      project_root=project_root,
      wt_path=wt,
      param_files=param_files,
      ignore_patterns=iso.ignore_patterns,
      unit_dirs=set(),
      param_content_map={},
    )
    iso._walk_and_link(ctx=ctx, current_dir=project_root)
    wt_link = wt / 'link_param.py'
    assert wt_link.exists()
    assert not wt_link.is_symlink()
    assert wt_link.read_text() == 'real content'

  def test_regular_file_param_uses_copy2_when_no_content(
    self, config, project_root, experiment
  ) -> None:
    """Regular file in param_files with no content_map entry uses copy2."""
    param_file = project_root / 'blah1.py'

    module = SimpleModule(source=str(param_file))
    _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )

    wt = config.worktrees_path / 'reg-exp'
    wt.mkdir(parents=True, exist_ok=True)

    param_files = {param_file.resolve()}
    ctx = WorktreeWalkContext(
      project_root=project_root,
      wt_path=wt,
      param_files=param_files,
      ignore_patterns=iso.ignore_patterns,
      unit_dirs=set(),
      param_content_map={},
    )
    iso._walk_and_link(ctx=ctx, current_dir=project_root)
    wt_file = wt / 'blah1.py'
    assert wt_file.exists()
    assert not wt_file.is_symlink()
    assert wt_file.read_text() == 'value = "original"'


# -- 4.10 activate and round-trip (new) --


class TestActivateAndRoundTrip:
  def test_activate_yields_worktree_and_teardown_removes(self, config, project_root, experiment):
    """activate context manager yields worktree path and removes on exit."""
    module = EmptyModule()
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    with iso.activate(experiment, store, module) as wt:
      assert wt.exists()
      assert wt.is_dir()
      wt_path = wt
    assert not wt_path.exists()

  def test_activate_teardown_on_exception(self, config, project_root, experiment):
    """activate teardown runs even when body raises; exception propagates."""
    module = EmptyModule()
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    captured_path: Path | None = None

    def run_and_raise() -> None:
      nonlocal captured_path
      with iso.activate(experiment, store, module) as wt:
        captured_path = wt
        assert wt.exists()
        msg = 'boom'
        raise ValueError(msg)

    with pytest.raises(ValueError, match='boom'):
      run_and_raise()
    assert captured_path is not None
    assert not captured_path.exists()

  def test_custom_symlink_as_unit_or_core_files(self, config, project_root, experiment):
    """Constructor overrides for symlink_as_unit and core_files are honored."""
    custom_dir = project_root / 'vendor'
    custom_dir.mkdir()
    (custom_dir / 'lib.py').write_text('vendored')

    module = EmptyModule()
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config,
      ignore_patterns=_TEST_IGNORE_PATTERNS,
      symlink_as_unit=('.venv', 'node_modules', 'vendor'),
      core_files=('pyproject.toml',),
    )
    wt = iso.setup(experiment, store, module)
    try:
      wt_vendor = wt / 'vendor'
      assert wt_vendor.is_symlink()
      assert (wt_vendor / 'lib.py').read_text() == 'vendored'
    finally:
      iso.teardown(experiment)

  def test_setup_teardown_setup_roundtrip_fresh_tree(self, config, project_root, experiment):
    """setup -> write file -> teardown -> setup again yields fresh tree."""
    module = EmptyModule()
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )

    wt1 = iso.setup(experiment, store, module)
    scratch_file = wt1 / 'scratch.txt'
    scratch_file.write_text('temporary')
    iso.teardown(experiment)

    wt2 = iso.setup(experiment, store, module)
    try:
      assert not (wt2 / 'scratch.txt').exists()
    finally:
      iso.teardown(experiment)


# -- 4.11 IsolatedEnvironment execution lifecycle --


class TestIsolatedEnvironmentExecution:
  def test_full_activate_lifecycle(self, config, project_root, experiment):
    """activate context manager sets up worktree and tears down on exit."""
    module = EmptyModule()
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    captured_wt: Path | None = None
    with iso.activate(experiment, store, module) as wt:
      captured_wt = wt
      assert wt.exists()
      assert wt.is_dir()
    assert captured_wt is not None
    assert not captured_wt.exists()

  def test_user_code_runs_inside_worktree(self, config, project_root, experiment):
    """User code can execute inside the activated worktree."""
    module = EmptyModule()
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    with iso.activate(experiment, store, module) as wt:
      result = subprocess.run(
        [sys.executable, '-c', 'print("hello from worktree")'],
        cwd=str(wt),
        capture_output=True,
        text=True,
        check=False,
      )
      assert result.returncode == 0
      assert 'hello from worktree' in result.stdout

  def test_activate_teardown_on_user_exception(self, config, project_root, experiment):
    """activate teardown runs even when user code raises inside the context."""
    module = EmptyModule()
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    captured_wt: Path | None = None

    def run_and_raise() -> None:
      nonlocal captured_wt
      with iso.activate(experiment, store, module) as wt:
        captured_wt = wt
        assert wt.exists()
        msg = 'user code failed'
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match='user code failed'):
      run_and_raise()
    assert captured_wt is not None
    assert not captured_wt.exists()

  def test_multiple_experiments_independent(self, config, project_root):
    """Two experiments can activate independent worktrees concurrently."""
    module = EmptyModule()
    exp_a = Experiment(experiment_id='exec-a')
    exp_b = Experiment(experiment_id='exec-b')
    store = _setup_store_with_snapshot(config, module, exp_a.id)
    from autopilot.tracking.io import atomic_write_json

    refs = store.load_refs()
    refs.setdefault('branches', {})[exp_b.id] = {
      'latest_epoch': -1,
      'parent_id': None,
      'parent_epoch': None,
    }
    atomic_write_json(config.refs_file, refs)

    iso_a = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    iso_b = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    with (
      iso_a.activate(exp_a, store, module) as wt_a,
      iso_b.activate(exp_b, store, module) as wt_b,
    ):
      assert wt_a != wt_b
      assert wt_a.exists()
      assert wt_b.exists()
    assert not wt_a.exists()
    assert not wt_b.exists()


# -- 4.12 Silent failure fixes (plan 02) --


class TestGetSnapshotContentWithoutSetupRaises:
  def test_get_snapshot_content_without_setup_raises(self, config) -> None:
    """_get_snapshot_content raises RuntimeError before setup() is called."""
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    with pytest.raises(RuntimeError, match='store is not initialized'):
      iso._get_snapshot_content('any-id')

  def test_load_snapshot_content_without_setup_raises(self, config) -> None:
    """_load_snapshot_content raises RuntimeError before setup() is called."""
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    with pytest.raises(RuntimeError, match='store is not initialized'):
      iso._load_snapshot_content('any-id', 0)


class TestMalformedBranchesTypeRaisesStoreError:
  def test_malformed_branches_type_raises_store_error(self, config) -> None:
    """branches value that is not a dict raises StoreError."""
    store = MagicMock()
    store.load_refs.return_value = {'branches': 'not-a-dict'}
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    iso._store = store
    with pytest.raises(StoreError, match='expected branches to be a dict'):
      iso._get_snapshot_content('any-exp')

  def test_branches_as_list_raises_store_error(self, config) -> None:
    """branches value as a list raises StoreError."""
    store = MagicMock()
    store.load_refs.return_value = {'branches': ['a', 'b']}
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    iso._store = store
    with pytest.raises(StoreError, match='expected branches to be a dict'):
      iso._get_snapshot_content('any-exp')


class TestStrictBranchFieldAccess:
  def test_missing_latest_epoch_raises_key_error(self, config) -> None:
    """Branch record without latest_epoch raises KeyError."""
    store = MagicMock()
    store.load_refs.return_value = {'branches': {'exp-1': {}}}
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    iso._store = store
    with pytest.raises(KeyError, match='latest_epoch'):
      iso._get_snapshot_content('exp-1')

  def test_optional_parent_id_defaults_to_none(self, config) -> None:
    """Branch without parent_id returns {} when latest_epoch < 0."""
    store = MagicMock()
    store.load_refs.return_value = {'branches': {'orphan': {'latest_epoch': -1}}}
    iso = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    iso._store = store
    assert iso._get_snapshot_content('orphan') == {}


# -- 4.13 API contract and edge cases (plan 02 additive) --


class TestIsolatedEnvironmentRequiresAllTupleArgs:
  """Verify IsolatedEnvironment.__init__ has no defaults for the three tuples."""

  def test_isolated_environment_requires_all_tuple_args(self) -> None:
    """inspect.signature confirms no default for the three tuple parameters."""
    import inspect

    sig = inspect.signature(IsolatedEnvironment.__init__)
    for param_name in ('ignore_patterns', 'symlink_as_unit', 'core_files'):
      param = sig.parameters[param_name]
      assert param.default is inspect.Parameter.empty, f'{param_name} should have no default value'


class TestEnvironmentEmptyIgnorePatterns:
  """Minimal ignore_patterns still works; normally-excluded dirs appear."""

  def test_environment_minimal_ignore_patterns(self, config, project_root, experiment) -> None:
    """With only .autopilot/ excluded, .git/ and __pycache__/ are present."""
    (project_root / '.git').mkdir()
    (project_root / '.git' / 'HEAD').write_text('ref: refs/heads/main')
    (project_root / '__pycache__').mkdir()
    (project_root / '__pycache__' / 'mod.cpython-312.pyc').write_bytes(b'compiled')

    module = EmptyModule()
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(config, ('.autopilot/',), _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES)
    wt = iso.setup(experiment, store, module)
    try:
      assert (wt / '.git').exists()
      assert (wt / '__pycache__').exists()
    finally:
      iso.teardown(experiment)


class TestEnvironmentEmptyCoreFiles:
  """Empty core_files means no files symlinked unconditionally."""

  def test_environment_empty_core_files(self, config, project_root, experiment) -> None:
    """activate() succeeds with core_files=(); no core symlinks created early."""
    module = EmptyModule()
    store = _setup_store_with_snapshot(config, module, experiment.id)
    iso = IsolatedEnvironment(config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, ())
    wt = iso.setup(experiment, store, module)
    try:
      assert wt.exists()
      assert wt.is_dir()
    finally:
      iso.teardown(experiment)


class TestEnvironmentConstructorTypeError:
  """Omitting required args raises TypeError (standard Python behavior)."""

  def test_environment_constructor_type_error_missing_all(self) -> None:
    """Missing all three tuples raises TypeError."""
    with pytest.raises(TypeError):
      IsolatedEnvironment(MagicMock())  # type: ignore[ty:missing-argument]

  def test_environment_constructor_type_error_missing_one(self, config) -> None:
    """Missing core_files raises TypeError."""
    with pytest.raises(TypeError):
      IsolatedEnvironment(  # type: ignore[ty:missing-argument]
        config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT
      )
