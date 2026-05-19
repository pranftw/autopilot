"""Tests for IsolatedEnvironment activation and PathParameter bind/unbind harness.

Covers worktree activation/teardown and harness-level bind/unbind after
activate() returns a path (mirrors Trainer ordering from Plan 08).
"""

from autopilot.ai.environment import IsolatedEnvironment
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.module.module import Module
from autopilot.core.parameter import Parameter
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

_TEST_CORE_FILES: tuple[str, ...] = ('pyproject.toml', 'README.md')


class SimpleModule(Module):
  """Module with a single PathParameter for testing."""

  def __init__(self, source: str, pattern: str = '**/*') -> None:
    super().__init__()
    self.param = PathParameter(source=source, pattern=pattern)


class TwoParamModule(Module):
  """Module with PathParameter and plain Parameter."""

  def __init__(self, source: str) -> None:
    super().__init__()
    self.path_param = PathParameter(source=source, pattern='*')
    self.plain = Parameter()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
  """Workspace with source files."""
  ws = tmp_path / 'project'
  ws.mkdir()
  src = ws / 'src'
  src.mkdir()
  (src / 'main.py').write_text('print("original")', encoding='utf-8')
  return ws


@pytest.fixture
def config(workspace: Path) -> AutoPilotConfig:
  """AutoPilotConfig for the workspace."""
  return AutoPilotConfig(workspace=workspace)


def _make_store_and_snapshot(
  config: AutoPilotConfig, module: Module, experiment_id: str
) -> FileStore:
  """Create store, register params, snapshot epoch 0."""
  store = FileStore(config)
  store.register_parameters(dict(module.named_parameters()))
  store.snapshot(experiment_id, 0)
  return store


class TestIsolatedEnvBinds:
  """After activate(), harness can bind PathParameters to worktree."""

  def test_isolated_env_binds(self, config: AutoPilotConfig, workspace: Path) -> None:
    module = SimpleModule(source=str(workspace / 'src'))
    store = _make_store_and_snapshot(config, module, 'exp-1')
    experiment = Experiment(experiment_id='exp-1')

    env = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt_path = env.setup(experiment, store, module)

    for _name, param in module.named_parameters():
      if isinstance(param, PathParameter):
        bound_root = str(wt_path / Path(param.source).relative_to(config.workspace))
        param.bind(bound_root)

    assert module.param.working_root != str(workspace / 'src')
    assert str(wt_path) in module.param.working_root

    for param in module.parameters():
      if isinstance(param, PathParameter):
        param.unbind()

    env.teardown(experiment)


class TestIsolatedEnvUnbinds:
  """unbind resets PathParameter to source after teardown."""

  def test_isolated_env_unbinds(self, config: AutoPilotConfig, workspace: Path) -> None:
    module = SimpleModule(source=str(workspace / 'src'))
    store = _make_store_and_snapshot(config, module, 'exp-1')
    experiment = Experiment(experiment_id='exp-1')

    env = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt_path = env.setup(experiment, store, module)
    module.param.bind(str(wt_path / 'src'))
    module.param.unbind()

    assert module.param.working_root == str(workspace / 'src')
    env.teardown(experiment)


class TestIsolatedEnvNonPathUnaffected:
  """Plain Parameters are not affected by bind/unbind."""

  def test_isolated_env_non_path_unaffected(self, config: AutoPilotConfig, workspace: Path) -> None:
    module = TwoParamModule(source=str(workspace / 'src'))
    store = _make_store_and_snapshot(config, module, 'exp-1')
    experiment = Experiment(experiment_id='exp-1')

    env = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt_path = env.setup(experiment, store, module)

    for _name, param in module.named_parameters():
      if isinstance(param, PathParameter):
        param.bind(str(wt_path / 'src'))

    assert not hasattr(module.plain, 'working_root')
    assert not hasattr(module.plain, 'bind')

    for param in module.parameters():
      if isinstance(param, PathParameter):
        param.unbind()
    env.teardown(experiment)


class TestSetupTeardownCycle:
  """Full setup/bind/teardown cycle works without errors."""

  def test_setup_teardown_cycle(self, config: AutoPilotConfig, workspace: Path) -> None:
    module = SimpleModule(source=str(workspace / 'src'))
    store = _make_store_and_snapshot(config, module, 'exp-1')
    experiment = Experiment(experiment_id='exp-1')

    env = IsolatedEnvironment(
      config, _TEST_IGNORE_PATTERNS, _TEST_SYMLINK_AS_UNIT, _TEST_CORE_FILES
    )
    wt_path = env.setup(experiment, store, module)
    assert wt_path.exists()

    bound_root = wt_path / Path(module.param.source).relative_to(config.workspace)
    module.param.bind(str(bound_root))
    assert module.param.working_root == str(bound_root)

    module.param.unbind()
    assert module.param.working_root == str(workspace / 'src')
    env.teardown(experiment)


class TestConcurrentOrInterleavedSnapshots:
  """BUG-004 regression: distinct experiment IDs produce non-corrupt manifests."""

  def test_concurrent_or_interleaved_snapshots(
    self, config: AutoPilotConfig, workspace: Path
  ) -> None:
    src = workspace / 'src'
    param_a = PathParameter(source=str(src), pattern='*')
    param_b = PathParameter(source=str(src), pattern='*')

    store_a = FileStore(config)
    store_a.register_parameters({'source': param_a})
    store_a.snapshot('exp-a', 0)

    store_b = FileStore(config)
    store_b.register_parameters({'source': param_b})
    store_b.snapshot('exp-b', 0)

    (src / 'main.py').write_text('modified-a', encoding='utf-8')
    store_a.snapshot('exp-a', 1)

    (src / 'main.py').write_text('modified-b', encoding='utf-8')
    store_b.snapshot('exp-b', 1)

    snap_a = store_a.load_snapshot('exp-a', 1)
    snap_b = store_b.load_snapshot('exp-b', 1)
    content_a = store_a.read_object(snap_a.entries['source/main.py'].digest)
    content_b = store_b.read_object(snap_b.entries['source/main.py'].digest)
    assert content_a.decode('utf-8') == 'modified-a'
    assert content_b.decode('utf-8') == 'modified-b'

    refs = store_a.load_refs()
    assert refs['branches']['exp-a']['latest_epoch'] == 1
    assert refs['branches']['exp-b']['latest_epoch'] == 1


class TestLoadStateDictRestoresPathParameterFiles:
  """BUG-007: state_dict -> load_state_dict round-trips file payloads."""

  def test_load_state_dict_restores_path_parameter_files(
    self, config: AutoPilotConfig, workspace: Path
  ) -> None:
    src_dir = workspace / 'src'
    module = SimpleModule(source=str(src_dir))

    state = module.state_dict()
    assert 'param' in state
    assert 'files' in state['param']
    assert state['param']['files']['main.py'] == 'print("original")'

    (src_dir / 'main.py').write_text('overwritten', encoding='utf-8')

    module2 = SimpleModule(source=str(src_dir))
    module2.load_state_dict(state)

    assert (src_dir / 'main.py').read_text(encoding='utf-8') == 'print("original")'
