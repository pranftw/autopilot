"""Tests for core/environment.py: Environment, LocalEnvironment, activate."""

from autopilot.core.environment import Environment, LocalEnvironment, WorktreeStore
from autopilot.core.experiment import Experiment
from autopilot.core.module.module import Module
from autopilot.core.types import Datum
from pathlib import Path
from typing import Any
import pytest


class _ConcreteModule(Module):
  """Minimal concrete Module for test fixtures."""

  def forward(self, *args: Any, **kwargs: Any) -> Datum:
    raise NotImplementedError


class _CountingEnvironment(Environment):
  """Stub that counts teardown calls and returns a fixed path from setup."""

  def __init__(self, setup_path: Path) -> None:
    self.setup_path = setup_path
    self.teardown_calls = 0

  def setup(self, experiment: Experiment, store: WorktreeStore, module: Module) -> Path:
    return self.setup_path

  def teardown(self, experiment: Experiment) -> None:
    self.teardown_calls += 1


class _FailingSetupEnvironment(Environment):
  """Stub whose setup raises RuntimeError."""

  def __init__(self) -> None:
    self.teardown_calls = 0

  def setup(self, experiment: Experiment, store: WorktreeStore, module: Module) -> Path:
    msg = 'setup failed'
    raise RuntimeError(msg)

  def teardown(self, experiment: Experiment) -> None:
    self.teardown_calls += 1


class TestEnvironmentSetupNotImplemented:
  def test_environment_cannot_be_instantiated(self) -> None:
    with pytest.raises(TypeError, match='abstract'):
      Environment()


class TestLocalEnvironmentSetup:
  def test_local_environment_setup_returns_cwd(self) -> None:
    env = LocalEnvironment()
    exp = Experiment('e')
    store: Any = object()
    module = _ConcreteModule()
    assert env.setup(exp, store, module) == Path.cwd()


class TestLocalEnvironmentTeardown:
  def test_local_environment_teardown_is_noop(self) -> None:
    env = LocalEnvironment()
    exp = Experiment('e')
    env.teardown(exp)


class TestActivateYieldsSetupPath:
  def test_activate_yields_setup_path(self, tmp_path: Path) -> None:
    expected_dir = tmp_path / 'workdir'
    expected_dir.mkdir()
    env = _CountingEnvironment(setup_path=expected_dir)
    exp = Experiment('e')
    store: Any = object()
    module = _ConcreteModule()
    with env.activate(exp, store, module) as path:
      assert path == expected_dir
      assert path.is_dir()


class TestActivateTeardownOnSuccess:
  def test_activate_teardown_on_success(self, tmp_path: Path) -> None:
    env = _CountingEnvironment(setup_path=tmp_path)
    exp = Experiment('e')
    store: Any = object()
    module = _ConcreteModule()
    with env.activate(exp, store, module):
      pass
    assert env.teardown_calls == 1


class TestActivateTeardownOnException:
  def test_activate_teardown_on_exception(self, tmp_path: Path) -> None:
    env = _CountingEnvironment(setup_path=tmp_path)
    exp = Experiment('e')
    store: Any = object()
    module = _ConcreteModule()
    exc = RuntimeError('x')
    with pytest.raises(RuntimeError, match='x'), env.activate(exp, store, module):
      raise exc
    assert env.teardown_calls == 1


class TestLocalEnvironmentAcceptsFileStore:
  def test_local_environment_accepts_file_store(self, tmp_path: Path) -> None:
    from autopilot.ai.store.file_store import FileStore
    from autopilot.core.config import AutoPilotConfig

    config = AutoPilotConfig(workspace=tmp_path)
    file_store = FileStore(config)
    exp = Experiment('e')
    module = _ConcreteModule()
    assert LocalEnvironment().setup(exp, file_store, module) == Path.cwd()


class TestActivateTeardownRunsWhenSetupRaises:
  def test_activate_teardown_runs_when_setup_raises(self) -> None:
    env = _FailingSetupEnvironment()
    exp = Experiment('e')
    store: Any = object()
    module = _ConcreteModule()
    with pytest.raises(RuntimeError, match='setup failed'), env.activate(exp, store, module):
      pass
    assert env.teardown_calls == 1


class TestLocalEnvironmentActivateEndToEnd:
  def test_local_environment_activate_end_to_end(self) -> None:
    """activate context manager yields cwd and tears down cleanly."""
    env = LocalEnvironment()
    exp = Experiment('e')
    store: Any = object()
    module = _ConcreteModule()
    with env.activate(exp, store, module) as path:
      assert path == Path.cwd()
      assert path.is_dir()
