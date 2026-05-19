"""Tests for Trainer-forest reconciliation (Plan 02).

Covers:
- FR-014: forest.save() called from fit() finally block
- BUG-TRAINER-CONFIG: dynamic environment class name in ConfigError
"""

from autopilot.core.environment import Environment
from autopilot.core.errors import ConfigError
from autopilot.core.experiment import Experiment
from autopilot.core.forest import Forest
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.tree import Tree
from autopilot.core.types import EvalDatum
from autopilot.data.dataloader import DataLoader
from contextlib import contextmanager
from pathlib import Path
from tests.doubles import NoopEvalModule
from unittest.mock import MagicMock, patch
import pytest


class _DummyEnv(Environment):
  """Trivial environment stub for testing error message class name."""

  def setup(self, experiment, store, module) -> Path:
    return Path.cwd()

  @contextmanager
  def activate(self, experiment, store, module):
    yield Path.cwd()


class _FailingTrainingModule(NoopEvalModule):
  """Module whose training_step raises to test failure path."""

  def training_step(self, batch, batch_idx) -> EvalDatum:
    msg = 'intentional training failure'
    raise RuntimeError(msg)


class TestEnvRequiresExperimentMessage:
  """BUG-TRAINER-CONFIG: error message should use actual environment class name."""

  def test_trainer_fit_env_requires_experiment_uses_actual_class_name(self) -> None:
    """ConfigError message must contain the concrete environment type name,
    not a hardcoded 'IsolatedEnvironment' string."""
    from autopilot.core.config import Config

    config = MagicMock(spec=Config)
    config.environment = _DummyEnv()
    trainer = Trainer(config=config)
    mod = NoopEvalModule()
    with pytest.raises(ConfigError, match='_DummyEnv') as exc_info:
      trainer.fit(mod, max_epochs=1)
    assert '_DummyEnv requires an experiment' in str(exc_info.value)

  def test_error_message_does_not_contain_hardcoded_isolated(self) -> None:
    """Verify the old hardcoded 'IsolatedEnvironment' string is gone."""
    from autopilot.core.config import Config

    config = MagicMock(spec=Config)
    config.environment = _DummyEnv()
    trainer = Trainer(config=config)
    mod = NoopEvalModule()
    with pytest.raises(ConfigError) as exc_info:
      trainer.fit(mod, max_epochs=1)
    assert 'IsolatedEnvironment' not in str(exc_info.value)


class TestForestSaveOnFit:
  """FR-014: Trainer.fit() should persist forest updates."""

  def test_trainer_fit_saves_forest_when_trainer_has_forest(self) -> None:
    """forest.save() must be called once after successful fit."""
    mock_forest = MagicMock(spec=Forest)
    exp = Experiment(experiment_id='exp-1', hypothesis='test')
    tree = MagicMock(spec=Tree)
    trainer = Trainer(forest=mock_forest, tree=tree, experiment=exp)
    mod = NoopEvalModule()
    dl = DataLoader([EvalDatum(success=True)], batch_size=1)
    trainer.fit(mod, train_dataloaders=dl, max_epochs=1)
    mock_forest.save.assert_called_once()

  def test_trainer_fit_no_crash_when_forest_none(self) -> None:
    """fit() must work when forest=None (no AttributeError)."""
    trainer = Trainer(forest=None)
    mod = NoopEvalModule()
    dl = DataLoader([EvalDatum(success=True)], batch_size=1)
    result = trainer.fit(mod, train_dataloaders=dl, max_epochs=1)
    assert result['total_epochs'] == 1

  def test_trainer_fit_saves_on_failure_path_when_forest_present(self) -> None:
    """forest.save() must be called even when fit() raises."""
    mock_forest = MagicMock(spec=Forest)
    exp = Experiment(experiment_id='exp-fail', hypothesis='test')
    tree = MagicMock(spec=Tree)
    trainer = Trainer(forest=mock_forest, tree=tree, experiment=exp)
    mod = _FailingTrainingModule()
    dl = DataLoader([EvalDatum(success=True)], batch_size=1)
    with pytest.raises(RuntimeError, match='intentional training failure'):
      trainer.fit(mod, train_dataloaders=dl, max_epochs=1)
    mock_forest.save.assert_called_once()

  def test_forest_save_error_propagates(self) -> None:
    """If forest.save() raises, the error must propagate (no catch)."""
    mock_forest = MagicMock(spec=Forest)
    mock_forest.save.side_effect = OSError('disk full')
    exp = Experiment(experiment_id='exp-2', hypothesis='test')
    tree = MagicMock(spec=Tree)
    trainer = Trainer(forest=mock_forest, tree=tree, experiment=exp)
    mod = NoopEvalModule()
    dl = DataLoader([EvalDatum(success=True)], batch_size=1)
    with pytest.raises(OSError, match='disk full'):
      trainer.fit(mod, train_dataloaders=dl, max_epochs=1)

  def test_forest_save_after_teardown(self) -> None:
    """forest.save() should run after _teardown_fit (ordering check)."""
    call_order: list[str] = []
    mock_forest = MagicMock(spec=Forest)
    mock_forest.save.side_effect = lambda: call_order.append('forest_save')

    exp = Experiment(experiment_id='exp-order', hypothesis='test')
    tree = MagicMock(spec=Tree)
    trainer = Trainer(forest=mock_forest, tree=tree, experiment=exp)

    original_teardown = trainer._teardown_fit

    def tracking_teardown(module, datamodule):
      original_teardown(module, datamodule)
      call_order.append('teardown')

    mod = NoopEvalModule()
    dl = DataLoader([EvalDatum(success=True)], batch_size=1)
    with patch.object(trainer, '_teardown_fit', side_effect=tracking_teardown):
      trainer.fit(mod, train_dataloaders=dl, max_epochs=1)
    assert call_order == ['teardown', 'forest_save']


class TestForestGracefulWithoutExperiment:
  """forest.save() still runs even without an experiment context manager."""

  def test_forest_save_without_experiment(self) -> None:
    """forest.save() runs when trainer has forest but no experiment."""
    mock_forest = MagicMock(spec=Forest)
    trainer = Trainer(forest=mock_forest)
    mod = NoopEvalModule()
    dl = DataLoader([EvalDatum(success=True)], batch_size=1)
    trainer.fit(mod, train_dataloaders=dl, max_epochs=1)
    mock_forest.save.assert_called_once()


class TestDryRunLifecycle:
  """Dry-run does not contradict experiment lifecycle (plan 02 §2.2)."""

  def test_dry_run_with_experiment_completes_normally(self) -> None:
    """dry_run=True still runs the loop; experiment completes on success."""
    exp = Experiment(experiment_id='dry-run-exp', hypothesis='test')
    trainer = Trainer(experiment=exp, dry_run=True)
    mod = NoopEvalModule()
    dl = DataLoader([EvalDatum(success=True)], batch_size=1)
    trainer.fit(mod, train_dataloaders=dl, max_epochs=1)
    assert exp.status.value == 'completed'

  def test_dry_run_with_forest_saves_forest(self) -> None:
    """dry_run=True with forest still persists forest on success."""
    mock_forest = MagicMock(spec=Forest)
    exp = Experiment(experiment_id='dry-forest', hypothesis='test')
    tree = MagicMock(spec=Tree)
    trainer = Trainer(forest=mock_forest, tree=tree, experiment=exp, dry_run=True)
    mod = NoopEvalModule()
    dl = DataLoader([EvalDatum(success=True)], batch_size=1)
    trainer.fit(mod, train_dataloaders=dl, max_epochs=1)
    mock_forest.save.assert_called_once()
