"""Tests for EpochOrchestrator."""

from autopilot.core.errors import OrchestratorError, StoreError
from autopilot.core.experiment import Experiment
from autopilot.core.loops.orchestrator import EpochOrchestrator, OrchestratorConfig
from autopilot.core.metric import Metric
from autopilot.core.models import Result
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import EvalDatum, GateResult
from autopilot.policy.policy import Policy
from typing import Any, cast
from unittest.mock import MagicMock
import pytest


class DummyModule(AutoPilotModule):
  def __init__(self):
    super().__init__()
    self._accuracy = 0.5
    self._step = 0

  def forward(self, batch):
    return EvalDatum(success=True)

  def training_step(self, batch, batch_idx):
    self._step += 1
    return EvalDatum(success=True, metrics={'accuracy': self._accuracy})

  def validation_step(self, batch, batch_idx):
    return EvalDatum(success=True, metrics={'accuracy': self._accuracy})

  def configure_optimizers(self):
    return None


class DummyMetric(Metric):
  def __init__(self):
    super().__init__()
    self.add_state('_correct', 0)
    self.add_state('_total', 0)

  def update(self, datum):
    self._total += 1
    if datum.success:
      self._correct += 1

  def compute(self):
    acc = self._correct / self._total if self._total else 0.0
    return {'accuracy': acc}


class MockExperiment(Experiment):
  def __init__(self, store=None):
    super().__init__(experiment_id='mock-orch')
    self.store = store
    self.should_rollback = False
    self.last_accepted_epoch = -1
    self.rollback_calls: list[int | None] = []

  def rollback(self, epoch: int | None) -> None:
    self.rollback_calls.append(epoch)
    if self.store:
      self.store.checkout(epoch)

  def on_epoch_complete(self, epoch: int, metrics: dict[str, float], **kwargs: Any) -> None:
    pass

  def on_validation_complete(self, epoch: int, metrics: dict[str, float], **kwargs: Any) -> None:
    pass


class TestEpochOrchestrator:
  def test_basic_loop_one_epoch(self):
    module = DummyModule()
    module.accuracy = DummyMetric()
    trainer = Trainer(loop=EpochOrchestrator())
    result = trainer.fit(module, train_dataloaders=[1], max_epochs=1)
    assert result['total_epochs'] == 1

  def test_basic_loop_three_epochs(self):
    module = DummyModule()
    module.accuracy = DummyMetric()
    trainer = Trainer(loop=EpochOrchestrator())
    result = trainer.fit(module, train_dataloaders=[1, 2, 3], max_epochs=3)
    assert result['total_epochs'] == 3

  def test_plateau_detection_stops(self):
    module = DummyModule()
    module.accuracy = DummyMetric()
    config = OrchestratorConfig(plateau_window=3, plateau_threshold=0.05, monitor='accuracy')
    trainer = Trainer(loop=EpochOrchestrator(config))
    result = trainer.fit(module, train_dataloaders=[1], max_epochs=10)
    assert result.get('stop_reason') == 'plateau'
    assert result['total_epochs'] <= 10

  def test_orchestrator_config_monitor(self):
    config = OrchestratorConfig(monitor='accuracy')
    assert config.monitor == 'accuracy'

  def test_plateau_no_monitor_raises_config_error(self):
    from autopilot.core.errors import ConfigError

    with pytest.raises(ConfigError, match='monitor is required'):
      OrchestratorConfig(plateau_window=3, plateau_threshold=0.05)

  def test_should_rollback_triggers_checkout(self):
    module = DummyModule()
    module.accuracy = DummyMetric()
    store = MagicMock()
    experiment = MockExperiment(store=store)
    config = OrchestratorConfig(auto_rollback=True, plateau_window=0)
    orch = EpochOrchestrator(config)

    class RegCallback:
      def on_epoch_end(self, trainer, module, epoch, result=None):
        if epoch == 1:
          trainer.experiment.should_rollback = True

    trainer = Trainer(
      loop=orch,
      experiment=experiment,
      callbacks=cast(Any, [RegCallback()]),
    )
    trainer.fit(module, train_dataloaders=[1], max_epochs=3)
    assert experiment.rollback_calls == [0]
    store.checkout.assert_called_with(0)

  def test_rollback_targets_last_good_epoch(self):
    module = DummyModule()
    module.accuracy = DummyMetric()
    store = MagicMock()
    experiment = MockExperiment(store=store)
    config = OrchestratorConfig(auto_rollback=True, plateau_window=0)
    orch = EpochOrchestrator(config)

    class RegCallback:
      def on_epoch_end(self, trainer, module, epoch, result=None):
        if epoch == 2:
          trainer.experiment.should_rollback = True

    trainer = Trainer(loop=orch, experiment=experiment, callbacks=cast(Any, [RegCallback()]))
    trainer.fit(module, train_dataloaders=[1], max_epochs=4)
    assert experiment.rollback_calls == [1]
    store.checkout.assert_called_with(1)

  def test_store_checkout_failure_raises(self):
    module = DummyModule()
    module.accuracy = DummyMetric()
    store = MagicMock()
    config = OrchestratorConfig(auto_rollback=True, plateau_window=0)
    orch = EpochOrchestrator(config)

    class RaisingExperiment(MockExperiment):
      def __init__(self, store):
        super().__init__(store=store)

      def rollback(self, epoch: int | None) -> None:
        msg = 'disk full'
        raise StoreError(msg)

    experiment = RaisingExperiment(store)

    class RegCallback:
      def on_epoch_end(self, trainer, module, epoch, result=None):
        if epoch == 1:
          trainer.experiment.should_rollback = True

    trainer = Trainer(loop=orch, experiment=experiment, callbacks=cast(Any, [RegCallback()]))
    with pytest.raises(OrchestratorError):
      trainer.fit(module, train_dataloaders=[1], max_epochs=3)

  def test_dry_run_no_side_effects(self):
    module = DummyModule()
    module.accuracy = DummyMetric()
    trainer = Trainer(loop=EpochOrchestrator(), dry_run=True)
    result = trainer.fit(module, train_dataloaders=[1], max_epochs=5)
    assert result.get('dry_run') is True
    assert result['total_epochs'] == 0

  def test_max_epochs_zero(self):
    module = DummyModule()
    module.accuracy = DummyMetric()
    trainer = Trainer(loop=EpochOrchestrator())
    result = trainer.fit(module, train_dataloaders=[1], max_epochs=0)
    assert result['total_epochs'] == 0

  def test_inherits_run_epoch(self):
    from autopilot.core.loops.epoch import EpochLoop

    orch = EpochOrchestrator()
    assert orch._run_epoch == EpochLoop._run_epoch.__get__(orch, type(orch))

  def test_should_rollback_false_at_train_epoch_starts_when_unset(self):
    module = DummyModule()
    module.accuracy = DummyMetric()
    flags = []
    experiment = MockExperiment()

    class FlagChecker:
      def on_train_epoch_start(self, trainer, module, epoch):
        flags.append(trainer.experiment.should_rollback)

    trainer = Trainer(
      loop=EpochOrchestrator(), experiment=experiment, callbacks=cast(Any, [FlagChecker()])
    )
    trainer.fit(module, train_dataloaders=[1], max_epochs=3)
    assert all(f is False for f in flags)

  def test_result_structure(self):
    module = DummyModule()
    module.accuracy = DummyMetric()
    trainer = Trainer(loop=EpochOrchestrator())
    result = trainer.fit(module, train_dataloaders=[1], max_epochs=2)
    assert 'epochs' in result
    assert 'total_epochs' in result
    assert 'last_good_epoch' in result

  def test_auto_rollback_false_skips_checkout(self):
    module = DummyModule()
    module.accuracy = DummyMetric()
    store = MagicMock()
    experiment = MockExperiment(store=store)
    config = OrchestratorConfig(auto_rollback=False, plateau_window=0)
    orch = EpochOrchestrator(config)

    class RegCallback:
      def on_epoch_end(self, trainer, module, epoch, result=None):
        if epoch == 1:
          trainer.experiment.should_rollback = True

    trainer = Trainer(loop=orch, experiment=experiment, callbacks=cast(Any, [RegCallback()]))
    trainer.fit(module, train_dataloaders=[1], max_epochs=3)
    assert experiment.rollback_calls == []
    store.checkout.assert_not_called()

  def test_policy_fail_stops_loop(self):
    class FailPolicy(Policy):
      def forward(self, result: Result) -> GateResult:
        return GateResult.FAIL

    module = DummyModule()
    module.accuracy = DummyMetric()
    store = MagicMock()
    experiment = MockExperiment(store=store)
    trainer = Trainer(
      loop=EpochOrchestrator(),
      policy=FailPolicy(),
      experiment=experiment,
    )
    result = trainer.fit(module, train_dataloaders=[1], max_epochs=5)
    assert result['total_epochs'] == 1
    assert result.get('stop_reason') == 'policy_fail'
    assert experiment.rollback_calls == [-1]
    store.checkout.assert_called_with(-1)


class TestOrchestratorStateReset:
  """Bug 47: EpochOrchestrator retains state across fit() calls."""

  def test_consecutive_fit_calls_have_clean_state(self):
    """Two consecutive fit() calls on same instance -- second run has clean state."""
    module = DummyModule()
    module.accuracy = DummyMetric()
    orch = EpochOrchestrator()
    trainer = Trainer(loop=orch)

    result1 = trainer.fit(module, train_dataloaders=[1, 2], max_epochs=3)
    assert result1['total_epochs'] == 3
    assert result1['last_good_epoch'] == 2

    result2 = trainer.fit(module, train_dataloaders=[1], max_epochs=2)
    assert result2['total_epochs'] == 2
    assert result2['last_good_epoch'] == 1

  def test_plateau_detection_resets_between_runs(self):
    """After first run() populates _metric_history, second run() starts empty."""
    module = DummyModule()
    module.accuracy = DummyMetric()
    config = OrchestratorConfig(
      plateau_window=3,
      plateau_threshold=0.05,
      monitor='accuracy',
    )
    orch = EpochOrchestrator(config)

    history_at_start: list[int] = []

    class HistoryProbe:
      def on_train_epoch_start(self, trainer, module, epoch):
        if epoch == 0:
          history_at_start.append(len(orch._metric_history))

    trainer = Trainer(loop=orch, callbacks=cast(Any, [HistoryProbe()]))

    trainer.fit(module, train_dataloaders=[1], max_epochs=5)
    assert history_at_start[0] == 0

    trainer.fit(module, train_dataloaders=[1], max_epochs=5)
    assert len(history_at_start) == 2
    assert history_at_start[1] == 0

  def test_last_good_epoch_resets_between_runs(self):
    """_last_good_epoch doesn't carry over from first run to second run."""
    module = DummyModule()
    module.accuracy = DummyMetric()
    orch = EpochOrchestrator()

    last_good_at_start: list[int] = []

    class EpochProbe:
      def on_train_epoch_start(self, trainer, module, epoch):
        if epoch == 0:
          last_good_at_start.append(orch._last_good_epoch)

    trainer = Trainer(loop=orch, callbacks=cast(Any, [EpochProbe()]))

    result1 = trainer.fit(module, train_dataloaders=[1], max_epochs=3)
    assert result1['last_good_epoch'] == 2
    assert last_good_at_start[0] == -1

    result2 = trainer.fit(module, train_dataloaders=[1], max_epochs=2)
    assert len(last_good_at_start) == 2
    assert last_good_at_start[1] == -1
    assert result2['last_good_epoch'] == 1
