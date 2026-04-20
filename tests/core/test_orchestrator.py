"""Tests for EpochOrchestrator."""

from autopilot.core.errors import OrchestratorError
from autopilot.core.metric import Metric
from autopilot.core.models import Datum, GateResult, Result
from autopilot.core.module import AutoPilotModule
from autopilot.core.orchestrator import EpochOrchestrator, OrchestratorConfig
from autopilot.core.trainer import Trainer
from autopilot.policy.policy import Policy
from unittest.mock import MagicMock
import pytest


class DummyModule(AutoPilotModule):
  def __init__(self):
    super().__init__()
    self._accuracy = 0.5
    self._step = 0

  def forward(self, batch):
    return Datum(success=True)

  def training_step(self, batch):
    self._step += 1
    return Datum(success=True, metrics={'accuracy': self._accuracy})

  def validation_step(self, batch):
    return Datum(success=True, metrics={'accuracy': self._accuracy})

  def configure_optimizers(self):
    return None


class DummyMetric(Metric):
  def __init__(self):
    super().__init__()
    self._correct = 0
    self._total = 0

  def update(self, datum):
    self._total += 1
    if datum.success:
      self._correct += 1

  def compute(self):
    acc = self._correct / self._total if self._total else 0.0
    return {'accuracy': acc}

  def reset(self):
    self._correct = 0
    self._total = 0


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
    config = OrchestratorConfig(plateau_window=3, plateau_threshold=0.05)
    trainer = Trainer(loop=EpochOrchestrator(config))
    result = trainer.fit(module, train_dataloaders=[1], max_epochs=10)
    assert result.get('stop_reason') == 'plateau'
    assert result['total_epochs'] <= 10

  def test_regression_rollback(self):
    module = DummyModule()
    module.accuracy = DummyMetric()
    store = MagicMock()
    config = OrchestratorConfig(auto_rollback=True)
    orch = EpochOrchestrator(config)

    class RegCallback:
      def on_epoch_end(self, trainer, epoch, result=None):
        if epoch == 2:
          trainer.regression_detected = True

    trainer = Trainer(
      loop=orch,
      store=store,
      callbacks=[RegCallback()],
    )
    trainer.fit(module, train_dataloaders=[1], max_epochs=3)
    store.checkout.assert_called()

  def test_rollback_targets_last_good_epoch(self):
    module = DummyModule()
    module.accuracy = DummyMetric()
    store = MagicMock()
    config = OrchestratorConfig(auto_rollback=True)
    orch = EpochOrchestrator(config)

    class RegCallback:
      def on_epoch_end(self, trainer, epoch, result=None):
        if epoch == 3:
          trainer.regression_detected = True

    trainer = Trainer(loop=orch, store=store, callbacks=[RegCallback()])
    trainer.fit(module, train_dataloaders=[1], max_epochs=4)
    store.checkout.assert_called_with(2)

  def test_store_checkout_failure_raises(self):
    module = DummyModule()
    module.accuracy = DummyMetric()
    store = MagicMock()
    store.checkout.side_effect = RuntimeError('disk full')
    config = OrchestratorConfig(auto_rollback=True)
    orch = EpochOrchestrator(config)

    class RegCallback:
      def on_epoch_end(self, trainer, epoch, result=None):
        if epoch == 2:
          trainer.regression_detected = True

    trainer = Trainer(loop=orch, store=store, callbacks=[RegCallback()])
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
    from autopilot.core.loops import EpochLoop

    orch = EpochOrchestrator()
    assert orch._run_epoch == EpochLoop._run_epoch.__get__(orch, type(orch))

  def test_regression_detected_reset_each_epoch(self):
    module = DummyModule()
    module.accuracy = DummyMetric()
    flags = []

    class FlagChecker:
      def on_train_epoch_start(self, trainer, epoch):
        flags.append(trainer.regression_detected)

    trainer = Trainer(loop=EpochOrchestrator(), callbacks=[FlagChecker()])
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
    config = OrchestratorConfig(auto_rollback=False)
    orch = EpochOrchestrator(config)

    class RegCallback:
      def on_epoch_end(self, trainer, epoch, result=None):
        if epoch == 2:
          trainer.regression_detected = True

    trainer = Trainer(loop=orch, store=store, callbacks=[RegCallback()])
    trainer.fit(module, train_dataloaders=[1], max_epochs=3)
    store.checkout.assert_not_called()

  def test_policy_fail_stops_loop(self):
    class FailPolicy(Policy):
      def forward(self, result: Result) -> GateResult:
        return GateResult.FAIL

    module = DummyModule()
    module.accuracy = DummyMetric()
    trainer = Trainer(loop=EpochOrchestrator(), policy=FailPolicy())
    result = trainer.fit(module, train_dataloaders=[1], max_epochs=5)
    assert result['total_epochs'] == 1
    assert result.get('stop_reason') == 'policy_fail'
