"""End-to-end integration tests for EpochOrchestrator with full callback stack."""

from autopilot.core.callbacks.callback import Callback
from autopilot.core.callbacks.cost import CostTrackerCallback
from autopilot.core.callbacks.data_recorder import DataRecorderCallback
from autopilot.core.callbacks.memory import MemoryCallback
from autopilot.core.checkpoint import JSONCheckpoint
from autopilot.core.experiment import PromotionExperiment
from autopilot.core.logger import JSONLogger
from autopilot.core.loops.orchestrator import EpochOrchestrator, OrchestratorConfig
from autopilot.core.memory import FileMemory
from autopilot.core.metric import Metric
from autopilot.core.module import AutoPilotModule
from autopilot.core.summary import build_experiment_summary, write_experiment_summary
from autopilot.core.trainer import Trainer
from autopilot.core.types import Datum
from unittest.mock import MagicMock
import pytest


class IntegrationModule(AutoPilotModule):
  def __init__(self, accuracy_schedule=None):
    super().__init__()
    self._schedule = accuracy_schedule or [0.6, 0.7, 0.8, 0.9]
    self._epoch_idx = 0

  def forward(self, batch):
    return Datum(success=True)

  def training_step(self, batch):
    acc = self._schedule[min(self._epoch_idx, len(self._schedule) - 1)]
    return Datum(success=acc > 0.5, metrics={'accuracy': acc})

  def validation_step(self, batch):
    acc = self._schedule[min(self._epoch_idx, len(self._schedule) - 1)]
    return Datum(success=acc > 0.5, metrics={'accuracy': acc})

  def configure_optimizers(self):
    return None

  def on_train_end(self):
    self._epoch_idx += 1


class IntegrationMetric(Metric):
  higher_is_better = True

  def __init__(self):
    super().__init__()
    self.add_state('_vals', list)

  def update(self, datum):
    if isinstance(datum, Datum) and 'accuracy' in datum.metrics:
      self._vals.append(datum.metrics['accuracy'])

  def compute(self):
    if not self._vals:
      return {'accuracy': 0.0}
    return {'accuracy': sum(self._vals) / len(self._vals)}


class TestFullLoopHappyPath:
  def test_two_epochs_with_callbacks(self, tmp_path):
    module = IntegrationModule(accuracy_schedule=[0.6, 0.8])
    module.metric = IntegrationMetric()
    memory = FileMemory(tmp_path)
    cost = CostTrackerCallback(tmp_path)
    recorder = DataRecorderCallback(tmp_path)
    mem_cb = MemoryCallback(memory)

    trainer = Trainer(
      loop=EpochOrchestrator(),
      callbacks=[recorder, mem_cb, cost],
    )
    result = trainer.fit(module, train_dataloaders=[1, 2], max_epochs=2)
    assert result['total_epochs'] == 2
    assert len(memory.recall()) == 2

  def test_artifacts_produced(self, tmp_path):
    module = IntegrationModule(accuracy_schedule=[0.7, 0.8])
    module.metric = IntegrationMetric()
    recorder = DataRecorderCallback(tmp_path)
    cost = CostTrackerCallback(tmp_path)

    trainer = Trainer(
      loop=EpochOrchestrator(),
      callbacks=[recorder, cost],
    )
    trainer.fit(module, train_dataloaders=[1], max_epochs=2)
    assert (tmp_path / 'epoch_1' / 'data.jsonl').exists()
    assert (tmp_path / 'cost_summary.json').exists()


class TestRegressionRollback:
  def test_regression_triggers_rollback(self, tmp_path):
    module = IntegrationModule(accuracy_schedule=[0.8, 0.5, 0.9])
    module.metric = IntegrationMetric()
    store = MagicMock()
    config = OrchestratorConfig(auto_rollback=True)

    experiment = PromotionExperiment(
      tmp_path,
      slug='orch-e2e',
      logger=JSONLogger(tmp_path),
      checkpoint=JSONCheckpoint(),
      store=store,
      threshold_pct=0.0,
    )
    experiment.best_baseline_artifact.write(
      {'epoch': 1, 'metrics': {'accuracy': 0.8}},
      tmp_path,
    )

    trainer = Trainer(
      loop=EpochOrchestrator(config),
      experiment=experiment,
    )
    trainer.fit(
      module,
      train_dataloaders=[1],
      val_dataloaders=[1],
      max_epochs=3,
    )
    store.checkout.assert_called()


class TestPlateauDetection:
  def test_stops_on_plateau(self, tmp_path):
    module = IntegrationModule(accuracy_schedule=[0.8, 0.8, 0.8, 0.8, 0.8])
    module.metric = IntegrationMetric()
    config = OrchestratorConfig(plateau_window=3, plateau_threshold=0.05, monitor='accuracy')

    trainer = Trainer(loop=EpochOrchestrator(config))
    result = trainer.fit(module, train_dataloaders=[1], max_epochs=10)
    assert result.get('stop_reason') == 'plateau'
    assert result['total_epochs'] < 10


class TestDryRun:
  def test_no_side_effects(self, tmp_path):
    module = IntegrationModule()
    module.metric = IntegrationMetric()
    recorder = DataRecorderCallback(tmp_path)

    trainer = Trainer(
      loop=EpochOrchestrator(),
      callbacks=[recorder],
      dry_run=True,
    )
    result = trainer.fit(module, train_dataloaders=[1], max_epochs=5)
    assert result.get('dry_run') is True
    assert not (tmp_path / 'epoch_1').exists()


class TestExperimentSummary:
  def test_summary_produced(self, tmp_path):
    module = IntegrationModule(accuracy_schedule=[0.6, 0.8])
    module.metric = IntegrationMetric()
    cost = CostTrackerCallback(tmp_path)

    trainer = Trainer(
      loop=EpochOrchestrator(),
      callbacks=[cost],
    )
    result = trainer.fit(module, train_dataloaders=[1], max_epochs=2)
    summary = build_experiment_summary(tmp_path, result, cost_tracker=cost)
    path = write_experiment_summary(tmp_path, summary)
    assert path.exists()
    assert summary.total_epochs == 2


class TestCallbackOrdering:
  def test_full_lifecycle_ordering(self, tmp_path):
    module = IntegrationModule(accuracy_schedule=[0.7])
    module.metric = IntegrationMetric()
    hooks: list[str] = []

    class OrderTracker(Callback):
      def on_fit_start(self, trainer):
        hooks.append('fit_start')

      def on_train_epoch_start(self, trainer, epoch):
        hooks.append(f'train_start_{epoch}')

      def on_train_epoch_end(self, trainer, epoch):
        hooks.append(f'train_end_{epoch}')

      def on_epoch_end(self, trainer, epoch, result=None):
        hooks.append(f'epoch_end_{epoch}')

      def on_fit_end(self, trainer):
        hooks.append('fit_end')

    trainer = Trainer(
      loop=EpochOrchestrator(),
      callbacks=[OrderTracker()],
    )
    trainer.fit(module, train_dataloaders=[1], max_epochs=1)
    assert hooks[0] == 'fit_start'
    assert 'train_start_1' in hooks
    assert 'train_end_1' in hooks
    assert 'epoch_end_1' in hooks
    assert hooks[-1] == 'fit_end'


class TestNoLossNoOptimizer:
  def test_forward_runs_without_loss(self, tmp_path):
    module = IntegrationModule(accuracy_schedule=[0.7])
    module.metric = IntegrationMetric()

    trainer = Trainer(loop=EpochOrchestrator())
    result = trainer.fit(module, train_dataloaders=[1, 2], max_epochs=1)
    assert result['total_epochs'] == 1
    assert result['epochs'][0]['metrics']['accuracy'] > 0


class TestCallbackException:
  def test_raising_callback_halts_loop(self, tmp_path):
    module = IntegrationModule()
    module.metric = IntegrationMetric()

    class BrokenCallback(Callback):
      def on_train_epoch_end(self, trainer, epoch):
        raise RuntimeError('callback failure')

    trainer = Trainer(
      loop=EpochOrchestrator(),
      callbacks=[BrokenCallback()],
    )
    with pytest.raises(RuntimeError, match='callback failure'):
      trainer.fit(module, train_dataloaders=[1], max_epochs=2)
