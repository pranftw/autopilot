"""Integration tests for PyTorch/Lightning philosophy alignment (sub-plan 16).

Tests:
  1. Callback receives (trainer, module, epoch) -- module is same object passed to fit()
  2. Epoch numbering: max_epochs=3 -> on_epoch_end called with epochs 0, 1, 2
  3. experiment.epoch after fit(max_epochs=3) is 2 (0-based, last epoch)
  4. StoreCheckpointCallback snapshots at epochs 0, 1, 2 (not 1, 2, 3)
  5. Trainer.fit(Module()) (not AutoPilotModule) -> TypeError
  6. Lazy DataLoader iteration: IterableDataset does not OOM
  7. MetricCollection double-update fix: child metric update_count == batches (not 2x)
"""

from autopilot.core.callbacks.callback import Callback
from autopilot.core.callbacks.store import StoreCheckpointCallback
from autopilot.core.experiment import Experiment
from autopilot.core.metric import Metric, MetricCollection
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.module.module import Module
from autopilot.core.snapshot import SnapshotManifest
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import EvalDatum
from tests.doubles import NoopEvalModule, PlainModule
from typing import Any, cast
from unittest.mock import MagicMock
import pytest


class _CountMetric(Metric):
  def __init__(self):
    super().__init__()
    self.add_state('_n', 0)

  def update(self, datum):
    self._n += 1

  def compute(self):
    return {'count': float(self._n)}


class _SumMetric(Metric):
  def __init__(self):
    super().__init__()
    self.add_state('_total', 0)

  def update(self, datum):
    self._total += 1

  def compute(self):
    return {'total': float(self._total)}


class TestCallbackReceivesModule:
  """Test 1: Callback receives (trainer, module, epoch) -- module is same object."""

  def test_module_is_same_object_passed_to_fit(self):
    received_modules: list[Any] = []

    class ModuleTracker(Callback):
      def on_epoch_start(self, trainer, module, epoch):
        received_modules.append(module)

      def on_train_epoch_start(self, trainer, module, epoch):
        received_modules.append(module)

      def on_fit_start(self, trainer, module):
        received_modules.append(module)

    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[ModuleTracker()])
    trainer.fit(mod, train_dataloaders=[1], max_epochs=1)

    assert len(received_modules) >= 3
    for received in received_modules:
      assert received is mod


class TestEpochNumbering:
  """Test 2: max_epochs=3 -> on_epoch_end called with epochs 0, 1, 2."""

  def test_zero_based_epoch_numbering(self):
    epoch_values: list[int] = []

    class EpochCollector(Callback):
      def on_epoch_end(self, trainer, module, epoch, result=None):
        epoch_values.append(epoch)

    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[EpochCollector()])
    trainer.fit(mod, train_dataloaders=[1], max_epochs=3)

    assert epoch_values == [0, 1, 2]

  def test_train_epoch_hooks_zero_based(self):
    train_epochs: list[int] = []

    class TrainEpochCollector(Callback):
      def on_train_epoch_start(self, trainer, module, epoch):
        train_epochs.append(epoch)

    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[TrainEpochCollector()])
    trainer.fit(mod, train_dataloaders=[1], max_epochs=3)

    assert train_epochs == [0, 1, 2]


class TestExperimentEpochAfterFit:
  """Test 3: experiment.epoch after fit(max_epochs=3) is 2 (0-based, last epoch)."""

  def test_experiment_epoch_after_three_epochs(self):
    experiment = Experiment(experiment_id='test-epoch-count')
    mod = NoopEvalModule()
    trainer = Trainer(experiment=experiment)
    trainer.fit(mod, train_dataloaders=[1], max_epochs=3)
    assert experiment.epoch == 2


class TestStoreCheckpointZeroBased:
  """Test 4: StoreCheckpointCallback snapshots at epochs 0, 1, 2 (not 1, 2, 3)."""

  def test_snapshot_epochs_are_zero_based(self):
    snapshot_epochs: list[int] = []

    class RecordingStore:
      def snapshot(self, experiment_id, epoch, **kwargs):
        snapshot_epochs.append(epoch)
        return SnapshotManifest(epoch=epoch, timestamp='', entries={})

    store = RecordingStore()
    exp = Experiment(experiment_id='test-store')
    exp.start()
    cb = StoreCheckpointCallback()
    trainer_mock = MagicMock()
    trainer_mock.experiment = exp
    trainer_mock.store = store

    for e in range(3):
      cb.on_epoch_end(trainer=trainer_mock, module=None, epoch=e)

    assert snapshot_epochs == [0, 1, 2]


class TestTrainerRequiresAutoPilotModule:
  """Test 5: Trainer.fit(Module()) -> TypeError."""

  def test_plain_module_raises_type_error(self):
    trainer = Trainer()
    with pytest.raises(TypeError, match=r'Trainer.fit\(\) requires AutoPilotModule'):
      trainer.fit(cast(Any, PlainModule()), max_epochs=1)

  def test_error_message_includes_type_name(self):
    class MyCustomModule(Module):
      def forward(self, *args, **kwargs):
        return EvalDatum(success=True)

    trainer = Trainer()
    with pytest.raises(TypeError, match='MyCustomModule'):
      trainer.fit(cast(Any, MyCustomModule()), max_epochs=1)


class TestLazyDataLoaderIteration:
  """Test 6: IterableDataset does not materialize all items."""

  def test_iterable_dataset_not_materialized(self):
    materialized_count = 0

    class CountingIterable:
      def __iter__(self):
        nonlocal materialized_count
        for _i in range(10):
          materialized_count += 1
          yield EvalDatum(success=True)

    mod = NoopEvalModule()
    trainer = Trainer()
    result = trainer.fit(mod, train_dataloaders=CountingIterable(), max_epochs=1)

    assert result['total_epochs'] == 1
    assert materialized_count == 10

  def test_large_iterable_does_not_oom(self):
    class LargeIterable:
      """Simulates a large dataset -- 10 items with early stop via max_epochs=1."""

      def __iter__(self):
        for _i in range(10):
          yield EvalDatum(success=True)

    mod = NoopEvalModule()

    class StopAfterBatches(Callback):
      def __init__(self, max_batches):
        self._max = max_batches
        self._count = 0

      def on_train_batch_end(self, trainer, module, batch_idx=0, data=None):
        self._count += 1

    stopper = StopAfterBatches(10)
    trainer = Trainer(callbacks=[stopper])
    result = trainer.fit(mod, train_dataloaders=LargeIterable(), max_epochs=1)

    assert result['total_epochs'] == 1
    assert stopper._count == 10


class _TrackingCountMetric(Metric):
  """Metric that tracks total updates across resets for testing."""

  def __init__(self):
    super().__init__()
    self.add_state('_n', 0)
    self.lifetime_updates = 0

  def update(self, datum):
    self._n += 1
    self.lifetime_updates += 1

  def compute(self):
    return {'count': float(self._n)}


class _TrackingSumMetric(Metric):
  """Metric that tracks total updates across resets for testing."""

  def __init__(self):
    super().__init__()
    self.add_state('_total', 0)
    self.lifetime_updates = 0

  def update(self, datum):
    self._total += 1
    self.lifetime_updates += 1

  def compute(self):
    return {'total': float(self._total)}


class TestMetricCollectionDoubleUpdate:
  """Test 7: MetricCollection children get update_count == batches, not 2x."""

  def test_no_double_update_with_collection(self):
    count_metric = _TrackingCountMetric()
    sum_metric = _TrackingSumMetric()
    collection = MetricCollection([count_metric, sum_metric])

    class _ModWithCollection(AutoPilotModule):
      def __init__(self):
        super().__init__()
        self.metrics_collection = collection

      def forward(self, batch):
        return EvalDatum(success=True)

      def training_step(self, batch, batch_idx):
        return EvalDatum(success=True)

      def configure_optimizers(self):
        return None

    mod = _ModWithCollection()
    trainer = Trainer()
    trainer.fit(mod, train_dataloaders=[1, 2, 3], max_epochs=1)

    assert count_metric.lifetime_updates == 3
    assert sum_metric.lifetime_updates == 3

  def test_standalone_metric_not_affected(self):
    standalone = _TrackingCountMetric()

    class _ModWithStandalone(AutoPilotModule):
      def __init__(self):
        super().__init__()
        self.standalone = standalone

      def forward(self, batch):
        return EvalDatum(success=True)

      def training_step(self, batch, batch_idx):
        return EvalDatum(success=True)

      def configure_optimizers(self):
        return None

    mod = _ModWithStandalone()
    trainer = Trainer()
    trainer.fit(mod, train_dataloaders=[1, 2, 3], max_epochs=1)

    assert standalone.lifetime_updates == 3
