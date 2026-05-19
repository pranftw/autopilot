"""Integration tests for Trainer + FileStore + StoreCheckpointCallback.

Verifies:
  - Trainer.fit() with FileStore + StoreCheckpointCallback creates snapshots at each epoch
  - Epoch numbers in snapshots match loop epochs
  - store.log shows correct history
  - StoreCheckpointCallback reads trainer.store and snapshots at each epoch
  - Trainer(store=None) with StoreCheckpointCallback skips silently
"""

from autopilot.ai.experiment import AutoPilotExperiment
from autopilot.ai.store.file_store import FileStore
from autopilot.core.callbacks.store import StoreCheckpointCallback
from autopilot.core.config import AutoPilotConfig
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.parameter import Parameter
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import EvalDatum
from autopilot.data.dataloader import DataLoader
from autopilot.data.datamodule import DataModule
from pathlib import Path
from tests.doubles import DirectNumericLoss, NoOpOptimizer


class _StubModule(AutoPilotModule):
  def __init__(self):
    super().__init__()
    self.param = Parameter(requires_grad=True)
    self.loss = DirectNumericLoss([self.param])
    self._opt = NoOpOptimizer([self.param])

  def forward(self, batch):
    return batch

  def training_step(self, batch, batch_idx):
    return batch

  def validation_step(self, batch, batch_idx):
    return batch

  def configure_optimizers(self):
    return self._opt


class _StubDataModule(DataModule):
  def train_dataloader(self) -> DataLoader:
    return DataLoader([EvalDatum(metadata={'i': i}) for i in range(2)], batch_size=1)

  def val_dataloader(self) -> DataLoader:
    return DataLoader([], batch_size=1)


def _setup(tmp_path: Path, exp_id: str = 'store-trainer-exp'):
  config = AutoPilotConfig(workspace=tmp_path)
  param = Parameter(requires_grad=True)
  store = FileStore(config)
  store.register_parameters({'source': param})
  experiment = AutoPilotExperiment(experiment_id=exp_id)
  cb = StoreCheckpointCallback()
  return config, store, experiment, cb


class TestTrainerFitWithFileStoreAndCallback:
  def test_snapshots_at_each_epoch(self, tmp_path: Path) -> None:
    """Trainer.fit with FileStore + StoreCheckpointCallback creates snapshots at each epoch."""
    config, store, experiment, cb = _setup(tmp_path)
    module = _StubModule()
    trainer = Trainer(
      callbacks=[cb],
      experiment=experiment,
      store=store,
      config=config,
    )
    trainer.fit(module, datamodule=_StubDataModule(), max_epochs=4)

    log = store.log(experiment.id)
    epochs_in_log = [entry.epoch for entry in log]
    assert epochs_in_log == [0, 1, 2, 3]

  def test_epoch_numbers_match_loop_epochs(self, tmp_path: Path) -> None:
    """Epoch numbers in snapshots match the 0-based loop epochs."""
    config, store, experiment, cb = _setup(tmp_path)
    module = _StubModule()
    trainer = Trainer(
      callbacks=[cb],
      experiment=experiment,
      store=store,
      config=config,
    )
    trainer.fit(module, datamodule=_StubDataModule(), max_epochs=3)

    log = store.log(experiment.id)
    for i, entry in enumerate(log):
      assert entry.epoch == i, f'Expected epoch {i}, got {entry.epoch}'

  def test_store_log_correct_history(self, tmp_path: Path) -> None:
    """store.log shows correct history after training."""
    config, store, experiment, cb = _setup(tmp_path)
    module = _StubModule()
    trainer = Trainer(
      callbacks=[cb],
      experiment=experiment,
      store=store,
      config=config,
    )
    trainer.fit(module, datamodule=_StubDataModule(), max_epochs=5)

    log = store.log(experiment.id)
    assert len(log) == 5
    for entry in log:
      assert entry.file_count >= 0
      assert entry.timestamp is not None

  def test_callback_reads_trainer_store(self, tmp_path: Path) -> None:
    """StoreCheckpointCallback reads trainer.store and snapshots at each epoch."""
    config, store, experiment, cb = _setup(tmp_path)
    module = _StubModule()
    trainer = Trainer(
      callbacks=[cb],
      experiment=experiment,
      store=store,
      config=config,
    )

    assert trainer.store is store

    trainer.fit(module, datamodule=_StubDataModule(), max_epochs=2)

    assert cb._last_epoch == 1
    log = store.log(experiment.id)
    assert len(log) == 2

  def test_single_epoch(self, tmp_path: Path) -> None:
    """Single-epoch training creates exactly one snapshot."""
    config, store, experiment, cb = _setup(tmp_path)
    module = _StubModule()
    trainer = Trainer(
      callbacks=[cb],
      experiment=experiment,
      store=store,
      config=config,
    )
    trainer.fit(module, datamodule=_StubDataModule(), max_epochs=1)

    log = store.log(experiment.id)
    assert len(log) == 1
    assert log[0].epoch == 0
    assert cb._last_epoch == 0


class TestTrainerNoStore:
  def test_no_store_callback_skips_silently(self, tmp_path: Path) -> None:
    """Trainer(store=None) with StoreCheckpointCallback skips silently (no crash)."""
    experiment = AutoPilotExperiment(experiment_id='no-store-exp')
    cb = StoreCheckpointCallback()
    module = _StubModule()

    trainer = Trainer(
      callbacks=[cb],
      experiment=experiment,
      store=None,
    )

    assert trainer.store is None

    result = trainer.fit(module, datamodule=_StubDataModule(), max_epochs=3)
    assert result['total_epochs'] == 3
    assert cb._last_epoch is None

  def test_no_store_no_experiment(self, tmp_path: Path) -> None:
    """Trainer with no store and no experiment completes without error."""
    cb = StoreCheckpointCallback()
    module = _StubModule()

    trainer = Trainer(
      callbacks=[cb],
      store=None,
    )

    result = trainer.fit(module, datamodule=_StubDataModule(), max_epochs=2)
    assert result['total_epochs'] == 2
    assert cb._last_epoch is None

  def test_store_property_is_none_by_default(self) -> None:
    """Trainer without store= has store property as None."""
    trainer = Trainer()
    assert trainer.store is None


class TestTrainerStoreProperty:
  def test_trainer_store_is_public_property(self, tmp_path: Path) -> None:
    """trainer.store is a public property, not a private attribute."""
    _config, store, _experiment, _cb = _setup(tmp_path)
    trainer = Trainer(store=store)
    assert trainer.store is store

  def test_trainer_store_kwarg(self, tmp_path: Path) -> None:
    """Trainer accepts store= keyword argument."""
    config = AutoPilotConfig(workspace=tmp_path)
    param = Parameter(requires_grad=True)
    store = FileStore(config)
    store.register_parameters({'source': param})
    trainer = Trainer(store=store)
    assert trainer.store is store
