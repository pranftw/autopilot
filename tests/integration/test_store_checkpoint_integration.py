"""Integration test: StoreCheckpointCallback with real Trainer.fit() flow.

Verifies that StoreCheckpointCallback creates snapshots at the correct epochs
when driven by a full Trainer.fit() run with FileStore and AutoPilotExperiment.
Proves the callback uses the epoch parameter from on_epoch_end(), not
experiment.epoch.
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


class _IntegModule(AutoPilotModule):
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


class _IntegDataModule(DataModule):
  def train_dataloader(self) -> DataLoader:
    return DataLoader([EvalDatum(metadata={'i': i}) for i in range(2)], batch_size=1)

  def val_dataloader(self) -> DataLoader:
    return DataLoader([], batch_size=1)


def _setup(tmp_path: Path, exp_id: str = 'integ-exp'):
  """Create config, store, experiment, and callback for integration tests."""
  config = AutoPilotConfig(workspace=tmp_path)
  param = Parameter(requires_grad=True)
  store = FileStore(config)
  store.register_parameters({'source': param})

  experiment = AutoPilotExperiment(experiment_id=exp_id)
  cb = StoreCheckpointCallback()
  return config, store, experiment, cb


def test_trainer_fit_creates_snapshots(tmp_path: Path) -> None:
  """Full Trainer.fit() with StoreCheckpointCallback + FileStore creates correct snapshots."""
  config, store, experiment, cb = _setup(tmp_path)

  module = _IntegModule()
  trainer = Trainer(
    callbacks=[cb],
    experiment=experiment,
    store=store,
    config=config,
  )
  trainer.fit(module, datamodule=_IntegDataModule(), max_epochs=3)

  log = store.log(experiment.id)
  epochs_in_log = [entry.epoch for entry in log]
  assert epochs_in_log == [0, 1, 2]
  assert cb._last_epoch == 2


def test_snapshot_epoch_sequence(tmp_path: Path) -> None:
  """Verifies the epoch sequence in store.log matches the training run."""
  config, store, experiment, cb = _setup(tmp_path)

  module = _IntegModule()
  trainer = Trainer(
    callbacks=[cb],
    experiment=experiment,
    store=store,
    config=config,
  )
  trainer.fit(module, datamodule=_IntegDataModule(), max_epochs=2)

  log = store.log(experiment.id)
  assert len(log) == 2
  assert log[0].epoch == 0
  assert log[1].epoch == 1


def test_no_store_no_error(tmp_path: Path) -> None:
  """Trainer.fit() with StoreCheckpointCallback but no store does not error."""
  experiment = AutoPilotExperiment(experiment_id='no-store-exp')
  cb = StoreCheckpointCallback()
  module = _IntegModule()

  trainer = Trainer(
    callbacks=[cb],
    experiment=experiment,
  )
  result = trainer.fit(module, datamodule=_IntegDataModule(), max_epochs=2)

  assert result['total_epochs'] == 2
  assert cb._last_epoch is None
