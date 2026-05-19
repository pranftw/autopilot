"""Tests for StoreCheckpointCallback rollback-stopped skip (section 4.13)."""

from autopilot.core.callbacks.store import StoreCheckpointCallback
from autopilot.core.experiment import Experiment
from unittest.mock import MagicMock


def test_store_checkpoint_callback_skips_snapshot_after_rollback_stopped() -> None:
  """When experiment.should_rollback is True, snapshot is not called."""
  cb = StoreCheckpointCallback()

  experiment = Experiment(experiment_id='exp-1')
  experiment.should_rollback = True

  store = MagicMock()
  trainer = MagicMock()
  trainer.experiment = experiment
  trainer.store = store

  cb.on_epoch_end(trainer, MagicMock(), epoch=0)

  store.snapshot.assert_not_called()


def test_store_checkpoint_callback_snapshots_when_not_rolled_back() -> None:
  """When should_rollback is False, snapshot proceeds normally."""
  cb = StoreCheckpointCallback()

  experiment = Experiment(experiment_id='exp-1')
  experiment.should_rollback = False

  store = MagicMock()
  trainer = MagicMock()
  trainer.experiment = experiment
  trainer.store = store

  cb.on_epoch_end(trainer, MagicMock(), epoch=0)

  store.snapshot.assert_called_once_with(
    'exp-1',
    0,
    experiment=experiment,
    force=True,
    context='epoch 0 checkpoint',
  )
