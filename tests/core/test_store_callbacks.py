"""Tests for store-related Trainer callbacks.

Proves StoreCheckpointCallback uses the epoch parameter from on_epoch_end(),
not experiment.epoch. No advance_epoch() calls needed -- the callback is
decoupled from experiment epoch tracking.
"""

from autopilot.core.callbacks.store import StoreCheckpointCallback
from autopilot.core.experiment import Experiment
from autopilot.core.snapshot import SnapshotManifest
from typing import Any
from unittest.mock import MagicMock
import pytest


class RecordingStore:
  """Minimal stand-in for Store snapshot hooks."""

  def __init__(self) -> None:
    self.snapshot_epochs: list[int] = []
    self.snapshot_ids: list[str] = []

  def snapshot(
    self,
    experiment_id: str,
    epoch: int,
    experiment: Any = None,
    **kwargs: Any,
  ) -> SnapshotManifest:
    self.snapshot_ids.append(experiment_id)
    self.snapshot_epochs.append(epoch)
    return SnapshotManifest(epoch=epoch, timestamp='', entries={})


def _trainer_with_experiment(store: RecordingStore, exp_id: str = 'exp-001') -> MagicMock:
  exp = Experiment(experiment_id=exp_id)
  exp.start()
  trainer = MagicMock()
  trainer.experiment = exp
  trainer.store = store
  return trainer


class TestStoreCheckpointCallback:
  def test_snapshot_per_epoch(self) -> None:
    """Callback uses epoch param, not experiment.epoch -- no advance_epoch() needed."""
    store = RecordingStore()
    cb = StoreCheckpointCallback()
    trainer = _trainer_with_experiment(store)
    for e in (0, 1, 2):
      cb.on_epoch_end(trainer=trainer, module=None, epoch=e)
    assert store.snapshot_epochs == [0, 1, 2]
    assert store.snapshot_ids == ['exp-001', 'exp-001', 'exp-001']

  def test_epoch_param_used_not_experiment_epoch(self) -> None:
    """Proves the epoch param is used, not experiment.epoch (which stays 0)."""
    store = RecordingStore()
    cb = StoreCheckpointCallback()
    trainer = _trainer_with_experiment(store)
    assert trainer.experiment.epoch == -1
    cb.on_epoch_end(trainer=trainer, module=None, epoch=5)
    assert store.snapshot_epochs == [5]
    assert trainer.experiment.epoch == -1

  def test_state_dict_returns_last_epoch(self) -> None:
    store = RecordingStore()
    cb = StoreCheckpointCallback()
    trainer = _trainer_with_experiment(store)
    cb.on_epoch_end(trainer=trainer, module=None, epoch=0)
    assert cb.state_dict()['last_epoch'] == 0

  def test_load_state_dict_restores_last_epoch(self) -> None:
    cb = StoreCheckpointCallback()
    cb.load_state_dict({'last_epoch': 9})
    assert cb._last_epoch == 9

  def test_state_dict_fresh_is_none(self) -> None:
    cb = StoreCheckpointCallback()
    assert cb.state_dict() == {'last_epoch': None}

  def test_no_experiment_is_noop(self) -> None:
    store = RecordingStore()
    cb = StoreCheckpointCallback()
    trainer = MagicMock()
    trainer.experiment = None
    cb.on_epoch_end(trainer=trainer, module=None, epoch=0)
    assert store.snapshot_epochs == []

  def test_no_store_is_noop(self) -> None:
    """When trainer.store is None, callback skips silently."""
    cb = StoreCheckpointCallback()
    trainer = MagicMock()
    trainer.experiment = Experiment(experiment_id='exp-001')
    trainer.experiment.start()
    trainer.store = None
    cb.on_epoch_end(trainer=trainer, module=None, epoch=0)
    assert cb._last_epoch is None

  def test_load_state_dict_missing_last_epoch_raises(self) -> None:
    """Missing last_epoch key raises KeyError."""
    cb = StoreCheckpointCallback()
    with pytest.raises(KeyError, match='last_epoch'):
      cb.load_state_dict({})
