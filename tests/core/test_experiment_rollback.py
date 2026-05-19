"""Tests for Experiment.rollback (BUG-038)."""

from autopilot.core.errors import StoreError
from autopilot.core.experiment import Experiment
from unittest.mock import MagicMock
import pytest


def test_experiment_rollback_calls_store_checkout() -> None:
  """rollback delegates to store.checkout with experiment id and epoch."""
  store = MagicMock()
  experiment = Experiment(experiment_id='exp-1')
  experiment.store = store
  experiment.rollback(3)
  store.checkout.assert_called_once_with(experiment.id, 3, context='rolled back to epoch 3')


def test_experiment_rollback_no_store_noop() -> None:
  """rollback with no store is a silent no-op."""
  experiment = Experiment(experiment_id='exp-2')
  experiment.store = None
  experiment.rollback(3)


def test_experiment_rollback_none_epoch_noop() -> None:
  """rollback(None) does not touch the store."""
  store = MagicMock()
  experiment = Experiment(experiment_id='exp-3')
  experiment.store = store
  experiment.rollback(None)
  store.checkout.assert_not_called()


def test_experiment_rollback_sets_epoch_attr() -> None:
  """rollback aligns experiment.epoch after checkout."""
  store = MagicMock()
  experiment = Experiment(experiment_id='exp-4')
  experiment.store = store
  experiment.rollback(5)
  assert experiment.epoch == 5


def test_experiment_rollback_missing_epoch_propagates_store_error() -> None:
  """checkout failures surface as StoreError."""
  store = MagicMock()
  store.checkout.side_effect = StoreError('missing epoch')
  experiment = Experiment(experiment_id='exp-5')
  experiment.store = store
  with pytest.raises(StoreError):
    experiment.rollback(99)
