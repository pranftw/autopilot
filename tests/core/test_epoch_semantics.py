"""Tests for epoch counter documentation (BUG-047) and rollback alignment."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from pathlib import Path
from unittest.mock import MagicMock
import autopilot.core.experiment as experiment_mod


def test_epoch_docstring_lists_three_counters() -> None:
  """Module doc lists store tip, Trainer loop, and Experiment logical epoch."""
  doc = experiment_mod.__doc__ or ''
  assert 'Trainer.current_epoch' in doc
  assert 'Experiment.epoch' in doc
  assert 'store' in doc


def test_after_rollback_experiment_epoch_matches_checkout_epoch() -> None:
  """rollback(epoch) sets experiment.epoch to the checkout target."""
  mock_store = MagicMock()
  exp = Experiment(experiment_id='exp-1')
  exp.store = mock_store
  exp.rollback(5)
  mock_store.checkout.assert_called_once_with('exp-1', 5, context='rolled back to epoch 5')
  assert exp.epoch == 5


def test_rollback_aligns_experiment_epoch_with_store_tip(tmp_path: Path) -> None:
  """Real FileStore checkout plus rollback keeps experiment.epoch in sync."""
  config = AutoPilotConfig(workspace=tmp_path)
  src = tmp_path / 'src'
  src.mkdir()
  (src / 'f.txt').write_text('hello')
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'source': param})
  store.snapshot('exp-1', 0)
  exp = Experiment(experiment_id='exp-1')
  exp.store = store
  exp.epoch = 3
  exp.rollback(0)
  assert exp.epoch == 0
