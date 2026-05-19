"""Tests for post-complete snapshot policy (BUG-046)."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.enums import Status
from autopilot.core.errors import ExperimentError
from autopilot.core.experiment import Experiment
from pathlib import Path
import pytest


def _store_with_param(tmp_path: Path) -> FileStore:
  src = tmp_path / 'src'
  src.mkdir()
  (src / 'f.txt').write_text('hello')
  config = AutoPilotConfig(workspace=tmp_path)
  store = FileStore(config)
  param = PathParameter(source=str(src), pattern='*')
  store.register_parameters({'source': param})
  store.snapshot('exp-1', 0)
  return store


def test_snapshot_after_complete_proceeds_silently(tmp_path: Path) -> None:
  """Default strict_snapshot_after_complete silently allows snapshot after complete."""
  store = _store_with_param(tmp_path)
  exp = Experiment(experiment_id='exp-1')
  exp.start()
  exp.complete()
  assert exp.status == Status.completed
  manifest = store.snapshot('exp-1', 1, experiment=exp, force=True)
  assert manifest.epoch == 1


def test_snapshot_after_complete_strict_raises_experiment_error(tmp_path: Path) -> None:
  """strict_snapshot_after_complete=True rejects snapshot after complete."""
  store = _store_with_param(tmp_path)
  exp = Experiment(experiment_id='exp-1', strict_snapshot_after_complete=True)
  exp.start()
  exp.complete()
  with pytest.raises(ExperimentError, match='completed'):
    store.snapshot('exp-1', 1, experiment=exp, force=True)
