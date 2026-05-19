"""Tests for materialize ref updates (BUG-039), log consistency, and checkout cleanup (BUG-075)."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from pathlib import Path


def test_materialize_updates_latest_epoch_in_refs(tmp_path: Path) -> None:
  """materialize rewinds tip and sets refs latest_epoch to the target epoch."""
  src = tmp_path / 'src'
  src.mkdir()
  (src / 'f.txt').write_text('v0')
  config = AutoPilotConfig(workspace=tmp_path)
  store = FileStore(config)
  param = PathParameter(source=str(src), pattern='*')
  store.register_parameters({'source': param})
  store.snapshot('exp-m', 0)
  (src / 'f.txt').write_text('v1')
  store.snapshot('exp-m', 1)
  (src / 'f.txt').write_text('v2')
  store.snapshot('exp-m', 2)

  store.materialize('exp-m', 1)
  refs = store.load_refs()
  assert refs['branches']['exp-m']['latest_epoch'] == 1


def test_materialize_sets_head_to_experiment_id(tmp_path: Path) -> None:
  """materialize keeps HEAD on the materialized branch."""
  src = tmp_path / 'src'
  src.mkdir()
  (src / 'f.txt').write_text('x')
  config = AutoPilotConfig(workspace=tmp_path)
  store = FileStore(config)
  param = PathParameter(source=str(src), pattern='*')
  store.register_parameters({'source': param})
  store.snapshot('exp-head', 0)
  (src / 'f.txt').write_text('y')
  store.snapshot('exp-head', 1)

  store.materialize('exp-head', 1)
  refs = store.load_refs()
  assert refs['HEAD'] == 'exp-head'


def test_log_latest_epoch_consistent_after_materialize(tmp_path: Path) -> None:
  """log lists on-disk epoch snapshots after materialize."""
  src = tmp_path / 'src'
  src.mkdir()
  (src / 'f.txt').write_text('a')
  config = AutoPilotConfig(workspace=tmp_path)
  store = FileStore(config)
  param = PathParameter(source=str(src), pattern='*')
  store.register_parameters({'source': param})
  store.snapshot('exp-log', 0)
  (src / 'f.txt').write_text('b')
  store.snapshot('exp-log', 1)
  (src / 'f.txt').write_text('c')
  store.snapshot('exp-log', 2)

  store.materialize('exp-log', 1)
  entries = store.log('exp-log')
  assert len(entries) >= 1
  epochs = {entry.epoch for entry in entries}
  assert 0 in epochs
  assert 1 in epochs
  assert 2 in epochs


def test_checkout_removes_extraneous_files_not_in_snapshot(tmp_path: Path) -> None:
  """checkout removes working-tree files absent from the target manifest (BUG-075)."""
  src = tmp_path / 'src'
  src.mkdir()
  (src / 'a.txt').write_text('alpha')
  (src / 'b.txt').write_text('beta')
  config = AutoPilotConfig(workspace=tmp_path)
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'source': param})
  store.snapshot('exp', 0)
  (src / 'c.txt').write_text('gamma')
  store.snapshot('exp', 1)
  store.checkout('exp', 0)
  assert not (src / 'c.txt').exists()
  assert (src / 'a.txt').exists()
