"""Tests for store checkout context threading (plan 01, dogfood-v8).

Verifies that ``Store.checkout(..., context=)`` records the context string
in the reflog entry, matching ``Store.snapshot(..., context=)`` behavior.
"""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError
from autopilot.core.experiment import Experiment
from autopilot.tracking.io import read_jsonl
from pathlib import Path
from unittest.mock import MagicMock
import pytest


def _make_store(
  tmp_path: Path,
  files: dict[str, str] | None = None,
) -> tuple[FileStore, Path, PathParameter]:
  """Create a FileStore with a single PathParameter for testing."""
  if files is None:
    files = {'main.py': 'print("hello")\n'}
  src = tmp_path / 'src'
  src.mkdir(parents=True, exist_ok=True)
  for name, content in files.items():
    (src / name).write_text(content)
  config = AutoPilotConfig(workspace=tmp_path)
  config.store_path = tmp_path / '.autopilot'
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'source': param})
  return store, src, param


def _read_reflog(store: FileStore) -> list[dict]:
  """Read all reflog entries from the store."""
  return read_jsonl(store.config.store_path / 'reflog.jsonl', strict=False)


# -- 4.1 Reflog recording (store API) ----------------------------------------


class TestCheckoutContextReflog:
  """Checkout context appears in reflog entries."""

  def test_checkout_context_recorded_in_reflog(self, tmp_path: Path) -> None:
    """Context string persists in reflog checkout entry."""
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    store.checkout('exp-1', 0, context='restore baseline')

    entries = _read_reflog(store)
    checkout_entries = [e for e in entries if e['operation'] == 'checkout']
    assert len(checkout_entries) == 1
    entry = checkout_entries[0]
    assert entry['operation'] == 'checkout'
    assert entry['context'] == 'restore baseline'
    assert entry['experiment_id'] == 'exp-1'
    assert entry['new_epoch'] == 0

  def test_checkout_null_context_backward_compat(self, tmp_path: Path) -> None:
    """Checkout without context arg defaults to None in reflog."""
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    store.checkout('exp-1', 0)

    entries = _read_reflog(store)
    checkout_entries = [e for e in entries if e['operation'] == 'checkout']
    assert len(checkout_entries) == 1
    assert checkout_entries[0].get('context') is None

  def test_checkout_context_with_multiple_epochs(self, tmp_path: Path) -> None:
    """Context recorded across epoch transitions."""
    store, src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    (src / 'main.py').write_text('print("v2")\n')
    store.snapshot('exp-1', 1)
    store.checkout('exp-1', 0, context='rewind')

    entries = _read_reflog(store)
    checkout_entries = [e for e in entries if e['operation'] == 'checkout']
    assert len(checkout_entries) == 1
    entry = checkout_entries[0]
    assert entry['old_epoch'] == 1
    assert entry['new_epoch'] == 0
    assert entry['context'] == 'rewind'

  def test_checkout_strict_schema_accepts_context(self, tmp_path: Path) -> None:
    """Context works alongside strict_schema=True when schema matches."""
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    store.checkout('exp-1', 0, context='schema ok', strict_schema=True)

    entries = _read_reflog(store)
    checkout_entries = [e for e in entries if e['operation'] == 'checkout']
    assert len(checkout_entries) == 1
    assert checkout_entries[0]['context'] == 'schema ok'


# -- 4.2 Error and edge cases ------------------------------------------------


class TestCheckoutContextEdgeCases:
  """Error paths and edge cases for checkout context."""

  def test_checkout_failure_does_not_append_reflog(self, tmp_path: Path) -> None:
    """Failed checkout (bad epoch) must not append a reflog entry."""
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    entries_before = len(_read_reflog(store))

    with pytest.raises(StoreError):
      store.checkout('exp-1', 99)

    entries_after = len(_read_reflog(store))
    assert entries_after == entries_before

  def test_checkout_missing_branch_no_reflog(self, tmp_path: Path) -> None:
    """Checkout of nonexistent branch appends no reflog entry."""
    store, _src, _param = _make_store(tmp_path)
    entries_before = len(_read_reflog(store))

    with pytest.raises(StoreError):
      store.checkout('ghost', 0)

    entries_after = len(_read_reflog(store))
    assert entries_after == entries_before

  def test_checkout_context_empty_string(self, tmp_path: Path) -> None:
    """Empty string context is preserved (store layer does not reject)."""
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    store.checkout('exp-1', 0, context='')

    entries = _read_reflog(store)
    checkout_entries = [e for e in entries if e['operation'] == 'checkout']
    assert len(checkout_entries) == 1
    assert checkout_entries[0]['context'] is not None
    assert len(checkout_entries[0]['context']) == 0


# -- 4.3 BranchHandle --------------------------------------------------------


class TestBranchHandleCheckoutContext:
  """BranchHandle.checkout forwards context to store."""

  def test_branch_handle_checkout_forwards_context(self, tmp_path: Path) -> None:
    """BranchHandle.checkout propagates context to reflog."""
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    handle = store.branch_handle('exp-1')
    handle.checkout(0, context='via handle')

    entries = _read_reflog(store)
    checkout_entries = [e for e in entries if e['operation'] == 'checkout']
    assert len(checkout_entries) == 1
    assert checkout_entries[0]['context'] == 'via handle'


# -- 4.4 Caller integration --------------------------------------------------


class TestExperimentRollbackCheckoutContext:
  """Experiment.rollback passes context through to store checkout."""

  def test_experiment_rollback_records_checkout_context(self, tmp_path: Path) -> None:
    """Rollback uses 'rolled back to epoch N' as checkout context."""
    store, src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    (src / 'main.py').write_text('print("v2")\n')
    store.snapshot('exp-1', 1)

    exp = Experiment(experiment_id='exp-1')
    exp.store = store
    exp.start()
    exp.rollback(0)

    entries = _read_reflog(store)
    checkout_entries = [e for e in entries if e['operation'] == 'checkout']
    assert len(checkout_entries) == 1
    assert checkout_entries[0]['context'] == 'rolled back to epoch 0'

  def test_experiment_rollback_noop_skips_checkout(self, tmp_path: Path) -> None:
    """rollback(None) and store=None produce no checkout reflog entry."""
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)

    exp = Experiment(experiment_id='exp-1')
    exp.store = store
    exp.start()

    entries_before = len(_read_reflog(store))
    exp.rollback(None)
    entries_after = len(_read_reflog(store))
    assert entries_after == entries_before

    exp2 = Experiment(experiment_id='exp-2')
    exp2.start()
    exp2.rollback(0)


class TestTrainerCheckpointRestoreContext:
  """Trainer checkpoint resume passes context to store checkout."""

  def test_trainer_checkpoint_restore_passes_context(self) -> None:
    """store.checkout called with 'checkpoint resume epoch N' context."""
    from autopilot.core.trainer.checkpoint import restore_path_parameter_files

    mock_store = MagicMock()
    mock_exp = MagicMock()
    mock_exp.id = 'exp-1'

    trainer = MagicMock()
    trainer.store = mock_store
    trainer._experiment = mock_exp
    trainer.profiler = None

    module = MagicMock()
    module.parameters.return_value = [PathParameter(source='/tmp/src', pattern='*')]

    state = {'experiment': {'epoch': 2}}
    restore_path_parameter_files(trainer, state, module)
    mock_store.checkout.assert_called_once_with('exp-1', 2, context='checkpoint resume epoch 2')

  def test_trainer_checkpoint_restore_skips_without_path_params(self) -> None:
    """No checkout when module has no PathParameter instances."""
    from autopilot.core.trainer.checkpoint import restore_path_parameter_files

    mock_store = MagicMock()
    mock_exp = MagicMock()
    mock_exp.id = 'exp-1'

    trainer = MagicMock()
    trainer.store = mock_store
    trainer._experiment = mock_exp

    module = MagicMock()
    module.parameters.return_value = [MagicMock(spec=[])]

    state = {'experiment': {'epoch': 2}}
    restore_path_parameter_files(trainer, state, module)
    mock_store.checkout.assert_not_called()
