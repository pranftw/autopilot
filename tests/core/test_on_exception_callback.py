"""Tests for OnExceptionCallback: crash checkpoint and store snapshot on exception.

Covers all scenarios from sub-plan 03: checkpoint save, store snapshot,
clean teardown cleanup, best-effort failure semantics, path resolution,
concurrent isolation, and crash file retention after failed fit.
"""

from autopilot.core.callbacks.on_exception import OnExceptionCallback
from autopilot.core.checkpoint import JSONCheckpointIO
from autopilot.core.config import Config
from autopilot.core.experiment import Experiment
from autopilot.core.trainer.trainer import Trainer
from pathlib import Path
from tests.doubles import NoopEvalModule
from typing import Any
from unittest.mock import MagicMock, patch
import json
import pytest


class FailingModule(NoopEvalModule):
  """Module that raises ValueError during training_step."""

  def training_step(self, batch: Any, batch_idx: int) -> Any:
    msg = 'intentional crash'
    raise ValueError(msg)


class TestOnExceptionSavesCheckpoint:
  """Test 1: on_exception saves a crash checkpoint file."""

  def test_on_exception_saves_checkpoint(self, tmp_path: Path) -> None:
    cb = OnExceptionCallback(directory=tmp_path)
    mod = FailingModule()
    trainer = Trainer(callbacks=[cb])
    with pytest.raises(ValueError, match='intentional crash'):
      trainer.fit(mod, max_epochs=1, train_dataloaders=[1])
    crash_path = tmp_path / 'crash_checkpoint.json'
    assert crash_path.exists()
    content = json.loads(crash_path.read_text())
    assert isinstance(content, dict)


class TestOnExceptionSavesStoreSnapshot:
  """Test 2: with store + experiment, snapshot is called with crash context."""

  def test_on_exception_saves_store_snapshot(self, tmp_path: Path) -> None:
    cb = OnExceptionCallback(directory=tmp_path)
    mod = FailingModule()
    mock_store = MagicMock()
    mock_store.snapshot = MagicMock()
    experiment = Experiment(experiment_id='exp-crash-test', hypothesis='test crash')
    trainer = Trainer(callbacks=[cb])
    trainer._store = mock_store
    trainer._experiment = experiment
    experiment.start()
    with pytest.raises(ValueError, match='intentional crash'):
      trainer.fit(mod, max_epochs=1, train_dataloaders=[1])
    mock_store.snapshot.assert_called_once()
    call_args = mock_store.snapshot.call_args
    context_arg = call_args.kwargs.get('context')
    if context_arg is None and len(call_args.args) > 2:
      context_arg = call_args.args[2]
    assert context_arg is not None
    assert 'crash:' in context_arg
    assert 'ValueError' in context_arg


class TestCleanTeardownRemovesCrashCheckpoint:
  """Test 3: successful fit removes any stale crash checkpoint."""

  def test_clean_teardown_removes_crash_checkpoint(self, tmp_path: Path) -> None:
    crash_path = tmp_path / 'crash_checkpoint.json'
    crash_path.write_text('{"stale": true}')
    cb = OnExceptionCallback(directory=tmp_path)
    cb._crash_path = crash_path
    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[cb])
    trainer.fit(mod, max_epochs=1)
    assert not crash_path.exists()


class TestOnExceptionWithoutStore:
  """Test 4: trainer.store is None -> only checkpoint, no snapshot."""

  def test_on_exception_without_store(self, tmp_path: Path) -> None:
    cb = OnExceptionCallback(directory=tmp_path)
    mod = FailingModule()
    trainer = Trainer(callbacks=[cb])
    assert trainer.store is None
    with pytest.raises(ValueError, match='intentional crash'):
      trainer.fit(mod, max_epochs=1, train_dataloaders=[1])
    crash_path = tmp_path / 'crash_checkpoint.json'
    assert crash_path.exists()


class TestOnExceptionWithoutExperiment:
  """Test 5: trainer.experiment is None -> checkpoint written, no snapshot."""

  def test_on_exception_without_experiment(self, tmp_path: Path) -> None:
    cb = OnExceptionCallback(directory=tmp_path)
    mod = FailingModule()
    mock_store = MagicMock()
    trainer = Trainer(callbacks=[cb])
    trainer._store = mock_store
    assert trainer.experiment is None
    with pytest.raises(ValueError, match='intentional crash'):
      trainer.fit(mod, max_epochs=1, train_dataloaders=[1])
    crash_path = tmp_path / 'crash_checkpoint.json'
    assert crash_path.exists()
    mock_store.snapshot.assert_not_called()


class TestResumeFromCrashCheckpoint:
  """Test 6: crash checkpoint can be used for fit resume."""

  def test_resume_from_crash_checkpoint(self, tmp_path: Path) -> None:
    cb = OnExceptionCallback(directory=tmp_path)
    mod = FailingModule()
    trainer = Trainer(callbacks=[cb])
    with pytest.raises(ValueError, match='intentional crash'):
      trainer.fit(mod, max_epochs=1, train_dataloaders=[1])
    crash_path = tmp_path / 'crash_checkpoint.json'
    assert crash_path.exists()
    loaded = JSONCheckpointIO().load(crash_path)
    assert isinstance(loaded, dict)
    assert 'module' in loaded
    resume_mod = NoopEvalModule()
    resume_trainer = Trainer(dry_run=True)
    result = resume_trainer.fit(resume_mod, max_epochs=1, ckpt_path=crash_path)
    assert result is not None


class TestOnExceptionDirectoryDefault:
  """Test 7: omit directory -> uses trainer.config.root when experiment present."""

  def test_on_exception_directory_default(self, tmp_path: Path) -> None:
    cb = OnExceptionCallback()
    mod = FailingModule()
    config = Config(workspace=tmp_path)
    config.root = tmp_path
    experiment = Experiment(experiment_id='exp-default-dir', hypothesis='default dir test')
    trainer = Trainer(callbacks=[cb], config=config)
    trainer._experiment = experiment
    experiment.start()
    with pytest.raises(ValueError, match='intentional crash'):
      trainer.fit(mod, max_epochs=1, train_dataloaders=[1])
    crash_path = tmp_path / 'crash_checkpoint.json'
    assert crash_path.exists()


class TestOnExceptionCustomDirectory:
  """Test 8: explicit directory -> file under custom dir."""

  def test_on_exception_custom_directory(self, tmp_path: Path) -> None:
    custom_dir = tmp_path / 'custom_crash_dir'
    custom_dir.mkdir()
    cb = OnExceptionCallback(directory=custom_dir)
    mod = FailingModule()
    trainer = Trainer(callbacks=[cb])
    with pytest.raises(ValueError, match='intentional crash'):
      trainer.fit(mod, max_epochs=1, train_dataloaders=[1])
    crash_path = custom_dir / 'crash_checkpoint.json'
    assert crash_path.exists()


class TestOnExceptionSaveFailureBestEffort:
  """Test 9: save_checkpoint raises -> original exception still propagates."""

  def test_on_exception_save_failure_best_effort(self, tmp_path: Path) -> None:
    cb = OnExceptionCallback(directory=tmp_path)
    mod = FailingModule()
    trainer = Trainer(callbacks=[cb])
    with (
      patch.object(Trainer, 'save_checkpoint', side_effect=RuntimeError('disk full')),
      pytest.raises(ValueError, match='intentional crash'),
    ):
      trainer.fit(mod, max_epochs=1, train_dataloaders=[1])


class TestOnExceptionSnapshotFailureBestEffort:
  """Test 10: store.snapshot raises -> original exception still propagates."""

  def test_on_exception_snapshot_failure_best_effort(self, tmp_path: Path) -> None:
    from autopilot.core.errors import StoreError

    cb = OnExceptionCallback(directory=tmp_path)
    mod = FailingModule()
    mock_store = MagicMock()
    mock_store.snapshot = MagicMock(side_effect=StoreError('snapshot failed'))
    experiment = Experiment(experiment_id='exp-snap-fail', hypothesis='snap fail test')
    trainer = Trainer(callbacks=[cb])
    trainer._store = mock_store
    trainer._experiment = experiment
    experiment.start()
    with pytest.raises(ValueError, match='intentional crash'):
      trainer.fit(mod, max_epochs=1, train_dataloaders=[1])
    crash_path = tmp_path / 'crash_checkpoint.json'
    assert crash_path.exists()


class TestOnExceptionConcurrentFit:
  """Test 11: two independent trainers crash -> no cross-run clobbering."""

  def test_on_exception_concurrent_fit(self, tmp_path: Path) -> None:
    dir_a = tmp_path / 'run_a'
    dir_b = tmp_path / 'run_b'
    dir_a.mkdir()
    dir_b.mkdir()
    cb_a = OnExceptionCallback(directory=dir_a)
    cb_b = OnExceptionCallback(directory=dir_b)
    mod_a = FailingModule()
    mod_b = FailingModule()
    trainer_a = Trainer(callbacks=[cb_a])
    trainer_b = Trainer(callbacks=[cb_b])
    with pytest.raises(ValueError, match='intentional crash'):
      trainer_a.fit(mod_a, max_epochs=1, train_dataloaders=[1])
    with pytest.raises(ValueError, match='intentional crash'):
      trainer_b.fit(mod_b, max_epochs=1, train_dataloaders=[1])
    crash_a = dir_a / 'crash_checkpoint.json'
    crash_b = dir_b / 'crash_checkpoint.json'
    assert crash_a.exists()
    assert crash_b.exists()
    content_a = json.loads(crash_a.read_text())
    content_b = json.loads(crash_b.read_text())
    assert isinstance(content_a, dict)
    assert isinstance(content_b, dict)


class TestFailedFitRetainsCrashFile:
  """Test 12: after fit raises, crash file is NOT deleted by teardown."""

  def test_failed_fit_retains_crash_file(self, tmp_path: Path) -> None:
    cb = OnExceptionCallback(directory=tmp_path)
    mod = FailingModule()
    trainer = Trainer(callbacks=[cb])
    with pytest.raises(ValueError, match='intentional crash'):
      trainer.fit(mod, max_epochs=1, train_dataloaders=[1])
    crash_path = tmp_path / 'crash_checkpoint.json'
    assert crash_path.exists()
    assert cb._exception_fired is True


class TestSetupResolvesCrashPath:
  """Test 13: setup() resolves _crash_path to a Path."""

  def test_setup_resolves_crash_path(self, tmp_path: Path) -> None:
    cb = OnExceptionCallback(directory=tmp_path)
    trainer = MagicMock()
    trainer.experiment = None
    from autopilot.data.datamodule import Stage

    cb.setup(trainer, MagicMock(), Stage.fit)
    assert cb._crash_path is not None
    assert isinstance(cb._crash_path, Path)


class TestStaleCrashFileRemovedOnCleanFit:
  """Test 14: stale crash file from prior run removed on clean fit."""

  def test_stale_crash_file_removed_on_clean_fit(self, tmp_path: Path) -> None:
    crash_path = tmp_path / 'crash_checkpoint.json'
    crash_path.write_text('{"stale": true}')
    cb = OnExceptionCallback(directory=tmp_path)
    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[cb])
    trainer.fit(mod, max_epochs=1)
    assert not crash_path.exists()


class TestCrashFilePreservedWhenExceptionFired:
  """Test 15: crash file preserved after exception during fit."""

  def test_crash_file_preserved_when_exception_fired(self, tmp_path: Path) -> None:
    crash_path = tmp_path / 'crash_checkpoint.json'
    crash_path.write_text('{"prior_crash": true}')
    cb = OnExceptionCallback(directory=tmp_path)
    mod = FailingModule()
    trainer = Trainer(callbacks=[cb])
    with pytest.raises(ValueError, match='intentional crash'):
      trainer.fit(mod, max_epochs=1, train_dataloaders=[1])
    assert crash_path.exists()


class TestSetupResetsExceptionFired:
  """Test 16: setup() resets _exception_fired to False."""

  def test_setup_resets_exception_fired(self, tmp_path: Path) -> None:
    cb = OnExceptionCallback(directory=tmp_path)
    cb._exception_fired = True
    trainer = MagicMock()
    trainer.experiment = None
    from autopilot.data.datamodule import Stage

    cb.setup(trainer, MagicMock(), Stage.fit)
    assert cb._exception_fired is False
