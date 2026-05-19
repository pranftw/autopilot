"""Tests for Trainer construction, callback dispatch, and fit()."""

from autopilot.ai.parameter import PathParameter
from autopilot.core.callbacks.callback import Callback
from autopilot.core.callbacks.checkpoint import CheckpointCallback
from autopilot.core.checkpoint import JSONCheckpointIO
from autopilot.core.errors import StoreError, TrackingError
from autopilot.core.experiment import Experiment
from autopilot.core.logger import Logger
from autopilot.core.models import Result
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.parameter import Parameter
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.tree import Tree
from autopilot.core.types import EvalDatum
from autopilot.data.dataloader import DataLoader
from autopilot.data.datamodule import DataModule, Stage
from autopilot.data.dataset import ListDataset
from pathlib import Path
from tests.doubles import NoopEvalModule, NoOpOptimizer, PlainModule
from typing import Any, cast
from unittest.mock import MagicMock, call
import json
import logging
import pytest


class TestTrainerConstruction:
  def test_minimal(self) -> None:
    trainer = Trainer()
    assert trainer.module is None
    assert trainer.callbacks == []
    assert trainer.logger is None

  def test_with_callbacks_and_dry_run(self) -> None:
    trainer = Trainer(callbacks=[Callback(), Callback()], dry_run=True)
    assert len(trainer.callbacks) == 2
    assert trainer.module is None
    assert trainer.dry_run is True

  def test_with_logger(self) -> None:
    logger = Logger()
    trainer = Trainer(logger=logger)
    assert trainer.logger is logger

  def test_no_run_method(self) -> None:
    trainer = Trainer()
    assert not hasattr(trainer, 'run') or not callable(trainer.run)

  def test_fit_sets_module_ref(self) -> None:
    mod = NoopEvalModule()
    trainer = Trainer()
    trainer.fit(mod, max_epochs=1)
    assert trainer.module is mod


class TestTrainerCallbackDispatch:
  def test_dispatch_invokes_all_callbacks(self) -> None:
    calls: list[str] = []

    class A(Callback):
      def on_epoch_start(self, trainer: object, module: object, epoch: int) -> None:
        calls.append('A')

    class B(Callback):
      def on_epoch_start(self, trainer: object, module: object, epoch: int) -> None:
        calls.append('B')

    trainer = Trainer(callbacks=[A(), B()])
    trainer.on_epoch_start(0)
    assert calls == ['A', 'B']

  def test_dispatch_skips_unimplemented_hooks(self) -> None:
    trainer = Trainer(callbacks=[Callback()])
    trainer.on_epoch_start(0)

  def test_dispatch_unknown_hook_is_safe(self) -> None:
    trainer = Trainer(callbacks=[Callback()])
    trainer.dispatch_callbacks('on_nonexistent_hook', x=1)


class TestTrainerFit:
  def test_fit_invokes_loop_hooks(self) -> None:
    events: list[str] = []

    class Track(Callback):
      def on_loop_start(self, trainer: object, module: object, max_epochs: int) -> None:
        events.append('loop_start')

      def on_loop_end(self, trainer: object, module: object, result: dict) -> None:
        events.append('loop_end')

    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[Track()], dry_run=True)
    trainer.fit(mod, max_epochs=2)
    assert 'loop_start' in events
    assert 'loop_end' in events


class TestTrainerRunLoopViaFit:
  def test_dispatches_epoch_hooks(self) -> None:
    epochs: list[tuple] = []

    class EpochTracker(Callback):
      def on_epoch_start(self, trainer: object, module: object, epoch: int) -> None:
        epochs.append(('start', epoch))

      def on_epoch_end(
        self,
        trainer: Any,
        module: Any,
        epoch: int,
        result: Result | None = None,
      ) -> None:
        epochs.append(('end', epoch))

    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[EpochTracker()], dry_run=True)
    result = trainer.fit(mod, max_epochs=2)
    assert result['total_epochs'] == 2
    assert epochs == [('start', 0), ('end', 0), ('start', 1), ('end', 1)]


class TestTrainerProperties:
  def test_trainer_optimizer_none_before_fit(self):
    trainer = Trainer()
    assert trainer.optimizer is None


class TestTrainerRequiresAutoPilotModule:
  def test_plain_module_raises_type_error(self):
    trainer = Trainer()
    with pytest.raises(TypeError, match=r'Trainer.fit\(\) requires AutoPilotModule'):
      trainer.fit(cast(Any, PlainModule()), max_epochs=1)


def test_trainer_fit_rejects_plain_module():
  mod = PlainModule()
  trainer = Trainer()
  with pytest.raises(TypeError) as exc_info:
    trainer.fit(cast(Any, mod), max_epochs=1)
  msg = str(exc_info.value)
  assert 'Trainer.fit() requires AutoPilotModule' in msg
  assert 'PlainModule' in msg


def test_trainer_fit_docstring_documents_loss_discovery():
  import inspect

  doc = inspect.getdoc(Trainer.fit)
  assert doc is not None
  assert 'module.modules()' in doc
  assert 'Loss' in doc
  assert 'loss_fn=None' in doc


class _TinyDataModule(DataModule):
  def train_dataloader(self) -> DataLoader:
    rows = [EvalDatum(success=True), EvalDatum(success=True)]
    return DataLoader(rows, batch_size=1)

  def val_dataloader(self) -> DataLoader:
    return DataLoader([], batch_size=1)


def test_fit_no_val_skips_validation_metrics_keys() -> None:
  mod = NoopEvalModule()
  trainer = Trainer()
  rows = [EvalDatum(success=True)] * 2
  dl = DataLoader(rows, batch_size=1)
  result = trainer.fit(mod, train_dataloaders=dl, val_dataloaders=None, max_epochs=1)
  assert 'val_metrics' not in result['epochs'][0]


def test_fit_datamodule_supplies_train_loader() -> None:
  mod = NoopEvalModule()
  trainer = Trainer()
  trainer.fit(mod, datamodule=_TinyDataModule(), max_epochs=1)
  assert trainer.fit_context == {}


def test_fit_ctx_exposed_via_fit_context() -> None:
  mod = NoopEvalModule()
  trainer = Trainer()
  ctx = {'run': 'tagged'}
  rows = [EvalDatum(success=True)]
  trainer.fit(
    mod,
    train_dataloaders=DataLoader(rows, batch_size=1),
    ctx=ctx,
    max_epochs=1,
  )
  assert trainer.fit_context == ctx


class _StopBeforeSecondEpochStart(Callback):
  def on_epoch_start(self, trainer, module, epoch: int):
    if epoch >= 1:
      return {'stop': True}
    return None


def test_fit_interrupt_via_should_stop_truncates_epochs() -> None:
  mod = NoopEvalModule()
  trainer = Trainer(callbacks=[_StopBeforeSecondEpochStart()])
  rows = [EvalDatum(success=True)]
  result = trainer.fit(
    mod,
    train_dataloaders=DataLoader(rows, batch_size=1),
    max_epochs=5,
  )
  assert result['total_epochs'] < 5


class TestTrainerLoggerFinalize:
  """Tests for logger.finalize wiring in Trainer.fit."""

  def test_fit_finalize_success_calls_logger(self) -> None:
    mock_logger = MagicMock(spec_set=['finalize', 'log_metrics'])
    mod = NoopEvalModule()
    trainer = Trainer(logger=mock_logger, dry_run=True)
    trainer.fit(mod, max_epochs=1)
    assert mock_logger.finalize.call_args == call('success')
    assert mock_logger.finalize.call_count == 1

  def test_fit_finalize_failed_on_exception(self) -> None:
    class FailingCb(Callback):
      def on_fit_start(self, trainer, module):
        msg = 'boom'
        raise RuntimeError(msg)

    mock_logger = MagicMock(spec_set=['finalize', 'log_metrics'])
    mod = NoopEvalModule()
    trainer = Trainer(logger=mock_logger, callbacks=[FailingCb()])
    with pytest.raises(RuntimeError, match='boom'):
      trainer.fit(mod, max_epochs=1, ctx={})
    assert mock_logger.finalize.call_args == call('failed')
    assert mock_logger.finalize.call_count == 1

  def test_fit_no_finalize_when_logger_none(self) -> None:
    mod = NoopEvalModule()
    trainer = Trainer(dry_run=True)
    result = trainer.fit(mod, max_epochs=1)
    assert result['total_epochs'] == 1


# Trainer checkpoint tests (uses NoOpOptimizer from tests.doubles)


# Trainer checkpoint tests (sub-plan 06, §4.3)


class TestTrainerSaveCheckpoint:
  """Trainer.save_checkpoint assembles component state."""

  def test_save_checkpoint_assembles_components(self, tmp_path: Path) -> None:
    exp = Experiment(experiment_id='e1', hypothesis='h')
    exp.start()
    exp.advance_epoch(metrics={'acc': 0.8})
    param = Parameter()
    opt = NoOpOptimizer([param], lr=0.5)
    opt.block_strategy('greedy')
    cb = Callback()

    mod = NoopEvalModule()
    trainer = Trainer(
      experiment=exp,
      callbacks=[cb],
      dry_run=True,
    )
    trainer._module = mod
    trainer._optimizer = opt

    path = tmp_path / 'ckpt.json'
    trainer.save_checkpoint(path)

    loaded = json.loads(path.read_text())
    assert 'experiment' in loaded
    assert loaded['experiment']['id'] == 'e1'
    assert loaded['experiment']['epoch'] == 0
    assert 'module' in loaded
    assert 'optimizer' in loaded
    assert loaded['optimizer']['defaults']['lr'] == 0.5
    assert loaded['optimizer']['blocked_strategies'] == ['greedy']
    assert 'callbacks' in loaded
    assert 'Callback_0' in loaded['callbacks']

  def test_save_checkpoint_no_experiment(self, tmp_path: Path) -> None:
    mod = NoopEvalModule()
    trainer = Trainer(dry_run=True)
    trainer._module = mod
    path = tmp_path / 'ckpt.json'
    trainer.save_checkpoint(path)
    loaded = json.loads(path.read_text())
    assert 'experiment' not in loaded
    assert 'module' in loaded


class TestTrainerRestoreFromCheckpoint:
  """_restore_from_checkpoint restores component state."""

  def test_restore_optimizer_and_experiment(self, tmp_path: Path) -> None:
    exp = Experiment(experiment_id='e1')
    exp.start()
    exp.advance_epoch(metrics={'m': 1.0})
    param = Parameter()
    opt = NoOpOptimizer([param], lr=0.5)
    opt.block_strategy('x')

    mod = NoopEvalModule()
    trainer = Trainer(experiment=exp, dry_run=True)
    trainer._module = mod
    trainer._optimizer = opt

    path = tmp_path / 'ckpt.json'
    trainer.save_checkpoint(path)

    exp2 = Experiment(experiment_id='placeholder')
    opt2 = NoOpOptimizer([param], lr=99.0)
    trainer2 = Trainer(experiment=exp2, dry_run=True)
    trainer2._optimizer = opt2

    io = JSONCheckpointIO()
    state = io.load(path)
    trainer2._callbacks = []
    trainer2._restore_from_checkpoint(state, mod)

    assert exp2.id == 'e1'
    assert exp2.epoch == 0
    assert opt2.defaults['lr'] == 0.5
    assert opt2.blocked_strategies == frozenset({'x'})

  def test_restore_missing_keys_is_safe(self) -> None:
    mod = NoopEvalModule()
    exp = Experiment(experiment_id='e1')
    trainer = Trainer(experiment=exp, dry_run=True)
    trainer._optimizer = NoOpOptimizer([], lr=5.0)
    trainer._callbacks = []
    trainer._restore_from_checkpoint({'experiment': exp.state_dict()}, mod)
    assert trainer._optimizer.defaults['lr'] == 5.0

  def test_restore_extra_keys_ignored(self) -> None:
    mod = NoopEvalModule()
    exp = Experiment(experiment_id='e1')
    exp.start()
    trainer = Trainer(experiment=exp, dry_run=True)
    trainer._optimizer = None
    trainer._callbacks = []
    state = {'experiment': exp.state_dict(), 'foo': 'bar', 'baz': [1, 2]}
    trainer._restore_from_checkpoint(state, mod)
    assert exp.status.value == 'running'


class TestFitCkptPathResumesMinEpoch:
  """fit(ckpt_path=...) resumes from checkpoint epoch."""

  def test_fit_ckpt_path_skips_completed_epochs(self, tmp_path: Path) -> None:
    exp = Experiment(experiment_id='e1')
    exp.start()
    exp.advance_epoch(metrics={'a': 1.0})
    exp.advance_epoch(metrics={'a': 2.0})
    assert exp.epoch == 1

    state = {
      'experiment': exp.state_dict(),
      'module': {},
    }
    ckpt_path = tmp_path / 'ckpt.json'
    JSONCheckpointIO().save(state, ckpt_path)

    exp2 = Experiment(experiment_id='placeholder')
    mod = NoopEvalModule()
    trainer = Trainer(experiment=exp2, dry_run=True)
    result = trainer.fit(mod, max_epochs=4, ckpt_path=ckpt_path)
    epochs = result['epochs']
    assert len(epochs) == 2
    for ep in epochs:
      assert ep['epoch'] >= 2
    assert exp2.epoch == 3

  def test_fit_ckpt_path_no_experiment_min_epoch_zero(self, tmp_path: Path) -> None:
    state = {'module': {}}
    ckpt_path = tmp_path / 'ckpt.json'
    JSONCheckpointIO().save(state, ckpt_path)

    mod = NoopEvalModule()
    trainer = Trainer(dry_run=True)
    result = trainer.fit(mod, max_epochs=2, ckpt_path=ckpt_path)
    assert result['total_epochs'] == 2

  def test_fit_ckpt_path_with_custom_io(self, tmp_path: Path) -> None:
    calls: list[str] = []

    class TrackingIO(JSONCheckpointIO):
      def load(self, path):
        calls.append('load')
        return super().load(path)

    state = {'module': {}}
    ckpt_path = tmp_path / 'ckpt.json'
    JSONCheckpointIO().save(state, ckpt_path)

    mod = NoopEvalModule()
    trainer = Trainer(dry_run=True)
    result = trainer.fit(mod, max_epochs=1, ckpt_path=ckpt_path, checkpoint_io=TrackingIO())
    assert 'load' in calls
    assert result['total_epochs'] == 1

  def test_fit_min_epoch_gte_max_epochs_zero_iterations(self, tmp_path: Path) -> None:
    exp = Experiment(experiment_id='e1')
    exp.start()
    for _ in range(3):
      exp.advance_epoch()
    assert exp.epoch == 2

    state = {'experiment': exp.state_dict()}
    ckpt_path = tmp_path / 'ckpt.json'
    JSONCheckpointIO().save(state, ckpt_path)

    exp2 = Experiment(experiment_id='placeholder')
    mod = NoopEvalModule()
    trainer = Trainer(experiment=exp2, dry_run=True)
    result = trainer.fit(mod, max_epochs=3, ckpt_path=ckpt_path)
    assert result['total_epochs'] == 0


# CheckpointCallback tests (sub-plan 06, §4.4)


class TestCheckpointCallback:
  """CheckpointCallback writes per-epoch files."""

  def test_writes_per_epoch(self, tmp_path: Path) -> None:
    ckpt_dir = tmp_path / 'checkpoints'
    ckpt_dir.mkdir()
    cb = CheckpointCallback(directory=ckpt_dir)
    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[cb], dry_run=True)
    result = trainer.fit(mod, max_epochs=3)
    assert result['total_epochs'] == 3
    for i in range(3):
      path = ckpt_dir / f'epoch-{i:04d}.json'
      assert path.is_file(), f'missing checkpoint for epoch {i}'
      data = json.loads(path.read_text())
      assert isinstance(data, dict)

  def test_checkpoint_contains_module_state(self, tmp_path: Path) -> None:
    ckpt_dir = tmp_path / 'checkpoints'
    ckpt_dir.mkdir()
    cb = CheckpointCallback(directory=ckpt_dir)
    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[cb], dry_run=True)
    trainer.fit(mod, max_epochs=1)
    path = ckpt_dir / 'epoch-0000.json'
    data = json.loads(path.read_text())
    assert 'module' in data


# Corrupt / missing checkpoint tests (sub-plan 04, section 2.9)


def test_fit_corrupt_checkpoint_raises(tmp_path: Path) -> None:
  """fit() with corrupt checkpoint propagates TrackingError."""
  bad_ckpt = tmp_path / 'bad.json'
  bad_ckpt.write_text('{invalid json', encoding='utf-8')
  trainer = Trainer(dry_run=True)
  module = NoopEvalModule()
  loader = DataLoader([EvalDatum(success=True)], batch_size=1)
  with pytest.raises(TrackingError, match='invalid JSON'):
    trainer.fit(module, train_dataloaders=loader, ckpt_path=bad_ckpt)


def test_fit_missing_checkpoint_raises(tmp_path: Path) -> None:
  """fit() with nonexistent checkpoint path propagates TrackingError."""
  missing_ckpt = tmp_path / 'nonexistent.json'
  trainer = Trainer(dry_run=True)
  module = NoopEvalModule()
  loader = DataLoader([EvalDatum(success=True)], batch_size=1)
  with pytest.raises(TrackingError):
    trainer.fit(module, train_dataloaders=loader, ckpt_path=missing_ckpt)


# _build_checkpoint_state unit tests (sub-plan 04, section 2.10)


class TestBuildCheckpointState:
  """Direct unit tests for _build_checkpoint_state."""

  def test_with_all_components(self, tmp_path: Path) -> None:
    """Checkpoint includes experiment, module, optimizer, callbacks keys."""
    experiment = Experiment(experiment_id='ckpt-test')
    experiment.start()
    module = NoopEvalModule()
    optimizer = NoOpOptimizer([Parameter()], lr=0.1)
    trainer = Trainer(
      experiment=experiment,
      callbacks=[CheckpointCallback(tmp_path / 'ck')],
    )
    trainer._module = module
    trainer._optimizer = optimizer
    state = trainer._build_checkpoint_state()
    assert 'experiment' in state
    assert 'module' in state
    assert 'optimizer' in state
    assert 'callbacks' in state
    assert state['experiment']['id'] == 'ckpt-test'
    assert state['optimizer']['defaults']['lr'] == 0.1

  def test_without_experiment(self) -> None:
    """No experiment key when trainer has no experiment."""
    trainer = Trainer()
    trainer._module = NoopEvalModule()
    state = trainer._build_checkpoint_state()
    assert 'experiment' not in state
    assert 'module' in state

  def test_without_optimizer(self) -> None:
    """No optimizer key when trainer has no optimizer."""
    trainer = Trainer()
    trainer._module = NoopEvalModule()
    state = trainer._build_checkpoint_state()
    assert 'optimizer' not in state


# Plan 05 §4.5 — Trainer: DataModule Stage + state_dict + test phase


class _StageSpy(DataModule):
  """Tracks which Stage values are passed to setup/teardown."""

  def __init__(self) -> None:
    self.setup_stages: list[Stage] = []
    self.teardown_stages: list[Stage] = []

  def setup(self, stage: Stage) -> None:
    self.setup_stages.append(stage)

  def teardown(self, stage: Stage) -> None:
    self.teardown_stages.append(stage)

  def train_dataloader(self) -> DataLoader:
    return DataLoader([EvalDatum(success=True)], batch_size=1)

  def val_dataloader(self) -> DataLoader:
    return DataLoader(ListDataset([]), batch_size=1)


def test_trainer_calls_datamodule_setup_teardown_with_stage():
  """Mock/spy datamodule records Stage enums on fit path."""
  dm = _StageSpy()
  mod = NoopEvalModule()
  trainer = Trainer()
  trainer.fit(mod, datamodule=dm, max_epochs=1)
  assert Stage.fit in dm.setup_stages
  assert Stage.fit in dm.teardown_stages


class _StatefulDataModule(DataModule):
  """DataModule with custom state for checkpoint tests."""

  def __init__(self) -> None:
    self.counter = 0
    self.split_seed = 42

  def state_dict(self) -> dict[str, Any]:
    return {'counter': self.counter, 'split_seed': self.split_seed}

  def load_state_dict(self, state: dict[str, Any]) -> None:
    self.counter = state['counter']
    self.split_seed = state['split_seed']

  def train_dataloader(self) -> DataLoader:
    return DataLoader([EvalDatum(success=True)], batch_size=1)

  def val_dataloader(self) -> DataLoader:
    return DataLoader(ListDataset([]), batch_size=1)


def test_save_checkpoint_includes_datamodule_state():
  """Checkpoint dict contains 'datamodule' key matching custom state."""
  dm = _StatefulDataModule()
  dm.counter = 7
  dm.split_seed = 99
  mod = NoopEvalModule()
  trainer = Trainer()
  trainer._module = mod
  trainer._datamodule = dm
  state = trainer._build_checkpoint_state()
  assert 'datamodule' in state
  assert state['datamodule']['counter'] == 7
  assert state['datamodule']['split_seed'] == 99


def test_load_checkpoint_restores_datamodule_state(tmp_path: Path):
  """Mutate -> save -> new trainer/datamodule -> load -> assert restored."""
  dm = _StatefulDataModule()
  dm.counter = 15
  dm.split_seed = 77

  mod = NoopEvalModule()
  trainer = Trainer()
  trainer._module = mod
  trainer._datamodule = dm
  ckpt_path = tmp_path / 'ckpt.json'
  trainer.save_checkpoint(ckpt_path)

  dm2 = _StatefulDataModule()
  assert dm2.counter == 0

  trainer2 = Trainer()
  trainer2._datamodule = dm2
  io = JSONCheckpointIO()
  state = io.load(ckpt_path)
  trainer2._restore_from_checkpoint(state, mod)

  assert dm2.counter == 15
  assert dm2.split_seed == 77


class _TestDataModule(DataModule):
  """DataModule that provides all three loaders plus stage tracking."""

  def __init__(self) -> None:
    self.setup_stages: list[Stage] = []
    self.teardown_stages: list[Stage] = []
    self.test_loader_called = False

  def setup(self, stage: Stage) -> None:
    self.setup_stages.append(stage)

  def teardown(self, stage: Stage) -> None:
    self.teardown_stages.append(stage)

  def train_dataloader(self) -> DataLoader:
    return DataLoader([EvalDatum(success=True)], batch_size=1)

  def val_dataloader(self) -> DataLoader:
    return DataLoader(ListDataset([]), batch_size=1)

  def test_dataloader(self) -> DataLoader:
    self.test_loader_called = True
    return DataLoader([EvalDatum(metadata={'test': True})], batch_size=1)


def test_fit_runs_test_phase_when_test_dataloaders_provided():
  """Assert test loop invoked when test_dataloaders is explicitly provided."""
  mod = NoopEvalModule()
  trainer = Trainer()
  test_dl = DataLoader([EvalDatum(metadata={'test': True})], batch_size=1)
  result = trainer.fit(
    mod,
    train_dataloaders=DataLoader([EvalDatum(success=True)], batch_size=1),
    max_epochs=1,
    test_dataloaders=test_dl,
  )
  assert 'test_results' in result
  assert isinstance(result['test_results'], dict)


def test_fit_runs_test_from_datamodule_test_dataloader():
  """Resolution: datamodule provides test loader."""
  dm = _TestDataModule()
  mod = NoopEvalModule()
  trainer = Trainer()
  result = trainer.fit(mod, datamodule=dm, max_epochs=1)
  assert 'test_results' in result
  assert Stage.test in dm.setup_stages
  assert Stage.test in dm.teardown_stages


def test_fit_test_explicit_overrides_datamodule():
  """When both provided, explicit test_dataloaders wins."""
  dm = _TestDataModule()
  mod = NoopEvalModule()
  explicit_dl = DataLoader([EvalDatum(metadata={'explicit': True})], batch_size=1)
  trainer = Trainer()
  result = trainer.fit(
    mod,
    datamodule=dm,
    max_epochs=1,
    test_dataloaders=explicit_dl,
  )
  assert 'test_results' in result
  assert not dm.test_loader_called


class _ValDataModule(DataModule):
  """DataModule with a non-empty val loader for Stage.validate testing."""

  def __init__(self) -> None:
    self.setup_stages: list[Stage] = []
    self.teardown_stages: list[Stage] = []

  def setup(self, stage: Stage) -> None:
    self.setup_stages.append(stage)

  def teardown(self, stage: Stage) -> None:
    self.teardown_stages.append(stage)

  def train_dataloader(self) -> DataLoader:
    return DataLoader([EvalDatum(success=True)], batch_size=1)

  def val_dataloader(self) -> DataLoader:
    return DataLoader([EvalDatum(success=True)], batch_size=1)

  def test_dataloader(self) -> DataLoader:
    raise NotImplementedError


def test_fit_calls_setup_stage_validate_before_validation():
  """Trainer calls datamodule.setup(Stage.validate) before the validation loop."""
  dm = _ValDataModule()

  class _ValMod(AutoPilotModule):
    def forward(self, *args, **kwargs):
      return EvalDatum(success=True)

    def training_step(self, batch, batch_idx):
      return EvalDatum(success=True)

    def validation_step(self, batch, batch_idx):
      return EvalDatum(success=True)

    def configure_optimizers(self):
      return None

  trainer = Trainer()
  trainer.fit(_ValMod(), datamodule=dm, max_epochs=1)
  assert Stage.fit in dm.setup_stages
  assert Stage.validate in dm.setup_stages


# Plan 02 §4.2 — _restore_path_parameter_files exception narrowing (BUG-004)


class _PathParamModule(AutoPilotModule):
  """Module with a PathParameter for restore tests."""

  def __init__(self, source: str):
    super().__init__()
    self.prompts = PathParameter(source=source)

  def forward(self, *args, **kwargs):
    return EvalDatum(success=True)

  def training_step(self, batch, batch_idx):
    return EvalDatum(success=True)

  def configure_optimizers(self):
    return None


def _restore_state(epoch: int = 2) -> dict[str, Any]:
  return {
    'experiment': {
      'id': 'exp-1',
      'epoch': epoch,
      'status': 'running',
      'hypothesis': None,
      'metrics': {},
      'notes': None,
      'created_at': '',
      'started_at': '',
      'completed_at': None,
      'failed_at': None,
      'cancelled_at': None,
      'error': None,
      'last_accepted_epoch': None,
      'strict_snapshot_after_complete': False,
    },
  }


class TestRestorePathParamsExceptionHandling:
  """BUG-004/BUG-A007: checkout errors propagate (no silent swallowing)."""

  def test_restore_path_parameters_propagates_store_error(self, tmp_path):
    source = tmp_path / 'src'
    source.mkdir()
    (source / 'file.txt').write_text('content')

    mock_store = MagicMock()
    mock_store.checkout.side_effect = StoreError('test error')
    exp = Experiment('exp-1')
    exp.start()
    trainer = Trainer(store=mock_store, experiment=exp)
    mod = _PathParamModule(str(source))
    trainer._module = mod
    with pytest.raises(StoreError, match='test error'):
      trainer._restore_path_parameter_files(_restore_state(), mod)

  def test_restore_path_parameters_propagates_os_error(self, tmp_path):
    source = tmp_path / 'src'
    source.mkdir()
    (source / 'file.txt').write_text('content')

    mock_store = MagicMock()
    mock_store.checkout.side_effect = OSError('disk full')
    exp = Experiment('exp-1')
    exp.start()
    trainer = Trainer(store=mock_store, experiment=exp)
    mod = _PathParamModule(str(source))
    trainer._module = mod
    with pytest.raises(OSError, match='disk full'):
      trainer._restore_path_parameter_files(_restore_state(), mod)

  def test_restore_path_parameters_skips_when_no_path_params(self):
    mock_store = MagicMock()
    exp = Experiment('exp-1')
    exp.start()
    trainer = Trainer(store=mock_store, experiment=exp)
    mod = NoopEvalModule()
    trainer._module = mod
    trainer._restore_path_parameter_files(_restore_state(), mod)
    mock_store.checkout.assert_not_called()

  def test_restore_path_parameters_skips_when_no_store(self, tmp_path):
    source = tmp_path / 'src'
    source.mkdir()
    (source / 'file.txt').write_text('content')

    exp = Experiment('exp-1')
    exp.start()
    trainer = Trainer(store=None, experiment=exp)
    mod = _PathParamModule(str(source))
    trainer._module = mod
    trainer._restore_path_parameter_files(_restore_state(), mod)

  def test_restore_path_parameters_skips_when_no_experiment(self, tmp_path):
    source = tmp_path / 'src'
    source.mkdir()
    (source / 'file.txt').write_text('content')

    mock_store = MagicMock()
    trainer = Trainer(store=mock_store, experiment=None)
    mod = _PathParamModule(str(source))
    trainer._module = mod
    trainer._restore_path_parameter_files(_restore_state(), mod)
    mock_store.checkout.assert_not_called()

  def test_restore_path_parameters_skips_when_no_epoch_in_state(self, tmp_path):
    source = tmp_path / 'src'
    source.mkdir()
    (source / 'file.txt').write_text('content')

    mock_store = MagicMock()
    exp = Experiment('exp-1')
    exp.start()
    trainer = Trainer(store=mock_store, experiment=exp)
    mod = _PathParamModule(str(source))
    trainer._module = mod
    state = {'experiment': {}}
    trainer._restore_path_parameter_files(state, mod)
    mock_store.checkout.assert_not_called()

  def test_restore_path_parameters_skips_when_negative_epoch(self, tmp_path):
    source = tmp_path / 'src'
    source.mkdir()
    (source / 'file.txt').write_text('content')

    mock_store = MagicMock()
    exp = Experiment('exp-1')
    exp.start()
    trainer = Trainer(store=mock_store, experiment=exp)
    mod = _PathParamModule(str(source))
    trainer._module = mod
    trainer._restore_path_parameter_files(_restore_state(epoch=-1), mod)
    mock_store.checkout.assert_not_called()


# Plan 02 §4.3 — _fit_failure_path logging (BUG-005)


class TestFitFailurePathLogging:
  """BUG-005: _fit_failure_path should log instead of silently suppressing."""

  def test_fit_failure_logs_experiment_error(self, caplog):
    exp = Experiment('fail-exp')
    exp.start()
    exp.complete()
    trainer = Trainer(experiment=exp)
    trainer._module = NoopEvalModule()
    with caplog.at_level(logging.WARNING):
      trainer._fit_failure_path(ValueError('training boom'))
    assert any('fail-exp' in r.message and 'fail()' in r.message for r in caplog.records)

  def test_fit_failure_logs_tree_update_error(self, caplog):
    exp = Experiment('tree-fail')
    exp.start()
    mock_tree = MagicMock(spec=Tree)
    mock_tree.update.side_effect = ValueError('node not found')
    trainer = Trainer(experiment=exp, tree=mock_tree)
    trainer._module = NoopEvalModule()
    with caplog.at_level(logging.WARNING):
      trainer._fit_failure_path(ValueError('training boom'))
    assert any('tree-fail' in r.message and 'tree update' in r.message for r in caplog.records)

  def test_fit_failure_original_exception_propagates(self):
    class _FailingModule(AutoPilotModule):
      def forward(self, *args, **kwargs):
        return EvalDatum(success=True)

      def training_step(self, batch, batch_idx):
        msg = 'training step exploded'
        raise ValueError(msg)

      def configure_optimizers(self):
        return None

    exp = Experiment('propagate-test')
    trainer = Trainer(experiment=exp)
    with pytest.raises(ValueError, match='training step exploded'):
      trainer.fit(
        _FailingModule(),
        train_dataloaders=DataLoader([EvalDatum(success=True)], batch_size=1),
        max_epochs=1,
      )


# Plan 02 §4.4 — datamodule property (BUG-006)


class TestDatamoduleProperty:
  """BUG-006: Trainer should expose a public datamodule property."""

  def test_datamodule_none_default(self):
    trainer = Trainer()
    assert trainer.datamodule is None

  def test_datamodule_set_during_fit(self):
    dm = _TinyDataModule()
    mod = NoopEvalModule()
    trainer = Trainer()
    trainer.fit(mod, datamodule=dm, max_epochs=1)
    assert trainer.datamodule is dm


# Plan 07 §2.2 — run_eval_phase empty/non-empty dataloader guard (BUG-015)


class TestRunEvalPhaseComputeGuard:
  """run_eval_phase returns {} for empty dataloaders, metrics for non-empty."""

  def test_run_eval_phase_empty_dataloader(self) -> None:
    module = NoopEvalModule()
    trainer = Trainer()
    trainer._module = module
    result = trainer.run_eval_phase(
      module,
      [],
      step_method='validation_step',
      hook_prefix='validation',
    )
    assert result == {}

  def test_run_eval_phase_with_data(self) -> None:
    module = NoopEvalModule()
    trainer = Trainer()
    trainer._module = module
    batches = [EvalDatum(success=True) for _ in range(3)]
    result = trainer.run_eval_phase(
      module,
      batches,
      step_method='validation_step',
      hook_prefix='validation',
    )
    assert isinstance(result, dict)
