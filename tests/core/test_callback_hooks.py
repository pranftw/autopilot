"""Tests for Plan 12: Callback lifecycle hooks.

Covers setup/teardown, on_exception, on_save_checkpoint, on_load_checkpoint,
and the new hook stubs (sanity, batch, predict). Verifies Trainer wiring
for all dispatched hooks and inheritance by subclasses.
"""

from autopilot.core.callbacks.callback import Callback
from autopilot.core.checkpoint import CheckpointIO
from autopilot.core.trainer.trainer import Trainer
from autopilot.data.datamodule import Stage
from pathlib import Path
from tests.doubles import NoopEvalModule
from typing import Any
from unittest.mock import patch
import pytest


class RecordingCallback(Callback):
  """Callback that records all Plan 12 hook invocations with arguments."""

  def __init__(self) -> None:
    self.events: list[str] = []
    self.last_stage: Stage | None = None
    self.last_exception: BaseException | None = None
    self.last_save_checkpoint: dict[str, Any] | None = None
    self.last_load_checkpoint: dict[str, Any] | None = None

  def setup(self, trainer: Any, module: Any, stage: Stage) -> None:
    self.events.append(f'setup:{stage.value}')
    self.last_stage = stage

  def teardown(self, trainer: Any, module: Any, stage: Stage) -> None:
    self.events.append(f'teardown:{stage.value}')

  def on_exception(self, trainer: Any, module: Any, exception: BaseException) -> None:
    self.events.append('on_exception')
    self.last_exception = exception

  def on_save_checkpoint(self, trainer: Any, module: Any, checkpoint: dict[str, Any]) -> None:
    self.events.append('on_save_checkpoint')
    self.last_save_checkpoint = checkpoint

  def on_load_checkpoint(self, trainer: Any, module: Any, checkpoint: dict[str, Any]) -> None:
    self.events.append('on_load_checkpoint')
    self.last_load_checkpoint = checkpoint


class CapturingCheckpointIO(CheckpointIO):
  """CheckpointIO that captures the saved dict instead of writing to disk."""

  def __init__(self) -> None:
    self.saved_state: dict[str, Any] | None = None

  def save(self, state: dict[str, Any], path: Path) -> None:
    self.saved_state = state

  def load(self, path: Path) -> dict[str, Any]:
    if self.saved_state is None:
      msg = 'no checkpoint was saved'
      raise FileNotFoundError(msg)
    return self.saved_state

  def remove(self, path: Path) -> None:
    self.saved_state = None

  def exists(self, path: Path) -> bool:
    return self.saved_state is not None


class TestCallbackSetupTeardown:
  """Tests for setup/teardown dispatch during Trainer.fit."""

  def test_setup_called_with_stage_fit(self) -> None:
    cb = RecordingCallback()
    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[cb])
    trainer.fit(mod, max_epochs=1)
    assert cb.last_stage is Stage.fit
    assert 'setup:fit' in cb.events

  def test_teardown_called_after_fit(self) -> None:
    cb = RecordingCallback()
    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[cb])
    trainer.fit(mod, max_epochs=1)
    assert 'teardown:fit' in cb.events

  def test_teardown_called_on_exception(self) -> None:
    cb = RecordingCallback()

    class FailingModule(NoopEvalModule):
      def training_step(self, batch, batch_idx):
        msg = 'boom'
        raise ValueError(msg)

    mod = FailingModule()
    trainer = Trainer(callbacks=[cb])
    with pytest.raises(ValueError, match='boom'):
      trainer.fit(mod, max_epochs=1, train_dataloaders=[1])
    assert 'teardown:fit' in cb.events

  def test_setup_before_on_fit_start(self) -> None:
    events: list[str] = []

    class OrderTracker(Callback):
      def setup(self, trainer: Any, module: Any, stage: Stage) -> None:
        events.append('setup')

      def on_fit_start(self, trainer: Any, module: Any) -> None:
        events.append('on_fit_start')

    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[OrderTracker()])
    trainer.fit(mod, max_epochs=0)
    assert events.index('setup') < events.index('on_fit_start')

  def test_teardown_after_module_teardown(self) -> None:
    events: list[str] = []

    class TrackingModule(NoopEvalModule):
      def teardown(self) -> None:
        events.append('module_teardown')

    class TrackingCb(Callback):
      def teardown(self, trainer: Any, module: Any, stage: Stage) -> None:
        events.append('callback_teardown')

    mod = TrackingModule()
    trainer = Trainer(callbacks=[TrackingCb()])
    trainer.fit(mod, max_epochs=0)
    assert events.index('module_teardown') < events.index('callback_teardown')


class TestOnException:
  """Tests for on_exception dispatch during Trainer.fit exception path."""

  def test_on_exception_fires_before_failure_path(self) -> None:
    sequence: list[str] = []

    class OrderCallback(Callback):
      def on_exception(self, trainer: Any, module: Any, exception: BaseException) -> None:
        sequence.append('on_exception')

    class FailingModule(NoopEvalModule):
      def training_step(self, batch, batch_idx):
        msg = 'test error'
        raise ValueError(msg)

    original_failure_path = Trainer._fit_failure_path

    def tracking_failure_path(self_t, exc):
      sequence.append('failure_path')
      return original_failure_path(self_t, exc)

    mod = FailingModule()
    trainer = Trainer(callbacks=[OrderCallback()])

    with (
      patch.object(Trainer, '_fit_failure_path', tracking_failure_path),
      pytest.raises(ValueError, match='test error'),
    ):
      trainer.fit(mod, max_epochs=1, train_dataloaders=[1])
    assert sequence == ['on_exception', 'failure_path']

  def test_on_exception_receives_exception(self) -> None:
    cb = RecordingCallback()

    class FailingModule(NoopEvalModule):
      def training_step(self, batch, batch_idx):
        msg = 'specific error'
        raise ValueError(msg)

    mod = FailingModule()
    trainer = Trainer(callbacks=[cb])
    with pytest.raises(ValueError, match='specific error'):
      trainer.fit(mod, max_epochs=1, train_dataloaders=[1])
    assert cb.last_exception is not None
    assert isinstance(cb.last_exception, ValueError)
    assert str(cb.last_exception) == 'specific error'

  def test_on_exception_propagation(self) -> None:
    """An exception raised inside on_exception propagates naturally."""

    class RaisingCallback(Callback):
      def on_exception(self, trainer: Any, module: Any, exception: BaseException) -> None:
        msg = 'callback error'
        raise RuntimeError(msg)

    class FailingModule(NoopEvalModule):
      def training_step(self, batch, batch_idx):
        msg = 'original error'
        raise ValueError(msg)

    mod = FailingModule()
    trainer = Trainer(callbacks=[RaisingCallback()])
    with pytest.raises(RuntimeError, match='callback error'):
      trainer.fit(mod, max_epochs=1, train_dataloaders=[1])

  def test_multiple_callbacks_all_receive_exception(self) -> None:
    counters = [0, 0]

    class CountA(Callback):
      def on_exception(self, trainer: Any, module: Any, exception: BaseException) -> None:
        counters[0] += 1

    class CountB(Callback):
      def on_exception(self, trainer: Any, module: Any, exception: BaseException) -> None:
        counters[1] += 1

    class FailingModule(NoopEvalModule):
      def training_step(self, batch, batch_idx):
        msg = 'boom'
        raise ValueError(msg)

    mod = FailingModule()
    trainer = Trainer(callbacks=[CountA(), CountB()])
    with pytest.raises(ValueError, match='boom'):
      trainer.fit(mod, max_epochs=1, train_dataloaders=[1])
    assert counters == [1, 1]


class TestOnSaveCheckpoint:
  """Tests for on_save_checkpoint dispatch during Trainer.save_checkpoint."""

  def test_on_save_checkpoint_mutates_dict(self) -> None:
    class MarkerCallback(Callback):
      def on_save_checkpoint(self, trainer: Any, module: Any, checkpoint: dict[str, Any]) -> None:
        checkpoint['marker'] = 42

    io = CapturingCheckpointIO()
    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[MarkerCallback()])
    trainer.fit(mod, max_epochs=0)
    trainer.save_checkpoint(Path('/tmp/ckpt.json'), checkpoint_io=io)
    assert io.saved_state is not None
    assert io.saved_state['marker'] == 42

  def test_on_save_checkpoint_sees_all_standard_keys(self) -> None:
    seen_keys: set[str] = set()

    class InspectorCallback(Callback):
      def on_save_checkpoint(self, trainer: Any, module: Any, checkpoint: dict[str, Any]) -> None:
        seen_keys.update(checkpoint.keys())

    io = CapturingCheckpointIO()
    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[InspectorCallback()])
    trainer.fit(mod, max_epochs=0)
    trainer.save_checkpoint(Path('/tmp/ckpt.json'), checkpoint_io=io)
    assert 'module' in seen_keys

  def test_on_save_checkpoint_raising_propagates(self) -> None:
    class BrokenCallback(Callback):
      def on_save_checkpoint(self, trainer: Any, module: Any, checkpoint: dict[str, Any]) -> None:
        msg = 'save hook failed'
        raise RuntimeError(msg)

    io = CapturingCheckpointIO()
    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[BrokenCallback()])
    trainer.fit(mod, max_epochs=0)
    with pytest.raises(RuntimeError, match='save hook failed'):
      trainer.save_checkpoint(Path('/tmp/ckpt.json'), checkpoint_io=io)


class TestOnLoadCheckpoint:
  """Tests for on_load_checkpoint dispatch during checkpoint resume."""

  def test_on_load_checkpoint_receives_dict(self) -> None:
    cb = RecordingCallback()
    io = CapturingCheckpointIO()
    mod = NoopEvalModule()

    trainer = Trainer(callbacks=[cb])
    trainer.fit(mod, max_epochs=0)
    trainer.save_checkpoint(Path('/tmp/ckpt.json'), checkpoint_io=io)

    loaded_ids: list[int] = []

    class IdTracker(Callback):
      def on_load_checkpoint(self, trainer: Any, module: Any, checkpoint: dict[str, Any]) -> None:
        loaded_ids.append(id(checkpoint))

    mod2 = NoopEvalModule()
    tracker = IdTracker()
    trainer2 = Trainer(callbacks=[tracker])
    trainer2.fit(
      mod2,
      max_epochs=0,
      ckpt_path=Path('/tmp/ckpt.json'),
      checkpoint_io=io,
    )
    assert len(loaded_ids) == 1
    assert loaded_ids[0] == id(io.saved_state)

  def test_on_load_checkpoint_called_before_restore(self) -> None:
    order: list[str] = []

    class TrackLoadCallback(Callback):
      def on_load_checkpoint(self, trainer: Any, module: Any, checkpoint: dict[str, Any]) -> None:
        order.append('on_load_checkpoint')

    io = CapturingCheckpointIO()
    mod = NoopEvalModule()
    trainer = Trainer()
    trainer.fit(mod, max_epochs=0)
    trainer.save_checkpoint(Path('/tmp/ckpt.json'), checkpoint_io=io)

    mod2 = NoopEvalModule()
    original_restore = Trainer._restore_from_checkpoint

    def tracking_restore(self, state, module):
      order.append('restore')
      return original_restore(self, state, module)

    trainer2 = Trainer(callbacks=[TrackLoadCallback()])
    with patch.object(Trainer, '_restore_from_checkpoint', tracking_restore):
      trainer2.fit(
        mod2,
        max_epochs=0,
        ckpt_path=Path('/tmp/ckpt.json'),
        checkpoint_io=io,
      )
    assert order == ['on_load_checkpoint', 'restore']


class TestMultipleCallbacks:
  """Tests for dispatch to multiple callbacks."""

  def test_multiple_callbacks_all_called(self) -> None:
    counters = [0, 0]

    class A(Callback):
      def setup(self, trainer: Any, module: Any, stage: Stage) -> None:
        counters[0] += 1

    class B(Callback):
      def setup(self, trainer: Any, module: Any, stage: Stage) -> None:
        counters[1] += 1

    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[A(), B()])
    trainer.fit(mod, max_epochs=0)
    assert counters == [1, 1]

  def test_multiple_callbacks_ordered(self) -> None:
    events: list[str] = []

    class First(Callback):
      def setup(self, trainer: Any, module: Any, stage: Stage) -> None:
        events.append('first_setup')

      def teardown(self, trainer: Any, module: Any, stage: Stage) -> None:
        events.append('first_teardown')

    class Second(Callback):
      def setup(self, trainer: Any, module: Any, stage: Stage) -> None:
        events.append('second_setup')

      def teardown(self, trainer: Any, module: Any, stage: Stage) -> None:
        events.append('second_teardown')

    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[First(), Second()])
    trainer.fit(mod, max_epochs=0)
    assert events == ['first_setup', 'second_setup', 'first_teardown', 'second_teardown']


class TestFullLifecycleOrder:
  """Tests for correct ordering of all Plan 12 hooks in a full fit."""

  def test_fit_lifecycle_order(self) -> None:
    events: list[str] = []

    class FullTracker(Callback):
      def setup(self, trainer: Any, module: Any, stage: Stage) -> None:
        events.append('setup')

      def on_fit_start(self, trainer: Any, module: Any) -> None:
        events.append('on_fit_start')

      def on_fit_end(self, trainer: Any, module: Any) -> None:
        events.append('on_fit_end')

      def teardown(self, trainer: Any, module: Any, stage: Stage) -> None:
        events.append('teardown')

    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[FullTracker()])
    trainer.fit(mod, max_epochs=1)
    assert events == ['setup', 'on_fit_start', 'on_fit_end', 'teardown']

  def test_exception_lifecycle_order(self) -> None:
    events: list[str] = []

    class FullTracker(Callback):
      def setup(self, trainer: Any, module: Any, stage: Stage) -> None:
        events.append('setup')

      def on_exception(self, trainer: Any, module: Any, exception: BaseException) -> None:
        events.append('on_exception')

      def teardown(self, trainer: Any, module: Any, stage: Stage) -> None:
        events.append('teardown')

    class FailingModule(NoopEvalModule):
      def training_step(self, batch, batch_idx):
        msg = 'fail'
        raise ValueError(msg)

    mod = FailingModule()
    trainer = Trainer(callbacks=[FullTracker()])
    with pytest.raises(ValueError, match='fail'):
      trainer.fit(mod, max_epochs=1, train_dataloaders=[1])
    assert events == ['setup', 'on_exception', 'teardown']


class TestStubHooksCallable:
  """Verify that all new hook stubs are callable as no-ops."""

  def test_sanity_check_stubs(self) -> None:
    cb = Callback()
    cb.on_sanity_check_start(trainer=None, module=None)
    cb.on_sanity_check_end(trainer=None, module=None)

  def test_validation_batch_stubs(self) -> None:
    cb = Callback()
    cb.on_validation_batch_start(trainer=None, module=None, batch=None, batch_idx=0)
    cb.on_validation_batch_end(trainer=None, module=None, batch=None, batch_idx=0)

  def test_test_batch_stubs(self) -> None:
    cb = Callback()
    cb.on_test_batch_start(trainer=None, module=None, batch=None, batch_idx=0)
    cb.on_test_batch_end(trainer=None, module=None, batch=None, batch_idx=0)

  def test_predict_stubs(self) -> None:
    cb = Callback()
    cb.on_predict_start(trainer=None, module=None)
    cb.on_predict_end(trainer=None, module=None)
    cb.on_predict_batch_start(trainer=None, module=None, batch=None, batch_idx=0)
    cb.on_predict_batch_end(trainer=None, module=None, batch=None, batch_idx=0)

  def test_lifecycle_stubs(self) -> None:
    cb = Callback()
    cb.setup(trainer=None, module=None, stage=Stage.fit)
    cb.teardown(trainer=None, module=None, stage=Stage.fit)
    cb.on_exception(trainer=None, module=None, exception=RuntimeError('test'))

  def test_checkpoint_stubs(self) -> None:
    cb = Callback()
    cb.on_save_checkpoint(trainer=None, module=None, checkpoint={})
    cb.on_load_checkpoint(trainer=None, module=None, checkpoint={})


class TestSubclassOverrideReceivesDispatch:
  """Verify that subclass overrides receive dispatch from Trainer."""

  def test_subclass_setup_receives_dispatch(self) -> None:
    received: list[Stage] = []

    class MyCallback(Callback):
      def setup(self, trainer: Any, module: Any, stage: Stage) -> None:
        received.append(stage)

    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[MyCallback()])
    trainer.fit(mod, max_epochs=0)
    assert received == [Stage.fit]

  def test_subclass_teardown_receives_dispatch(self) -> None:
    received: list[Stage] = []

    class MyCallback(Callback):
      def teardown(self, trainer: Any, module: Any, stage: Stage) -> None:
        received.append(stage)

    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[MyCallback()])
    trainer.fit(mod, max_epochs=0)
    assert received == [Stage.fit]

  def test_subclass_on_save_checkpoint_receives_dispatch(self) -> None:
    received: list[dict] = []

    class MyCallback(Callback):
      def on_save_checkpoint(self, trainer: Any, module: Any, checkpoint: dict[str, Any]) -> None:
        received.append(dict(checkpoint))

    io = CapturingCheckpointIO()
    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[MyCallback()])
    trainer.fit(mod, max_epochs=0)
    trainer.save_checkpoint(Path('/tmp/ckpt.json'), checkpoint_io=io)
    assert len(received) == 1
    assert 'module' in received[0]


class TestStageValidateNotDispatchedDuringFitOnly:
  """Verify that Stage.validate is not dispatched during fit-only runs."""

  def test_only_fit_stage_dispatched(self) -> None:
    stages: list[Stage] = []

    class StageTracker(Callback):
      def setup(self, trainer: Any, module: Any, stage: Stage) -> None:
        stages.append(stage)

      def teardown(self, trainer: Any, module: Any, stage: Stage) -> None:
        stages.append(stage)

    mod = NoopEvalModule()
    trainer = Trainer(callbacks=[StageTracker()])
    trainer.fit(mod, max_epochs=1)
    assert all(s is Stage.fit for s in stages)
    assert Stage.validate not in stages
