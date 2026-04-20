"""Tests for Trainer construction, callback dispatch, and fit()."""

from autopilot.core.callbacks import Callback
from autopilot.core.logger import Logger
from autopilot.core.models import Datum
from autopilot.core.module import Module
from autopilot.core.trainer import Trainer


class _StubModule(Module):
  def forward(self, *args, **kwargs) -> Datum:
    return Datum(success=True)


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
    assert not hasattr(trainer, 'run') or not callable(getattr(trainer, 'run', None))

  def test_fit_sets_module_ref(self) -> None:
    mod = _StubModule()
    trainer = Trainer()
    trainer.fit(mod, max_epochs=1)
    assert trainer.module is mod


class TestTrainerCallbackDispatch:
  def test_dispatch_invokes_all_callbacks(self) -> None:
    calls: list[str] = []

    class A(Callback):
      def on_epoch_start(self, trainer: object, epoch: int) -> None:
        calls.append('A')

    class B(Callback):
      def on_epoch_start(self, trainer: object, epoch: int) -> None:
        calls.append('B')

    trainer = Trainer(callbacks=[A(), B()])
    trainer.on_epoch_start(1)
    assert calls == ['A', 'B']

  def test_dispatch_skips_unimplemented_hooks(self) -> None:
    trainer = Trainer(callbacks=[Callback()])
    trainer.on_epoch_start(1)

  def test_dispatch_unknown_hook_is_safe(self) -> None:
    trainer = Trainer(callbacks=[Callback()])
    trainer._dispatch('on_nonexistent_hook', x=1)


class TestTrainerFit:
  def test_fit_invokes_loop_hooks(self) -> None:
    events: list[str] = []

    class Track(Callback):
      def on_loop_start(self, trainer: object, max_epochs: int) -> None:
        events.append('loop_start')

      def on_loop_end(self, trainer: object, result: dict) -> None:
        events.append('loop_end')

    mod = _StubModule()
    trainer = Trainer(callbacks=[Track()], dry_run=True)
    trainer.fit(mod, max_epochs=2)
    assert 'loop_start' in events
    assert 'loop_end' in events


class TestTrainerRunLoopViaFit:
  def test_dispatches_epoch_hooks(self) -> None:
    epochs: list[tuple] = []

    class EpochTracker(Callback):
      def on_epoch_start(self, trainer: object, epoch: int) -> None:
        epochs.append(('start', epoch))

      def on_epoch_end(self, trainer: object, epoch: int, result: object) -> None:
        epochs.append(('end', epoch))

    mod = _StubModule()
    trainer = Trainer(callbacks=[EpochTracker()], dry_run=True)
    result = trainer.fit(mod, max_epochs=2)
    assert result['total_epochs'] == 2
    assert epochs == [('start', 1), ('end', 1), ('start', 2), ('end', 2)]
