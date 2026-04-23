"""Tests for Callback base class and Lightning-style hooks."""

from autopilot.core.callbacks.callback import Callback
from autopilot.core.module import Module
from autopilot.core.trainer import Trainer
from autopilot.core.types import Datum
from typing import Any


class _StubModule(Module):
  def forward(self, *args, **kwargs) -> Datum:
    return Datum(success=True)


class TestCallbackDefaults:
  def test_all_hooks_are_noop(self) -> None:
    cb = Callback()
    cb.on_fit_start(trainer=None)
    cb.on_fit_end(trainer=None)
    cb.on_train_epoch_start(trainer=None, epoch=1)
    cb.on_train_epoch_end(trainer=None, epoch=1)
    cb.on_validation_epoch_start(trainer=None, epoch=1)
    cb.on_validation_epoch_end(trainer=None, epoch=1)
    cb.on_test_epoch_start(trainer=None, epoch=1)
    cb.on_test_epoch_end(trainer=None, epoch=1)
    cb.on_epoch_start(trainer=None, epoch=1)
    cb.on_epoch_end(trainer=None, epoch=1)
    cb.on_loop_start(trainer=None, max_epochs=1)
    cb.on_loop_end(trainer=None, result={})
    cb.on_train_batch_start(trainer=None, batch_idx=0)
    cb.on_train_batch_end(trainer=None, batch_idx=0, data=None)
    cb.on_before_backward(trainer=None)
    cb.on_after_backward(trainer=None)
    cb.on_before_optimizer_step(trainer=None)
    cb.on_before_zero_grad(trainer=None)

  def test_no_status_transition_hook(self) -> None:
    cb = Callback()
    assert not hasattr(cb, 'on_status_transition')

  def test_no_experiment_created_hook(self) -> None:
    cb = Callback()
    assert not hasattr(cb, 'on_experiment_created')

  def test_no_result_computed_hook(self) -> None:
    cb = Callback()
    assert not hasattr(cb, 'on_result_computed')

  def test_no_policy_evaluated_hook(self) -> None:
    cb = Callback()
    assert not hasattr(cb, 'on_policy_evaluated')

  def test_state_dict_empty_by_default(self) -> None:
    assert Callback().state_dict() == {}

  def test_load_state_dict_is_noop(self) -> None:
    Callback().load_state_dict({'key': 'value'})


class TestCallbackDispatchInTrainer:
  def test_fit_dispatches_fit_start_end(self) -> None:
    events: list[str] = []

    class Track(Callback):
      def on_fit_start(self, trainer: Any) -> None:
        events.append('fit_start')

      def on_fit_end(self, trainer: Any) -> None:
        events.append('fit_end')

    mod = _StubModule()
    trainer = Trainer(callbacks=[Track()], dry_run=True)
    trainer.fit(mod, max_epochs=1)
    assert events == ['fit_start', 'fit_end']

  def test_fit_dispatches_loop_hooks(self) -> None:
    events: list[str] = []

    class Track(Callback):
      def on_loop_start(self, trainer: Any, max_epochs: int) -> None:
        events.append('loop_start')

      def on_loop_end(self, trainer: Any, result: dict) -> None:
        events.append('loop_end')

    mod = _StubModule()
    trainer = Trainer(callbacks=[Track()], dry_run=True)
    trainer.fit(mod, max_epochs=1)
    assert events == ['loop_start', 'loop_end']

  def test_fit_dispatches_epoch_hooks(self) -> None:
    epochs: list[tuple] = []

    class Track(Callback):
      def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        epochs.append(('start', epoch))

      def on_epoch_end(self, trainer: Any, epoch: int, result: Any = None) -> None:
        epochs.append(('end', epoch))

    mod = _StubModule()
    trainer = Trainer(callbacks=[Track()], dry_run=True)
    trainer.fit(mod, max_epochs=2)
    assert epochs == [('start', 1), ('end', 1), ('start', 2), ('end', 2)]

  def test_fit_order(self) -> None:
    events: list[str] = []

    class Track(Callback):
      def on_fit_start(self, trainer: Any) -> None:
        events.append('fit_start')

      def on_loop_start(self, trainer: Any, max_epochs: int) -> None:
        events.append('loop_start')

      def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        events.append(f'epoch_start_{epoch}')

      def on_epoch_end(self, trainer: Any, epoch: int, result: Any = None) -> None:
        events.append(f'epoch_end_{epoch}')

      def on_loop_end(self, trainer: Any, result: dict) -> None:
        events.append('loop_end')

      def on_fit_end(self, trainer: Any) -> None:
        events.append('fit_end')

    mod = _StubModule()
    trainer = Trainer(callbacks=[Track()], dry_run=True)
    trainer.fit(mod, max_epochs=1)
    assert events == [
      'fit_start',
      'loop_start',
      'epoch_start_1',
      'epoch_end_1',
      'loop_end',
      'fit_end',
    ]

  def test_multiple_callbacks(self) -> None:
    calls: list[str] = []

    class A(Callback):
      def on_fit_start(self, trainer: Any) -> None:
        calls.append('A')

    class B(Callback):
      def on_fit_start(self, trainer: Any) -> None:
        calls.append('B')

    mod = _StubModule()
    trainer = Trainer(callbacks=[A(), B()])
    trainer.fit(mod, max_epochs=0)
    assert calls == ['A', 'B']


class TestLightningStyleHooks:
  def test_train_epoch_hooks(self) -> None:
    events: list[str] = []

    class Track(Callback):
      def on_train_epoch_start(self, trainer: Any, epoch: int) -> None:
        events.append(f'train_start_{epoch}')

      def on_train_epoch_end(self, trainer: Any, epoch: int) -> None:
        events.append(f'train_end_{epoch}')

    trainer = Trainer(callbacks=[Track()])
    trainer._dispatch('on_train_epoch_start', epoch=1)
    trainer._dispatch('on_train_epoch_end', epoch=1)
    assert events == ['train_start_1', 'train_end_1']

  def test_validation_epoch_hooks(self) -> None:
    events: list[str] = []

    class Track(Callback):
      def on_validation_epoch_start(self, trainer: Any, epoch: int) -> None:
        events.append(f'val_start_{epoch}')

      def on_validation_epoch_end(self, trainer: Any, epoch: int) -> None:
        events.append(f'val_end_{epoch}')

    trainer = Trainer(callbacks=[Track()])
    trainer._dispatch('on_validation_epoch_start', epoch=1)
    trainer._dispatch('on_validation_epoch_end', epoch=1)
    assert events == ['val_start_1', 'val_end_1']

  def test_test_epoch_hooks(self) -> None:
    events: list[str] = []

    class Track(Callback):
      def on_test_epoch_start(self, trainer: Any, epoch: int) -> None:
        events.append(f'test_start_{epoch}')

      def on_test_epoch_end(self, trainer: Any, epoch: int) -> None:
        events.append(f'test_end_{epoch}')

    trainer = Trainer(callbacks=[Track()])
    trainer._dispatch('on_test_epoch_start', epoch=1)
    trainer._dispatch('on_test_epoch_end', epoch=1)
    assert events == ['test_start_1', 'test_end_1']
