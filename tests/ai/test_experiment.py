"""Tests for AutoPilotExperiment -- lifecycle hooks for Trainer integration."""

from autopilot.ai.experiment import AutoPilotExperiment
from autopilot.core.enums import Status
from autopilot.core.errors import ExperimentError
from autopilot.core.experiment import Experiment
from typing import Any
import pytest


class TestAutoPilotExperimentStart:
  def test_start_calls_on_start(self) -> None:
    calls: list[str] = []

    class Hooked(AutoPilotExperiment):
      def on_start(self) -> None:
        calls.append('on_start')

    exp = Hooked(experiment_id='t')
    exp.start()
    assert calls == ['on_start']

  def test_on_start_fires_after_running(self) -> None:
    status_at_hook: list[Status] = []

    class Hooked(AutoPilotExperiment):
      def on_start(self) -> None:
        status_at_hook.append(self.status)

    exp = Hooked(experiment_id='t')
    exp.start()
    assert status_at_hook == [Status.running]


class TestAutoPilotExperimentComplete:
  def test_complete_with_metrics(self) -> None:
    calls: list[str] = []

    class Hooked(AutoPilotExperiment):
      def on_complete(self) -> None:
        calls.append('on_complete')

    exp = Hooked(experiment_id='t')
    exp.start()
    exp.complete(metrics={'accuracy': 0.9})
    assert calls == ['on_complete']
    assert exp.metrics == {'accuracy': 0.9}

  def test_complete_without_metrics_calls_build_result(self) -> None:
    calls: list[str] = []

    class Hooked(AutoPilotExperiment):
      def build_result(self) -> dict[str, float]:
        calls.append('build_result')
        return {'computed': 1.0}

      def on_complete(self) -> None:
        calls.append('on_complete')

    exp = Hooked(experiment_id='t')
    exp.start()
    exp.complete()
    assert calls == ['build_result', 'on_complete']
    assert exp.metrics == {'computed': 1.0}

  def test_complete_with_metrics_skips_build_result(self) -> None:
    calls: list[str] = []

    class Hooked(AutoPilotExperiment):
      def build_result(self) -> dict[str, float]:
        calls.append('build_result')
        return {'computed': 1.0}

    exp = Hooked(experiment_id='t')
    exp.start()
    exp.complete(metrics={'provided': 2.0})
    assert 'build_result' not in calls
    assert exp.metrics == {'provided': 2.0}

  def test_on_complete_fires_after_completed(self) -> None:
    status_at_hook: list[Status] = []

    class Hooked(AutoPilotExperiment):
      def on_complete(self) -> None:
        status_at_hook.append(self.status)

    exp = Hooked(experiment_id='t')
    exp.start()
    exp.complete()
    assert status_at_hook == [Status.completed]


class TestAutoPilotExperimentFail:
  def test_fail_calls_on_fail(self) -> None:
    calls: list[tuple] = []

    class Hooked(AutoPilotExperiment):
      def on_fail(self, error: str | None) -> None:
        calls.append(('on_fail', error))

    exp = Hooked(experiment_id='t')
    exp.start()
    exp.fail(error='broken')
    assert calls == [('on_fail', 'broken')]

  def test_fail_without_error(self) -> None:
    calls: list[tuple] = []

    class Hooked(AutoPilotExperiment):
      def on_fail(self, error: str | None) -> None:
        calls.append(('on_fail', error))

    exp = Hooked(experiment_id='t')
    exp.start()
    exp.fail()
    assert calls == [('on_fail', None)]


class TestAutoPilotExperimentCancel:
  def test_cancel_calls_on_cancel(self) -> None:
    calls: list[str] = []

    class Hooked(AutoPilotExperiment):
      def on_cancel(self) -> None:
        calls.append('on_cancel')

    exp = Hooked(experiment_id='t')
    exp.cancel()
    assert calls == ['on_cancel']

  def test_cancel_from_running(self) -> None:
    calls: list[str] = []

    class Hooked(AutoPilotExperiment):
      def on_cancel(self) -> None:
        calls.append('on_cancel')

    exp = Hooked(experiment_id='t')
    exp.start()
    exp.cancel()
    assert calls == ['on_cancel']
    assert exp.status == Status.cancelled


class TestAutoPilotExperimentAdvanceEpoch:
  def test_advance_does_not_call_on_epoch_complete(self) -> None:
    """advance_epoch only increments epoch. EpochLoop is the single caller of on_epoch_complete."""
    calls: list[tuple] = []

    class Hooked(AutoPilotExperiment):
      def on_epoch_complete(self, epoch: int, metrics: dict[str, float], **kwargs) -> None:
        calls.append(('on_epoch_complete', epoch, metrics))

    exp = Hooked(experiment_id='t')
    exp.start()
    exp.advance_epoch(metrics={'accuracy': 0.7})
    assert calls == []
    assert exp.epoch == 0
    assert exp.metrics == {'accuracy': 0.7}

  def test_advance_epoch_increments_counter(self) -> None:
    exp = AutoPilotExperiment(experiment_id='t')
    exp.start()
    assert exp.epoch == -1
    exp.advance_epoch()
    assert exp.epoch == 0
    exp.advance_epoch()
    assert exp.epoch == 1
    exp.advance_epoch()
    assert exp.epoch == 2

  def test_advance_without_metrics_keeps_existing(self) -> None:
    exp = AutoPilotExperiment(experiment_id='t')
    exp.start()
    exp.advance_epoch(metrics={'accuracy': 0.7})
    assert exp.metrics == {'accuracy': 0.7}
    exp.advance_epoch()
    assert exp.epoch == 1


class TestAutoPilotExperimentValidation:
  def test_on_validation_complete_is_noop_by_default(self) -> None:
    exp = AutoPilotExperiment(experiment_id='t')
    exp.start()
    result = exp.on_validation_complete(0, {'accuracy': 0.8})
    assert result is None

  def test_on_validation_complete_override(self) -> None:
    calls: list[tuple] = []

    class Hooked(AutoPilotExperiment):
      def on_validation_complete(
        self,
        epoch: int,
        metrics: dict[str, float],
        **kwargs: Any,
      ) -> None:
        calls.append(('on_validation_complete', epoch, metrics))

    exp = Hooked(experiment_id='t')
    exp.start()
    exp.on_validation_complete(0, {'accuracy': 0.9})
    assert calls == [('on_validation_complete', 0, {'accuracy': 0.9})]


class TestOnLoopCompleteRemoved:
  def test_no_on_loop_complete_method(self) -> None:
    exp = AutoPilotExperiment(experiment_id='t')
    assert not hasattr(exp, 'on_loop_complete')


class TestBuildResult:
  def test_default_returns_self_metrics(self) -> None:
    exp = AutoPilotExperiment(experiment_id='t')
    exp.metrics = {'accuracy': 0.8}
    assert exp.build_result() == {'accuracy': 0.8}

  def test_default_returns_empty_dict_when_no_metrics(self) -> None:
    exp = AutoPilotExperiment(experiment_id='t')
    assert exp.build_result() == {}

  def test_overridable(self) -> None:
    class Custom(AutoPilotExperiment):
      def build_result(self) -> dict[str, float]:
        return {'custom': 99.0}

    exp = Custom(experiment_id='t')
    assert exp.build_result() == {'custom': 99.0}


class TestHookFiringOrder:
  def test_on_start_before_on_complete(self) -> None:
    """advance_epoch does not call on_epoch_complete; only EpochLoop does."""
    order: list[str] = []

    class Hooked(AutoPilotExperiment):
      def on_start(self) -> None:
        order.append('on_start')

      def on_complete(self) -> None:
        order.append('on_complete')

    exp = Hooked(experiment_id='t')
    exp.start()
    exp.advance_epoch()
    exp.advance_epoch()
    exp.complete()
    assert order == ['on_start', 'on_complete']

  def test_full_lifecycle_with_all_hooks(self) -> None:
    """on_epoch_complete and on_validation_complete are called by EpochLoop,
    not by advance_epoch. Direct calls still work for manual invocation."""
    order: list[str] = []

    class FullHooks(AutoPilotExperiment):
      def on_start(self) -> None:
        order.append('start')

      def on_epoch_complete(self, epoch: int, metrics: dict[str, float], **kwargs) -> None:
        order.append(f'epoch_{epoch}')

      def on_validation_complete(self, epoch: int, metrics: dict[str, float], **kwargs) -> None:
        order.append(f'val_{epoch}')

      def on_complete(self) -> None:
        order.append('complete')

      def build_result(self) -> dict[str, float]:
        order.append('build_result')
        return self.metrics

    exp = FullHooks(experiment_id='t')
    exp.start()
    exp.on_epoch_complete(0, {'acc': 0.7})
    exp.on_validation_complete(0, {'acc': 0.7})
    exp.advance_epoch(metrics={'acc': 0.7})
    exp.on_epoch_complete(1, {'acc': 0.8})
    exp.on_validation_complete(1, {'acc': 0.8})
    exp.advance_epoch(metrics={'acc': 0.8})
    exp.complete()

    assert order == [
      'start',
      'epoch_0',
      'val_0',
      'epoch_1',
      'val_1',
      'build_result',
      'complete',
    ]


class TestContextManagerCallsOnStartOnEnter:
  def test_context_manager_calls_on_start_on_enter(self) -> None:
    calls: list[str] = []

    class Hooked(AutoPilotExperiment):
      def on_start(self) -> None:
        calls.append('on_start')

    exp = Hooked(experiment_id='cm-start')
    with exp:
      pass
    assert calls == ['on_start']

  def test_context_manager_on_start_fires_once(self) -> None:
    call_count: list[int] = []

    class Hooked(AutoPilotExperiment):
      def on_start(self) -> None:
        call_count.append(1)

    exp = Hooked(experiment_id='cm-once')
    with exp:
      pass
    assert len(call_count) == 1


class TestContextManagerNormalExitCallsOnComplete:
  def test_context_manager_normal_exit_calls_on_complete(self) -> None:
    order: list[str] = []

    class Hooked(AutoPilotExperiment):
      def on_start(self) -> None:
        order.append('on_start')

      def on_complete(self) -> None:
        order.append('on_complete')

    exp = Hooked(experiment_id='cm-complete')
    with exp:
      pass
    assert order == ['on_start', 'on_complete']
    assert exp.status == Status.completed

  def test_context_manager_build_result_called_on_normal_exit(self) -> None:
    calls: list[str] = []

    class Hooked(AutoPilotExperiment):
      def build_result(self) -> dict[str, float]:
        calls.append('build_result')
        return {'from_build': 1.0}

    exp = Hooked(experiment_id='cm-build')
    with exp:
      pass
    assert 'build_result' in calls
    assert exp.metrics == {'from_build': 1.0}


class TestContextManagerExceptionExitCallsOnFail:
  def test_context_manager_exception_exit_calls_on_fail(self) -> None:
    fail_args: list[str | None] = []

    class Hooked(AutoPilotExperiment):
      def on_fail(self, error: str | None) -> None:
        fail_args.append(error)

    exp = Hooked(experiment_id='cm-fail')
    msg = 'kaboom'
    with pytest.raises(RuntimeError, match=msg), exp:
      raise RuntimeError(msg)
    assert fail_args == ['kaboom']
    assert exp.status == Status.failed

  def test_context_manager_exception_not_suppressed(self) -> None:
    exp = AutoPilotExperiment(experiment_id='cm-no-suppress')
    msg = 'test error'
    with pytest.raises(ValueError, match=msg), exp:
      raise ValueError(msg)

  def test_context_manager_exit_returns_false(self) -> None:
    exp = AutoPilotExperiment(experiment_id='cm-ret')
    exp.start()
    result = exp.__exit__(None, None, None)
    assert result is False


class TestAutoPilotSubclass:
  def test_subclass_with_all_hooks(self) -> None:
    """advance_epoch does not fire on_epoch_complete. Only EpochLoop does."""
    calls: list[str] = []

    class MyExperiment(AutoPilotExperiment):
      def __init__(self, experiment_id: str, store: Any = None) -> None:
        super().__init__(experiment_id=experiment_id)
        self.store = store

      def on_start(self) -> None:
        calls.append('start')

      def on_epoch_complete(self, epoch: int, metrics: dict[str, float], **kwargs) -> None:
        calls.append(f'epoch_{epoch}')

      def on_complete(self) -> None:
        calls.append('complete')

      def on_fail(self, error: str | None) -> None:
        calls.append(f'fail:{error}')

      def on_cancel(self) -> None:
        calls.append('cancel')

      def build_result(self) -> dict[str, float]:
        return {'result': 1.0}

    exp = MyExperiment(experiment_id='mine', store='fake')
    exp.start()
    exp.advance_epoch()
    exp.complete()
    assert calls == ['start', 'complete']
    assert exp.metrics == {'result': 1.0}

  def test_inherits_from_experiment(self) -> None:
    exp = AutoPilotExperiment(experiment_id='t')
    assert isinstance(exp, Experiment)

  def test_transition_errors_propagate(self) -> None:
    exp = AutoPilotExperiment(experiment_id='t')
    exp.start()
    exp.complete()
    with pytest.raises(ExperimentError, match='cannot complete'):
      exp.complete()

  def test_cancel_propagates_error_on_terminal(self) -> None:
    exp = AutoPilotExperiment(experiment_id='t')
    exp.start()
    exp.complete()
    with pytest.raises(ExperimentError, match='cannot cancel'):
      exp.cancel()
