"""Tests for new base Experiment entity with lifecycle state transitions."""

from autopilot.core.enums import Status
from autopilot.core.errors import ExperimentError
from autopilot.core.experiment import Experiment
from typing import Any
import pytest


class TestExperimentCreation:
  def test_create_with_id(self) -> None:
    exp = Experiment(experiment_id='test-1')
    assert exp.id == 'test-1'

  def test_create_with_hypothesis(self) -> None:
    exp = Experiment(experiment_id='test-1', hypothesis='test hypothesis')
    assert exp.hypothesis == 'test hypothesis'

  def test_hypothesis_default_none(self) -> None:
    exp = Experiment(experiment_id='test-1')
    assert exp.hypothesis is None

  def test_default_status_pending(self) -> None:
    exp = Experiment(experiment_id='test-1')
    assert exp.status == Status.pending

  def test_default_metrics_empty(self) -> None:
    exp = Experiment(experiment_id='test-1')
    assert exp.metrics == {}

  def test_default_epoch_minus_one(self) -> None:
    exp = Experiment(experiment_id='test-1')
    assert exp.epoch == -1

  def test_created_at_set_automatically(self) -> None:
    exp = Experiment(experiment_id='test-1')
    assert exp.created_at is not None
    assert 'T' in exp.created_at

  def test_default_notes_none(self) -> None:
    exp = Experiment(experiment_id='test-1')
    assert exp.notes is None

  def test_default_error_none(self) -> None:
    exp = Experiment(experiment_id='test-1')
    assert exp.error is None

  def test_default_timestamps_none(self) -> None:
    exp = Experiment(experiment_id='test-1')
    assert exp.started_at is None
    assert exp.completed_at is None
    assert exp.failed_at is None
    assert exp.cancelled_at is None

  def test_repr(self) -> None:
    exp = Experiment(experiment_id='test-1')
    r = repr(exp)
    assert 'test-1' in r
    assert 'pending' in r


class TestIsTerminal:
  def test_pending_not_terminal(self) -> None:
    exp = Experiment(experiment_id='t')
    assert exp.is_terminal is False

  def test_running_not_terminal(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    assert exp.is_terminal is False

  def test_completed_is_terminal(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.complete()
    assert exp.is_terminal is True

  def test_failed_is_terminal(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.fail()
    assert exp.is_terminal is True

  def test_cancelled_is_terminal(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.cancel()
    assert exp.is_terminal is True


class TestStartTransition:
  def test_start_from_pending(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    assert exp.status == Status.running

  def test_start_sets_started_at(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    assert exp.started_at is not None

  def test_start_from_running_raises(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    with pytest.raises(ExperimentError, match='cannot start'):
      exp.start()

  def test_start_from_completed_raises(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.complete()
    with pytest.raises(ExperimentError, match='cannot start'):
      exp.start()

  def test_start_from_failed_raises(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.fail()
    with pytest.raises(ExperimentError, match='cannot start'):
      exp.start()

  def test_start_from_cancelled_raises(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.cancel()
    with pytest.raises(ExperimentError, match='cannot start'):
      exp.start()


class TestCompleteTransition:
  def test_complete_from_running(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.complete()
    assert exp.status == Status.completed

  def test_complete_sets_completed_at(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.complete()
    assert exp.completed_at is not None

  def test_complete_with_metrics(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.complete(metrics={'accuracy': 0.9})
    assert exp.metrics == {'accuracy': 0.9}

  def test_complete_without_metrics_preserves_existing(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.metrics = {'loss': 0.1}
    exp.complete()
    assert exp.metrics == {'loss': 0.1}

  def test_complete_from_pending_succeeds(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.complete({'x': 1.0})
    assert exp.status == Status.completed
    assert exp.metrics == {'x': 1.0}

  def test_complete_from_completed_raises(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.complete()
    with pytest.raises(ExperimentError, match='cannot complete'):
      exp.complete()

  def test_complete_from_failed_raises(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.fail()
    with pytest.raises(ExperimentError, match='cannot complete'):
      exp.complete()

  def test_complete_from_cancelled_raises(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.cancel()
    with pytest.raises(ExperimentError, match='cannot complete'):
      exp.complete()


class TestFailTransition:
  def test_fail_from_running(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.fail()
    assert exp.status == Status.failed

  def test_fail_sets_failed_at(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.fail()
    assert exp.failed_at is not None

  def test_fail_with_error(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.fail(error='something broke')
    assert exp.error == 'something broke'

  def test_fail_without_error_leaves_none(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.fail()
    assert exp.error is None

  def test_fail_from_pending_succeeds(self) -> None:
    """BUG-DFV1-002: fail() accepts pending status for CLI-only workflows."""
    exp = Experiment(experiment_id='t')
    exp.fail('test reason')
    assert exp.status == Status.failed
    assert exp.error == 'test reason'
    assert exp.failed_at is not None

  def test_fail_from_pending_stores_error_and_timestamp(self) -> None:
    """BUG-DFV1-002: fail from pending sets error and failed_at correctly."""
    exp = Experiment(experiment_id='t')
    exp.fail('data was corrupted')
    assert exp.error == 'data was corrupted'
    assert exp.failed_at is not None
    assert exp.started_at is None

  def test_fail_from_completed_raises(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.complete()
    with pytest.raises(ExperimentError, match='cannot fail'):
      exp.fail()

  def test_fail_from_failed_raises(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.fail()
    with pytest.raises(ExperimentError, match='cannot fail'):
      exp.fail()

  def test_fail_from_cancelled_raises(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.cancel()
    with pytest.raises(ExperimentError, match='cannot fail'):
      exp.fail()


class TestCancelTransition:
  def test_cancel_from_pending(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.cancel()
    assert exp.status == Status.cancelled

  def test_cancel_from_running(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.cancel()
    assert exp.status == Status.cancelled

  def test_cancel_sets_cancelled_at(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.cancel()
    assert exp.cancelled_at is not None

  def test_cancel_from_completed_raises(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.complete()
    with pytest.raises(ExperimentError, match='cannot cancel'):
      exp.cancel()

  def test_cancel_from_failed_raises(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.fail()
    with pytest.raises(ExperimentError, match='cannot cancel'):
      exp.cancel()

  def test_cancel_from_cancelled_raises(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.cancel()
    with pytest.raises(ExperimentError, match='cannot cancel'):
      exp.cancel()


class TestAdvanceEpoch:
  def test_advance_increments_epoch(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.advance_epoch()
    assert exp.epoch == 0

  def test_advance_multiple(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.advance_epoch()
    exp.advance_epoch()
    exp.advance_epoch()
    assert exp.epoch == 2

  def test_advance_with_metrics(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.advance_epoch(metrics={'accuracy': 0.7})
    assert exp.metrics == {'accuracy': 0.7}

  def test_advance_without_metrics_preserves_existing(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.metrics = {'loss': 0.5}
    exp.advance_epoch()
    assert exp.metrics == {'loss': 0.5}

  def test_advance_from_pending_raises(self) -> None:
    exp = Experiment(experiment_id='t')
    with pytest.raises(ExperimentError, match='cannot advance epoch'):
      exp.advance_epoch()

  def test_advance_from_completed_raises(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.complete()
    with pytest.raises(ExperimentError, match='cannot advance epoch'):
      exp.advance_epoch()

  def test_advance_from_failed_raises(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.fail()
    with pytest.raises(ExperimentError, match='cannot advance epoch'):
      exp.advance_epoch()

  def test_advance_from_cancelled_raises(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.cancel()
    with pytest.raises(ExperimentError, match='cannot advance epoch'):
      exp.advance_epoch()


class TestOnLoopCompleteRemoved:
  def test_no_on_loop_complete_method(self) -> None:
    exp = Experiment(experiment_id='t')
    assert not hasattr(exp, 'on_loop_complete')


class TestStateDict:
  def test_state_dict_contains_all_fields(self) -> None:
    exp = Experiment(experiment_id='test-1', hypothesis='h')
    state = exp.state_dict()
    assert state['id'] == 'test-1'
    assert state['hypothesis'] == 'h'
    assert state['status'] == 'pending'
    assert state['metrics'] == {}
    assert state['notes'] is None
    assert state['epoch'] == -1
    assert state['error'] is None
    assert state['created_at'] is not None
    assert state['started_at'] is None
    assert state['completed_at'] is None
    assert state['failed_at'] is None
    assert state['cancelled_at'] is None
    assert state['last_accepted_epoch'] is None

  def test_state_dict_status_as_string(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    state = exp.state_dict()
    assert state['status'] == 'running'
    assert isinstance(state['status'], str)

  def test_load_state_dict_restores_all(self) -> None:
    exp = Experiment(experiment_id='t', hypothesis='h')
    exp.start()
    exp.advance_epoch(metrics={'accuracy': 0.8})
    exp.notes = 'some note'
    state = exp.state_dict()

    exp2 = Experiment(experiment_id='placeholder')
    exp2.load_state_dict(state)
    assert exp2.id == 't'
    assert exp2.hypothesis == 'h'
    assert exp2.status == Status.running
    assert exp2.metrics == {'accuracy': 0.8}
    assert exp2.notes == 'some note'
    assert exp2.epoch == 0
    assert exp2.started_at == exp.started_at

  def test_round_trip(self) -> None:
    exp = Experiment(experiment_id='round-trip', hypothesis='test')
    exp.start()
    exp.advance_epoch(metrics={'loss': 0.3})
    exp.complete(metrics={'accuracy': 0.95, 'loss': 0.2})
    state = exp.state_dict()

    exp2 = Experiment(experiment_id='temp')
    exp2.load_state_dict(state)
    assert exp2.state_dict() == state

  def test_status_survives_round_trip(self) -> None:
    for status_method in ['start', 'cancel']:
      exp = Experiment(experiment_id='t')
      if status_method == 'start':
        exp.start()
      else:
        exp.cancel()
      state = exp.state_dict()
      exp2 = Experiment(experiment_id='temp')
      exp2.load_state_dict(state)
      assert exp2.status == exp.status

  def test_completed_status_round_trip(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.complete(metrics={'m': 1.0})
    state = exp.state_dict()
    exp2 = Experiment(experiment_id='temp')
    exp2.load_state_dict(state)
    assert exp2.status == Status.completed
    assert exp2.metrics == {'m': 1.0}

  def test_failed_status_round_trip(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.start()
    exp.fail(error='boom')
    state = exp.state_dict()
    exp2 = Experiment(experiment_id='temp')
    exp2.load_state_dict(state)
    assert exp2.status == Status.failed
    assert exp2.error == 'boom'


class TestFullTransitionMatrix:
  """Parametrized test covering the entire invalid transition matrix."""

  @pytest.mark.parametrize(
    ('setup_status', 'method', 'expected_match'),
    [
      ('running', 'start', 'cannot start'),
      ('completed', 'start', 'cannot start'),
      ('failed', 'start', 'cannot start'),
      ('cancelled', 'start', 'cannot start'),
      ('completed', 'complete', 'cannot complete'),
      ('failed', 'complete', 'cannot complete'),
      ('cancelled', 'complete', 'cannot complete'),
      ('completed', 'fail', 'cannot fail'),
      ('failed', 'fail', 'cannot fail'),
      ('cancelled', 'fail', 'cannot fail'),
      ('completed', 'cancel', 'cannot cancel'),
      ('failed', 'cancel', 'cannot cancel'),
      ('cancelled', 'cancel', 'cannot cancel'),
      ('pending', 'advance_epoch', 'cannot advance epoch'),
      ('completed', 'advance_epoch', 'cannot advance epoch'),
      ('failed', 'advance_epoch', 'cannot advance epoch'),
      ('cancelled', 'advance_epoch', 'cannot advance epoch'),
    ],
  )
  def test_invalid_transition(self, setup_status: str, method: str, expected_match: str) -> None:
    exp = Experiment(experiment_id='t')
    if setup_status == 'running':
      exp.start()
    elif setup_status == 'completed':
      exp.start()
      exp.complete()
    elif setup_status == 'failed':
      exp.start()
      exp.fail()
    elif setup_status == 'cancelled':
      exp.cancel()

    with pytest.raises(ExperimentError, match=expected_match):
      getattr(exp, method)()


class TestContextManagerNormalExit:
  def test_context_manager_normal_exit_completes(self) -> None:
    exp = Experiment(experiment_id='cm-ok')
    with exp:
      assert exp.status == Status.running
    assert exp.status == Status.completed
    assert exp.completed_at is not None

  def test_context_manager_returns_self(self) -> None:
    exp = Experiment(experiment_id='cm-self')
    with exp as ctx:
      assert ctx is exp


class TestContextManagerExceptionExit:
  def test_context_manager_exception_exit_fails(self) -> None:
    exp = Experiment(experiment_id='cm-fail')
    msg = 'boom'
    with pytest.raises(RuntimeError, match=msg), exp:
      raise RuntimeError(msg)
    assert exp.status == Status.failed
    assert exp.error == 'boom'

  def test_context_manager_preserves_error_message(self) -> None:
    exp = Experiment(experiment_id='cm-msg')
    msg = 'details'
    with pytest.raises(ValueError, match=msg), exp:
      raise ValueError(msg)
    assert exp.error == 'details'


class TestContextManagerExitNoSuppress:
  def test_exception_propagates_through_context_manager(self) -> None:
    exp = Experiment(experiment_id='cm-propagate')
    msg = 'not suppressed'
    with pytest.raises(RuntimeError, match=msg), exp:
      raise RuntimeError(msg)

  def test_exit_returns_false_directly(self) -> None:
    exp = Experiment(experiment_id='cm-ret')
    exp.start()
    result = exp.__exit__(None, None, None)
    assert result is False

  def test_exit_returns_false_on_exception(self) -> None:
    exp = Experiment(experiment_id='cm-ret-exc')
    exp.start()
    exc = RuntimeError('test')
    result = exp.__exit__(type(exc), exc, None)
    assert result is False


class TestContextManagerGuardAlreadyCompleted:
  def test_guard_skips_when_already_completed(self) -> None:
    exp = Experiment(experiment_id='cm-guard-c')
    exp.start()
    exp.complete()
    result = exp.__exit__(None, None, None)
    assert result is False
    assert exp.status == Status.completed

  def test_no_experiment_error_when_already_completed(self) -> None:
    exp = Experiment(experiment_id='cm-guard-c2')
    exp.start()
    exp.complete(metrics={'x': 1.0})
    exp.__exit__(None, None, None)
    assert exp.metrics == {'x': 1.0}


class TestContextManagerGuardAlreadyFailed:
  def test_guard_skips_when_already_failed(self) -> None:
    exp = Experiment(experiment_id='cm-guard-f')
    exp.start()
    exp.fail('original error')
    exc = RuntimeError('second error')
    result = exp.__exit__(type(exc), exc, None)
    assert result is False
    assert exp.status == Status.failed
    assert exp.error == 'original error'

  def test_no_double_fail(self) -> None:
    exp = Experiment(experiment_id='cm-guard-f2')
    exp.start()
    exp.fail('first')
    exp.__exit__(RuntimeError, RuntimeError('second'), None)
    assert exp.error == 'first'


class TestContextManagerRunningResumeSkipsStart:
  """Checkpoint resume: status=running before __enter__ is allowed."""

  def test_enter_running_skips_start(self) -> None:
    exp = Experiment(experiment_id='cm-resume')
    exp.start()
    assert exp.status == Status.running
    started_at = exp.started_at
    with exp as ctx:
      assert ctx is exp
      assert exp.status == Status.running
      assert exp.started_at == started_at
    assert exp.status == Status.completed

  def test_enter_completed_raises(self) -> None:
    exp = Experiment(experiment_id='cm-completed')
    exp.start()
    exp.complete()
    with pytest.raises(ExperimentError, match='cannot enter context'), exp:
      pass

  def test_enter_failed_raises(self) -> None:
    exp = Experiment(experiment_id='cm-failed')
    exp.start()
    exp.fail('oops')
    with pytest.raises(ExperimentError, match='cannot enter context'), exp:
      pass

  def test_enter_cancelled_raises(self) -> None:
    exp = Experiment(experiment_id='cm-cancelled')
    exp.cancel()
    with pytest.raises(ExperimentError, match='cannot enter context'), exp:
      pass


class TestContextManagerReentryRaises:
  def test_reentry_after_terminal_raises(self) -> None:
    exp = Experiment(experiment_id='cm-reentry')
    with exp:
      pass
    assert exp.status == Status.completed
    with pytest.raises(ExperimentError, match='cannot enter context'), exp:
      pass

  def test_nested_enter_raises(self) -> None:
    exp = Experiment(experiment_id='cm-nested')
    with exp, pytest.raises(ExperimentError, match='cannot start'):
      exp.start()


class TestNestedContextManager:
  """Re-entrant (nested) context manager behavior with depth counter."""

  def test_nested_context_manager_defers_completion(self) -> None:
    """Inner exit does not finalize; only outer exit completes."""
    e = Experiment(experiment_id='nested')
    with e:
      assert e.status == Status.running
      with e:
        assert e.status == Status.running
      assert e.status == Status.running
    assert e.status == Status.completed

  def test_nested_context_manager_failure_propagates(self) -> None:
    """Exception in inner block propagates; outer exit calls fail."""
    e = Experiment(experiment_id='nested-fail')
    msg = 'boom'
    with pytest.raises(ValueError, match=msg), e, e:
      raise ValueError(msg)
    assert e.status == Status.failed
    assert msg in e.error

  def test_triple_nesting(self) -> None:
    """Three levels of nesting; only outermost exit finalizes."""
    e = Experiment(experiment_id='triple')
    with e:
      with e:
        with e:
          assert e.status == Status.running
        assert e.status == Status.running
      assert e.status == Status.running
    assert e.status == Status.completed

  def test_depth_resets_after_exit(self) -> None:
    """After full exit, depth is 0 and re-entering works from pending."""
    e = Experiment(experiment_id='depth-reset')
    with e:
      pass
    assert e.status == Status.completed
    with pytest.raises(ExperimentError, match='cannot enter context'), e:
      pass


class TestExcValNoneEdgeCase:
  """Edge cases for __exit__ when exc_val is None."""

  def test_exit_with_exc_val_none_uses_type_name(self) -> None:
    """When exc_val is None but exc_type is set, error uses type name."""
    e = Experiment(experiment_id='exc-none')
    e.start()
    e._context_depth = 1
    e.__exit__(ValueError, None, None)
    assert e.status == Status.failed
    assert e.error == 'ValueError'
    assert e.error != 'None'

  def test_exit_with_both_exc_type_and_val(self) -> None:
    """When both exc_type and exc_val are set, error uses str(exc_val)."""
    e = Experiment(experiment_id='exc-both')
    e.start()
    e._context_depth = 1
    exc = RuntimeError('kaboom')
    e.__exit__(type(exc), exc, None)
    assert e.status == Status.failed
    assert e.error == 'kaboom'

  def test_exit_normal_no_exception(self) -> None:
    """Normal exit (no exception) completes the experiment."""
    e = Experiment(experiment_id='exc-normal')
    e.start()
    e._context_depth = 1
    e.__exit__(None, None, None)
    assert e.status == Status.completed


class TestContextDepthNotSerialized:
  """Transient _context_depth must not appear in state_dict."""

  def test_context_depth_not_in_state_dict(self) -> None:
    """Transient _context_depth must not appear in state_dict."""
    e = Experiment(experiment_id='depth-test')
    with e:
      state = e.state_dict()
      assert '_context_depth' not in state
      assert 'context_depth' not in state

  def test_load_state_dict_ignores_depth(self) -> None:
    """load_state_dict does not restore _context_depth even if present."""
    e = Experiment(experiment_id='load-depth')
    e.start()
    state = e.state_dict()
    state['_context_depth'] = 5

    e2 = Experiment(experiment_id='temp')
    e2.load_state_dict(state)
    assert e2._context_depth == 0


class TestSubclass:
  def test_custom_experiment_with_extra_fields(self) -> None:
    class CustomExperiment(Experiment):
      def __init__(
        self, experiment_id: str, hypothesis: str | None = None, extra: str | None = None
      ) -> None:
        super().__init__(experiment_id=experiment_id, hypothesis=hypothesis)
        self.extra = extra

      def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state['extra'] = self.extra
        return state

      def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.extra = state['extra']

    exp = CustomExperiment(experiment_id='custom', hypothesis='test', extra='value')
    assert exp.extra == 'value'
    exp.start()
    assert exp.status == Status.running
    state = exp.state_dict()
    assert state['extra'] == 'value'

    exp2 = CustomExperiment(experiment_id='temp')
    exp2.load_state_dict(state)
    assert exp2.extra == 'value'
    assert exp2.status == Status.running


class TestRequireStatusMessagesStable:
  """Verify _require_status preserves error message format across all guarded methods."""

  @pytest.mark.parametrize(
    ('setup_status', 'method', 'action', 'expected_status'),
    [
      ('running', 'start', 'start', 'pending'),
      ('pending', 'advance_epoch', 'advance epoch', 'running'),
      ('completed', 'start', 'start', 'pending'),
      ('failed', 'complete', 'complete', 'pending or running'),
      ('completed', 'fail', 'fail', 'pending or running'),
      ('cancelled', 'fail', 'fail', 'pending or running'),
      ('completed', 'advance_epoch', 'advance epoch', 'running'),
    ],
  )
  def test_message_format(
    self,
    setup_status: str,
    method: str,
    action: str,
    expected_status: str,
  ) -> None:
    exp = Experiment(experiment_id='msg-test')
    if setup_status == 'running':
      exp.start()
    elif setup_status == 'completed':
      exp.start()
      exp.complete()
    elif setup_status == 'failed':
      exp.start()
      exp.fail()
    elif setup_status == 'cancelled':
      exp.cancel()

    with pytest.raises(ExperimentError) as exc_info:
      getattr(exp, method)()

    error_msg = str(exc_info.value)
    assert f'cannot {action}:' in error_msg
    assert "experiment id='msg-test'" in error_msg
    assert f'expected {expected_status}' in error_msg


class TestStrictLoadStateDict:
  def test_load_state_dict_missing_last_accepted_epoch_raises(self) -> None:
    """Missing last_accepted_epoch key raises KeyError."""
    e = Experiment(experiment_id='strict')
    state = e.state_dict()
    del state['last_accepted_epoch']
    with pytest.raises(KeyError, match='last_accepted_epoch'):
      e.load_state_dict(state)
