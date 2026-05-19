"""Edge-case tests for Experiment lifecycle, state transitions, and hooks."""

from autopilot.core.enums import Status
from autopilot.core.errors import ExperimentError
from autopilot.core.experiment import Experiment
import math
import pytest


class TestCancelFromPending:
  def test_cancel_from_pending_allowed(self) -> None:
    exp = Experiment(experiment_id='e1')
    assert exp.status == Status.pending
    exp.cancel()
    assert exp.status == Status.cancelled
    assert exp.cancelled_at is not None


class TestNanMetrics:
  def test_complete_with_nan_metrics(self) -> None:
    exp = Experiment(experiment_id='e1')
    exp.start()
    exp.complete(metrics={'loss': float('nan'), 'accuracy': 0.9})
    assert exp.status == Status.completed
    assert math.isnan(exp.metrics['loss'])
    assert exp.metrics['accuracy'] == 0.9

  def test_advance_epoch_with_nan_metrics(self) -> None:
    exp = Experiment(experiment_id='e1')
    exp.start()
    exp.advance_epoch(metrics={'val': float('nan')})
    assert math.isnan(exp.metrics['val'])

  def test_nan_metrics_survive_state_dict_round_trip(self) -> None:
    exp = Experiment(experiment_id='e1')
    exp.start()
    exp.complete(metrics={'loss': float('nan')})
    state = exp.state_dict()
    exp2 = Experiment(experiment_id='tmp')
    exp2.load_state_dict(state)
    assert math.isnan(exp2.metrics['loss'])


class TestVeryLargeEpochNumbers:
  def test_large_epoch_number(self) -> None:
    exp = Experiment(experiment_id='e1')
    exp.start()
    for _ in range(1000):
      exp.advance_epoch()
    assert exp.epoch == 999

  def test_large_epoch_in_state_dict(self) -> None:
    exp = Experiment(experiment_id='e1')
    exp.start()
    for _ in range(100):
      exp.advance_epoch()
    state = exp.state_dict()
    assert state['epoch'] == 99
    exp2 = Experiment(experiment_id='tmp')
    exp2.load_state_dict(state)
    assert exp2.epoch == 99


class TestStateDictWithEveryStatus:
  @pytest.mark.parametrize('status', list(Status))
  def test_state_dict_round_trip_per_status(self, status: Status) -> None:
    exp = Experiment(experiment_id='e1')
    if status == Status.pending:
      pass
    elif status == Status.running:
      exp.start()
    elif status == Status.completed:
      exp.start()
      exp.complete(metrics={'m': 1.0})
    elif status == Status.failed:
      exp.start()
      exp.fail(error='test error')
    elif status == Status.cancelled:
      exp.cancel()
    elif status == Status.invalidated:
      exp.start()
      exp.complete(metrics={'m': 1.0})
      exp.invalidate(reason='test invalidation')

    assert exp.status == status
    state = exp.state_dict()
    assert state['status'] == status.value

    exp2 = Experiment(experiment_id='tmp')
    exp2.load_state_dict(state)
    assert exp2.status == status
    assert exp2.state_dict() == state


class TestHookNoOps:
  def test_on_epoch_complete_returns_none(self) -> None:
    exp = Experiment(experiment_id='e1')
    result = exp.on_epoch_complete(epoch=0, metrics={'loss': 0.5})
    assert result is None

  def test_on_epoch_complete_does_not_raise(self) -> None:
    exp = Experiment(experiment_id='e1')
    exp.on_epoch_complete(epoch=0, metrics={})
    exp.on_epoch_complete(epoch=99, metrics={'a': 1.0, 'b': 2.0}, extra='kwarg')

  def test_on_validation_complete_returns_none(self) -> None:
    exp = Experiment(experiment_id='e1')
    result = exp.on_validation_complete(epoch=0, metrics={'val_loss': 0.3})
    assert result is None

  def test_on_validation_complete_does_not_raise(self) -> None:
    exp = Experiment(experiment_id='e1')
    exp.on_validation_complete(epoch=0, metrics={})
    exp.on_validation_complete(epoch=5, metrics={'v': 0.1}, key='value')

  def test_rollback_is_noop(self) -> None:
    exp = Experiment(experiment_id='e1')
    exp.start()
    exp.advance_epoch()
    result = exp.rollback(epoch=0)
    assert result is None


class TestDoubleStart:
  def test_double_start_raises_experiment_error(self) -> None:
    exp = Experiment(experiment_id='e1')
    exp.start()
    with pytest.raises(ExperimentError, match='cannot start'):
      exp.start()

  def test_double_start_error_message_includes_running(self) -> None:
    exp = Experiment(experiment_id='e1')
    exp.start()
    with pytest.raises(ExperimentError, match='running'):
      exp.start()


class TestDoubleComplete:
  def test_double_complete_raises_experiment_error(self) -> None:
    exp = Experiment(experiment_id='e1')
    exp.start()
    exp.complete()
    with pytest.raises(ExperimentError, match='cannot complete'):
      exp.complete()

  def test_double_complete_error_message_includes_completed(self) -> None:
    exp = Experiment(experiment_id='e1')
    exp.start()
    exp.complete()
    with pytest.raises(ExperimentError, match='completed'):
      exp.complete()


class TestStartAfterComplete:
  def test_start_after_complete_raises(self) -> None:
    exp = Experiment(experiment_id='e1')
    exp.start()
    exp.complete()
    with pytest.raises(ExperimentError, match='cannot start'):
      exp.start()

  def test_start_after_complete_message_includes_completed(self) -> None:
    exp = Experiment(experiment_id='e1')
    exp.start()
    exp.complete()
    with pytest.raises(ExperimentError, match='completed'):
      exp.start()


class TestStoreAndLastAcceptedEpochProperties:
  def test_store_default_none(self) -> None:
    exp = Experiment(experiment_id='e1')
    assert exp.store is None

  def test_store_setter(self) -> None:
    exp = Experiment(experiment_id='e1')
    sentinel = object()
    exp.store = sentinel
    assert exp.store is sentinel

  def test_last_accepted_epoch_default_none(self) -> None:
    exp = Experiment(experiment_id='e1')
    assert exp.last_accepted_epoch is None

  def test_last_accepted_epoch_setter(self) -> None:
    exp = Experiment(experiment_id='e1')
    exp.last_accepted_epoch = 5
    assert exp.last_accepted_epoch == 5

  def test_last_accepted_epoch_reset_to_none(self) -> None:
    exp = Experiment(experiment_id='e1')
    exp.last_accepted_epoch = 3
    exp.last_accepted_epoch = None
    assert exp.last_accepted_epoch is None


class TestExperimentStateDictUsesLastAcceptedEpoch:
  def test_state_dict_includes_last_accepted_epoch(self) -> None:
    exp = Experiment(experiment_id='e1')
    exp.last_accepted_epoch = 7
    d = exp.state_dict()
    assert d['last_accepted_epoch'] == 7

  def test_state_dict_uses_last_accepted_epoch_key(self) -> None:
    exp = Experiment(experiment_id='e1')
    exp.last_accepted_epoch = 5
    d = exp.state_dict()
    assert 'last_accepted_epoch' in d
    assert d['last_accepted_epoch'] == 5

  def test_load_state_dict_restores_last_accepted_epoch(self) -> None:
    exp = Experiment(experiment_id='e1')
    exp.last_accepted_epoch = 7
    d = exp.state_dict()
    exp2 = Experiment(experiment_id='e2')
    exp2.load_state_dict(d)
    assert exp2.last_accepted_epoch == 7

  def test_round_trip_last_accepted_epoch(self) -> None:
    exp = Experiment(experiment_id='e1')
    exp.last_accepted_epoch = 4
    exp3 = Experiment(experiment_id='e3')
    exp3.load_state_dict(exp.state_dict())
    assert exp3.last_accepted_epoch == exp.last_accepted_epoch

  def test_state_dict_default_none(self) -> None:
    exp = Experiment(experiment_id='e1')
    d = exp.state_dict()
    assert d['last_accepted_epoch'] is None
