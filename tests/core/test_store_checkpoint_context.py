"""Tests for StoreCheckpointCallback snapshot context strings (plan 16)."""

from autopilot.core.callbacks.store import StoreCheckpointCallback, _build_snapshot_context
from autopilot.core.experiment import Experiment
from autopilot.core.models import Result
from unittest.mock import MagicMock


def test_store_checkpoint_context_present() -> None:
  """Snapshot receives a non-empty context string when experiment is present."""
  cb = StoreCheckpointCallback()

  experiment = Experiment(experiment_id='ctx-exp')
  experiment.should_rollback = False

  store = MagicMock()
  trainer = MagicMock()
  trainer.experiment = experiment
  trainer.store = store
  trainer.profiler = None

  result = Result(metrics={'accuracy': 0.95})
  cb.on_epoch_end(trainer, MagicMock(), epoch=0, result=result)

  store.snapshot.assert_called_once()
  call_kwargs = store.snapshot.call_args
  context = call_kwargs.kwargs.get('context') or call_kwargs[1].get('context')
  if context is None:
    _, kwargs = store.snapshot.call_args
    context = kwargs.get('context')
  assert context is not None
  assert len(context) > 0


def test_store_checkpoint_context_has_epoch() -> None:
  """Context string contains 'epoch 0' for the first checkpointed epoch."""
  cb = StoreCheckpointCallback()

  experiment = Experiment(experiment_id='epoch-exp')
  experiment.should_rollback = False

  store = MagicMock()
  trainer = MagicMock()
  trainer.experiment = experiment
  trainer.store = store
  trainer.profiler = None

  result = Result(metrics={'loss': 0.5})
  cb.on_epoch_end(trainer, MagicMock(), epoch=0, result=result)

  _, kwargs = store.snapshot.call_args
  context = kwargs['context']
  assert 'epoch 0' in context


def test_store_checkpoint_context_has_metrics() -> None:
  """Context string includes at least one metric_name=value fragment."""
  cb = StoreCheckpointCallback()

  experiment = Experiment(experiment_id='metric-exp')
  experiment.should_rollback = False

  store = MagicMock()
  trainer = MagicMock()
  trainer.experiment = experiment
  trainer.store = store
  trainer.profiler = None

  result = Result(metrics={'accuracy': 0.85, 'loss': 0.3})
  cb.on_epoch_end(trainer, MagicMock(), epoch=1, result=result)

  _, kwargs = store.snapshot.call_args
  context = kwargs['context']
  assert 'accuracy=0.85' in context


def test_store_checkpoint_unchanged_without_experiment() -> None:
  """No crash and no snapshot when trainer has no experiment."""
  cb = StoreCheckpointCallback()

  store = MagicMock()
  trainer = MagicMock()
  trainer.experiment = None
  trainer.store = store
  trainer.profiler = None

  cb.on_epoch_end(trainer, MagicMock(), epoch=0, result=Result(metrics={'x': 1.0}))

  store.snapshot.assert_not_called()


def test_build_snapshot_context_no_result() -> None:
  """Context without a result produces epoch-only string."""
  ctx = _build_snapshot_context(2, None)
  assert ctx == 'epoch 2 checkpoint'


def test_build_snapshot_context_empty_metrics() -> None:
  """Context with empty metrics produces epoch-only string."""
  ctx = _build_snapshot_context(0, Result(metrics={}))
  assert ctx == 'epoch 0 checkpoint'


def test_build_snapshot_context_lexicographic_order() -> None:
  """Metrics are sorted lexicographically by key."""
  result = Result(metrics={'zebra': 1.0, 'alpha': 2.0, 'middle': 3.0})
  ctx = _build_snapshot_context(0, result)
  assert 'alpha=2.0, middle=3.0, zebra=1.0' in ctx


def test_build_snapshot_context_max_three_metrics() -> None:
  """At most three metrics are included in context."""
  result = Result(
    metrics={
      'a': 1.0,
      'b': 2.0,
      'c': 3.0,
      'd': 4.0,
      'e': 5.0,
    }
  )
  ctx = _build_snapshot_context(0, result)
  assert 'a=1.0' in ctx
  assert 'b=2.0' in ctx
  assert 'c=3.0' in ctx
  assert 'd=' not in ctx
  assert 'e=' not in ctx


def test_build_snapshot_context_skips_non_numeric() -> None:
  """Non-numeric metric values are excluded."""
  metrics: dict = {'loss': 0.5, 'label': 'not_a_number'}
  result = Result(metrics=metrics)
  ctx = _build_snapshot_context(0, result)
  assert 'loss=0.5' in ctx
  assert 'label' not in ctx


def test_store_checkpoint_context_epoch_2() -> None:
  """Context string reflects the actual epoch index passed."""
  cb = StoreCheckpointCallback()

  experiment = Experiment(experiment_id='e2')
  experiment.should_rollback = False

  store = MagicMock()
  trainer = MagicMock()
  trainer.experiment = experiment
  trainer.store = store
  trainer.profiler = None

  result = Result(metrics={'val_acc': 0.9})
  cb.on_epoch_end(trainer, MagicMock(), epoch=2, result=result)

  _, kwargs = store.snapshot.call_args
  context = kwargs['context']
  assert 'epoch 2' in context
  assert 'val_acc=0.9' in context
