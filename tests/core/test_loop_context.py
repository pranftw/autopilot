"""Tests for training loop and experiment lifecycle context emission (sub-plan 07).

Covers:
  - test_policy_gate_accept_emits_context (2.1)
  - test_policy_gate_reject_emits_context (2.1)
  - test_experiment_completion_emits_context (2.2)
  - test_experiment_failure_emits_context (2.2)
  - test_rollback_adds_context_directly (2.3)
  - test_rollback_context_has_target_epoch (2.3)
  - test_full_epoch_loop_accumulates_context (2.1)
  - test_no_context_without_experiment (2.1 / 2.2)
"""

from autopilot.core.callbacks.callback import Callback
from autopilot.core.context import ContextEntry
from autopilot.core.decision import DecisionEntry
from autopilot.core.experiment import Experiment
from autopilot.core.metric import Metric
from autopilot.core.models import Result
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import GateResult
from autopilot.policy.policy import Policy
from tests.doubles import NoopEvalModule
from unittest.mock import MagicMock

# -- helpers --


class SpyCallback(Callback):
  """Records on_context_emit calls for assertion."""

  def __init__(self) -> None:
    super().__init__()
    self.entries: list[ContextEntry] = []

  def on_context_emit(self, trainer, module, entry) -> None:
    """Capture emitted context entries."""
    self.entries.append(entry)


class AccuracyMetric(Metric):
  """Test metric that returns a fixed accuracy value."""

  higher_is_better = True

  def __init__(self, value: float = 0.9) -> None:
    super().__init__()
    self._value = value
    self._updated = False

  def update(self, datum) -> None:
    """Mark as updated."""
    self._updated = True

  def compute(self) -> dict[str, float]:
    """Return the fixed accuracy."""
    return {'accuracy': self._value}

  def reset(self) -> None:
    """Reset update flag."""
    self._updated = False


class AlwaysPassPolicy(Policy):
  """Policy that always passes for testing."""

  def forward(self, result: Result) -> GateResult:
    """Always pass."""
    return GateResult.PASSED


class AlwaysFailPolicy(Policy):
  """Policy that always fails for testing."""

  def forward(self, result: Result) -> GateResult:
    """Always fail."""
    return GateResult.FAIL


class ModuleWithAccuracy(NoopEvalModule):
  """Module that reports a configurable accuracy metric."""

  def __init__(self, accuracy: float = 0.9) -> None:
    super().__init__()
    self.accuracy_metric = AccuracyMetric(value=accuracy)


class FailingModule(NoopEvalModule):
  """Module whose training_step raises to simulate failure."""

  def training_step(self, batch, batch_idx):
    """Raise on first call to trigger failure path."""
    msg = 'deliberate training failure'
    raise RuntimeError(msg)


# -- 2.1 tests: policy gate context --


def test_policy_gate_accept_emits_context():
  """After gate pass, context log has entry with 'accepted' and metrics in metadata."""
  exp = Experiment('gate-accept')
  spy = SpyCallback()
  module = ModuleWithAccuracy(accuracy=0.9)
  trainer = Trainer(
    callbacks=[spy],
    experiment=exp,
    policy=AlwaysPassPolicy(),
  )

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=1)

  policy_entries = [e for e in spy.entries if e.source == 'policy']
  assert len(policy_entries) == 1
  accept_entry = policy_entries[0]
  assert 'accepted' in accept_entry.reason
  assert 'metrics' in accept_entry.metadata
  assert isinstance(accept_entry.metadata['metrics'], dict)

  log_entries = exp.context_log.search('accepted')
  assert len(log_entries) == 1


def test_policy_gate_reject_emits_context():
  """After gate reject, log has 'rejected' with _type and gates (no metrics/gate_result)."""
  exp = Experiment('gate-reject')
  spy = SpyCallback()
  module = ModuleWithAccuracy(accuracy=0.3)
  trainer = Trainer(
    callbacks=[spy],
    experiment=exp,
    policy=AlwaysFailPolicy(),
  )

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=1)

  policy_entries = [e for e in spy.entries if e.source == 'policy']
  assert len(policy_entries) == 1
  reject_entry = policy_entries[0]
  assert 'rejected' in reject_entry.reason
  assert reject_entry.metadata['_type'] == DecisionEntry.POLICY_GATE_TYPE
  assert isinstance(reject_entry.metadata['gates'], list)
  assert 'metrics' not in reject_entry.metadata

  log_reject = exp.context_log.search('rejected epoch')
  assert len(log_reject) == 1
  assert log_reject[0].metadata['_type'] == DecisionEntry.POLICY_GATE_TYPE


def test_full_epoch_loop_accumulates_context():
  """Run 3 epochs with policy gate; multiple policy entries accumulate in order."""
  exp = Experiment('multi-epoch')
  spy = SpyCallback()
  module = ModuleWithAccuracy(accuracy=0.95)
  trainer = Trainer(
    callbacks=[spy],
    experiment=exp,
    policy=AlwaysPassPolicy(),
  )

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=3)

  policy_entries = [e for e in spy.entries if e.source == 'policy']
  assert len(policy_entries) == 3
  for i, entry in enumerate(policy_entries):
    assert 'accepted' in entry.reason
    assert f'epoch {i}' in entry.reason

  log_policy = exp.context_log.filter_by_source('policy')
  assert len(log_policy) == 3


def test_no_context_without_experiment():
  """Without experiment, ContextLogCallback is not attached, but emit_context
  still dispatches to other callbacks (e.g. SpyCallback)."""
  spy = SpyCallback()
  module = ModuleWithAccuracy(accuracy=0.9)
  trainer = Trainer(
    callbacks=[spy],
    policy=AlwaysPassPolicy(),
  )

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=1)

  policy_entries = [e for e in spy.entries if e.source == 'policy']
  assert len(policy_entries) == 1


# -- 2.2 tests: experiment completion and failure --


def test_experiment_completion_emits_context():
  """Successful completion path; entry with source=='trainer' and final_metrics populated."""
  exp = Experiment('completion-test')
  spy = SpyCallback()
  module = ModuleWithAccuracy(accuracy=0.85)
  trainer = Trainer(
    callbacks=[spy],
    experiment=exp,
  )

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=1)

  completion_entries = [
    e for e in spy.entries if e.source == 'trainer' and 'completed successfully' in e.reason
  ]
  assert len(completion_entries) == 1
  assert 'final_metrics' in completion_entries[0].metadata

  log_completion = exp.context_log.search('completed successfully')
  assert len(log_completion) == 1
  assert log_completion[0].source == 'trainer'


def test_experiment_failure_emits_context():
  """Forced failure path; entry includes error info in metadata['error']."""
  exp = Experiment('failure-test')
  spy = SpyCallback()
  module = FailingModule()
  trainer = Trainer(
    callbacks=[spy],
    experiment=exp,
  )

  caught = False
  try:
    trainer.fit(module, train_dataloaders=[[1]], max_epochs=1)
  except RuntimeError:
    caught = True
  assert caught

  failure_entries = [e for e in spy.entries if e.source == 'trainer' and 'failed' in e.reason]
  assert len(failure_entries) == 1
  assert 'error' in failure_entries[0].metadata
  assert 'deliberate training failure' in failure_entries[0].metadata['error']


# -- 2.3 tests: rollback context --


def test_rollback_adds_context_directly():
  """After rollback, new entry exists via add_context (no trainer.emit_context)."""
  store = MagicMock()
  exp = Experiment('rollback-test')
  exp.store = store

  exp.rollback(2)

  assert len(exp.context_log) == 1
  entry = exp.context_log.entries[0]
  assert 'rolled back to epoch 2' in entry.reason
  assert entry.source == 'trainer'
  store.checkout.assert_called_once_with(exp.id, 2, context='rolled back to epoch 2')


def test_rollback_context_has_target_epoch():
  """metadata['target_epoch'] equals rollback argument; epoch reflects new state."""
  store = MagicMock()
  exp = Experiment('rollback-epoch')
  exp.store = store

  exp.rollback(5)

  entry = exp.context_log.entries[0]
  assert entry.metadata['target_epoch'] == 5
  assert entry.epoch == 5
  assert exp.epoch == 5


def test_rollback_noop_no_store():
  """rollback with no store appends no context."""
  exp = Experiment('no-store')
  exp.store = None

  exp.rollback(2)

  assert len(exp.context_log) == 0


def test_rollback_noop_none_epoch():
  """rollback(None) appends no context."""
  store = MagicMock()
  exp = Experiment('none-epoch')
  exp.store = store

  exp.rollback(None)

  assert len(exp.context_log) == 0
  store.checkout.assert_not_called()


class PassThenFailPolicy(Policy):
  """Policy that passes the first call and fails all subsequent calls."""

  def __init__(self) -> None:
    super().__init__()
    self._call_count = 0

  def forward(self, result: Result) -> GateResult:
    """Pass first, fail after."""
    self._call_count += 1
    if self._call_count == 1:
      return GateResult.PASSED
    return GateResult.FAIL


def test_first_epoch_reject_no_rollback():
  """First-epoch reject: no previous accepted epoch, so rollback is a no-op."""
  store = MagicMock()
  exp = Experiment('reject-no-rollback')
  exp.store = store
  spy = SpyCallback()
  module = ModuleWithAccuracy(accuracy=0.3)
  trainer = Trainer(
    callbacks=[spy],
    experiment=exp,
    policy=AlwaysFailPolicy(),
  )

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=1)

  policy_entries = exp.context_log.filter_by_source('policy')
  assert len(policy_entries) == 1
  assert 'rejected' in policy_entries[0].reason

  rollback_entries = exp.context_log.search('rolled back')
  assert len(rollback_entries) == 0
  store.checkout.assert_not_called()


def test_multi_epoch_reject_triggers_rollback_context():
  """Epoch 0 accepted, epoch 1 rejected -> rollback to epoch 0 with context."""
  store = MagicMock()
  exp = Experiment('multi-reject-rollback')
  exp.store = store
  spy = SpyCallback()
  module = ModuleWithAccuracy(accuracy=0.5)
  trainer = Trainer(
    callbacks=[spy],
    experiment=exp,
    policy=PassThenFailPolicy(),
  )

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=2)

  policy_entries = exp.context_log.filter_by_source('policy')
  assert len(policy_entries) == 2
  assert 'accepted' in policy_entries[0].reason
  assert 'rejected' in policy_entries[1].reason

  rollback_entries = exp.context_log.search('rolled back')
  assert len(rollback_entries) == 1
  assert rollback_entries[0].metadata['target_epoch'] == 0
  store.checkout.assert_called_once_with(exp.id, 0, context='rolled back to epoch 0')


def test_completion_and_policy_context_coexist():
  """Both policy accept and completion entries appear after a successful epoch."""
  exp = Experiment('coexist')
  module = ModuleWithAccuracy(accuracy=0.9)
  trainer = Trainer(
    experiment=exp,
    policy=AlwaysPassPolicy(),
  )

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=1)

  policy_entries = exp.context_log.filter_by_source('policy')
  trainer_entries = exp.context_log.filter_by_source('trainer')
  assert len(policy_entries) == 1
  assert len(trainer_entries) == 3
  assert 'accepted' in policy_entries[0].reason
  reasons = [e.reason for e in trainer_entries]
  assert any('optimizer step completed' in r for r in reasons)
  assert any('max_epochs reached' in r for r in reasons)
  assert any('completed successfully' in r for r in reasons)
