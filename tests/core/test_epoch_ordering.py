"""Tests for epoch loop ordering, policy gate placement, and context emission.

BUG-010: Policy gate now runs after validation with merged train/val metrics.
BUG-013: trainer.current_epoch is set at the start of each epoch.
Ctx-1:   Context emitted when training completes via max_epochs.
Ctx-2:   Context emitted when AgentOptimizer agentic step fails.
Ctx-5:   Context emitted on early-stopping callback signal.
TS-grad-no-opt: Gradient summaries journaled without AgentOptimizer.

Organized by subplan:
  4.1 BUG-010 policy ordering and metrics (tests 1-4)
  4.2 BUG-013 and context epoch (tests 5-6)
  4.3 Stop and optimizer traceability (tests 7-11)
"""

from autopilot.core.callbacks.callback import Callback
from autopilot.core.context import ContextEntry
from autopilot.core.experiment import Experiment
from autopilot.core.gradient import NumericGradient
from autopilot.core.metric import Metric
from autopilot.core.models import Result
from autopilot.core.parameter import Parameter
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import EvalDatum, GateResult
from autopilot.policy.policy import Policy
from tests.doubles import DirectNumericLoss, NoopEvalModule, NoOpOptimizer
from unittest.mock import MagicMock

# -- shared helpers --


class SpyCallback(Callback):
  """Records on_context_emit calls."""

  def __init__(self) -> None:
    super().__init__()
    self.entries: list[ContextEntry] = []

  def on_context_emit(self, trainer, module, entry) -> None:
    """Capture emitted context entries."""
    self.entries.append(entry)


class FixedMetric(Metric):
  """Metric returning a fixed value keyed by name."""

  higher_is_better = True

  def __init__(self, name: str, value: float) -> None:
    super().__init__()
    self._name = name
    self._value = value

  def update(self, datum) -> None:
    """No-op accumulation."""

  def compute(self) -> dict[str, float]:
    """Return the fixed metric."""
    return {self._name: self._value}

  def reset(self) -> None:
    """No-op reset."""


class ThresholdPolicy(Policy):
  """Policy that requires a specific metric to meet a threshold."""

  def __init__(self, metric_key: str, threshold: float) -> None:
    super().__init__()
    self._metric_key = metric_key
    self._threshold = threshold
    self.received_metrics: list[dict[str, float]] = []

  def forward(self, result: Result) -> GateResult:
    """Pass when metric >= threshold, fail otherwise."""
    self.received_metrics.append(dict(result.metrics))
    val = result.metrics.get(self._metric_key, 0.0)
    if val >= self._threshold:
      return GateResult.PASSED
    return GateResult.FAIL


class AlwaysPassPolicy(Policy):
  """Policy that always passes."""

  def __init__(self) -> None:
    super().__init__()
    self.received_metrics: list[dict[str, float]] = []

  def forward(self, result: Result) -> GateResult:
    """Always pass, recording received metrics."""
    self.received_metrics.append(dict(result.metrics))
    return GateResult.PASSED


class SingleMetricModule(NoopEvalModule):
  """Module with a single metric that reports a fixed accuracy."""

  def __init__(self, value: float) -> None:
    super().__init__()
    self.accuracy = FixedMetric('accuracy', value)

  def validation_step(self, batch, batch_idx) -> EvalDatum:
    """Return constant EvalDatum for validation."""
    return EvalDatum(success=True)


class TrainOnlyModule(NoopEvalModule):
  """Module with only a train metric, no val metric."""

  def __init__(self, train_value: float = 0.9) -> None:
    super().__init__()
    self.accuracy = FixedMetric('accuracy', train_value)


class ModuleWithParam(NoopEvalModule):
  """Module with an optimizable parameter for gradient tests."""

  def __init__(self) -> None:
    super().__init__()
    self.param = Parameter(requires_grad=True)
    self.loss = DirectNumericLoss([self.param])

  def configure_optimizers(self):
    """Return a no-op optimizer over the module parameter."""
    return NoOpOptimizer([self.param])


class EarlyStopCallback(Callback):
  """Callback that signals stop before a given epoch."""

  def __init__(self, stop_before_epoch: int) -> None:
    super().__init__()
    self._stop_before = stop_before_epoch

  def on_epoch_start(self, trainer, module, epoch, **kwargs):
    """Signal stop when epoch reaches threshold."""
    if epoch >= self._stop_before:
      return {'stop': True}
    return None


# -- 4.1 BUG-010 policy ordering and metrics --


def test_policy_gate_receives_validation_metrics():
  """Policy requires val_accuracy >= 0.8; accuracy is 0.5 -> val_accuracy 0.5 -> gate fails.

  With a single metric, train and val both produce {'accuracy': 0.5}. After
  merging, the gate receives 'val_accuracy' = 0.5 which is below 0.8.
  """
  exp = Experiment('val-gate-test')
  module = SingleMetricModule(value=0.5)
  policy = ThresholdPolicy('val_accuracy', 0.8)
  spy = SpyCallback()
  trainer = Trainer(
    callbacks=[spy],
    experiment=exp,
    policy=policy,
  )

  trainer.fit(
    module,
    train_dataloaders=[[1]],
    val_dataloaders=[[1]],
    max_epochs=1,
  )

  reject_entries = [e for e in spy.entries if e.reason.startswith('policy gate rejected epoch')]
  assert len(reject_entries) == 1
  assert policy.received_metrics[0].get('val_accuracy') == 0.5


def test_policy_gate_runs_after_validation():
  """Validation pass must complete before policy gate runs."""
  call_order: list[str] = []

  class OrderTrackingModule(SingleMetricModule):
    """Module that tracks call ordering."""

    def validation_step(self, batch, batch_idx):
      call_order.append('validation')
      return EvalDatum(success=True)

  class OrderTrackingPolicy(Policy):
    """Policy that tracks when the gate is called."""

    def forward(self, result: Result) -> GateResult:
      call_order.append('policy_gate')
      return GateResult.PASSED

  exp = Experiment('order-test')
  module = OrderTrackingModule(value=0.9)
  trainer = Trainer(experiment=exp, policy=OrderTrackingPolicy())

  trainer.fit(
    module,
    train_dataloaders=[[1]],
    val_dataloaders=[[1]],
    max_epochs=1,
  )

  assert 'validation' in call_order
  assert 'policy_gate' in call_order
  val_idx = call_order.index('validation')
  gate_idx = call_order.index('policy_gate')
  assert val_idx < gate_idx


def test_gate_metrics_use_train_val_prefixes():
  """When both train and val metrics exist, gate receives train_* and val_* keys."""
  policy = AlwaysPassPolicy()
  exp = Experiment('prefix-test')
  module = SingleMetricModule(value=0.85)
  trainer = Trainer(experiment=exp, policy=policy)

  trainer.fit(
    module,
    train_dataloaders=[[1]],
    val_dataloaders=[[1]],
    max_epochs=1,
  )

  gate_input = policy.received_metrics[0]
  assert 'train_accuracy' in gate_input
  assert 'val_accuracy' in gate_input
  assert gate_input['train_accuracy'] == 0.85
  assert gate_input['val_accuracy'] == 0.85
  assert 'accuracy' not in gate_input


def test_train_only_metrics_unprefixed():
  """Without val loader, metrics passed to gate are unprefixed."""
  policy = AlwaysPassPolicy()
  exp = Experiment('train-only-test')
  module = TrainOnlyModule(train_value=0.95)
  trainer = Trainer(experiment=exp, policy=policy)

  trainer.fit(
    module,
    train_dataloaders=[[1]],
    max_epochs=1,
  )

  gate_input = policy.received_metrics[0]
  assert 'accuracy' in gate_input
  assert gate_input['accuracy'] == 0.95
  assert 'train_accuracy' not in gate_input
  assert 'val_accuracy' not in gate_input


# -- 4.2 BUG-013 and context epoch --


def test_current_epoch_reflects_last_epoch_index():
  """After 3-epoch fit, trainer.current_epoch == 2."""
  module = TrainOnlyModule()
  trainer = Trainer()

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=3)

  assert trainer.current_epoch == 2


def test_context_entry_epoch_matches_loop_index():
  """Context entries emitted during epochs carry correct epoch indices."""
  exp = Experiment('epoch-ctx-test')
  spy = SpyCallback()
  policy = AlwaysPassPolicy()
  module = TrainOnlyModule()
  trainer = Trainer(
    callbacks=[spy],
    experiment=exp,
    policy=policy,
  )

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=3)

  policy_entries = [e for e in spy.entries if e.source == 'policy']
  assert len(policy_entries) == 3
  for i, entry in enumerate(policy_entries):
    assert entry.epoch == i, f'entry {i} has epoch={entry.epoch}, expected {i}'


# -- 4.3 Stop and optimizer traceability --


def test_stop_reason_context_emitted():
  """After normal fit for max_epochs=N, context log contains 'max_epochs reached'."""
  exp = Experiment('max-stop-test')
  spy = SpyCallback()
  module = TrainOnlyModule()
  trainer = Trainer(
    callbacks=[spy],
    experiment=exp,
  )

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=2)

  max_stop = [e for e in spy.entries if 'max_epochs reached' in e.reason]
  assert len(max_stop) == 1
  assert max_stop[0].source == 'trainer'

  log_stop = exp.context_log.search('max_epochs reached')
  assert len(log_stop) == 1


def test_failed_optimizer_step_context():
  """Failed agentic step produces context entry with 'optimizer step failed'."""
  spy = SpyCallback()
  exp = Experiment('failed-opt-test')
  trainer = Trainer(callbacks=[spy], experiment=exp)
  trainer.current_epoch = 2

  mock_agent = MagicMock()
  mock_agent.limiter = None
  mock_agent.run.return_value = None

  from autopilot.ai.optimizer import AgentOptimizer

  param = Parameter(requires_grad=True)
  param.grad = NumericGradient(value=1.0)
  opt = AgentOptimizer(
    agent=mock_agent,
    params=[param],
    agentic=True,
    feedback_dir='/tmp/test-feedback',
    context={'trainer': trainer, 'epoch': 2},
  )

  opt.step()

  failure_entries = [e for e in spy.entries if 'optimizer step failed' in e.reason]
  assert len(failure_entries) == 1
  assert failure_entries[0].source == 'agent-optimizer'


def test_early_stop_emits_context_entry():
  """Early stopping callback triggers context entry with source='early-stopping'."""
  exp = Experiment('early-stop-test')
  spy = SpyCallback()
  early_cb = EarlyStopCallback(stop_before_epoch=1)
  module = TrainOnlyModule()
  trainer = Trainer(
    callbacks=[spy, early_cb],
    experiment=exp,
  )

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=5)

  es_entries = [e for e in spy.entries if e.source == 'early-stopping']
  assert len(es_entries) == 1
  assert 'early stopping' in es_entries[0].reason

  log_es = exp.context_log.filter_by_source('early-stopping')
  assert len(log_es) == 1


def test_early_stop_does_not_emit_max_epochs_reached():
  """Early stopping exits without 'max_epochs reached' context."""
  exp = Experiment('early-no-max')
  spy = SpyCallback()
  early_cb = EarlyStopCallback(stop_before_epoch=1)
  module = TrainOnlyModule()
  trainer = Trainer(
    callbacks=[spy, early_cb],
    experiment=exp,
  )

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=5)

  max_stop = [e for e in spy.entries if 'max_epochs reached' in e.reason]
  assert len(max_stop) == 0


def test_gradient_journaled_without_agent_optimizer():
  """Fit with plain Optimizer and gradient-producing loss includes gradient summary."""
  exp = Experiment('grad-journal-test')
  spy = SpyCallback()
  module = ModuleWithParam()
  trainer = Trainer(
    callbacks=[spy],
    experiment=exp,
  )

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=1)

  completion_entries = [e for e in spy.entries if e.reason == 'gradient feedback recorded']
  assert len(completion_entries) == 1
  assert 'gradient_summaries' in completion_entries[0].metadata
  summaries = completion_entries[0].metadata['gradient_summaries']
  assert len(summaries) >= 1
  assert isinstance(summaries[0], dict)
  assert 'param_name' in summaries[0]
  assert 'param_type' in summaries[0]
  assert 'gradient_type' in summaries[0]
  assert 'summary' in summaries[0]


def test_multi_source_context_entries():
  """After fit with policy + optimizer, context contains entries from multiple sources."""
  exp = Experiment('multi-source-test')
  spy = SpyCallback()
  policy = AlwaysPassPolicy()
  module = ModuleWithParam()
  trainer = Trainer(
    callbacks=[spy],
    experiment=exp,
    policy=policy,
  )

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=1)

  sources = {e.source for e in spy.entries if e.source is not None}
  assert 'trainer' in sources
  assert 'policy' in sources
  assert len(sources) >= 2


# -- 4.4 BUG-A001 current_epoch timing --


class EpochRecorderCallback(Callback):
  """Records trainer.current_epoch at on_epoch_start."""

  def __init__(self) -> None:
    super().__init__()
    self.recorded_epochs: list[int] = []

  def on_epoch_start(self, trainer, module, epoch, **kwargs):
    """Capture current_epoch during on_epoch_start."""
    self.recorded_epochs.append(trainer.current_epoch)


def test_current_epoch_set_before_on_epoch_start():
  """Callback recording trainer.current_epoch in on_epoch_start sees [0, 1, 2]."""
  recorder = EpochRecorderCallback()
  module = TrainOnlyModule()
  trainer = Trainer(callbacks=[recorder])

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=3)

  assert recorder.recorded_epochs == [0, 1, 2]


def test_early_stop_context_has_correct_epoch():
  """Early stop at epoch 2 emits context entry with entry.epoch == 2."""
  exp = Experiment('early-epoch-ctx')
  spy = SpyCallback()
  early_cb = EarlyStopCallback(stop_before_epoch=2)
  module = TrainOnlyModule()
  trainer = Trainer(
    callbacks=[spy, early_cb],
    experiment=exp,
  )

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=5)

  es_entries = [e for e in spy.entries if e.source == 'early-stopping']
  assert len(es_entries) == 1
  assert es_entries[0].epoch == 2


def test_current_epoch_correct_after_early_stop():
  """After early stopping at epoch 2, trainer.current_epoch == 2."""
  early_cb = EarlyStopCallback(stop_before_epoch=2)
  module = TrainOnlyModule()
  trainer = Trainer(callbacks=[early_cb])

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=5)

  assert trainer.current_epoch == 2


def test_single_epoch_current_epoch():
  """With max_epochs=1, on_epoch_start records [0]."""
  recorder = EpochRecorderCallback()
  module = TrainOnlyModule()
  trainer = Trainer(callbacks=[recorder])

  trainer.fit(module, train_dataloaders=[[1]], max_epochs=1)

  assert recorder.recorded_epochs == [0]
