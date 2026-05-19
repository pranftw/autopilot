"""Tests for MonotonicGate and _prev_ metric injection in EpochLoop."""

from autopilot.core.enums import Status
from autopilot.core.experiment import Experiment
from autopilot.core.loops.epoch import EpochLoop
from autopilot.core.metric import Metric
from autopilot.core.models import Result
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.parameter import Parameter
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import Datum, EvalDatum, GateResult
from autopilot.data.dataloader import DataLoader
from autopilot.policy.gates import MinGate, MonotonicGate
from autopilot.policy.quality_first import QualityFirstPolicy
from tests.doubles import DirectNumericLoss, NoOpOptimizer
from typing import Any
from unittest.mock import MagicMock
import math
import pytest


class TestMonotonicNonDecreasing:
  """Unit tests for non_decreasing direction (default)."""

  def test_pass_when_increases(self) -> None:
    gate = MonotonicGate('score')
    result = Result(metrics={'score': 1.0, '_prev_score': 0.5})
    assert gate.forward(result) == GateResult.PASSED

  def test_fail_when_decreases(self) -> None:
    gate = MonotonicGate('score')
    result = Result(metrics={'score': 0.4, '_prev_score': 0.5})
    assert gate.forward(result) == GateResult.FAIL

  def test_same_value_passes(self) -> None:
    gate = MonotonicGate('score')
    result = Result(metrics={'score': 0.5, '_prev_score': 0.5})
    assert gate.forward(result) == GateResult.PASSED


class TestMonotonicNonIncreasing:
  """Unit tests for non_increasing direction."""

  def test_pass_when_decreases(self) -> None:
    gate = MonotonicGate('loss', direction='non_increasing')
    result = Result(metrics={'loss': 0.3, '_prev_loss': 0.5})
    assert gate.forward(result) == GateResult.PASSED

  def test_fail_when_increases(self) -> None:
    gate = MonotonicGate('loss', direction='non_increasing')
    result = Result(metrics={'loss': 0.7, '_prev_loss': 0.5})
    assert gate.forward(result) == GateResult.FAIL

  def test_same_value_passes(self) -> None:
    gate = MonotonicGate('loss', direction='non_increasing')
    result = Result(metrics={'loss': 0.5, '_prev_loss': 0.5})
    assert gate.forward(result) == GateResult.PASSED


class TestMonotonicEdgeCases:
  """Edge cases: missing metrics, first epoch, NaN, invalid direction."""

  def test_missing_current_fails(self) -> None:
    gate = MonotonicGate('score')
    result = Result(metrics={'_prev_score': 0.5})
    assert gate.forward(result) == GateResult.FAIL

  def test_missing_current_empty_dict_fails(self) -> None:
    gate = MonotonicGate('score')
    result = Result(metrics={})
    assert gate.forward(result) == GateResult.FAIL

  def test_missing_prev_passes(self) -> None:
    """First epoch: no _prev_ key means baseline establishment -> PASS."""
    gate = MonotonicGate('score')
    result = Result(metrics={'score': 0.5})
    assert gate.forward(result) == GateResult.PASSED

  def test_invalid_direction_raises(self) -> None:
    with pytest.raises(ValueError, match='non_decreasing'):
      MonotonicGate('score', direction='up')

  def test_nan_current_fails_non_decreasing(self) -> None:
    gate = MonotonicGate('score')
    result = Result(metrics={'score': math.nan, '_prev_score': 0.5})
    assert gate.forward(result) == GateResult.FAIL

  def test_nan_prev_fails_non_decreasing(self) -> None:
    gate = MonotonicGate('score')
    result = Result(metrics={'score': 0.5, '_prev_score': math.nan})
    assert gate.forward(result) == GateResult.FAIL

  def test_nan_current_fails_non_increasing(self) -> None:
    gate = MonotonicGate('loss', direction='non_increasing')
    result = Result(metrics={'loss': math.nan, '_prev_loss': 0.5})
    assert gate.forward(result) == GateResult.FAIL

  def test_required_defaults_to_true(self) -> None:
    gate = MonotonicGate('score')
    assert gate.required is True

  def test_required_false(self) -> None:
    gate = MonotonicGate('score', required=False)
    assert gate.required is False


class TestMonotonicExplain:
  """Tests for MonotonicGate.explain() output."""

  def test_explain_missing_metric(self) -> None:
    gate = MonotonicGate('score')
    result = Result(metrics={})
    explanation = gate.explain(result)
    assert 'MonotonicGate(score): missing -> FAIL' in explanation

  def test_explain_no_prior(self) -> None:
    gate = MonotonicGate('score')
    result = Result(metrics={'score': 0.8})
    text = gate.explain(result)
    assert 'no prior' in text
    assert 'PASSED' in text

  def test_explain_non_decreasing_pass(self) -> None:
    gate = MonotonicGate('score')
    result = Result(metrics={'score': 0.9, '_prev_score': 0.5})
    text = gate.explain(result)
    assert '>=' in text
    assert 'PASSED' in text

  def test_explain_non_decreasing_fail(self) -> None:
    gate = MonotonicGate('score')
    result = Result(metrics={'score': 0.3, '_prev_score': 0.5})
    text = gate.explain(result)
    assert '>=' in text
    assert 'FAIL' in text

  def test_explain_non_increasing_pass(self) -> None:
    gate = MonotonicGate('loss', direction='non_increasing')
    result = Result(metrics={'loss': 0.3, '_prev_loss': 0.5})
    text = gate.explain(result)
    assert '<=' in text
    assert 'PASSED' in text


class TestMonotonicCallable:
  """Verify __call__ delegates to forward."""

  def test_call_wraps_forward(self) -> None:
    gate = MonotonicGate('score')
    result = Result(metrics={'score': 1.0, '_prev_score': 0.5})
    assert gate(result) == gate.forward(result)


class TestMonotonicRepr:
  """Verify repr includes useful info."""

  def test_repr(self) -> None:
    gate = MonotonicGate('accuracy', direction='non_increasing', required=False)
    rep = repr(gate)
    assert 'MonotonicGate' in rep
    assert 'accuracy' in rep


class TestEpochLoopInjectsPrevMetrics:
  """Integration: EpochLoop._check_policy_gate injects _prev_ from experiment.metrics."""

  def test_prev_metrics_injected(self) -> None:
    """_prev_train_loss present when experiment.metrics has train_loss."""
    loop = EpochLoop()
    trainer = MagicMock()
    captured_result = {}

    def capture_policy(result: Result) -> GateResult:
      captured_result['metrics'] = dict(result.metrics)
      return GateResult.PASSED

    trainer.policy = capture_policy
    trainer.emit_context = MagicMock()

    experiment = Experiment('exp-1', hypothesis='test')
    experiment.start()
    experiment.metrics = {'train_loss': 0.5, 'val_accuracy': 0.8}

    gate_metrics = {'train_loss': 0.3, 'val_accuracy': 0.9}
    loop._check_policy_gate(trainer, 1, gate_metrics, experiment)

    assert captured_result['metrics']['_prev_train_loss'] == 0.5
    assert captured_result['metrics']['_prev_val_accuracy'] == 0.8
    assert captured_result['metrics']['train_loss'] == 0.3
    assert captured_result['metrics']['val_accuracy'] == 0.9

  def test_no_prev_when_experiment_is_none(self) -> None:
    """No _prev_ keys injected when experiment is None."""
    loop = EpochLoop()
    trainer = MagicMock()
    captured_result = {}

    def capture_policy(result: Result) -> GateResult:
      captured_result['metrics'] = dict(result.metrics)
      return GateResult.PASSED

    trainer.policy = capture_policy
    trainer.emit_context = MagicMock()

    gate_metrics = {'score': 0.9}
    loop._check_policy_gate(trainer, 0, gate_metrics, None)

    assert '_prev_score' not in captured_result['metrics']

  def test_no_prev_when_experiment_metrics_empty(self) -> None:
    """No _prev_ keys injected when experiment.metrics is empty (first epoch)."""
    loop = EpochLoop()
    trainer = MagicMock()
    captured_result = {}

    def capture_policy(result: Result) -> GateResult:
      captured_result['metrics'] = dict(result.metrics)
      return GateResult.PASSED

    trainer.policy = capture_policy
    trainer.emit_context = MagicMock()

    experiment = Experiment('exp-1', hypothesis='test')
    experiment.start()

    gate_metrics = {'score': 0.9}
    loop._check_policy_gate(trainer, 0, gate_metrics, experiment)

    assert '_prev_score' not in captured_result['metrics']

  def test_current_metrics_not_overwritten_by_prev(self) -> None:
    """Current-epoch keys win over _prev_ if there's a naming collision."""
    loop = EpochLoop()
    trainer = MagicMock()
    captured_result = {}

    def capture_policy(result: Result) -> GateResult:
      captured_result['metrics'] = dict(result.metrics)
      return GateResult.PASSED

    trainer.policy = capture_policy
    trainer.emit_context = MagicMock()

    experiment = Experiment('exp-1', hypothesis='test')
    experiment.start()
    experiment.metrics = {'score': 0.5}

    gate_metrics = {'score': 0.9, '_prev_score': 999.0}
    loop._check_policy_gate(trainer, 1, gate_metrics, experiment)

    assert captured_result['metrics']['_prev_score'] == 999.0


class TestMonotonicMultiEpochIntegration:
  """Integration: two-epoch simulation with MonotonicGate."""

  def test_epoch0_pass_epoch1_pass_when_improves(self) -> None:
    """Epoch 0 passes (no prior), epoch 1 passes (score improved)."""
    gate = MonotonicGate('val_score')
    policy = QualityFirstPolicy(gates=[gate])
    loop = EpochLoop()
    trainer = MagicMock()
    trainer.policy = policy
    trainer.emit_context = MagicMock()

    experiment = Experiment('exp-1', hypothesis='test')
    experiment.start()

    gate_metrics_e0 = {'val_score': 0.5}
    result_e0 = loop._check_policy_gate(trainer, 0, gate_metrics_e0, experiment)
    assert result_e0 is None

    experiment.metrics = {'val_score': 0.5}

    gate_metrics_e1 = {'val_score': 0.6}
    result_e1 = loop._check_policy_gate(trainer, 1, gate_metrics_e1, experiment)
    assert result_e1 is None

  def test_epoch0_pass_epoch1_fail_when_regresses(self) -> None:
    """Epoch 0 passes (no prior), epoch 1 fails (score decreased)."""
    gate = MonotonicGate('val_score')
    policy = QualityFirstPolicy(gates=[gate])
    loop = EpochLoop()
    trainer = MagicMock()
    trainer.policy = policy
    trainer.emit_context = MagicMock()

    experiment = Experiment('exp-1', hypothesis='test')
    experiment.start()

    gate_metrics_e0 = {'val_score': 0.5}
    result_e0 = loop._check_policy_gate(trainer, 0, gate_metrics_e0, experiment)
    assert result_e0 is None

    experiment.metrics = {'val_score': 0.5}

    gate_metrics_e1 = {'val_score': 0.4}
    result_e1 = loop._check_policy_gate(trainer, 1, gate_metrics_e1, experiment)
    assert result_e1 is not None
    assert result_e1['stopped'] is True


class TestMonotonicCombinedWithMinGate:
  """Integration: MonotonicGate + MinGate in a single policy."""

  def test_min_gate_fails_while_monotonic_passes(self) -> None:
    """MinGate rejects low absolute value even though monotonic is satisfied."""
    min_gate = MinGate('score', 0.8)
    mono_gate = MonotonicGate('score')
    policy = QualityFirstPolicy(gates=[min_gate, mono_gate])

    result = Result(metrics={'score': 0.6, '_prev_score': 0.5})
    assert policy(result) == GateResult.FAIL

  def test_monotonic_fails_while_min_passes(self) -> None:
    """MonotonicGate rejects regression even though absolute value is above minimum."""
    min_gate = MinGate('score', 0.5)
    mono_gate = MonotonicGate('score')
    policy = QualityFirstPolicy(gates=[min_gate, mono_gate])

    result = Result(metrics={'score': 0.7, '_prev_score': 0.9})
    assert policy(result) == GateResult.FAIL

  def test_both_pass(self) -> None:
    """Both gates pass: above minimum and non-decreasing."""
    min_gate = MinGate('score', 0.5)
    mono_gate = MonotonicGate('score')
    policy = QualityFirstPolicy(gates=[min_gate, mono_gate])

    result = Result(metrics={'score': 0.9, '_prev_score': 0.8})
    assert policy(result) == GateResult.PASSED


class _DegradeValMetric(Metric):
  """Accuracy metric that degrades across epochs.

  All metrics participate in both train and val phases, so compute()
  is called 4 times over 2 epochs (train0, val0, train1, val1).
  Returns 0.9 for the first two calls (epoch 0) and 0.4 for the
  next two (epoch 1), ensuring val_accuracy regresses across epochs.
  """

  higher_is_better = True

  def __init__(self) -> None:
    super().__init__()
    self._call_count = 0

  def update(self, datum) -> None:
    pass

  def compute(self) -> dict[str, float]:
    self._call_count += 1
    if self._call_count <= 2:
      return {'accuracy': 0.9}
    return {'accuracy': 0.4}

  def reset(self) -> None:
    super().reset()


class _ConstantTrainMetric(Metric):
  """Train metric: returns constant loss=0.1."""

  higher_is_better = False

  def __init__(self) -> None:
    super().__init__()

  def update(self, datum) -> None:
    pass

  def compute(self) -> dict[str, float]:
    return {'loss': 0.1}

  def reset(self) -> None:
    super().reset()


class _MonotonicTestModule(AutoPilotModule):
  """Module with separate train/val metrics for MonotonicGate integration tests."""

  def __init__(
    self,
    train_metric: Metric | None = None,
    val_metric: Metric | None = None,
  ):
    super().__init__()
    self.param = Parameter(requires_grad=True)
    self.loss = DirectNumericLoss([self.param])
    self._opt = NoOpOptimizer([self.param])
    if train_metric is not None:
      self.train_met = train_metric
    if val_metric is not None:
      self.val_met = val_metric

  def forward(self, batch):
    return batch

  def training_step(self, batch, batch_idx):
    return EvalDatum(success=True)

  def validation_step(self, batch, batch_idx):
    return EvalDatum(success=True)

  def configure_optimizers(self):
    return self._opt


class _MonotonicTestExperiment(Experiment):
  """Experiment subclass bypassing store for integration tests."""

  def __init__(self, experiment_id: str = 'monotonic-test'):
    super().__init__(experiment_id=experiment_id, hypothesis='h')

  def on_epoch_complete(self, epoch: int, metrics: dict[str, float], **kwargs: Any) -> None:
    pass

  def on_validation_complete(self, epoch: int, metrics: dict[str, float], **kwargs: Any) -> None:
    pass


def _single_batch_loader() -> DataLoader:
  return DataLoader([Datum()], batch_size=1)


class TestMonotonicGateTrainerIntegration:
  """BUG-DFV1-001: MonotonicGate integration with real Trainer.fit()."""

  def test_monotonic_gate_rejects_val_metric_regression_via_trainer_fit(self) -> None:
    """BUG-DFV1-001: MonotonicGate on val metrics must detect regression through Trainer.fit().

    Epoch 0: val_accuracy=0.9 (no prior -> accepted).
    Epoch 1: val_accuracy=0.4 (regression from 0.9 -> rejected).
    """
    module = _MonotonicTestModule(
      train_metric=_ConstantTrainMetric(),
      val_metric=_DegradeValMetric(),
    )
    gate = MonotonicGate('val_accuracy', direction='non_decreasing')
    policy = QualityFirstPolicy(gates=[gate])
    experiment = _MonotonicTestExperiment()
    experiment.start()

    trainer = Trainer(policy=policy, experiment=experiment, num_sanity_val_steps=0)
    trainer.fit(
      module,
      train_dataloaders=_single_batch_loader(),
      val_dataloaders=_single_batch_loader(),
      max_epochs=2,
    )

    assert experiment.last_accepted_epoch == 0
    assert experiment.status == Status.completed
    assert 'val_accuracy' in experiment.metrics
    context_reasons = [e.reason for e in experiment.context_log]
    accepted = [r for r in context_reasons if 'accepted' in r and 'epoch 0' in r]
    rejected = [r for r in context_reasons if 'rejected' in r and 'epoch 1' in r]
    assert len(accepted) >= 1
    assert len(rejected) >= 1

  def test_monotonic_gate_train_only_no_regression(self) -> None:
    """Train-only metrics (no val loader), MonotonicGate on loss still works."""

    class _StableLossMetric(Metric):
      """Train loss decreasing across epochs: 0.5 then 0.3."""

      higher_is_better = False

      def __init__(self) -> None:
        super().__init__()
        self._call_count = 0

      def update(self, datum) -> None:
        pass

      def compute(self) -> dict[str, float]:
        self._call_count += 1
        if self._call_count <= 1:
          return {'loss': 0.5}
        return {'loss': 0.3}

      def reset(self) -> None:
        super().reset()

    module = _MonotonicTestModule(train_metric=_StableLossMetric())
    gate = MonotonicGate('loss', direction='non_increasing')
    policy = QualityFirstPolicy(gates=[gate])
    experiment = _MonotonicTestExperiment('train-only-test')
    experiment.start()

    trainer = Trainer(policy=policy, experiment=experiment)
    trainer.fit(
      module,
      train_dataloaders=_single_batch_loader(),
      max_epochs=2,
    )

    assert experiment.status == Status.completed
    assert experiment.last_accepted_epoch == 1
    assert 'loss' in experiment.metrics

  def test_monotonic_gate_single_epoch_always_passes(self) -> None:
    """max_epochs=1: first epoch has no prior, MonotonicGate always passes."""
    module = _MonotonicTestModule(
      train_metric=_ConstantTrainMetric(),
      val_metric=_DegradeValMetric(),
    )
    gate = MonotonicGate('val_accuracy', direction='non_decreasing')
    policy = QualityFirstPolicy(gates=[gate])
    experiment = _MonotonicTestExperiment('single-epoch-test')
    experiment.start()

    trainer = Trainer(policy=policy, experiment=experiment)
    trainer.fit(
      module,
      train_dataloaders=_single_batch_loader(),
      val_dataloaders=_single_batch_loader(),
      max_epochs=1,
    )

    assert experiment.status == Status.completed
    assert experiment.last_accepted_epoch == 0


class TestFinalizeEpochStoresMergedMetrics:
  """BUG-DFV1-001: _finalize_epoch must store merged train_*/val_* metrics."""

  def test_finalize_epoch_stores_merged_metrics_in_experiment(self) -> None:
    """After gate pass, experiment.metrics must contain merged train_*/val_* keys."""
    module = _MonotonicTestModule(
      train_metric=_ConstantTrainMetric(),
      val_metric=_DegradeValMetric(),
    )
    gate = MonotonicGate('val_accuracy', direction='non_decreasing')
    policy = QualityFirstPolicy(gates=[gate])
    experiment = _MonotonicTestExperiment()
    experiment.start()

    trainer = Trainer(policy=policy, experiment=experiment)
    trainer.fit(
      module,
      train_dataloaders=_single_batch_loader(),
      val_dataloaders=_single_batch_loader(),
      max_epochs=1,
    )

    assert 'train_loss' in experiment.metrics
    assert 'val_accuracy' in experiment.metrics
    assert 'loss' not in experiment.metrics
    assert 'accuracy' not in experiment.metrics
    assert experiment.last_accepted_epoch == 0
