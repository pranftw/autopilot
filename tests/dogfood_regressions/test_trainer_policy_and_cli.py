"""Trainer, policy gate, metric normalization, and CLI smoke regression tests.

Covers:
  - BUG-F-002: gate reject marks experiment failed (not completed)
  - BUG-F-003: metric normalization avoids double-prefix (train_train_*)
  - MonotonicGate: non-decreasing / non-increasing direction validation
  - BudgetGate: cost enforcement boundaries
  - Policy CLI exit code contract
  - DataModule checkpoint edge: empty val loader
  - Optimize/AI CLI smoke (parser construction, help)
"""

from autopilot.cli.main import AutoPilotCLI, build_parser
from autopilot.core.enums import Status
from autopilot.core.experiment import Experiment
from autopilot.core.metric import Metric
from autopilot.core.metric_utils import strip_metric_prefix
from autopilot.core.models import Result
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.parameter import Parameter
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import Datum, EvalDatum, GateResult
from autopilot.data.dataloader import DataLoader
from autopilot.policy.gates import BudgetGate, MinGate, MonotonicGate
from autopilot.policy.policy import Policy
from tests.doubles import DirectNumericLoss, NoOpOptimizer
from typing import Any
import pytest


class _FixedMetric(Metric):
  """Metric returning a constant keyed by name."""

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


class _GateModule(AutoPilotModule):
  """Module with configurable metric for gate tests."""

  def __init__(self, metric_name: str = 'accuracy', metric_value: float = 0.5):
    super().__init__()
    self.param = Parameter(requires_grad=True)
    self.loss = DirectNumericLoss([self.param])
    self._opt = NoOpOptimizer([self.param])
    self.metric_obj = _FixedMetric(metric_name, metric_value)

  def forward(self, batch):
    return batch

  def training_step(self, batch, batch_idx):
    return EvalDatum(success=True)

  def validation_step(self, batch, batch_idx):
    return EvalDatum(success=True)

  def configure_optimizers(self):
    return self._opt


class _SequencePolicy(Policy):
  """Policy returning a predetermined GateResult sequence."""

  def __init__(self, sequence: list[GateResult]):
    super().__init__()
    self._sequence = list(sequence)
    self._idx = 0

  def forward(self, result: Result) -> GateResult:
    out = self._sequence[self._idx]
    self._idx += 1
    return out


def _single_batch() -> DataLoader:
  return DataLoader([Datum()], batch_size=1)


class _GateExperiment(Experiment):
  """Experiment bypassing store for gate lifecycle tests."""

  def __init__(self):
    super().__init__(experiment_id='gate-exp')

  def on_epoch_complete(self, epoch: int, metrics: dict[str, float], **kwargs: Any) -> None:
    pass

  def on_validation_complete(self, epoch: int, metrics: dict[str, float], **kwargs: Any) -> None:
    pass


class TestGateRejectExperimentLifecycle:
  """BUG-F-002: gate reject must mark experiment as failed, not completed."""

  @pytest.mark.timeout(1)
  def test_all_epochs_rejected_marks_failed(self) -> None:
    """When policy gate rejects every epoch, experiment status is failed."""
    module = _GateModule()
    policy = _SequencePolicy([GateResult.FAIL])
    exp = _GateExperiment()
    exp.start()

    trainer = Trainer(policy=policy, experiment=exp)
    trainer.fit(module, train_dataloaders=_single_batch(), max_epochs=1)

    assert exp.status == Status.failed
    assert exp.error == 'policy gate rejected all epochs'

  @pytest.mark.timeout(1)
  def test_partial_acceptance_completes(self) -> None:
    """First epoch accepted, second rejected -> experiment completes."""
    module = _GateModule()
    policy = _SequencePolicy([GateResult.PASSED, GateResult.FAIL])
    exp = _GateExperiment()
    exp.start()

    trainer = Trainer(policy=policy, experiment=exp)
    trainer.fit(module, train_dataloaders=_single_batch(), max_epochs=2)

    assert exp.status == Status.completed
    assert exp.last_accepted_epoch == 0

  @pytest.mark.timeout(1)
  def test_no_success_context_when_all_rejected(self) -> None:
    """'experiment completed successfully' must not appear when all rejected."""
    module = _GateModule()
    policy = _SequencePolicy([GateResult.FAIL])
    exp = _GateExperiment()
    exp.start()

    trainer = Trainer(policy=policy, experiment=exp)
    trainer.fit(module, train_dataloaders=_single_batch(), max_epochs=1)

    reasons = [e.reason for e in exp.context_log]
    assert 'experiment completed successfully' not in reasons


class TestTrainerMetricNormalization:
  """BUG-F-003: no double metric prefix train_train_* or val_val_*."""

  @pytest.mark.timeout(1)
  def test_strip_metric_prefix_train(self) -> None:
    """train_ prefix stripped correctly."""
    base, prefix = strip_metric_prefix('train_loss')
    assert base == 'loss'
    assert prefix == 'train_'

  @pytest.mark.timeout(1)
  def test_strip_metric_prefix_val(self) -> None:
    """val_ prefix stripped correctly."""
    base, prefix = strip_metric_prefix('val_accuracy')
    assert base == 'accuracy'
    assert prefix == 'val_'

  @pytest.mark.timeout(1)
  def test_strip_metric_prefix_bare(self) -> None:
    """Bare key returns empty prefix."""
    base, prefix = strip_metric_prefix('accuracy')
    assert base == 'accuracy'
    assert not prefix

  @pytest.mark.timeout(1)
  def test_no_double_prefix_in_final_metrics(self) -> None:
    """BUG-F-003: complete_experiment_success must not double-prefix."""
    exp = Experiment('test-exp')
    exp.start()
    trainer = Trainer(experiment=exp)
    trainer._experiment = exp
    loop_result = {
      'epochs': [
        {
          'epoch': 0,
          'metrics': {'train_loss': 0.1, 'accuracy': 0.9},
          'val_metrics': {'val_f1': 0.8, 'recall': 0.7},
        }
      ]
    }
    trainer._complete_experiment_success(loop_result)
    all_keys = ' '.join(exp.metrics)
    assert 'train_train_' not in all_keys
    assert 'val_val_' not in all_keys
    assert 'train_loss' in exp.metrics
    assert 'train_accuracy' in exp.metrics
    assert 'val_f1' in exp.metrics
    assert 'val_recall' in exp.metrics


class TestMonotonicGate:
  """MonotonicGate regression tests for direction validation and edge cases."""

  @pytest.mark.timeout(1)
  def test_non_decreasing_passes_when_current_gte_prev(self) -> None:
    """non_decreasing: current >= prev -> PASS."""
    gate = MonotonicGate('accuracy', direction='non_decreasing')
    result = Result(
      metrics={'accuracy': 0.9, '_prev_accuracy': 0.8},
      summary='ok',
    )
    assert gate.forward(result) == GateResult.PASSED

  @pytest.mark.timeout(1)
  def test_non_decreasing_fails_when_current_lt_prev(self) -> None:
    """non_decreasing: current < prev -> FAIL."""
    gate = MonotonicGate('accuracy', direction='non_decreasing')
    result = Result(
      metrics={'accuracy': 0.7, '_prev_accuracy': 0.8},
      summary='ok',
    )
    assert gate.forward(result) == GateResult.FAIL

  @pytest.mark.timeout(1)
  def test_non_increasing_passes_when_current_lte_prev(self) -> None:
    """non_increasing: current <= prev -> PASS."""
    gate = MonotonicGate('loss', direction='non_increasing')
    result = Result(
      metrics={'loss': 0.3, '_prev_loss': 0.5},
      summary='ok',
    )
    assert gate.forward(result) == GateResult.PASSED

  @pytest.mark.timeout(1)
  def test_non_increasing_fails_when_current_gt_prev(self) -> None:
    """non_increasing: current > prev -> FAIL."""
    gate = MonotonicGate('loss', direction='non_increasing')
    result = Result(
      metrics={'loss': 0.6, '_prev_loss': 0.5},
      summary='ok',
    )
    assert gate.forward(result) == GateResult.FAIL

  @pytest.mark.timeout(1)
  def test_first_epoch_passes_when_no_prev(self) -> None:
    """No _prev_ key means first epoch -> PASS (baseline)."""
    gate = MonotonicGate('accuracy')
    result = Result(metrics={'accuracy': 0.5}, summary='ok')
    assert gate.forward(result) == GateResult.PASSED

  @pytest.mark.timeout(1)
  def test_missing_current_metric_fails(self) -> None:
    """Missing current metric key -> FAIL."""
    gate = MonotonicGate('accuracy')
    result = Result(metrics={}, summary='ok')
    assert gate.forward(result) == GateResult.FAIL

  @pytest.mark.timeout(1)
  def test_invalid_direction_raises(self) -> None:
    """Invalid direction string raises ValueError."""
    with pytest.raises(ValueError, match='non_decreasing'):
      MonotonicGate('accuracy', direction='sideways')


class TestBudgetGate:
  """BudgetGate regression tests for cost enforcement boundaries."""

  @pytest.mark.timeout(1)
  def test_under_budget_passes(self) -> None:
    """Cost below max_usd -> PASS."""
    gate = BudgetGate(max_usd=50.0)
    result = Result(metrics={'cost_usd': 25.0}, summary='ok')
    assert gate.forward(result) == GateResult.PASSED

  @pytest.mark.timeout(1)
  def test_at_boundary_passes(self) -> None:
    """Cost == max_usd -> PASS (boundary inclusive)."""
    gate = BudgetGate(max_usd=50.0)
    result = Result(metrics={'cost_usd': 50.0}, summary='ok')
    assert gate.forward(result) == GateResult.PASSED

  @pytest.mark.timeout(1)
  def test_over_budget_fails(self) -> None:
    """Cost > max_usd -> FAIL."""
    gate = BudgetGate(max_usd=50.0)
    result = Result(metrics={'cost_usd': 50.01}, summary='ok')
    assert gate.forward(result) == GateResult.FAIL

  @pytest.mark.timeout(1)
  def test_missing_cost_metric_fails(self) -> None:
    """Missing cost_usd metric -> FAIL (fail-closed)."""
    gate = BudgetGate(max_usd=50.0)
    result = Result(metrics={}, summary='ok')
    assert gate.forward(result) == GateResult.FAIL


class TestPolicyExitCodeContract:
  """Policy gate success/failure result contract."""

  @pytest.mark.timeout(1)
  def test_threshold_gate_pass_returns_passed(self) -> None:
    """ThresholdGate above threshold returns PASSED."""
    gate = MinGate('accuracy', threshold=0.5)
    result = Result(metrics={'accuracy': 0.8}, summary='ok')
    assert gate.forward(result) == GateResult.PASSED

  @pytest.mark.timeout(1)
  def test_min_gate_fail_returns_fail(self) -> None:
    """MinGate below threshold returns FAIL."""
    gate = MinGate('accuracy', threshold=0.9)
    result = Result(metrics={'accuracy': 0.5}, summary='ok')
    assert gate.forward(result) == GateResult.FAIL


class TestDataModuleCheckpointEdge:
  """DataModule edge case: empty validation loader."""

  @pytest.mark.timeout(1)
  def test_trainer_handles_no_val_loader(self) -> None:
    """Training without a val_dataloader should complete without errors."""
    module = _GateModule()
    trainer = Trainer()
    result = trainer.fit(module, train_dataloaders=_single_batch(), max_epochs=1)
    assert result['total_epochs'] == 1


class TestCLISmokeMatrix:
  """Section 2.5: AI CLI / optimize / data edge matrix smoke tests."""

  @pytest.mark.timeout(1)
  def test_optimize_dry_run_parser_accepts_flags(self) -> None:
    """Optimize parser accepts --dry-run and --json without import errors."""
    parser = build_parser()
    args = parser.parse_args(
      [
        'optimize',
        'preflight',
        '--experiment',
        'dummy',
        '--workspace',
        '/tmp/ws',
        '--dry-run',
        '--json',
        '--context',
        'test',
      ]
    )
    assert args.dry_run is True
    assert args.use_json is True

  @pytest.mark.timeout(1)
  def test_ai_cli_group_registered(self) -> None:
    """AI CLI group is registered and lists subcommands without import errors."""
    cli = AutoPilotCLI()
    assert 'ai' in cli.commands
