"""Tests for BUG-F-002: gate reject lifecycle fix.

When the policy gate rejects all epochs, the experiment must be marked as
``failed`` (not ``completed``).  When some epochs are accepted (partial
acceptance), the experiment completes normally with the last accepted
metrics.

Regression IDs:
  - BUG-F-002: _fit_success_path always called _complete_experiment_success
    even when every epoch ended with ``stopped: True`` from the policy gate.
"""

from autopilot.core.enums import Status
from autopilot.core.experiment import Experiment
from autopilot.core.metric import Metric
from autopilot.core.models import Result
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.parameter import Parameter
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import Datum, EvalDatum, GateResult
from autopilot.data.dataloader import DataLoader
from autopilot.policy.policy import Policy
from tests.doubles import DirectNumericLoss, NoOpOptimizer
from typing import Any


class _FixedMetric(Metric):
  """Metric returning a fixed value for gate lifecycle tests."""

  higher_is_better = True

  def __init__(self, value: float) -> None:
    super().__init__()
    self._value = value

  def update(self, datum) -> None:
    """No-op accumulation."""

  def compute(self) -> dict[str, float]:
    """Return the fixed accuracy metric."""
    return {'accuracy': self._value}

  def reset(self) -> None:
    """No-op reset."""


class _GateModule(AutoPilotModule):
  """Minimal module with a metric for gate lifecycle tests."""

  def __init__(self, accuracy: float = 0.5):
    super().__init__()
    self.param = Parameter(requires_grad=True)
    self.loss = DirectNumericLoss([self.param])
    self._opt = NoOpOptimizer([self.param])
    self.accuracy = _FixedMetric(accuracy)

  def forward(self, batch):
    return batch

  def training_step(self, batch, batch_idx):
    return EvalDatum(success=True)

  def validation_step(self, batch, batch_idx):
    return EvalDatum(success=True)

  def configure_optimizers(self):
    return self._opt


class _SequencePolicy(Policy):
  """Policy that returns a predetermined sequence of gate results."""

  def __init__(self, sequence: list[GateResult]):
    super().__init__()
    self._sequence = list(sequence)
    self._idx = 0

  def forward(self, result: Result) -> GateResult:
    out = self._sequence[self._idx]
    self._idx += 1
    return out


def _single_batch_loader() -> DataLoader:
  return DataLoader([Datum()], batch_size=1)


class _GateExperiment(Experiment):
  """Experiment subclass that bypasses store for gate lifecycle tests."""

  def __init__(self):
    super().__init__(experiment_id='gate-exp')

  def on_epoch_complete(self, epoch: int, metrics: dict[str, float], **kwargs: Any) -> None:
    pass

  def on_validation_complete(self, epoch: int, metrics: dict[str, float], **kwargs: Any) -> None:
    pass


class TestGateRejectExperimentLifecycle:
  """BUG-F-002 regression tests for gate reject lifecycle."""

  def test_all_epochs_rejected_by_gate_marks_experiment_failed(self):
    """When the policy gate rejects every epoch, experiment.fail() is called.

    BUG-F-002: previously _complete_experiment_success was called
    unconditionally, marking rejected experiments as completed.
    """
    module = _GateModule()
    policy = _SequencePolicy([GateResult.FAIL])
    experiment = _GateExperiment()
    experiment.start()

    trainer = Trainer(policy=policy, experiment=experiment)
    trainer.fit(module, train_dataloaders=_single_batch_loader(), max_epochs=1)

    assert experiment.status == Status.failed
    assert experiment.error == 'policy gate rejected all epochs'

  def test_partial_gate_acceptance_completes_with_last_accepted_epoch(self):
    """When first epoch passes and second is rejected, experiment completes.

    Context order: gate-stop context first, then completion context.
    """
    module = _GateModule()
    policy = _SequencePolicy([GateResult.PASSED, GateResult.FAIL])
    experiment = _GateExperiment()
    experiment.start()

    trainer = Trainer(policy=policy, experiment=experiment)
    trainer.fit(module, train_dataloaders=_single_batch_loader(), max_epochs=2)

    assert experiment.status == Status.completed
    assert experiment.last_accepted_epoch == 0
    assert experiment.metrics

  def test_no_success_context_when_all_epochs_rejected_by_gate(self):
    """No 'experiment completed successfully' context when all epochs rejected.

    BUG-F-002: the premature success context entry must not appear when the
    policy gate rejects every epoch.
    """
    module = _GateModule()
    policy = _SequencePolicy([GateResult.FAIL])
    experiment = _GateExperiment()
    experiment.start()

    trainer = Trainer(policy=policy, experiment=experiment)
    trainer.fit(module, train_dataloaders=_single_batch_loader(), max_epochs=1)

    context_reasons = [e.reason for e in experiment.context_log]
    assert 'experiment completed successfully' not in context_reasons

    gate_fail_entries = [
      e for e in experiment.context_log if 'policy gate rejected all epochs' in e.reason
    ]
    assert len(gate_fail_entries) >= 1

  def test_no_experiment_skips_failure_path(self):
    """When experiment is None, gate rejection does not raise AttributeError.

    The no-experiment path must call _complete_experiment_success (which
    no-ops when experiment is None) without attempting experiment.fail().
    """
    module = _GateModule()
    policy = _SequencePolicy([GateResult.FAIL])

    trainer = Trainer(policy=policy)
    result = trainer.fit(module, train_dataloaders=_single_batch_loader(), max_epochs=1)

    assert result['total_epochs'] == 1
    assert result['epochs'][0].get('stopped') is True
