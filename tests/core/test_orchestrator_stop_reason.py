"""Tests for EpochOrchestrator.stop_reason public property."""

from autopilot.core.loops.orchestrator import EpochOrchestrator, OrchestratorConfig
from autopilot.core.metric import Metric
from autopilot.core.models import Result
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import EvalDatum, GateResult
from autopilot.policy.policy import Policy
from typing import Any


class _SimpleModule(AutoPilotModule):
  """Module that returns configurable accuracy."""

  def __init__(self, accuracy: float = 0.9) -> None:
    super().__init__()
    self._acc = accuracy

  def forward(self, batch: Any) -> EvalDatum:
    return EvalDatum(success=True)

  def training_step(self, batch: Any, batch_idx: int) -> EvalDatum:
    return EvalDatum(success=True, metrics={'accuracy': self._acc})

  def configure_optimizers(self) -> None:
    return None


class _AccMetric(Metric):
  """Simple accuracy metric that returns a fixed value from data."""

  def __init__(self) -> None:
    super().__init__()
    self.add_state('_value', 0.0)
    self.add_state('_count', 0)

  def update(self, datum: Any) -> None:
    self._count += 1
    if hasattr(datum, 'metrics') and datum.metrics:
      self._value = datum.metrics.get('accuracy', self._value)

  def compute(self) -> dict[str, float]:
    return {'accuracy': self._value}


class _FailPolicy(Policy):
  """Policy that always rejects."""

  def __init__(self, gates: list[Any] | None = None) -> None:
    self._gates = gates or []

  def forward(self, result: Result) -> GateResult:
    from autopilot.core.constraint import ConstraintResult

    result.gates = [
      ConstraintResult(
        name='MinGate',
        passed=False,
        metric='accuracy',
        value=result.metrics.get('accuracy'),
        threshold='>= 0.5',
      )
    ]
    return GateResult.FAIL


class _PassPolicy(Policy):
  """Policy that always passes."""

  def forward(self, result: Result) -> GateResult:
    return GateResult.PASSED


class _CallbackStopTrainer(Trainer):
  """Trainer that signals stop at epoch 1 via should_stop_at."""

  def __init__(self, **kwargs: Any) -> None:
    super().__init__(**kwargs)
    self._stop_at_epoch = 1

  def should_stop_at(self, hook_method: Any, **kwargs: Any) -> bool:
    epoch = kwargs.get('epoch', 0)
    return epoch >= self._stop_at_epoch


class TestStopReasonProperty:
  """Tests for the public stop_reason property on EpochOrchestrator."""

  def test_stop_reason_before_run_none(self) -> None:
    orch = EpochOrchestrator()
    assert orch.stop_reason is None

  def test_stop_reason_after_max_epochs(self) -> None:
    module = _SimpleModule(accuracy=0.9)
    module.acc_metric = _AccMetric()
    orch = EpochOrchestrator()
    trainer = Trainer(loop=orch)
    trainer.fit(module, train_dataloaders=[1, 2], max_epochs=3)
    assert orch.stop_reason is None

  def test_stop_reason_after_callback_stop(self) -> None:
    module = _SimpleModule(accuracy=0.9)
    module.acc_metric = _AccMetric()
    orch = EpochOrchestrator()
    trainer = _CallbackStopTrainer(loop=orch)
    trainer.fit(module, train_dataloaders=[1, 2], max_epochs=5)
    assert orch.stop_reason == 'callback_stop'

  def test_stop_reason_after_plateau(self) -> None:
    module = _SimpleModule(accuracy=0.5)
    module.acc_metric = _AccMetric()
    orch_config = OrchestratorConfig(
      plateau_window=3,
      plateau_threshold=0.05,
      monitor='accuracy',
    )
    orch = EpochOrchestrator(config=orch_config)
    trainer = Trainer(loop=orch)
    trainer.fit(module, train_dataloaders=[1, 2, 3], max_epochs=10)
    assert orch.stop_reason == 'plateau'

  def test_stop_reason_after_policy_reject(self) -> None:
    module = _SimpleModule(accuracy=0.3)
    module.acc_metric = _AccMetric()
    policy = _FailPolicy()
    orch = EpochOrchestrator()
    trainer = Trainer(loop=orch, policy=policy)
    trainer.fit(module, train_dataloaders=[1, 2], max_epochs=5)
    assert orch.stop_reason == 'policy_fail'

  def test_stop_reason_matches_run_result(self) -> None:
    module = _SimpleModule(accuracy=0.3)
    module.acc_metric = _AccMetric()
    policy = _FailPolicy()
    orch = EpochOrchestrator()
    trainer = Trainer(loop=orch, policy=policy)
    result = trainer.fit(module, train_dataloaders=[1, 2], max_epochs=5)
    assert result['stop_reason'] == orch.stop_reason
