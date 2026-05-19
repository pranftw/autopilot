"""Shared behavioral test doubles for the minimal training stack.

Holds canonical noop Module / Loss / Optimizer implementations reused across
core, unit, and integration tests.  Local specialized stubs (counting losses,
stateful metrics, graph modules) should remain in individual test files when
tests assert on counters, metrics, or graph shape.

Evaluation doubles (``make_run_config``, ``EvalDatumIterable``) live alongside
production doubles from plan 10.
"""

from autopilot.ai.evaluation.schemas import RetryConfig, RunConfig
from autopilot.core.experiment import Experiment
from autopilot.core.gradient import NumericGradient
from autopilot.core.loss import Loss
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.module.module import Module
from autopilot.core.optimizer import Optimizer
from autopilot.core.types import Datum, EvalDatum
from autopilot.data.dataset import IterableDataset
from dataclasses import dataclass, field


class NoopEvalModule(AutoPilotModule):
  """Minimal module: constant EvalDatum(success=True), no optimizer."""

  def forward(self, *args, **kwargs) -> Datum:
    return EvalDatum(success=True)

  def training_step(self, batch, batch_idx) -> EvalDatum:
    return EvalDatum(success=True)

  def validation_step(self, batch, batch_idx) -> EvalDatum:
    return EvalDatum(success=True)

  def test_step(self, batch, batch_idx) -> EvalDatum:
    return EvalDatum(success=True)

  def predict_step(self, batch, batch_idx) -> EvalDatum:
    return EvalDatum(success=True)

  def configure_optimizers(self):
    return None


class NoOpOptimizer(Optimizer):
  """Optimizer that does nothing on step."""

  def step(self) -> None:
    pass


class DirectNumericLoss(Loss):
  """Shared test double: assigns NumericGradient(value=1.0) to scoped parameters."""

  def forward(self, data: Datum, targets: Datum | None = None) -> None:
    pass

  def backward(self) -> None:
    for p in self._loss_parameters:
      if p.requires_grad:
        p.grad = NumericGradient(value=1.0)

  def reset(self) -> None:
    pass


class TrackingNumericLoss(Loss):
  """Loss that assigns NumericGradient(1.0) and counts forward/backward/reset calls."""

  def __init__(self, params=None):
    super().__init__(params)
    self.forward_calls = 0
    self.backward_calls = 0
    self.reset_calls = 0

  def forward(self, data: Datum, targets: Datum | None = None) -> None:
    self.forward_calls += 1

  def backward(self) -> None:
    self.backward_calls += 1
    for p in self._loss_parameters:
      if p.requires_grad:
        p.grad = NumericGradient(value=1.0)

  def reset(self) -> None:
    self.reset_calls += 1


class TrackingOptimizer(Optimizer):
  """Optimizer that counts step() and zero_grad() calls."""

  def __init__(self, params, lr=1.0):
    super().__init__(params, lr=lr)
    self.step_calls = 0
    self.zero_grad_calls = 0

  def step(self) -> None:
    self.step_calls += 1

  def zero_grad(self) -> None:
    self.zero_grad_calls += 1
    super().zero_grad()


class StepCountingOptimizer(Optimizer):
  """Optimizer that counts step() calls only (no zero_grad tracking)."""

  def __init__(self, params, lr=1.0):
    super().__init__(params, lr=lr)
    self.step_count = 0

  def step(self) -> None:
    self.step_count += 1


class StateTrackingOptimizer(Optimizer):
  """Optimizer that writes a per-parameter marker into ``self.state`` on step.

  Proves ``state_dict`` / ``load_state_dict`` round-trip of per-parameter
  optimizer state keyed by ``Parameter.id`` strings (Plan 25).
  """

  def __init__(self, params, lr=1.0):
    super().__init__(params, lr=lr)
    self._step_count = 0

  def step(self) -> None:
    self._step_count += 1
    for param in self.parameters:
      entry = self.state.setdefault(id(param), {})
      entry['plan25_marker'] = self._step_count


class PlainModule(Module):
  """Plain Module (not AutoPilotModule) for Trainer.fit type-error tests."""

  def forward(self, *args, **kwargs):
    return EvalDatum(success=True)


@dataclass
class MockEvaluationOutput:
  """Shared test double for EvaluationOutputProtocol."""

  infos: list[str] = field(default_factory=list)
  warns: list[str] = field(default_factory=list)
  results: list[tuple[dict, bool]] = field(default_factory=list)

  def info(self, message: str) -> None:
    self.infos.append(message)

  def warn(self, message: str) -> None:
    self.warns.append(message)

  def result(self, payload: dict, ok: bool = True) -> None:
    self.results.append((payload, ok))


def make_run_config(*, num_parallel: int = 1) -> RunConfig:
  """Shared RunConfig factory for evaluation tests."""
  return RunConfig(
    model='test-model',
    num_parallel=num_parallel,
    max_rpm=100,
    rpm_safety_margin=1.0,
    retry=RetryConfig(
      max_retries=1,
      min_timeout_ms=100,
      max_timeout_ms=1000,
      backoff_factor=2,
    ),
    max_tool_steps=5,
    max_output_tokens=1024,
  )


def make_completed_experiment(
  exp_id: str,
  hypothesis: str,
  metrics: dict[str, float],
) -> Experiment:
  """Build a completed experiment with given metrics.

  Canonical shared helper for tests that need a completed experiment
  with specific metrics.  Replaces per-file ``_make_completed`` stubs.

  Args:
    exp_id: Unique experiment identifier.
    hypothesis: Experiment hypothesis string.
    metrics: Metric name-to-value mapping.

  Returns:
    A completed ``Experiment`` instance.
  """
  exp = Experiment(experiment_id=exp_id, hypothesis=hypothesis)
  exp.start()
  exp.complete(metrics=metrics)
  return exp


class EvalDatumIterable(IterableDataset[EvalDatum]):
  """Canonical iterable yielding EvalDatum(metadata={'idx': i}).

  Replaces the prior ``_DatumIterable`` / ``_CountingIterable`` helpers
  where those helpers were identical.
  """

  def __init__(self, n: int) -> None:
    self._n = n

  def __iter__(self):
    for i in range(self._n):
      yield EvalDatum(metadata={'idx': i})
