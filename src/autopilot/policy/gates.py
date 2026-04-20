"""Composable scoring gates. Like nn.Module for loss functions."""

from autopilot.core.models import GateResult, Result
from typing import Any


class Gate:
  """Base class for scoring gates. Subclass and override forward()."""

  def __init__(self, metric: str, *, required: bool = True) -> None:
    self.metric = metric
    self.required = required

  def forward(self, result: Result) -> GateResult:
    raise NotImplementedError

  def __call__(self, result: Result) -> GateResult:
    return self.forward(result)

  def explain(self, result: Result) -> str:
    gate_out = self.forward(result)
    return f'{type(self).__name__}({self.metric}): {gate_out.value}'

  def __repr__(self) -> str:
    return f'{type(self).__name__}({self.metric!r}, required={self.required})'


class MinGate(Gate):
  """Passes if metric >= threshold."""

  def __init__(self, metric: str, threshold: float, *, required: bool = True) -> None:
    super().__init__(metric, required=required)
    self.threshold = threshold

  def forward(self, result: Result) -> GateResult:
    value = result.metrics.get(self.metric)
    if value is None:
      return GateResult.SKIP
    return GateResult.PASS if value >= self.threshold else GateResult.FAIL

  def explain(self, result: Result) -> str:
    value = result.metrics.get(self.metric)
    if value is None:
      return f'{self.metric}: skipped (not in result)'
    passed = value >= self.threshold
    outcome = 'pass' if passed else 'fail'
    return f'{self.metric}: {value:.3f} >= {self.threshold} -> {outcome}'


class MaxGate(Gate):
  """Passes if metric <= threshold."""

  def __init__(self, metric: str, threshold: float, *, required: bool = True) -> None:
    super().__init__(metric, required=required)
    self.threshold = threshold

  def forward(self, result: Result) -> GateResult:
    value = result.metrics.get(self.metric)
    if value is None:
      return GateResult.SKIP
    return GateResult.PASS if value <= self.threshold else GateResult.FAIL


class RangeGate(Gate):
  """Passes if min <= metric <= max."""

  def __init__(
    self,
    metric: str,
    min: float,
    max: float,
    *,
    required: bool = True,
  ) -> None:
    super().__init__(metric, required=required)
    self.min = min
    self.max = max

  def forward(self, result: Result) -> GateResult:
    value = result.metrics.get(self.metric)
    if value is None:
      return GateResult.SKIP
    return GateResult.PASS if self.min <= value <= self.max else GateResult.FAIL


class CustomGate(Gate):
  """Gate with a custom evaluation function."""

  def __init__(
    self,
    metric: str,
    fn: Any,
    *,
    required: bool = True,
  ) -> None:
    super().__init__(metric, required=required)
    self._fn = fn

  def forward(self, result: Result) -> GateResult:
    value = result.metrics.get(self.metric)
    if value is None:
      return GateResult.SKIP
    return GateResult.PASS if self._fn(value) else GateResult.FAIL
