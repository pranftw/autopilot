"""torchmetrics-style metric base classes.

Metric extends Module (like torchmetrics.Metric extends nn.Module).
Metrics assigned as attributes on a Module auto-register into _modules.
"""

from autopilot.core.models import Datum
from autopilot.core.module import Module
from typing import Any


class Metric(Module):
  """Base metric. Extends Module so it auto-registers as a child module.

  update() per batch, compute() per epoch, reset() between epochs.
  forward() calls update() and returns compute() (torchmetrics pattern).
  """

  def __init__(self) -> None:
    super().__init__()
    self._state: dict[str, Any] = {}

  def name(self) -> str:
    return type(self).__name__

  def forward(self, datum: Datum) -> dict[str, float]:
    """Call update() then return compute(). Like torchmetrics.Metric.forward()."""
    self.update(datum)
    return self.compute()

  def update(self, datum: Datum) -> None:
    raise NotImplementedError

  def compute(self) -> dict[str, float]:
    raise NotImplementedError

  def reset(self) -> None:
    self._state = {}

  def __add__(self, other: 'Metric') -> 'CompositeMetric':
    return CompositeMetric([self, other])

  def __repr__(self) -> str:
    return f'{type(self).__name__}()'


class CompositeMetric(Metric):
  """Multiple metrics composed via __add__."""

  def __init__(self, metrics: list['Metric']) -> None:
    super().__init__()
    self._parts = list(metrics)
    for i, m in enumerate(self._parts):
      setattr(self, f'metric_{i}', m)

  def forward(self, datum: Datum) -> dict[str, float]:
    self.update(datum)
    return self.compute()

  def update(self, datum: Datum) -> None:
    for m in self._parts:
      m.update(datum)

  def compute(self) -> dict[str, float]:
    result: dict[str, float] = {}
    for m in self._parts:
      result.update(m.compute())
    return result

  def reset(self) -> None:
    for m in self._parts:
      m.reset()

  def __repr__(self) -> str:
    names = ', '.join(type(m).__name__ for m in self._parts)
    return f'CompositeMetric([{names}])'
