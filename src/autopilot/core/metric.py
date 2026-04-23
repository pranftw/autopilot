"""torchmetrics-style metric base classes.

Metric extends Module (like torchmetrics.Metric extends nn.Module).
Metrics assigned as attributes on a Module auto-register into _modules.
"""

from autopilot.core.module import Module
from autopilot.core.types import Datum
from copy import deepcopy
from typing import Any


class Metric(Module):
  """torchmetrics-style metric base.

  Extends Module so it auto-registers as a child module on the parent.
  Trainer.fit() collects Metric instances from module.named_modules()
  (excluding Loss) and builds metric_metadata from higher_is_better.

  Extension points:
    update(datum)  -- accumulate from one batch (must implement)
    compute()      -- return metric dict from accumulated state (must implement)

  State management:
    add_state(name, default)  -- register accumulator with reset default
    reset()                   -- restore all states to defaults (no override needed)

  Class attributes:
    higher_is_better (bool | None) -- True/False for regression detection,
                                      None if directionality is unknown

  Auto-wrapping: __init_subclass__ wraps update() to increment _update_count.
  The update_count property tracks calls since last reset.

  Composition: Metric + Metric -> MetricCollection via __add__.
  MetricCollection dispatches update/compute/reset to children with
  optional prefix/postfix namespacing. Raises on key collision.
  """

  higher_is_better: bool | None = None

  def __init_subclass__(cls, **kwargs: Any) -> None:
    super().__init_subclass__(**kwargs)
    if 'update' in cls.__dict__:
      original = cls.__dict__['update']

      def wrapped_update(self, *args, **kw):
        self._update_count += 1
        return original(self, *args, **kw)

      cls.update = wrapped_update

  def __init__(self) -> None:
    super().__init__()
    self._defaults: dict[str, Any] = {}
    self._update_count: int = 0

  def add_state(self, name: str, default: Any) -> None:
    """Register metric state with a default value.

    Call in __init__ after super().__init__().
    default can be a value (int, float) or a callable factory (list, dict).
    The state is accessible as self.<name> and auto-reset by reset().
    """
    self._defaults[name] = default
    value = default() if callable(default) else default
    object.__setattr__(self, name, value)

  def update(self, datum: Datum) -> None:
    """Accumulate metric state from one datum/batch. Must override."""
    raise NotImplementedError

  def compute(self) -> dict[str, float]:
    """Compute metric values from accumulated state. Must override."""
    raise NotImplementedError

  def reset(self) -> None:
    """Reset all registered states to defaults. Subclasses should NOT need to override."""
    self._update_count = 0
    for name, default in self._defaults.items():
      value = default() if callable(default) else default
      object.__setattr__(self, name, value)

  @property
  def update_count(self) -> int:
    """Number of times update() has been called since last reset."""
    return self._update_count

  def name(self) -> str:
    """Metric identity for logging keys."""
    return type(self).__name__

  def clone(self) -> 'Metric':
    """Deep copy this metric (fresh state, same config)."""
    return deepcopy(self)

  def __add__(self, other: 'Metric') -> 'MetricCollection':
    """Compose two metrics into a MetricCollection."""
    return MetricCollection([self, other])

  def __repr__(self) -> str:
    return f'{type(self).__name__}()'


class MetricCollection(Metric):
  """Named collection of metrics with prefix/postfix namespacing.

  Like torchmetrics.MetricCollection: accepts dict or list of metrics,
  dispatches update/compute/reset to children, raises on key collision.
  """

  higher_is_better: bool | None = None

  def __init__(
    self,
    metrics: 'dict[str, Metric] | list[Metric]',
    prefix: str | None = None,
    postfix: str | None = None,
  ) -> None:
    super().__init__()
    if isinstance(metrics, list):
      names = [m.name() for m in metrics]
      if len(names) != len(set(names)):
        raise ValueError(f'duplicate metric names: {names}')
      metrics = dict(zip(names, metrics, strict=True))
    self._prefix = prefix
    self._postfix = postfix
    self._metric_keys: list[str] = list(metrics.keys())
    for key, m in metrics.items():
      setattr(self, key, m)

  def update(self, datum: Datum) -> None:
    for key in self._metric_keys:
      getattr(self, key).update(datum)

  def compute(self) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in self._metric_keys:
      m = getattr(self, key)
      for mk, mv in m.compute().items():
        full_key = f'{self._prefix or ""}{mk}{self._postfix or ""}'
        if full_key in result:
          raise ValueError(f'metric key collision: {full_key!r}')
        result[full_key] = mv
    return result

  def reset(self) -> None:
    super().reset()
    for key in self._metric_keys:
      getattr(self, key).reset()

  def clone(self) -> 'MetricCollection':
    return deepcopy(self)

  def __repr__(self) -> str:
    names = ', '.join(self._metric_keys)
    pre = f'prefix={self._prefix!r}, ' if self._prefix else ''
    post = f'postfix={self._postfix!r}, ' if self._postfix else ''
    return f'MetricCollection({pre}{post}[{names}])'
