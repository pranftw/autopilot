"""torchmetrics-style metric base classes.

Metric extends Module (like torchmetrics.Metric extends nn.Module).
Metrics assigned as attributes on a Module auto-register into _modules.
"""

from autopilot.core.module.module import Module
from autopilot.core.types import Datum
from copy import deepcopy
from typing import Any, cast
import functools


class Metric(Module):
  """torchmetrics-style metric base.

  Unlike PyTorch Lightning's ``Metric.update(preds, targets)`` two-argument
  pattern, AutoPilot metrics take a **single Datum**: ``update(datum)``.
  Extract predictions and targets via ``datum.items`` or subclass fields.

  ``compute()`` raises ``RuntimeError`` when called without prior ``update()``.
  This prevents silent returns of meaningless initial values.

  Attributes:
    higher_is_better: Optional directional hint used by trainers and policies.
    _defaults: Registered state factories for ``reset``.
    _update_count: Number of ``update`` calls since last ``reset``.
    _computed: Cached result of ``compute()``; invalidated on ``update`` or ``reset``.

  Extends Module so it auto-registers as a child module on the parent.
  Trainer.fit() collects Metric instances from module.named_modules()
  (excluding Loss) and builds metric_metadata from higher_is_better.

  Extension points:
    update(datum)  -- accumulate from one batch (must implement)
    compute()      -- return metric dict from accumulated state (must implement)

  State management:
    add_state(name, default)  -- register accumulator with reset default
    reset()                   -- restore all states to defaults (no override needed)

  Auto-wrapping: __init_subclass__ wraps update() to increment _update_count.
  The update_count property tracks calls since last reset.

  Composition: Metric + Metric -> MetricCollection via __add__.
  MetricCollection dispatches update/compute/reset to children with
  optional prefix/postfix namespacing. Raises on key collision.

  Example:
    >>> from autopilot.core.metric import Metric
    >>> from autopilot.core.types import Datum, EvalDatum
    >>>
    >>> class SuccessRate(Metric):
    ...   def __init__(self):
    ...     super().__init__()
    ...     self.add_state('correct', lambda: 0)
    ...     self.add_state('total', lambda: 0)
    ...
    ...   def update(self, datum):
    ...     for item in datum.items:
    ...       if isinstance(item, EvalDatum):
    ...         self.correct += int(item.success)
    ...         self.total += 1
    ...
    ...   def compute(self):
    ...     return {'success_rate': self.correct / max(self.total, 1)}
    >>>
    >>> metric = SuccessRate()
    >>> metric.update(Datum(items=[EvalDatum(success=True)]))
    >>> metric.compute()
    {'success_rate': 1.0}
  """

  higher_is_better: bool | None = None

  def __init_subclass__(cls, **kwargs: Any) -> None:
    """Wrap ``update``/``compute``/``reset`` to track call counts and memoization."""
    super().__init_subclass__(**kwargs)
    if 'update' in cls.__dict__:
      original = cls.__dict__['update']

      @functools.wraps(original)
      def wrapped_update(self: Any, *args: Any, **kw: Any) -> None:
        self._computed = None
        self._update_count += 1
        return original(self, *args, **kw)

      cls.update = cast(Any, wrapped_update)
    if 'compute' in cls.__dict__:
      original_compute = cls.__dict__['compute']

      @functools.wraps(original_compute)
      def wrapped_compute(self: Any, *args: Any, **kw: Any) -> dict[str, float]:
        if self._computed is not None:
          return self._computed
        if self._update_count == 0:
          msg = (
            f'{type(self).__name__}.compute() called without prior update(). '
            f'Call update() at least once before compute().'
          )
          raise RuntimeError(msg)
        out = original_compute(self, *args, **kw)
        self._computed = out
        return out

      cls.compute = cast(Any, wrapped_compute)
    if 'reset' in cls.__dict__:
      original_reset = cls.__dict__['reset']

      @functools.wraps(original_reset)
      def wrapped_reset(self: Any, *args: Any, **kw: Any) -> None:
        self._computed = None
        self._update_count = 0
        return original_reset(self, *args, **kw)

      cls.reset = cast(Any, wrapped_reset)

  def __init__(self) -> None:
    """Initialize metric state containers and memoized compute slot."""
    super().__init__()
    self._defaults: dict[str, Any] = {}
    self._update_count: int = 0
    self._computed: dict[str, float] | None = None

  def forward(self, *args: Any, **kwargs: Any) -> Datum:
    """Not used -- metrics mutate state via ``update`` / ``compute``.

    In the non-tensor domain there is no forward graph for metric objects;
    calling ``forward`` is unsupported and mirrors ``nn.Module``'s requirement
    to override ``forward`` elsewhere.

    Raises:
      NotImplementedError: Always -- intentionally not implemented for metrics.
    """
    raise NotImplementedError

  def add_state(self, name: str, default: Any) -> None:
    """Register metric state with a default value.

    Call in __init__ after super().__init__().
    default can be a value (int, float) or a callable factory (list, dict).
    The state is accessible as self.<name> and auto-reset by reset().
    """
    self._defaults[name] = default
    value = default() if callable(default) else default
    setattr(self, name, value)

  def update(self, datum: Datum) -> None:
    """Accumulate metric state from one datum/batch. Must override.

    Unlike PyTorch Lightning's ``update(preds, targets)`` two-argument
    convention, AutoPilot metrics accept a **single** ``Datum``. Extract
    predictions and targets from ``datum.items`` or subclass-specific
    fields. Passing two positional arguments (e.g. ``update(preds, targets)``)
    raises a ``TypeError`` from Python's argument binding.

    Args:
      datum: The input Datum containing batch data for metric accumulation.
    """
    raise NotImplementedError

  def compute(self) -> dict[str, float]:
    """Compute metric values from accumulated state. Must override.

    Returns:
      Dict mapping metric names to float values.

    Raises:
      RuntimeError: When called without prior ``update()``.
    """
    raise NotImplementedError

  def reset(self) -> None:
    """Reset all registered states to defaults. Subclasses should NOT need to override."""
    self._update_count = 0
    self._computed = None
    for name, default in self._defaults.items():
      value = default() if callable(default) else default
      setattr(self, name, value)

  @property
  def update_count(self) -> int:
    """Number of times update() has been called since last reset."""
    return self._update_count

  def name(self) -> str:
    """Metric identity for logging keys.

    Returns:
      This class's ``__name__``.
    """
    return type(self).__name__

  def clone(self) -> 'Metric':
    """Deep copy this metric (fresh state, same config).

    Returns:
      Independent duplicate suitable for a separate run.
    """
    return deepcopy(self)

  def __add__(self, other: 'Metric') -> 'MetricCollection':
    """Compose two metrics into a ``MetricCollection``.

    Returns:
      Collection containing ``self`` and ``other``.
    """
    return MetricCollection([self, other])

  def __repr__(self) -> str:
    """Return the concrete metric class name."""
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
    """Register child metrics with optional key prefix/postfix.

    Args:
      metrics: Mapping of name to metric, or list (names from ``metric.name()``).
      prefix: Optional prefix for flattened compute keys.
      postfix: Optional postfix for flattened compute keys.

    Raises:
      ValueError: When list metrics yield duplicate names or flattened keys collide
        during ``compute``.
    """
    super().__init__()
    if isinstance(metrics, list):
      names = [m.name() for m in metrics]
      if len(names) != len(set(names)):
        msg = f'duplicate metric names: {names}'
        raise ValueError(msg)
      metrics = dict(zip(names, metrics, strict=True))
    self._prefix = prefix
    self._postfix = postfix
    self._metric_keys: list[str] = list(metrics)
    for key, m in metrics.items():
      setattr(self, key, m)

  def update(self, datum: Datum) -> None:
    """Forward ``update`` to every child metric."""
    for key in self._metric_keys:
      getattr(self, key).update(datum)

  def compute(self) -> dict[str, float]:
    """Merge child ``compute`` results with prefix/postfix namespacing.

    Returns:
      Flattened dict of all child metrics' scalar outputs.

    Raises:
      ValueError: On key collision after namespacing.
    """
    result: dict[str, float] = {}
    for key in self._metric_keys:
      m = getattr(self, key)
      for mk, mv in m.compute().items():
        pre = '' if self._prefix is None else self._prefix
        post = '' if self._postfix is None else self._postfix
        full_key = f'{pre}{mk}{post}'
        if full_key in result:
          msg = f'metric key collision: {full_key!r}'
          raise ValueError(msg)
        result[full_key] = mv
    return result

  def reset(self) -> None:
    """Reset this collection and every child metric."""
    super().reset()
    for key in self._metric_keys:
      getattr(self, key).reset()

  def clone(self) -> 'MetricCollection':
    """Deep copy the whole collection and child metrics.

    Returns:
      Independent ``MetricCollection`` with copied child state.
    """
    return deepcopy(self)

  def __repr__(self) -> str:
    """Describe prefix/postfix and child metric keys.

    Returns:
      A ``MetricCollection(...)`` debug string.
    """
    names = ', '.join(self._metric_keys)
    pre = f'prefix={self._prefix!r}, ' if self._prefix else ''
    post = f'postfix={self._postfix!r}, ' if self._postfix else ''
    return f'MetricCollection({pre}{post}[{names}])'
