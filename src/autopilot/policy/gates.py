"""Composable scoring gates. Like nn.Module for loss functions.

``forward(result)`` clears ``hint``, reads the configured metric with an exact
``Result.metrics`` key (no fuzzy resolution at decision time - downstream code
must align keys with Trainer/EpochLoop output), then compares using
normal float ordering - NaN fails because comparisons return False. A missing key
sets ``GateResult.FAIL`` and stores a ``difflib``-derived ``hint`` string for
operators/policies to surface without re-deriving suggestions from scratch.
"""

from autopilot.core.models import Result
from autopilot.core.types import GateResult
from collections.abc import Callable
from typing import Any
import difflib

METRIC_SIMILARITY_CUTOFF = 0.4

# --- metric lookup hints ---


def _suggest_closest_metric(target: str, available: list[str]) -> str | None:
  """Return the single best metric name match using sequence matching.

  Args:
    target: Metric name the gate was configured with.
    available: Sorted list of metric names present in the result.

  Returns:
    Closest match above the similarity cutoff, or ``None``.
  """
  matches = difflib.get_close_matches(target, available, n=1, cutoff=METRIC_SIMILARITY_CUTOFF)
  return matches[0] if matches else None


def _suggest_closest_metrics_tied(target: str, available: list[str]) -> list[str]:
  """Return all metric names tied for the best similarity score above cutoff.

  Uses ``SequenceMatcher.ratio()`` to score each candidate and returns every
  name whose ratio equals the best score (above the cutoff), sorted for
  determinism.

  Args:
    target: Metric name the gate was configured with.
    available: List of metric names present in the result.

  Returns:
    Sorted list of equally-close matches, or empty list.
  """
  scored: list[tuple[float, str]] = []
  for name in available:
    ratio = difflib.SequenceMatcher(None, target, name).ratio()
    if ratio >= METRIC_SIMILARITY_CUTOFF:
      scored.append((ratio, name))
  if not scored:
    return []
  best_ratio = max(s[0] for s in scored)
  return sorted(name for ratio, name in scored if ratio == best_ratio)


def _format_metric_unavailable_hint(requested: str, available: list[str]) -> str:
  """Build an actionable hint when a gate's metric is not found in the result.

  Args:
    requested: Metric name the gate was configured with.
    available: All metric keys present in the result.

  Returns:
    Multi-line hint listing available keys and closest-match suggestions.
  """
  sorted_keys = sorted(available)
  hint = f'metric {requested!r} not found; available: {sorted_keys}'
  suggestions = _suggest_closest_metrics_tied(requested, sorted_keys)
  if suggestions:
    hint += f'; did you mean: {suggestions}'
  return hint


def _check_metric_available(
  gate: 'Gate',
  metric_value: float | None,
  result: Result,
) -> GateResult | None:
  """Return FAIL and set ``gate.hint`` when ``metric_value`` is ``None``.

  Centralizes the repeated ``if value is None: return GateResult.FAIL`` prelude
  shared by ``MinGate``, ``MaxGate``, ``RangeGate``, ``CustomGate``, and
  ``BudgetGate``.

  Keeps miss handling in one place so every numeric gate emits the same hint
  shape and fail-closed semantics when Trainer/EpochLoop keys drift.

  Args:
    gate: Gate instance whose ``hint`` is set on metric mismatch.
    metric_value: Value extracted from ``Result.metrics.get(metric_name)``.
    result: Full ``Result`` providing ``metrics`` keys for hint generation.

  Returns:
    ``GateResult.FAIL`` when ``metric_value`` is ``None``, or ``None`` so the
    caller proceeds with the numeric path.
  """
  if metric_value is None:
    gate.hint = _format_metric_unavailable_hint(gate.metric, list(result.metrics.keys()))
    return GateResult.FAIL
  return None


# --- gate base ---


class Gate:
  """Base class for scoring gates. Subclass and override forward().

  Gates are objects with forward(result) -> GateResult, not dict-driven helpers.
  Wire gates inside Policy subclasses (e.g. QualityFirstPolicy(gates=[...])).

  NaN metric values result in FAIL (NaN comparisons are always False).
  Missing metrics (key not in Result) result in FAIL with an ephemeral
  ``hint`` populated via ``difflib`` closest-match suggestions.

  Built-ins: MinGate (>= threshold), MaxGate (<= threshold),
  RangeGate (min <= value <= max), CustomGate (fn(value) -> bool).

  explain() format: '{GateName}({metric}): {value} {op} {threshold} -> {PASS|FAIL}'.
  Missing metric: '{GateName}({metric}): missing -> FAIL; {hint}'.

  Attributes:
    metric: Key in ``Result.metrics`` to evaluate.
    required: When false, failures downgrade to warnings at policy level.
    hint: Ephemeral diagnostic string set when the configured metric is
      not found in ``Result.metrics``. Cleared at the start of each
      ``forward()`` call. Not serialized.

  Clearing ``hint`` on every ``forward()`` keeps reused gate instances from
  surfacing diagnostics tied to an older ``Result`` when metrics keys move.

  See ``ConstraintResult`` for structured pass/fail output. See ``Policy``
  for gate composition.

  Example:
    >>> from autopilot.policy.gates import RangeGate, MinGate
    >>> from autopilot.core.models import Result
    >>>
    >>> gate = MinGate('accuracy', threshold=0.8)
    >>> result = Result(metrics={'accuracy': 0.85}, gates=[])
    >>> gate.forward(result).value
    'pass'
  """

  def __init__(self, metric: str, *, required: bool = True) -> None:
    """Configure which metric this gate reads and whether it is required.

    Args:
      metric: Key in ``Result.metrics`` to evaluate.
      required: When false, failures downgrade to warnings at policy level.
    """
    self.metric = metric
    self.required = required
    self.hint: str | None = None

  def forward(self, result: Result) -> GateResult:
    """Evaluate the gate; subclasses must implement.

    Raises:
      NotImplementedError: On the base class.
    """
    raise NotImplementedError

  def __call__(self, result: Result) -> GateResult:
    """``forward`` alias for callable syntax.

    Returns:
      Outcome from ``forward``.
    """
    return self.forward(result)

  def explain(self, result: Result) -> str:
    """Return a human-readable explanation of the gate decision for ``result``.

    Returns:
      Short status line including gate class and evaluated value.
    """
    gate_out = self.forward(result)
    return f'{type(self).__name__}({self.metric}): {gate_out.value}'

  def format_missing_explanation(self) -> str:
    """Build the missing-metric explanation line, appending hint when set.

    Returns:
      ``'{GateName}({metric}): missing -> FAIL'`` with optional ``'; {hint}'``.
    """
    base = f'{type(self).__name__}({self.metric}): missing -> FAIL'
    if self.hint is not None:
      return f'{base}; {self.hint}'
    return base

  def state_dict(self) -> dict:
    """Serialize gate configuration to a dictionary.

    Raises:
      NotImplementedError: On the base class.
    """
    raise NotImplementedError

  @classmethod
  def from_dict(cls, data: dict) -> 'Gate':
    """Deserialize a gate from its dictionary representation.

    Args:
      data: Dictionary produced by ``state_dict()``.

    Raises:
      NotImplementedError: On the base class.
    """
    raise NotImplementedError

  def __repr__(self) -> str:
    """Return a debug representation with metric and required flag."""
    return f'{type(self).__name__}({self.metric!r}, required={self.required})'


# --- numeric threshold gates ---


class MinGate(Gate):
  """Passes if metric >= threshold.

  NaN metric values result in FAIL (NaN >= threshold is False).
  Missing metrics (key not in Result) result in FAIL.
  """

  def __init__(self, metric: str, threshold: float, *, required: bool = True) -> None:
    """Create a lower-bound gate on ``metric``.

    Args:
      metric: Key in ``Result.metrics``.
      threshold: Inclusive minimum passing value.
      required: Whether failures block promotion.
    """
    super().__init__(metric, required=required)
    self.threshold = threshold

  def forward(self, result: Result) -> GateResult:
    """Return PASS when present metric is at least ``threshold``."""
    self.hint = None
    value = result.metrics.get(self.metric)
    unavailable = _check_metric_available(self, value, result)
    if unavailable is not None:
      return unavailable
    assert value is not None
    return GateResult.PASSED if value >= self.threshold else GateResult.FAIL

  def explain(self, result: Result) -> str:
    """Return a line comparing the metric to the minimum threshold.

    Returns:
      Explanation string with PASS/FAIL suffix.
    """
    self.forward(result)
    value = result.metrics.get(self.metric)
    if value is None:
      return self.format_missing_explanation()
    gate_result = self(result)
    return f'MinGate({self.metric}): {value} >= {self.threshold} -> {gate_result.name}'

  def state_dict(self) -> dict:
    """Serialize gate configuration for checkpoint persistence.

    Returns:
      Dict with type discriminator and gate parameters.
    """
    return {
      'type': 'MinGate',
      'metric': self.metric,
      'threshold': self.threshold,
      'required': self.required,
    }

  @classmethod
  def from_dict(cls, data: dict) -> 'MinGate':
    """Restore a MinGate from serialized state.

    Args:
      data: Dict previously produced by ``state_dict()``.

    Returns:
      New MinGate instance with restored configuration.
    """
    return cls(
      metric=data['metric'],
      threshold=data['threshold'],
      required=data['required'],
    )


class MaxGate(Gate):
  """Passes if metric <= threshold.

  NaN metric values result in FAIL (NaN <= threshold is False).
  Missing metrics (key not in Result) result in FAIL.
  """

  def __init__(self, metric: str, threshold: float, *, required: bool = True) -> None:
    """Create an upper-bound gate on ``metric``.

    Args:
      metric: Key in ``Result.metrics``.
      threshold: Inclusive maximum passing value.
      required: Whether failures block promotion.
    """
    super().__init__(metric, required=required)
    self.threshold = threshold

  def forward(self, result: Result) -> GateResult:
    """Return PASS when present metric is at most ``threshold``."""
    self.hint = None
    value = result.metrics.get(self.metric)
    unavailable = _check_metric_available(self, value, result)
    if unavailable is not None:
      return unavailable
    assert value is not None
    return GateResult.PASSED if value <= self.threshold else GateResult.FAIL

  def explain(self, result: Result) -> str:
    """Return a line comparing the metric to the maximum threshold.

    Returns:
      Explanation string with PASS/FAIL suffix.
    """
    self.forward(result)
    value = result.metrics.get(self.metric)
    if value is None:
      return self.format_missing_explanation()
    gate_result = self(result)
    return f'MaxGate({self.metric}): {value} <= {self.threshold} -> {gate_result.name}'

  def state_dict(self) -> dict:
    """Serialize gate configuration for checkpoint persistence.

    Returns:
      Dict with type discriminator and gate parameters.
    """
    return {
      'type': 'MaxGate',
      'metric': self.metric,
      'threshold': self.threshold,
      'required': self.required,
    }

  @classmethod
  def from_dict(cls, data: dict) -> 'MaxGate':
    """Restore a MaxGate from serialized state.

    Args:
      data: Dict previously produced by ``state_dict()``.

    Returns:
      New MaxGate instance with restored configuration.
    """
    return cls(
      metric=data['metric'],
      threshold=data['threshold'],
      required=data['required'],
    )


# --- range gate ---


_RANGE_GATE_SENTINEL = object()


class RangeGate(Gate):
  """Passes if min_value <= metric <= max_value.

  NaN metric values result in FAIL (NaN comparisons are always False).
  Missing metrics (key not in Result) result in FAIL.

  Common typos (``low``/``high``) raise guided ``TypeError``
  suggesting the correct ``min_value``/``max_value`` names.
  """

  def __init__(
    self,
    metric: str,
    min_value: float = _RANGE_GATE_SENTINEL,  # type: ignore[assignment]  # ty: ignore[invalid-parameter-default]
    max_value: float = _RANGE_GATE_SENTINEL,  # type: ignore[assignment]  # ty: ignore[invalid-parameter-default]
    *,
    required: bool = True,
    **kwargs: Any,
  ) -> None:
    """Create a range gate checking that a metric falls within bounds.

    Common typos ``low``/``high`` raise guided ``TypeError`` suggesting
    ``min_value``/``max_value``.

    Args:
      metric: Name of the metric to check.
      min_value: Lower bound (inclusive).
      max_value: Upper bound (inclusive).
      required: Whether failures block promotion at the policy level.
      **kwargs: Caught for did-you-mean hints.

    Raises:
      TypeError: When bounds are omitted or wrong kwarg names are used.
      ValueError: When ``min_value > max_value``.
    """
    hints = {'low': 'min_value', 'high': 'max_value'}
    for wrong, right in hints.items():
      if wrong in kwargs:
        msg = (
          f'{type(self).__name__}() got unexpected keyword argument {wrong!r}. '
          f'Use {right!r} instead.'
        )
        raise TypeError(msg)
    if kwargs:
      msg = f'{type(self).__name__}() got unexpected keyword arguments: {", ".join(sorted(kwargs))}'
      raise TypeError(msg)
    missing = []
    if min_value is _RANGE_GATE_SENTINEL:
      missing.append('min_value')
    if max_value is _RANGE_GATE_SENTINEL:
      missing.append('max_value')
    if missing:
      names = ' and '.join(missing)
      msg = f'RangeGate() missing required argument(s): {names}.'
      raise TypeError(msg)
    if min_value > max_value:
      msg = f'RangeGate min_value ({min_value}) must be <= max_value ({max_value})'
      raise ValueError(msg)
    super().__init__(metric, required=required)
    self.min_value = min_value
    self.max_value = max_value

  def forward(self, result: Result) -> GateResult:
    """Return PASS when the metric lies inside ``[min_value, max_value]``."""
    self.hint = None
    value = result.metrics.get(self.metric)
    unavailable = _check_metric_available(self, value, result)
    if unavailable is not None:
      return unavailable
    assert value is not None
    return GateResult.PASSED if self.min_value <= value <= self.max_value else GateResult.FAIL

  def explain(self, result: Result) -> str:
    """Return a line showing interval membership for the metric.

    Returns:
      Explanation string with PASS/FAIL suffix.
    """
    self.forward(result)
    value = result.metrics.get(self.metric)
    if value is None:
      return self.format_missing_explanation()
    gate_result = self(result)
    return (
      f'RangeGate({self.metric}): {value} in'
      f' [{self.min_value}, {self.max_value}] -> {gate_result.name}'
    )

  def state_dict(self) -> dict:
    """Serialize gate configuration for checkpoint persistence.

    Returns:
      Dict with type discriminator and gate parameters.
    """
    return {
      'type': 'RangeGate',
      'metric': self.metric,
      'min_value': self.min_value,
      'max_value': self.max_value,
      'required': self.required,
    }

  @classmethod
  def from_dict(cls, data: dict) -> 'RangeGate':
    """Restore a RangeGate from serialized state.

    Args:
      data: Dict previously produced by ``state_dict()``.

    Returns:
      New RangeGate instance with restored configuration.
    """
    return cls(
      metric=data['metric'],
      min_value=data['min_value'],
      max_value=data['max_value'],
      required=data['required'],
    )


# --- custom predicate gate ---


class CustomGate(Gate):
  """Gate with a custom evaluation function.

  NaN metric values result in FAIL if the function returns False for NaN.
  Missing metrics (key not in Result) result in FAIL.
  """

  def __init__(
    self,
    metric: str,
    fn: Callable[[float], bool],
    *,
    required: bool = True,
  ) -> None:
    """Create a gate backed by arbitrary predicate ``fn``.

    Args:
      metric: Key in ``Result.metrics``.
      fn: Callable returning true when the metric passes.
      required: Whether failures block promotion.
    """
    super().__init__(metric, required=required)
    self._fn = fn

  def forward(self, result: Result) -> GateResult:
    """Return PASS when ``fn(metric_value)`` is true."""
    self.hint = None
    value = result.metrics.get(self.metric)
    unavailable = _check_metric_available(self, value, result)
    if unavailable is not None:
      return unavailable
    assert value is not None
    return GateResult.PASSED if self._fn(value) else GateResult.FAIL

  def explain(self, result: Result) -> str:
    """Return a line naming the callable and its outcome.

    Returns:
      Explanation string with PASS/FAIL suffix.
    """
    self.forward(result)
    value = result.metrics.get(self.metric)
    if value is None:
      return self.format_missing_explanation()
    gate_result = self(result)
    fn_name = getattr(self._fn, '__name__', repr(self._fn))
    return f'CustomGate({self.metric}): {value} via {fn_name} -> {gate_result.name}'


# --- monotonic and budget gates ---


class MonotonicGate(Gate):
  """Require a metric to never decrease (or never increase) across epochs.

  Compares the current metric value against the prior epoch's value, which
  is injected into the ``Result.metrics`` dict by ``EpochLoop`` under the
  reserved ``_prev_<metric>`` key prefix (single leading underscore).

  An optional absolute ``epsilon`` tolerance absorbs small drifts from noisy
  metrics without triggering a gate failure:

    - ``non_decreasing``: pass iff ``current >= prev - epsilon``
    - ``non_increasing``: pass iff ``current <= prev + epsilon``

  Default ``epsilon=0.0`` preserves strict comparison behavior.

  Semantics:
    - **First epoch / no history:** ``_prev_`` key absent -> **PASS** (baseline).
    - **Missing current** (metric key absent or ``None``): **FAIL**.
    - **NaN:** Python float comparisons make ``NaN >= x`` False, so NaN
      in either position causes a monotonic check to fail.

  Attributes:
    metric: Key in ``Result.metrics`` to evaluate (inherited from ``Gate``).
    required: Whether failures block promotion (inherited from ``Gate``).
  """

  VALID_DIRECTIONS = frozenset(('non_decreasing', 'non_increasing'))

  def __init__(
    self,
    metric: str,
    *,
    direction: str = 'non_decreasing',
    required: bool = True,
    epsilon: float = 0.0,
  ) -> None:
    """Create a monotonic constraint gate.

    Args:
      metric: Key in ``Result.metrics`` to evaluate.
      direction: Either ``'non_decreasing'`` (current >= prev) or
        ``'non_increasing'`` (current <= prev).
      required: Whether failures block promotion.
      epsilon: Absolute tolerance for noisy metrics. Must be non-negative.
        Default ``0.0`` preserves strict comparison behavior.

    Raises:
      ValueError: When ``direction`` is not a recognized value or
        ``epsilon`` is negative.
    """
    super().__init__(metric, required=required)
    if direction not in self.VALID_DIRECTIONS:
      valid = ', '.join(sorted(self.VALID_DIRECTIONS))
      msg = f'MonotonicGate direction must be one of {valid}, got {direction!r}'
      raise ValueError(msg)
    if epsilon < 0:
      msg = f'MonotonicGate epsilon must be non-negative, got {epsilon!r}'
      raise ValueError(msg)
    self._direction = direction
    self._epsilon = epsilon

  @property
  def direction(self) -> str:
    """Monotonic comparison direction ('non_decreasing' or 'non_increasing').

    Returns:
      Direction configured at construction time.
    """
    return self._direction

  @property
  def epsilon(self) -> float:
    """Absolute tolerance for noisy metric comparisons.

    Returns:
      Epsilon configured at construction time.
    """
    return self._epsilon

  def forward(self, result: Result) -> GateResult:
    """Return PASS when the metric satisfies the monotonic constraint.

    Returns:
      ``GateResult.PASSED`` when the constraint holds (or no prior exists),
      ``GateResult.FAIL`` when the metric is missing or violates monotonicity.
    """
    self.hint = None
    current = result.metrics.get(self.metric)
    if current is None:
      self.hint = _format_metric_unavailable_hint(self.metric, list(result.metrics.keys()))
      return GateResult.FAIL
    prev = result.metrics.get(f'_prev_{self.metric}')
    if prev is None:
      return GateResult.PASSED
    if self._direction == 'non_decreasing':
      return GateResult.PASSED if current >= prev - self._epsilon else GateResult.FAIL
    return GateResult.PASSED if current <= prev + self._epsilon else GateResult.FAIL

  def explain(self, result: Result) -> str:
    """Return a human-readable explanation of the monotonic check.

    When a prior value exists, the comparison bound includes ``epsilon`` exactly
    as in ``forward()`` (``prev - epsilon`` for ``non_decreasing``,
    ``prev + epsilon`` for ``non_increasing``).

    Returns:
      Explanation string including direction, values, epsilon-adjusted bound when
      applicable, and outcome.
    """
    gate_result = self.forward(result)
    current = result.metrics.get(self.metric)
    if current is None:
      return self.format_missing_explanation()
    prev = result.metrics.get(f'_prev_{self.metric}')
    if prev is None:
      return (
        f'MonotonicGate({self.metric}): {current}'
        f' {self._direction} (no prior) -> {gate_result.name}'
      )
    if self._direction == 'non_decreasing':
      bound = prev - self._epsilon
      op = '>='
    else:
      bound = prev + self._epsilon
      op = '<='
    return f'MonotonicGate({self.metric}): {current} {op} {bound} -> {gate_result.name}'

  def state_dict(self) -> dict:
    """Serialize gate configuration for checkpoint persistence.

    Returns:
      Dict with type discriminator and gate parameters.
    """
    return {
      'type': 'MonotonicGate',
      'metric': self.metric,
      'direction': self._direction,
      'required': self.required,
      'epsilon': self._epsilon,
    }

  @classmethod
  def from_dict(cls, data: dict) -> 'MonotonicGate':
    """Restore a MonotonicGate from serialized state.

    Args:
      data: Dict previously produced by ``state_dict()``.

    Returns:
      New MonotonicGate instance with restored configuration.
    """
    return cls(
      metric=data['metric'],
      direction=data['direction'],
      required=data['required'],
      epsilon=data['epsilon'],
    )


_BUDGET_GATE_SENTINEL = object()


class BudgetGate(Gate):
  """Reject epochs where cumulative cost exceeds a budget.

  Reads ``cost_usd`` from ``Result.metrics`` (injected by ``EpochLoop``
  from ``CostTrackerCallback.cumulative_usd``). Opt-in only -- never
  auto-attached by Trainer. Wire via policy composition:

  .. code-block:: python

      policy = QualityFirstPolicy(
        gates=[
          MinGate('val_accuracy', 0.8),
          BudgetGate(max_usd=50.0),
        ]
      )

  Semantics:
    - **cost <= max_usd:** PASS (boundary inclusive).
    - **cost > max_usd:** FAIL.
    - **Missing ``cost_usd``:** FAIL (fail-closed, same as other gates).
  """

  def __init__(
    self,
    max_usd: float = _BUDGET_GATE_SENTINEL,  # type: ignore[assignment]  # ty: ignore[invalid-parameter-default]
    *,
    required: bool = True,
    **kwargs: Any,
  ) -> None:
    """Create a budget-enforcement gate.

    Common typo ``budget`` raises guided ``TypeError`` suggesting ``max_usd``.

    Args:
      max_usd: Maximum cumulative budget in USD (inclusive).
      required: Whether failures block promotion at the policy level.
      **kwargs: Caught for did-you-mean hints (e.g. ``budget`` -> ``max_usd``).

    Raises:
      TypeError: When ``max_usd`` is omitted, or a known-wrong kwarg
        (e.g. ``budget``) or any unexpected keyword argument is provided.
    """
    hints = {'budget': 'max_usd'}
    for wrong, right in hints.items():
      if wrong in kwargs:
        msg = (
          f'{type(self).__name__}() got unexpected keyword argument {wrong!r}. '
          f'Use {right!r} instead.'
        )
        raise TypeError(msg)
    if kwargs:
      msg = f'{type(self).__name__}() got unexpected keyword arguments: {", ".join(sorted(kwargs))}'
      raise TypeError(msg)
    if max_usd is _BUDGET_GATE_SENTINEL:
      msg = (
        'BudgetGate() missing required argument: max_usd. '
        'Pass max_usd=<amount> to set the budget limit.'
      )
      raise TypeError(msg)
    super().__init__('cost_usd', required=required)
    self._max_usd = max_usd

  @property
  def max_usd(self) -> float:
    """Maximum cumulative budget in USD (inclusive).

    Returns:
      Budget ceiling configured at construction time.
    """
    return self._max_usd

  def forward(self, result: Result) -> GateResult:
    """Return PASS when cumulative cost is at or below ``max_usd``.

    Returns:
      ``GateResult.PASSED`` when ``cost_usd <= max_usd``,
      ``GateResult.FAIL`` when missing or over budget.
    """
    self.hint = None
    cost = result.metrics.get(self.metric)
    unavailable = _check_metric_available(self, cost, result)
    if unavailable is not None:
      return unavailable
    assert cost is not None
    return GateResult.PASSED if cost <= self._max_usd else GateResult.FAIL

  def explain(self, result: Result) -> str:
    """Return a line comparing cumulative cost to the budget.

    Returns:
      Explanation string with PASS/FAIL suffix.
    """
    self.forward(result)
    cost = result.metrics.get(self.metric)
    if cost is None:
      return self.format_missing_explanation()
    gate_result = self(result)
    return f'BudgetGate({self.metric}): {cost} <= {self._max_usd} -> {gate_result.name}'

  def state_dict(self) -> dict:
    """Serialize gate configuration for checkpoint persistence.

    Returns:
      Dict with type discriminator and gate parameters.
    """
    return {
      'type': 'BudgetGate',
      'max_usd': self._max_usd,
      'required': self.required,
    }

  @classmethod
  def from_dict(cls, data: dict) -> 'BudgetGate':
    """Restore a BudgetGate from serialized state.

    Args:
      data: Dict previously produced by ``state_dict()``.

    Returns:
      New BudgetGate instance with restored configuration.
    """
    return cls(
      max_usd=data['max_usd'],
      required=data['required'],
    )


# --- hint aggregation ---


def collect_gate_hints(gates: list[Gate]) -> dict[str, str]:
  """Collect metric-mismatch hints from evaluated gates.

  Args:
    gates: List of gates that have been evaluated via ``forward()``.

  Returns:
    Mapping of gate metric name to hint string for gates that
    have a non-None hint after the last ``forward()`` call.

  Keys use each gate's configured ``metric`` field; if multiple gates share the
  same key, the last gate in ``gates`` wins - callers should keep names distinct.
  """
  hints: dict[str, str] = {}
  for gate in gates:
    if gate.hint is not None:
      hints[gate.metric] = gate.hint
  return hints
