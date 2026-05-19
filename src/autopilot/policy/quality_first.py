"""Quality-first policy: gates must pass before promotion is considered.

Triggers human review on warn-level gate results when configured.
"""

from autopilot.core.constraint import ConstraintResult, gate_to_constraint
from autopilot.core.metric import Metric
from autopilot.core.models import Result
from autopilot.core.types import EvalDatum, GateResult
from autopilot.policy.gates import (
  BudgetGate,
  Gate,
  MaxGate,
  MinGate,
  MonotonicGate,
  RangeGate,
  collect_gate_hints,
)
from autopilot.policy.policy import Policy


def _gate_threshold_str(gate: Gate) -> str:
  """Human-readable threshold for ConstraintResult payloads.

  Returns:
    Operator-prefixed threshold for MinGate/MaxGate, bracketed range for
    RangeGate, direction/epsilon for MonotonicGate, USD amount for BudgetGate.

  Raises:
    TypeError: If gate is not a recognized gate type.
  """
  if isinstance(gate, RangeGate):
    return f'[{gate.min_value}, {gate.max_value}]'
  if isinstance(gate, MinGate):
    return f'>= {gate.threshold}'
  if isinstance(gate, MaxGate):
    return f'<= {gate.threshold}'
  if isinstance(gate, MonotonicGate):
    return f'{gate.direction} (epsilon={gate.epsilon})'
  if isinstance(gate, BudgetGate):
    return f'{gate.max_usd} USD'
  msg = (
    f'unrecognized gate type {type(gate).__name__}; expected one of '
    f'MinGate, MaxGate, RangeGate, MonotonicGate, BudgetGate'
  )
  raise TypeError(msg)


GATE_TYPE_MAP: dict[str, type[Gate]] = {
  'MinGate': MinGate,
  'MaxGate': MaxGate,
  'RangeGate': RangeGate,
  'MonotonicGate': MonotonicGate,
  'BudgetGate': BudgetGate,
}


class QualityFirstPolicy(Policy):
  """All required gates must pass. Warn triggers human review if configured."""

  def __init__(
    self,
    gates: list[Gate] | None = None,
    human_review_on_warn: bool = True,
  ) -> None:
    """Configure ordered gates and warning behavior.

    Args:
      gates: Gates evaluated on each ``Result`` (empty means unconditional pass).
      human_review_on_warn: Whether optional-gate failures mention human review.
    """
    self._gates = gates if gates is not None else []
    self._human_review_on_warn = human_review_on_warn
    self._last_results: dict[Gate, GateResult] | None = None

  @property
  def gates(self) -> list[Gate]:
    """Return the ordered list of gates for inspection.

    Returns:
      Gates evaluated on each ``Result``.
    """
    return self._gates

  def gate_hints(self) -> dict[str, str]:
    """Return metric-mismatch hints from gates after evaluation."""
    return collect_gate_hints(self._gates)

  def name(self) -> str:
    """Return the canonical ``quality_first`` slug."""
    return 'quality_first'

  def forward(self, result: Result) -> GateResult:
    """Run all gates and summarize required vs optional failures.

    Populates ``result.gates`` with per-gate structured constraint rows (parity with
    :meth:`QualityFirstMetric.to_result`) for trainer policy-gate journaling.

    Returns:
      PASS when all gates succeed, WARN for optional-only failures, FAIL otherwise.
    """
    self._last_results = {g: g(result) for g in self._gates}
    assert self._last_results is not None
    constraints: list[ConstraintResult] = []
    for gate in self._gates:
      gate_out = self._last_results[gate]
      constraints.append(
        gate_to_constraint(
          type(gate).__name__,
          gate_out,
          gate.metric,
          result.metrics.get(gate.metric),
          _gate_threshold_str(gate),
        )
      )
    result.gates = constraints
    required_failed = any(r == GateResult.FAIL for g, r in self._last_results.items() if g.required)
    optional_failed = any(
      r == GateResult.FAIL for g, r in self._last_results.items() if not g.required
    )
    if required_failed:
      return GateResult.FAIL
    if optional_failed:
      return GateResult.WARN
    return GateResult.PASSED

  def __call__(self, result: Result) -> GateResult:
    """Delegate to ``forward``.

    Returns:
      Same ``GateResult`` as ``forward``.
    """
    return self.forward(result)

  def explain(self, result: Result) -> str:
    """Human-readable summary; optional failures are not labeled as required.

    Appends metric-mismatch hints from failed gates when available so
    plain-text explain output matches JSON ``gate_hints``.

    Returns:
      Human-readable summary of which gate groups failed, with required and
      optional failures reported separately.
    """
    if self._last_results is None:
      self.forward(result)
    assert self._last_results is not None
    outcome_map = self._last_results
    failed = [g.metric for g, r in outcome_map.items() if r == GateResult.FAIL and g.required]
    optional_failed = [
      g.metric for g, r in outcome_map.items() if r == GateResult.FAIL and not g.required
    ]
    if not failed and not optional_failed:
      return 'all gates passed'

    hint_lines = [
      g.hint for g, r in outcome_map.items() if r == GateResult.FAIL and g.hint is not None
    ]

    if failed and optional_failed:
      msg = f'required gates failed: {failed}; optional gates also failed: {optional_failed}'
    elif failed:
      msg = f'required gates failed: {failed}'
    else:
      msg = f'optional gate(s) failed: {optional_failed}'
      if self._human_review_on_warn:
        msg = f'{msg} - human review triggered'

    if hint_lines:
      msg = msg + '\n' + '\n'.join(hint_lines)
    return msg

  @property
  def human_review_on_warn(self) -> bool:
    """Whether optional-gate failures trigger human review mention.

    Returns:
      Configuration flag set at construction time.
    """
    return self._human_review_on_warn

  def state_dict(self) -> dict:
    """Serialize policy configuration for checkpoint persistence.

    Returns:
      Dict with gates list (each serialized via gate.state_dict()) and
      human_review_on_warn flag.
    """
    return {
      'gates': [g.state_dict() for g in self._gates],
      'human_review_on_warn': self._human_review_on_warn,
    }

  def load_state_dict(self, state: dict) -> None:
    """Restore policy configuration from checkpoint data.

    Replaces current gates with deserialized instances from ``state``.
    Uses ``GATE_TYPE_MAP`` to dispatch to the correct ``from_dict`` classmethod.
    Unrecognized gate type strings propagate as ``KeyError``.

    Args:
      state: Dict previously produced by ``state_dict()``.
    """
    gate_dicts = state['gates']
    self._gates = [GATE_TYPE_MAP[g['type']].from_dict(g) for g in gate_dicts]
    self._human_review_on_warn = state['human_review_on_warn']


class QualityFirstMetric(Metric):
  """Metric that accumulates datum metrics and applies quality-first gates on compute()."""

  higher_is_better = True

  def __init__(self, gates: list[Gate] | None = None) -> None:
    """Create metric state plus optional gate list for ``to_result``."""
    super().__init__()
    self._gates = gates if gates is not None else []
    self.add_state('_accumulated', dict)

  def name(self) -> str:
    """Return the canonical ``quality_first`` metric name."""
    return 'quality_first'

  def update(self, datum: EvalDatum) -> None:  # type: ignore[ty:invalid-method-override]  # intentional: metric requires EvalDatum for .metrics
    """Append per-key metric lists from one ``EvalDatum``."""
    for key, value in datum.metrics.items():
      self._accumulated.setdefault(key, []).append(value)

  def compute(self) -> dict[str, float]:
    """Average each accumulated metric list into scalar floats.

    Returns:
      Mapping from metric name to mean value (0.0 for empty lists).
    """
    metrics: dict[str, float] = {}
    for key, values in self._accumulated.items():
      metrics[key] = sum(values) / len(values) if values else 0.0
    return metrics

  def to_result(self, metrics: dict[str, float] | None = None) -> Result:
    """Build a Result by applying gates to the given or computed metrics.

    Each gate is evaluated exactly once; the cached ``GateResult`` is reused
    for both the per-gate constraint results and the computed ``passed`` property.

    Returns:
      ``Result`` annotated with per-gate ``ConstraintResult`` entries.
    """
    if metrics is None:
      metrics = self.compute()
    eval_result = Result(metrics=metrics)
    constraints: list[ConstraintResult] = []
    for gate in self._gates:
      gate_out = gate(eval_result)
      value = eval_result.metrics.get(gate.metric)
      threshold = self._threshold_str(gate)
      constraints.append(
        gate_to_constraint(
          type(gate).__name__,
          gate_out,
          gate.metric,
          value,
          threshold,
        )
      )
    eval_result.gates = constraints
    return eval_result

  def _threshold_str(self, gate: Gate) -> str:
    """Extract a human-readable threshold description from a gate.

    Args:
      gate: Gate instance to describe.

    Returns:
      Threshold string like '>= 0.8'.
    """
    return _gate_threshold_str(gate)
