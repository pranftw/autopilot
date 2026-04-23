"""Quality-first policy: gates must pass before promotion is considered.

Triggers human review on warn-level gate results when configured.
"""

from autopilot.core.metric import Metric
from autopilot.core.models import Result
from autopilot.core.types import Datum, GateResult
from autopilot.policy.gates import Gate
from autopilot.policy.policy import Policy


class QualityFirstPolicy(Policy):
  """All required gates must pass. Warn triggers human review if configured."""

  def __init__(
    self,
    gates: list[Gate] | None = None,
    human_review_on_warn: bool = True,
  ) -> None:
    self._gates = gates or []
    self._human_review_on_warn = human_review_on_warn

  def name(self) -> str:
    return 'quality_first'

  def forward(self, result: Result) -> GateResult:
    required_failed = False
    optional_failed = False
    for gate in self._gates:
      gate_out = gate(result)
      if gate_out == GateResult.FAIL:
        if gate.required:
          required_failed = True
        else:
          optional_failed = True
    if required_failed:
      return GateResult.FAIL
    if optional_failed:
      return GateResult.WARN
    return GateResult.PASS

  def __call__(self, result: Result) -> GateResult:
    return self.forward(result)

  def explain(self, result: Result) -> str:
    outcome = self.forward(result)
    if outcome == GateResult.PASS:
      return 'all gates passed'
    if outcome == GateResult.WARN:
      prefix = 'optional gate(s) failed'
      if self._human_review_on_warn:
        return f'{prefix} - human review triggered'
      return prefix
    failed = [g.metric for g in self._gates if g(result) == GateResult.FAIL]
    return f'required gates failed: {failed}'


class QualityFirstMetric(Metric):
  """Metric that accumulates datum metrics and applies quality-first gates on compute()."""

  higher_is_better = True

  def __init__(self, gates: list[Gate] | None = None) -> None:
    super().__init__()
    self._gates = gates or []
    self.add_state('_accumulated', dict)

  def name(self) -> str:
    return 'quality_first'

  def update(self, datum: Datum) -> None:
    for key, value in datum.metrics.items():
      self._accumulated.setdefault(key, []).append(value)

  def compute(self) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, values in self._accumulated.items():
      metrics[key] = sum(values) / len(values) if values else 0.0
    return metrics

  def to_result(self, metrics: dict[str, float] | None = None) -> Result:
    """Build a Result by applying gates to the given or computed metrics."""
    if metrics is None:
      metrics = self.compute()
    eval_result = Result(metrics=metrics)
    for gate in self._gates:
      gate_out = gate(eval_result)
      eval_result.gates[gate.metric] = gate_out.value
    eval_result.passed = all(
      gate(eval_result) != GateResult.FAIL for gate in self._gates if gate.required
    )
    return eval_result
