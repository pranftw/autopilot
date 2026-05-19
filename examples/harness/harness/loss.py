"""Loss and gradient for the agent harness optimization loop.

``HarnessGradient`` carries categorized failure information (tool failures,
communication gaps, policy violations, efficiency issues) and produces
file-specific recommendations for the optimizer agent.

``HarnessLoss`` accumulates failed evaluation rows via ``forward()``, then
emits a ``HarnessGradient`` from ``compute_seed_gradient()``. Call ``reset()``
between epochs to clear accumulation state.

Workflow:
  1. ``forward(datum, targets)`` -- accumulate failures (success rows skipped)
  2. ``compute_seed_gradient()`` -- build categorized gradient from failures
  3. ``backward()`` -- distribute gradient through the computation graph
  4. ``reset()`` -- clear ``_eval_results`` and base ``Loss`` state
"""

from autopilot.core.gradient import Gradient
from autopilot.core.loss import Loss
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from dataclasses import dataclass, field
from harness.evaluator import EvaluationResult
from typing import Any

EFFICIENCY_TURN_THRESHOLD = 10


@dataclass
class HarnessGradient(Gradient):
  """Structured gradient with categorized failure information.

  Each bucket holds a list of dicts with at least ``task_id`` and
  ``description`` keys. ``render()`` emits markdown sections only for
  non-empty buckets; ``_generate_recommendations()`` maps failure classes
  to file-specific action items.

  Attributes:
    tool_failures: Scenarios with tool_recall < 1.0.
    communication_gaps: Scenarios with communication_recall < 1.0.
    policy_violations: Scenarios with policy_compliance < 1.0.
    efficiency_issues: Scenarios exceeding the turn threshold.
    metadata: Arbitrary context (e.g. total_failures count).
  """

  tool_failures: list[dict[str, Any]] = field(default_factory=list)
  communication_gaps: list[dict[str, Any]] = field(default_factory=list)
  policy_violations: list[dict[str, Any]] = field(default_factory=list)
  efficiency_issues: list[dict[str, Any]] = field(default_factory=list)
  metadata: dict[str, Any] = field(default_factory=dict)

  def accumulate(self, other: 'Gradient') -> 'HarnessGradient':
    """Merge two HarnessGradients by concatenating buckets.

    Args:
      other: Must be a ``HarnessGradient``.

    Returns:
      New ``HarnessGradient`` with merged lists and right-hand metadata wins.

    Raises:
      TypeError: When ``other`` is not ``HarnessGradient``.
    """
    if not isinstance(other, HarnessGradient):
      msg = (
        f'Cannot accumulate HarnessGradient with {type(other).__name__}. '
        f'Insert a conversion operator to coerce types before fan-in.'
      )
      raise TypeError(msg)
    return HarnessGradient(
      tool_failures=self.tool_failures + other.tool_failures,
      communication_gaps=self.communication_gaps + other.communication_gaps,
      policy_violations=self.policy_violations + other.policy_violations,
      efficiency_issues=self.efficiency_issues + other.efficiency_issues,
      metadata={**self.metadata, **other.metadata},
    )

  def render(self) -> str:
    """Produce markdown summary of categorized failures.

    Sections appear only for non-empty buckets. A trailing
    ``## Recommendations`` section lists file-specific action items when
    any bucket has entries.

    Returns:
      Markdown string, or ``'No issues found.'`` when all buckets are empty
      and no recommendations exist.
    """
    sections: list[str] = []

    buckets = [
      ('Tool Call Failures', self.tool_failures),
      ('Communication Gaps', self.communication_gaps),
      ('Policy Violations', self.policy_violations),
      ('Efficiency Issues', self.efficiency_issues),
    ]

    for header, items in buckets:
      if items:
        lines = [f'## {header} ({len(items)} scenarios)']
        for item in items:
          task_id = item.get('task_id', 'unknown')
          description = item.get('description', '')
          lines.append(f'- Task {task_id}: {description}')
        sections.append('\n'.join(lines))

    recommendations = self._generate_recommendations()
    if recommendations:
      rec_lines = ['## Recommendations']
      for rec in recommendations:
        rec_lines.append(f'- {rec}')
      sections.append('\n'.join(rec_lines))

    if not sections:
      return 'No issues found.'

    return '\n\n'.join(sections)

  def _generate_recommendations(self) -> list[str]:
    """Generate file-specific recommendations based on non-empty buckets.

    Returns:
      List of recommendation strings (one per failure class present).
    """
    recs: list[str] = []
    if self.tool_failures:
      recs.append(
        'Update retail_tools.py: improve tool docstrings and return formatting for failed tools'
      )
    if self.communication_gaps:
      recs.append(
        'Update system_prompt.md: add instructions to communicate all required information'
      )
    if self.policy_violations:
      recs.append('Update policies.md: clarify policy rules that were violated')
    if self.efficiency_issues:
      recs.append('Update system_prompt.md: add guidance for more efficient conversation flow')
    return recs

  def todo_items(self) -> list[str]:
    """Return file-specific recommendations as todo items.

    Returns:
      Output of ``_generate_recommendations()`` (not the base heuristics).
    """
    return self._generate_recommendations()


class HarnessLoss(Loss):
  """Accumulates failed evaluation rows and emits HarnessGradient.

  Only rows where ``datum.success is False`` contribute to gradient
  computation. Each failure is parsed into an ``EvaluationResult`` using
  ``EvaluationResult.from_metadata`` and paired with its scenario dict.

  Expected metadata keys on each ``EvalDatum``:
    - ``metadata['eval_result']``: serialized ``EvaluationResult`` (via ``to_dict()``)
    - ``metadata['scenario']``: scenario dict with at least ``task_id``

  Failure categorization uses strict thresholds:
    - tool_recall < 1.0 -> tool_failures bucket
    - communication_recall < 1.0 -> communication_gaps bucket
    - policy_compliance < 1.0 -> policy_violations bucket
    - turns > EFFICIENCY_TURN_THRESHOLD -> efficiency_issues bucket
  """

  def __init__(self, parameters: list[Parameter] | None = None) -> None:
    """Initialize with empty eval results accumulator.

    Args:
      parameters: Optional explicit parameter scope for gradient targets.
    """
    super().__init__(parameters)
    self._eval_results: list[tuple[dict, EvaluationResult]] = []

  def forward(self, data: Datum, targets: Any = None) -> None:
    """Accumulate a datum; parse failures into eval_results.

    Always calls ``super().forward()`` to maintain base ``Loss`` invariants.
    Only appends to ``_eval_results`` when ``data.success`` is ``False``.

    Args:
      data: An ``EvalDatum`` with ``success`` and ``metadata`` fields.
      targets: Optional targets (passed through to base).
    """
    super().forward(data, targets)
    if not data.success:
      eval_result = EvaluationResult.from_metadata(data.metadata)
      scenario = data.metadata['scenario']
      self._eval_results.append((scenario, eval_result))

  def compute_seed_gradient(self) -> HarnessGradient:
    """Build categorized gradient from accumulated failure results.

    Iterates stored ``(scenario, result)`` pairs and routes each into the
    appropriate bucket based on metric thresholds.

    Returns:
      ``HarnessGradient`` with categorized failures and metadata.
    """
    tool_failures: list[dict[str, Any]] = []
    communication_gaps: list[dict[str, Any]] = []
    policy_violations: list[dict[str, Any]] = []
    efficiency_issues: list[dict[str, Any]] = []

    for scenario, result in self._eval_results:
      task_id = scenario.get('task_id', 'unknown')

      if result.tool_recall < 1.0:
        tool_failures.append(
          {
            'task_id': task_id,
            'description': f'tool_recall={result.tool_recall:.2f}',
            'details': result.details.get('tool_matching', {}),
          }
        )

      if result.communication_recall < 1.0:
        communication_gaps.append(
          {
            'task_id': task_id,
            'description': f'communication_recall={result.communication_recall:.2f}',
            'details': result.details.get('communication', {}),
          }
        )

      if result.policy_compliance < 1.0:
        policy_violations.append(
          {
            'task_id': task_id,
            'description': f'policy_compliance={result.policy_compliance:.2f}',
            'details': result.details.get('assertions', {}),
          }
        )

      if result.turns > EFFICIENCY_TURN_THRESHOLD:
        efficiency_issues.append(
          {
            'task_id': task_id,
            'description': f'took {result.turns} turns (threshold {EFFICIENCY_TURN_THRESHOLD})',
          }
        )

    return HarnessGradient(
      tool_failures=tool_failures,
      communication_gaps=communication_gaps,
      policy_violations=policy_violations,
      efficiency_issues=efficiency_issues,
      metadata={'total_failures': len(self._eval_results)},
    )

  def reset(self) -> None:
    """Clear accumulated failures and base Loss state."""
    super().reset()
    self._eval_results = []
