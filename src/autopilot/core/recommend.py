"""Recommendation engine base types.

Recommender is the base class for recommendation engines. Subclasses
override recommend() to provide concrete logic using Forest data and
MetricsComparator for evidence.

Recommendation is the structured output: action, confidence, reasoning,
alternatives, and evidence dict. Validated at construction time via
__post_init__.
"""

from autopilot.core.comparison import MetricsComparator
from autopilot.core.forest import Forest
from autopilot.core.serialization import DictMixin
from dataclasses import dataclass, field
from typing import Any

VALID_RECOMMENDATION_ACTIONS = frozenset(
  {
    'deploy',
    'rollback',
    'continue',
    'branch',
    'investigate',
  }
)

VALID_CONFIDENCE_LEVELS = frozenset({'high', 'medium', 'low'})


@dataclass
class Recommendation(DictMixin):
  """Structured recommendation with evidence and alternatives.

  Attributes:
    action: One of VALID_RECOMMENDATION_ACTIONS.
    experiment_id: Target experiment, or None when action is investigative.
    confidence: One of VALID_CONFIDENCE_LEVELS.
    reasoning: Human-readable justification lines.
    alternatives: Experiment IDs or strategies worth considering.
    evidence: Metric dicts, comparator outputs, or other supporting data.
  """

  action: str
  experiment_id: str | None
  confidence: str
  reasoning: list[str] = field(default_factory=list)
  alternatives: list[str] = field(default_factory=list)
  evidence: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    """Validate action and confidence against allowed values.

    Raises:
      ValueError: When action or confidence is not in the allowed set.
    """
    if self.action not in VALID_RECOMMENDATION_ACTIONS:
      msg = (
        f'Invalid recommendation action {self.action!r}. '
        f'Must be one of: {", ".join(sorted(VALID_RECOMMENDATION_ACTIONS))}'
      )
      raise ValueError(msg)
    if self.confidence not in VALID_CONFIDENCE_LEVELS:
      msg = (
        f'Invalid confidence level {self.confidence!r}. '
        f'Must be one of: {", ".join(sorted(VALID_CONFIDENCE_LEVELS))}'
      )
      raise ValueError(msg)


class Recommender:
  """Base class for recommendation engines.

  Subclasses override recommend() to provide concrete recommendation
  logic. The base implementation raises NotImplementedError.
  """

  def recommend(
    self,
    forest: Forest,
    comparator: MetricsComparator,
    *,
    scope: str | None = None,
  ) -> Recommendation:
    """Produce a recommendation from forest data and metric comparisons.

    Args:
      forest: Forest containing experiment trees to analyze.
      comparator: MetricsComparator for evidence-based ranking.
      scope: Optional scope filter (e.g. 'tree:<name>' for single tree).

    Returns:
      Structured Recommendation with action, confidence, and evidence.

    Raises:
      NotImplementedError: Always on the base class.
    """
    raise NotImplementedError
