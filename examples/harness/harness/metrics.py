"""Harness metrics for agent evaluation scoring.

Twelve ``Metric`` subclasses cover task success, tool matching, communication,
policy compliance, efficiency, error rate, a composite tau reward, and
cost-attribution token/API-call sums. All are composed in
``HarnessMetrics`` (a ``MetricCollection``).

Metric keys (stable, used in policy gates and comparators):
  - ``task_success_rate`` -- fraction of successful datums
  - ``tool_recall`` -- mean tool recall across datums
  - ``tool_precision`` -- mean tool precision across datums
  - ``tool_argument_accuracy`` -- mean tool argument accuracy
  - ``communication_recall`` -- mean communication recall
  - ``policy_compliance`` -- mean policy compliance
  - ``avg_turns`` -- mean conversation turns (lower is better)
  - ``error_rate`` -- fraction of errored datums (lower is better)
  - ``tau_reward`` -- mean product of five component metrics (0 if errored)
  - ``total_input_tokens`` -- sum of input tokens from ``EvalDatum.metadata``
  - ``total_output_tokens`` -- sum of output tokens from ``EvalDatum.metadata``
  - ``total_api_calls`` -- sum of API calls from ``EvalDatum.metadata``

``HarnessMetrics`` collection child keys (attribute names):
  task_success, tool_recall, tool_precision, tool_arg_accuracy,
  communication, policy_compliance, efficiency, error_rate, tau_reward,
  input_tokens, output_tokens, api_calls.

Tau reward formula:
  tau = tool_recall * tool_precision * tool_argument_accuracy
        * communication_recall * policy_compliance
  (0.0 when result.errored is True)
"""

from autopilot.core.metric import Metric, MetricCollection
from autopilot.core.types import EvalDatum
from harness.evaluator import EvaluationResult


class TaskSuccessRate(Metric):
  """Fraction of datums where ``datum.success`` is True."""

  def __init__(self) -> None:
    """Initialize correct/total counters."""
    super().__init__()
    self.add_state('correct', 0)
    self.add_state('total', 0)

  def name(self) -> str:
    """Return stable metric key."""
    return 'task_success_rate'

  @property
  def higher_is_better(self) -> bool:
    """Higher success rate is better."""
    return True

  def update(self, datum: EvalDatum) -> None:
    """Increment counters based on datum.success.

    Args:
      datum: Evaluation datum with ``success`` field.
    """
    self.total += 1
    if datum.success:
      self.correct += 1

  def compute(self) -> dict[str, float]:
    """Compute success rate.

    Returns:
      Dict with ``task_success_rate`` float (0.0 when no updates).
    """
    if self.total == 0:
      return {'task_success_rate': 0.0}
    return {'task_success_rate': self.correct / self.total}


class ToolRecallMetric(Metric):
  """Mean tool recall across evaluation datums."""

  def __init__(self) -> None:
    """Initialize sum/count accumulators."""
    super().__init__()
    self.add_state('sum', 0.0)
    self.add_state('count', 0)

  def name(self) -> str:
    """Return stable metric key."""
    return 'tool_recall'

  @property
  def higher_is_better(self) -> bool:
    """Higher recall is better."""
    return True

  def update(self, datum: EvalDatum) -> None:
    """Add tool_recall from the datum's EvaluationResult.

    Args:
      datum: Evaluation datum with ``metadata['eval_result']``.
    """
    result = EvaluationResult.from_metadata(datum.metadata)
    self.sum += result.tool_recall
    self.count += 1

  def compute(self) -> dict[str, float]:
    """Compute mean tool recall.

    Returns:
      Dict with ``tool_recall`` float (0.0 when no updates).
    """
    if self.count == 0:
      return {'tool_recall': 0.0}
    return {'tool_recall': self.sum / self.count}


class ToolPrecisionMetric(Metric):
  """Mean tool precision across evaluation datums."""

  def __init__(self) -> None:
    """Initialize sum/count accumulators."""
    super().__init__()
    self.add_state('sum', 0.0)
    self.add_state('count', 0)

  def name(self) -> str:
    """Return stable metric key."""
    return 'tool_precision'

  @property
  def higher_is_better(self) -> bool:
    """Higher precision is better."""
    return True

  def update(self, datum: EvalDatum) -> None:
    """Add tool_precision from the datum's EvaluationResult.

    Args:
      datum: Evaluation datum with ``metadata['eval_result']``.
    """
    result = EvaluationResult.from_metadata(datum.metadata)
    self.sum += result.tool_precision
    self.count += 1

  def compute(self) -> dict[str, float]:
    """Compute mean tool precision.

    Returns:
      Dict with ``tool_precision`` float (0.0 when no updates).
    """
    if self.count == 0:
      return {'tool_precision': 0.0}
    return {'tool_precision': self.sum / self.count}


class ToolArgumentAccuracy(Metric):
  """Mean tool argument accuracy across evaluation datums."""

  def __init__(self) -> None:
    """Initialize sum/count accumulators."""
    super().__init__()
    self.add_state('sum', 0.0)
    self.add_state('count', 0)

  def name(self) -> str:
    """Return stable metric key."""
    return 'tool_argument_accuracy'

  @property
  def higher_is_better(self) -> bool:
    """Higher accuracy is better."""
    return True

  def update(self, datum: EvalDatum) -> None:
    """Add tool_argument_accuracy from the datum's EvaluationResult.

    Args:
      datum: Evaluation datum with ``metadata['eval_result']``.
    """
    result = EvaluationResult.from_metadata(datum.metadata)
    self.sum += result.tool_argument_accuracy
    self.count += 1

  def compute(self) -> dict[str, float]:
    """Compute mean tool argument accuracy.

    Returns:
      Dict with ``tool_argument_accuracy`` float (0.0 when no updates).
    """
    if self.count == 0:
      return {'tool_argument_accuracy': 0.0}
    return {'tool_argument_accuracy': self.sum / self.count}


class CommunicationRecall(Metric):
  """Mean communication recall across evaluation datums."""

  def __init__(self) -> None:
    """Initialize sum/count accumulators."""
    super().__init__()
    self.add_state('sum', 0.0)
    self.add_state('count', 0)

  def name(self) -> str:
    """Return stable metric key."""
    return 'communication_recall'

  @property
  def higher_is_better(self) -> bool:
    """Higher recall is better."""
    return True

  def update(self, datum: EvalDatum) -> None:
    """Add communication_recall from the datum's EvaluationResult.

    Args:
      datum: Evaluation datum with ``metadata['eval_result']``.
    """
    result = EvaluationResult.from_metadata(datum.metadata)
    self.sum += result.communication_recall
    self.count += 1

  def compute(self) -> dict[str, float]:
    """Compute mean communication recall.

    Returns:
      Dict with ``communication_recall`` float (0.0 when no updates).
    """
    if self.count == 0:
      return {'communication_recall': 0.0}
    return {'communication_recall': self.sum / self.count}


class PolicyComplianceRate(Metric):
  """Mean policy compliance across evaluation datums."""

  def __init__(self) -> None:
    """Initialize sum/count accumulators."""
    super().__init__()
    self.add_state('sum', 0.0)
    self.add_state('count', 0)

  def name(self) -> str:
    """Return stable metric key."""
    return 'policy_compliance'

  @property
  def higher_is_better(self) -> bool:
    """Higher compliance is better."""
    return True

  def update(self, datum: EvalDatum) -> None:
    """Add policy_compliance from the datum's EvaluationResult.

    Args:
      datum: Evaluation datum with ``metadata['eval_result']``.
    """
    result = EvaluationResult.from_metadata(datum.metadata)
    self.sum += result.policy_compliance
    self.count += 1

  def compute(self) -> dict[str, float]:
    """Compute mean policy compliance.

    Returns:
      Dict with ``policy_compliance`` float (0.0 when no updates).
    """
    if self.count == 0:
      return {'policy_compliance': 0.0}
    return {'policy_compliance': self.sum / self.count}


class ConversationEfficiency(Metric):
  """Mean conversation turns (lower is better)."""

  def __init__(self) -> None:
    """Initialize sum_turns/count accumulators."""
    super().__init__()
    self.add_state('sum_turns', 0)
    self.add_state('count', 0)

  def name(self) -> str:
    """Return stable metric key."""
    return 'avg_turns'

  @property
  def higher_is_better(self) -> bool:
    """Lower turn count is better."""
    return False

  def update(self, datum: EvalDatum) -> None:
    """Add turns from the datum's EvaluationResult.

    Args:
      datum: Evaluation datum with ``metadata['eval_result']``.
    """
    result = EvaluationResult.from_metadata(datum.metadata)
    self.sum_turns += result.turns
    self.count += 1

  def compute(self) -> dict[str, float]:
    """Compute mean turns.

    Returns:
      Dict with ``avg_turns`` float (0.0 when no updates).
    """
    if self.count == 0:
      return {'avg_turns': 0.0}
    return {'avg_turns': self.sum_turns / self.count}


class ErrorRateMetric(Metric):
  """Fraction of datums where the conversation errored."""

  def __init__(self) -> None:
    """Initialize errors/total counters."""
    super().__init__()
    self.add_state('errors', 0)
    self.add_state('total', 0)

  def name(self) -> str:
    """Return stable metric key."""
    return 'error_rate'

  @property
  def higher_is_better(self) -> bool:
    """Lower error rate is better."""
    return False

  def update(self, datum: EvalDatum) -> None:
    """Increment counters; mark errored via EvaluationResult.

    Args:
      datum: Evaluation datum with ``metadata['eval_result']``.
    """
    result = EvaluationResult.from_metadata(datum.metadata)
    self.total += 1
    if result.errored:
      self.errors += 1

  def compute(self) -> dict[str, float]:
    """Compute error rate.

    Returns:
      Dict with ``error_rate`` float (0.0 when no updates).
    """
    if self.total == 0:
      return {'error_rate': 0.0}
    return {'error_rate': self.errors / self.total}


class TauRewardMetric(Metric):
  """Mean tau reward: product of five component metrics (0 if errored).

  tau = tool_recall * tool_precision * tool_argument_accuracy
        * communication_recall * policy_compliance
  """

  def __init__(self) -> None:
    """Initialize sum_tau/count accumulators."""
    super().__init__()
    self.add_state('sum_tau', 0.0)
    self.add_state('count', 0)

  def name(self) -> str:
    """Return stable metric key."""
    return 'tau_reward'

  @property
  def higher_is_better(self) -> bool:
    """Higher tau reward is better."""
    return True

  def update(self, datum: EvalDatum) -> None:
    """Compute per-datum tau and accumulate.

    Args:
      datum: Evaluation datum with ``metadata['eval_result']``.
    """
    result = EvaluationResult.from_metadata(datum.metadata)
    if result.errored:
      tau = 0.0
    else:
      tau = (
        result.tool_recall
        * result.tool_precision
        * result.tool_argument_accuracy
        * result.communication_recall
        * result.policy_compliance
      )
    self.sum_tau += tau
    self.count += 1

  def compute(self) -> dict[str, float]:
    """Compute mean tau reward.

    Returns:
      Dict with ``tau_reward`` float (0.0 when no updates).
    """
    if self.count == 0:
      return {'tau_reward': 0.0}
    return {'tau_reward': self.sum_tau / self.count}


class TotalInputTokens(Metric):
  """Sum of input tokens across evaluation datums.

  Reads ``metadata['input_tokens']`` from each ``EvalDatum``. Missing keys
  are treated as zero additional usage.
  """

  def __init__(self) -> None:
    """Initialize token accumulator."""
    super().__init__()
    self.add_state('total', 0)

  def name(self) -> str:
    """Return stable metric key."""
    return 'total_input_tokens'

  @property
  def higher_is_better(self) -> bool:
    """Lower token usage is better (cost metric)."""
    return False

  def update(self, datum: EvalDatum) -> None:
    """Accumulate input tokens from datum metadata.

    Args:
      datum: Evaluation datum with optional ``metadata['input_tokens']``.
    """
    metadata = datum.metadata if datum.metadata is not None else {}
    self.total += int(metadata.get('input_tokens') or 0)

  def compute(self) -> dict[str, float]:
    """Return total input tokens as float.

    Returns:
      Dict with ``total_input_tokens`` float.
    """
    return {'total_input_tokens': float(self.total)}


class TotalOutputTokens(Metric):
  """Sum of output tokens across evaluation datums.

  Reads ``metadata['output_tokens']`` from each ``EvalDatum``. Missing keys
  are treated as zero additional usage.
  """

  def __init__(self) -> None:
    """Initialize token accumulator."""
    super().__init__()
    self.add_state('total', 0)

  def name(self) -> str:
    """Return stable metric key."""
    return 'total_output_tokens'

  @property
  def higher_is_better(self) -> bool:
    """Lower token usage is better (cost metric)."""
    return False

  def update(self, datum: EvalDatum) -> None:
    """Accumulate output tokens from datum metadata.

    Args:
      datum: Evaluation datum with optional ``metadata['output_tokens']``.
    """
    metadata = datum.metadata if datum.metadata is not None else {}
    self.total += int(metadata.get('output_tokens') or 0)

  def compute(self) -> dict[str, float]:
    """Return total output tokens as float.

    Returns:
      Dict with ``total_output_tokens`` float.
    """
    return {'total_output_tokens': float(self.total)}


class TotalApiCalls(Metric):
  """Sum of API calls across evaluation datums.

  Reads ``metadata['api_calls']`` from each ``EvalDatum``. Missing keys
  are treated as zero additional usage.
  """

  def __init__(self) -> None:
    """Initialize API call accumulator."""
    super().__init__()
    self.add_state('total', 0)

  def name(self) -> str:
    """Return stable metric key."""
    return 'total_api_calls'

  @property
  def higher_is_better(self) -> bool:
    """Lower API call count is better (cost metric)."""
    return False

  def update(self, datum: EvalDatum) -> None:
    """Accumulate API calls from datum metadata.

    Args:
      datum: Evaluation datum with optional ``metadata['api_calls']``.
    """
    metadata = datum.metadata if datum.metadata is not None else {}
    self.total += int(metadata.get('api_calls') or 0)

  def compute(self) -> dict[str, float]:
    """Return total API calls as float.

    Returns:
      Dict with ``total_api_calls`` float.
    """
    return {'total_api_calls': float(self.total)}


class HarnessMetrics(MetricCollection):
  """Collection of all twelve harness evaluation metrics.

  Child keys (attribute names on the collection):
    task_success, tool_recall, tool_precision, tool_arg_accuracy,
    communication, policy_compliance, efficiency, error_rate, tau_reward,
    input_tokens, output_tokens, api_calls.

  The first nine are evaluation quality metrics. The last three are
  cost-attribution sum metrics sourced from ``EvalDatum.metadata`` token
  keys (``input_tokens``, ``output_tokens``, ``api_calls``).

  Flattened ``compute()`` returns twelve distinct scalar keys with no collision.
  """

  def __init__(self) -> None:
    """Initialize with all twelve metrics."""
    super().__init__(
      {
        'task_success': TaskSuccessRate(),
        'tool_recall': ToolRecallMetric(),
        'tool_precision': ToolPrecisionMetric(),
        'tool_arg_accuracy': ToolArgumentAccuracy(),
        'communication': CommunicationRecall(),
        'policy_compliance': PolicyComplianceRate(),
        'efficiency': ConversationEfficiency(),
        'error_rate': ErrorRateMetric(),
        'tau_reward': TauRewardMetric(),
        'input_tokens': TotalInputTokens(),
        'output_tokens': TotalOutputTokens(),
        'api_calls': TotalApiCalls(),
      }
    )
