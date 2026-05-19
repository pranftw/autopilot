"""Tests for harness metrics (twelve Metric subclasses + HarnessMetrics collection)."""

from autopilot.core.types import EvalDatum
from harness.evaluator import EvaluationResult
from harness.metrics import (
  CommunicationRecall,
  ConversationEfficiency,
  ErrorRateMetric,
  HarnessMetrics,
  PolicyComplianceRate,
  TaskSuccessRate,
  TauRewardMetric,
  ToolArgumentAccuracy,
  ToolPrecisionMetric,
  ToolRecallMetric,
)
import pytest


def datum(success: bool, result: EvaluationResult) -> EvalDatum:
  """Build an EvalDatum with embedded EvaluationResult."""
  return EvalDatum(
    success=success,
    metadata={'eval_result': result.to_dict()},
  )


def _result(
  tool_recall=1.0,
  tool_precision=1.0,
  tool_argument_accuracy=1.0,
  communication_recall=1.0,
  policy_compliance=1.0,
  turns=3,
  errored=False,
):
  """Build an EvaluationResult with given scores."""
  return EvaluationResult(
    task_success=not errored,
    tool_recall=tool_recall,
    tool_precision=tool_precision,
    tool_argument_accuracy=tool_argument_accuracy,
    communication_recall=communication_recall,
    policy_compliance=policy_compliance,
    turns=turns,
    errored=errored,
  )


class TestTaskSuccessRate:
  """Tests for TaskSuccessRate metric."""

  def test_task_success_rate_all_pass(self):
    """Three successful datums yield 1.0."""
    m = TaskSuccessRate()
    for _ in range(3):
      m.update(datum(True, _result()))
    assert m.compute()['task_success_rate'] == 1.0

  def test_task_success_rate_partial(self):
    """One success out of three yields ~1/3."""
    m = TaskSuccessRate()
    m.update(datum(True, _result()))
    m.update(datum(False, _result()))
    m.update(datum(False, _result()))
    assert abs(m.compute()['task_success_rate'] - 1 / 3) < 1e-9

  def test_task_success_rate_all_fail(self):
    """Three failures yield 0.0."""
    m = TaskSuccessRate()
    for _ in range(3):
      m.update(datum(False, _result()))
    assert m.compute()['task_success_rate'] == 0.0

  def test_task_success_rate_empty(self):
    """No updates yields 0.0 with UserWarning."""
    m = TaskSuccessRate()
    with pytest.warns(UserWarning):
      result = m.compute()
    assert result['task_success_rate'] == 0.0


class TestToolMetrics:
  """Tests for tool-related metrics."""

  def test_tool_recall_metric(self):
    """Mean of 0.4 and 0.8 is 0.6."""
    m = ToolRecallMetric()
    m.update(datum(False, _result(tool_recall=0.4)))
    m.update(datum(False, _result(tool_recall=0.8)))
    assert abs(m.compute()['tool_recall'] - 0.6) < 1e-9

  def test_tool_precision_metric(self):
    """Mean of 0.4 and 0.8 is 0.6."""
    m = ToolPrecisionMetric()
    m.update(datum(False, _result(tool_precision=0.4)))
    m.update(datum(False, _result(tool_precision=0.8)))
    assert abs(m.compute()['tool_precision'] - 0.6) < 1e-9

  def test_tool_argument_accuracy_metric(self):
    """Mean of 0.4 and 0.8 is 0.6."""
    m = ToolArgumentAccuracy()
    m.update(datum(False, _result(tool_argument_accuracy=0.4)))
    m.update(datum(False, _result(tool_argument_accuracy=0.8)))
    assert abs(m.compute()['tool_argument_accuracy'] - 0.6) < 1e-9


class TestCommunicationAndPolicy:
  """Tests for communication and policy metrics."""

  def test_communication_recall(self):
    """Mean of 0.4 and 0.8 is 0.6."""
    m = CommunicationRecall()
    m.update(datum(False, _result(communication_recall=0.4)))
    m.update(datum(False, _result(communication_recall=0.8)))
    assert abs(m.compute()['communication_recall'] - 0.6) < 1e-9

  def test_policy_compliance(self):
    """Mean of 0.4 and 0.8 is 0.6."""
    m = PolicyComplianceRate()
    m.update(datum(False, _result(policy_compliance=0.4)))
    m.update(datum(False, _result(policy_compliance=0.8)))
    assert abs(m.compute()['policy_compliance'] - 0.6) < 1e-9


class TestConversationEfficiency:
  """Tests for ConversationEfficiency metric."""

  def test_conversation_efficiency(self):
    """Mean of 4 and 8 turns is 6.0."""
    m = ConversationEfficiency()
    m.update(datum(True, _result(turns=4)))
    m.update(datum(True, _result(turns=8)))
    assert m.compute()['avg_turns'] == 6.0
    assert m.higher_is_better is False


class TestErrorRate:
  """Tests for ErrorRateMetric."""

  def test_error_rate(self):
    """One errored out of three yields ~1/3."""
    m = ErrorRateMetric()
    m.update(datum(False, _result(errored=True)))
    m.update(datum(True, _result(errored=False)))
    m.update(datum(True, _result(errored=False)))
    assert abs(m.compute()['error_rate'] - 1 / 3) < 1e-9


class TestTauReward:
  """Tests for TauRewardMetric."""

  def test_tau_reward(self):
    """Tau is product of five metrics; errored contributes 0."""
    m = TauRewardMetric()
    # datum A: all 1.0 -> tau=1.0
    m.update(datum(True, _result()))
    # datum B: tool_recall=0.5, rest 1.0 -> tau=0.5
    m.update(datum(False, _result(tool_recall=0.5)))
    # datum C: errored -> tau=0.0
    m.update(datum(False, EvaluationResult.error()))
    expected = (1.0 + 0.5 + 0.0) / 3
    assert abs(m.compute()['tau_reward'] - expected) < 1e-9


class TestHarnessMetricsCollection:
  """Tests for HarnessMetrics MetricCollection."""

  def test_harness_metrics_collection(self):
    """Collection produces all twelve keys with correct types."""
    coll = HarnessMetrics()
    datums = [
      datum(True, _result()),
      datum(False, _result(tool_recall=0.5, communication_recall=0.8)),
      datum(False, EvaluationResult.error()),
    ]
    for d in datums:
      coll.update(d)
    out = coll.compute()
    expected_keys = {
      'task_success_rate',
      'tool_recall',
      'tool_precision',
      'tool_argument_accuracy',
      'communication_recall',
      'policy_compliance',
      'avg_turns',
      'error_rate',
      'tau_reward',
      'total_input_tokens',
      'total_output_tokens',
      'total_api_calls',
    }
    assert set(out.keys()) == expected_keys
    for value in out.values():
      assert isinstance(value, float)


class TestMetricReset:
  """Tests for metric reset behavior."""

  def test_metric_reset(self):
    """Reset clears state so only post-reset values contribute."""
    m = ToolRecallMetric()
    m.update(datum(True, _result(tool_recall=0.9)))
    m.update(datum(True, _result(tool_recall=0.7)))
    m.reset()
    m.update(datum(True, _result(tool_recall=0.25)))
    assert m.compute()['tool_recall'] == 0.25


class TestHigherIsBetter:
  """Tests for higher_is_better property on all metrics."""

  def test_metric_higher_is_better(self):
    """Verify higher_is_better matches the spec table."""
    higher_true = [
      TaskSuccessRate,
      ToolRecallMetric,
      ToolPrecisionMetric,
      ToolArgumentAccuracy,
      CommunicationRecall,
      PolicyComplianceRate,
      TauRewardMetric,
    ]
    higher_false = [
      ConversationEfficiency,
      ErrorRateMetric,
    ]
    for cls in higher_true:
      assert cls().higher_is_better is True, f'{cls.__name__} should be True'
    for cls in higher_false:
      assert cls().higher_is_better is False, f'{cls.__name__} should be False'
