"""Tests for HarnessGradient and HarnessLoss."""

from autopilot.ai.gradient import TextGradient
from autopilot.core.types import EvalDatum
from harness.evaluator import EvaluationResult
from harness.loss import EFFICIENCY_TURN_THRESHOLD, HarnessGradient, HarnessLoss
import pytest


def _make_eval_result(
  tool_recall=1.0,
  tool_precision=1.0,
  tool_argument_accuracy=1.0,
  communication_recall=1.0,
  policy_compliance=1.0,
  turns=3,
  errored=False,
):
  """Build an EvaluationResult with the given scores."""
  return EvaluationResult(
    task_success=(
      tool_recall == 1.0
      and communication_recall == 1.0
      and policy_compliance == 1.0
      and not errored
    ),
    tool_recall=tool_recall,
    tool_precision=tool_precision,
    tool_argument_accuracy=tool_argument_accuracy,
    communication_recall=communication_recall,
    policy_compliance=policy_compliance,
    turns=turns,
    errored=errored,
  )


def _make_datum(success, task_id='t1', **eval_kwargs):
  """Build an EvalDatum with embedded EvaluationResult."""
  result = _make_eval_result(**eval_kwargs)
  return EvalDatum(
    success=success,
    metadata={
      'scenario': {'task_id': task_id},
      'eval_result': result.to_dict(),
    },
  )


class TestHarnessGradient:
  """Tests for HarnessGradient."""

  def test_gradient_accumulate(self):
    """Accumulate merges lists and metadata."""
    g1 = HarnessGradient(
      tool_failures=[{'task_id': 'a', 'description': 'x'}],
      metadata={'a': 1},
    )
    g2 = HarnessGradient(
      tool_failures=[{'task_id': 'b', 'description': 'y'}],
      metadata={'b': 2},
    )
    g3 = g1.accumulate(g2)
    assert len(g3.tool_failures) == 2
    assert len(g3.communication_gaps) == 0
    assert len(g3.policy_violations) == 0
    assert len(g3.efficiency_issues) == 0
    assert g3.metadata == {'a': 1, 'b': 2}

  def test_gradient_accumulate_metadata_collision(self):
    """Right-hand metadata wins on key collision."""
    g1 = HarnessGradient(metadata={'k': 1})
    g2 = HarnessGradient(metadata={'k': 2})
    g3 = g1.accumulate(g2)
    assert g3.metadata['k'] == 2

  def test_gradient_render_with_failures(self):
    """Render shows sections for non-empty buckets only."""
    g = HarnessGradient(
      tool_failures=[{'task_id': 't1', 'description': 'missing cancel_order'}],
    )
    rendered = g.render()
    assert '## Tool Call Failures (1 scenarios)' in rendered
    assert '- Task t1: missing cancel_order' in rendered
    assert 'Communication' not in rendered

  def test_gradient_render_empty(self):
    """Default-constructed gradient renders to sentinel string."""
    g = HarnessGradient()
    assert g.render() == 'No issues found.'

  def test_gradient_todo_items(self):
    """todo_items returns recommendations for non-empty buckets."""
    g = HarnessGradient(
      tool_failures=[{'task_id': 'x', 'description': 'bad'}],
    )
    expected = [
      'Update retail_tools.py: improve tool docstrings and return formatting for failed tools'
    ]
    assert g.todo_items() == expected

  def test_gradient_todo_items_empty(self):
    """Empty gradient has no todo items."""
    g = HarnessGradient()
    assert g.todo_items() == []

  def test_gradient_render_recommendations(self):
    """Render includes recommendations section when failures exist."""
    g = HarnessGradient(
      tool_failures=[{'task_id': 'a', 'description': 'x'}],
      policy_violations=[{'task_id': 'b', 'description': 'y'}],
    )
    rendered = g.render()
    assert '## Recommendations' in rendered
    assert 'retail_tools.py' in rendered
    assert 'policies.md' in rendered

  def test_gradient_accumulate_type_error(self):
    """Accumulating incompatible gradient type raises TypeError."""
    g = HarnessGradient()
    with pytest.raises(TypeError, match='HarnessGradient'):
      g.accumulate(TextGradient(text='test'))


class TestHarnessLoss:
  """Tests for HarnessLoss."""

  def test_loss_forward_accumulates_failures(self):
    """Forward with failed datum populates _eval_results."""
    loss = HarnessLoss()
    datum = _make_datum(success=False, task_id='a', tool_recall=0.5)
    loss.forward(datum)
    assert len(loss._eval_results) == 1
    scenario, _ = loss._eval_results[0]
    assert scenario['task_id'] == 'a'

  def test_loss_forward_skips_success(self):
    """Forward with successful datum does not populate _eval_results."""
    loss = HarnessLoss()
    datum = _make_datum(success=True, task_id='b')
    loss.forward(datum)
    assert loss._eval_results == []

  def test_loss_forward_still_calls_super(self):
    """Forward always maintains base Loss invariants."""
    loss = HarnessLoss()
    datum = _make_datum(success=False, task_id='c', tool_recall=0.5)
    loss.forward(datum)
    assert len(loss._accumulated) == 1
    assert loss._last_data is not None

  def test_loss_compute_seed_gradient(self):
    """compute_seed_gradient categorizes failures correctly."""
    loss = HarnessLoss()
    datum = _make_datum(
      success=False,
      task_id='t1',
      tool_recall=0.5,
      communication_recall=1.0,
      policy_compliance=1.0,
      turns=3,
    )
    loss.forward(datum)
    grad = loss.compute_seed_gradient()
    assert len(grad.tool_failures) == 1
    assert '0.50' in grad.tool_failures[0]['description']
    assert grad.communication_gaps == []
    assert grad.policy_violations == []
    assert grad.efficiency_issues == []
    assert grad.metadata['total_failures'] == 1

  def test_loss_efficiency_bucket(self):
    """Scenarios exceeding turn threshold go to efficiency_issues."""
    loss = HarnessLoss()
    datum = _make_datum(
      success=False,
      task_id='slow',
      turns=EFFICIENCY_TURN_THRESHOLD + 1,
    )
    loss.forward(datum)
    grad = loss.compute_seed_gradient()
    assert len(grad.efficiency_issues) == 1
    assert str(EFFICIENCY_TURN_THRESHOLD) in grad.efficiency_issues[0]['description']
    assert str(EFFICIENCY_TURN_THRESHOLD + 1) in grad.efficiency_issues[0]['description']

  def test_loss_reset_clears(self):
    """Reset clears both _eval_results and base Loss state."""
    loss = HarnessLoss()
    datum = _make_datum(success=False, task_id='r', tool_recall=0.5)
    loss.forward(datum)
    loss.reset()
    assert loss._eval_results == []
    assert loss._accumulated == []

  def test_loss_multiple_scenarios(self):
    """Multiple failures yield entries in the gradient buckets."""
    loss = HarnessLoss()
    for tid in ['x', 'y', 'z']:
      datum = _make_datum(success=False, task_id=tid, tool_recall=0.3)
      loss.forward(datum)
    grad = loss.compute_seed_gradient()
    assert len(grad.tool_failures) == 3
    ids = [f['task_id'] for f in grad.tool_failures]
    assert ids == ['x', 'y', 'z']
