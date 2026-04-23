"""Tests for JudgeLoss with GradientCollator."""

from autopilot.ai.gradient import CollationResult, GradientCollator, TextGradient
from autopilot.ai.loss import JudgeLoss
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from unittest.mock import MagicMock
import pytest


def _mock_judge() -> MagicMock:
  return MagicMock()


def _collator_returning(params: list[Parameter], context: str = 'ctx') -> MagicMock:
  gradients = {p.id: TextGradient(attribution=f'fix {p.id}') for p in params}
  result = CollationResult(context=context, gradients=gradients)
  collator = MagicMock(spec=GradientCollator)
  collator.collate.return_value = result
  return collator


class TestJudgeLossForward:
  def test_forward_accumulates(self):
    judge = _mock_judge()
    collator = MagicMock(spec=GradientCollator)
    loss = JudgeLoss(judge, collator)
    for i in range(3):
      loss.forward(Datum(feedback=f'f{i}'))
    assert len(loss._accumulated) == 3

  def test_forward_after_backward_without_reset(self):
    judge = _mock_judge()
    params = [Parameter(requires_grad=True)]
    collator = _collator_returning(params)
    loss = JudgeLoss(judge, collator, params)
    loss.forward(Datum(feedback='a'))
    loss.backward()
    loss.forward(Datum(feedback='b'))
    assert len(loss._accumulated) == 2


class TestJudgeLossBackward:
  def test_backward_empty_is_noop(self):
    judge = _mock_judge()
    collator = MagicMock(spec=GradientCollator)
    loss = JudgeLoss(judge, collator)
    loss.backward()
    collator.collate.assert_not_called()

  def test_backward_calls_collator(self):
    judge = _mock_judge()
    params = [Parameter(requires_grad=True)]
    collator = _collator_returning(params)
    loss = JudgeLoss(judge, collator, params)
    loss.forward(Datum(feedback='f1'))
    loss.forward(Datum(feedback='f2'))
    loss.backward()
    collator.collate.assert_called_once()
    args = collator.collate.call_args[0]
    assert len(args[0]) == 2
    assert args[1] is loss._loss_parameters

  def test_backward_assigns_per_param_gradients(self):
    params = [Parameter(requires_grad=True), Parameter(requires_grad=True)]
    collator = _collator_returning(params)
    judge = _mock_judge()
    loss = JudgeLoss(judge, collator, params)
    loss.forward(Datum(feedback='f'))
    loss.backward()
    for p in params:
      assert p.grad is not None
      assert isinstance(p.grad, TextGradient)

  def test_backward_skips_missing_param(self):
    params = [Parameter(requires_grad=True), Parameter(requires_grad=True)]
    gradients = {params[0].id: TextGradient(attribution='only first')}
    collator = MagicMock(spec=GradientCollator)
    collator.collate.return_value = CollationResult(context='c', gradients=gradients)
    judge = _mock_judge()
    loss = JudgeLoss(judge, collator, params)
    loss.forward(Datum(feedback='f'))
    loss.backward()
    assert params[0].grad is not None
    assert params[1].grad is None

  def test_backward_skips_requires_grad_false(self):
    p = Parameter(requires_grad=False)
    collator = MagicMock(spec=GradientCollator)
    collator.collate.return_value = CollationResult(
      context='c', gradients={p.id: TextGradient(attribution='x')}
    )
    judge = _mock_judge()
    loss = JudgeLoss(judge, collator, [p])
    loss.forward(Datum(feedback='f'))
    loss.backward()
    assert p.grad is None

  def test_backward_params_always_have_unique_ids(self):
    params = [Parameter(requires_grad=True), Parameter(requires_grad=True)]
    assert params[0].id != params[1].id
    collator = _collator_returning(params)
    judge = _mock_judge()
    loss = JudgeLoss(judge, collator, params)
    loss.forward(Datum(feedback='f'))
    loss.backward()
    assert params[0].grad.attribution != params[1].grad.attribution

  def test_collator_raises_propagates(self):
    collator = MagicMock(spec=GradientCollator)
    collator.collate.side_effect = RuntimeError('collator broke')
    judge = _mock_judge()
    loss = JudgeLoss(judge, collator, [Parameter(requires_grad=True)])
    loss.forward(Datum(feedback='f'))
    with pytest.raises(RuntimeError, match='collator broke'):
      loss.backward()

  def test_backward_collator_returns_gradient_for_unknown_id(self):
    params = [Parameter(requires_grad=True)]
    gradients = {
      params[0].id: TextGradient(attribution='known'),
      'nonexistent-id': TextGradient(attribution='unknown'),
    }
    collator = MagicMock(spec=GradientCollator)
    collator.collate.return_value = CollationResult(context='c', gradients=gradients)
    judge = _mock_judge()
    loss = JudgeLoss(judge, collator, params)
    loss.forward(Datum(feedback='f'))
    loss.backward()
    assert params[0].grad is not None


class TestJudgeLossGradientsProperty:
  def test_gradients_none_before_backward(self):
    judge = _mock_judge()
    collator = MagicMock(spec=GradientCollator)
    loss = JudgeLoss(judge, collator)
    assert loss.gradients is None

  def test_gradients_property_after_backward(self):
    params = [Parameter(requires_grad=True)]
    collator = _collator_returning(params)
    judge = _mock_judge()
    loss = JudgeLoss(judge, collator, params)
    loss.forward(Datum(feedback='f'))
    loss.backward()
    assert isinstance(loss.gradients, CollationResult)
    assert loss.gradients.context == 'ctx'

  def test_gradients_persists_after_reset(self):
    params = [Parameter(requires_grad=True)]
    collator = _collator_returning(params)
    judge = _mock_judge()
    loss = JudgeLoss(judge, collator, params)
    loss.forward(Datum(feedback='f'))
    loss.backward()
    loss.reset()
    assert loss._accumulated == []
    assert loss.gradients is not None


class TestJudgeLossReset:
  def test_reset_clears_accumulated(self):
    judge = _mock_judge()
    collator = MagicMock(spec=GradientCollator)
    loss = JudgeLoss(judge, collator)
    loss.forward(Datum(feedback='f'))
    loss.forward(Datum(feedback='g'))
    loss.forward(Datum(feedback='h'))
    loss.reset()
    assert loss._accumulated == []
