"""Tests for JudgeLoss with GradientCollator.

Updated for graph-seeded backward: JudgeLoss.backward() now requires data
with grad_fn (from a Module.__call__ graph recording). Tests that exercise
backward() run in fresh contexts with proper graph wiring.
"""

from autopilot.ai.gradient import CollationResult, GradientCollator, TextGradient
from autopilot.ai.loss import JudgeLoss
from autopilot.core.graph import get_current_graph
from autopilot.core.module.module import Module
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum, EvalDatum
from contextvars import copy_context
from typing import Any, cast
from unittest.mock import MagicMock
import pytest


def _mock_judge() -> MagicMock:
  return MagicMock()


def _collator_returning(params: list[Parameter], context: str = 'ctx') -> MagicMock:
  gradients = {p.id: TextGradient(attribution=f'fix {p.id}') for p in params}
  result = CollationResult(context=context, gradients=cast(Any, gradients))
  collator = MagicMock(spec=GradientCollator)
  collator.collate.return_value = result
  return collator


def _fresh_context_run(fn: Any) -> Any:
  ctx = copy_context()
  return ctx.run(fn)


class _LeafModule(Module):
  def __init__(self) -> None:
    super().__init__()
    self.w = Parameter(requires_grad=True)

  def forward(self, x: Datum) -> Datum:
    return Datum()


class _TwoParamModule(Module):
  def __init__(self) -> None:
    super().__init__()
    self.a = Parameter(requires_grad=True)
    self.b = Parameter(requires_grad=True)

  def forward(self, x: Datum) -> Datum:
    return Datum()


class TestJudgeLossForward:
  def test_forward_accumulates(self):
    judge = _mock_judge()
    collator = MagicMock(spec=GradientCollator)
    loss = JudgeLoss(judge, collator)
    for i in range(3):
      loss.forward(EvalDatum(feedback=f'f{i}'))
    assert len(loss._accumulated) == 3

  def test_forward_after_backward_without_reset(self):
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False
      m = _LeafModule()
      judge = _mock_judge()
      params = [m.w]
      collator = _collator_returning(params)
      loss = JudgeLoss(judge, collator, params)
      output = m(Datum())
      loss.forward(output)
      loss.backward()

      g._freed = False
      output2 = m(Datum())
      loss.forward(output2)
      assert len(loss._accumulated) == 2

    _fresh_context_run(run)


class TestJudgeLossBackward:
  def test_backward_empty_raises(self):
    judge = _mock_judge()
    collator = MagicMock(spec=GradientCollator)
    loss = JudgeLoss(judge, collator)
    with pytest.raises(RuntimeError, match=r'Loss.backward.*called without prior forward'):
      loss.backward()

  def test_backward_calls_collator(self):
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False
      m = _LeafModule()
      judge = _mock_judge()
      params = [m.w]
      collator = _collator_returning(params)
      loss = JudgeLoss(judge, collator, params)
      output = m(Datum())
      loss.forward(output)
      loss.forward(m(Datum()))
      g._freed = False
      loss.backward()
      collator.collate.assert_called_once()
      args = collator.collate.call_args[0]
      assert len(args[0]) == 2
      assert args[1] is loss._loss_parameters

    _fresh_context_run(run)

  def test_backward_distributes_grads_via_graph(self):
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False
      m = _TwoParamModule()
      params = [m.a, m.b]
      collator = _collator_returning(params)
      judge = _mock_judge()
      loss = JudgeLoss(judge, collator, params)
      output = m(Datum())
      loss.forward(output)
      loss.backward()
      assert m.a.grad is not None
      assert m.b.grad is not None

    _fresh_context_run(run)

  def test_backward_no_grad_fn_raises(self):
    judge = _mock_judge()
    params = [Parameter(requires_grad=True)]
    collator = _collator_returning(params)
    loss = JudgeLoss(judge, collator, params)
    loss.forward(EvalDatum(feedback='f'))
    with pytest.raises(RuntimeError, match='cannot backward: data has no grad_fn'):
      loss.backward()

  def test_backward_params_always_have_unique_ids(self):
    params = [Parameter(requires_grad=True), Parameter(requires_grad=True)]
    assert params[0].id != params[1].id

  def test_collator_raises_propagates(self):
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False
      m = _LeafModule()
      collator = MagicMock(spec=GradientCollator)
      collator.collate.side_effect = RuntimeError('collator broke')
      judge = _mock_judge()
      loss = JudgeLoss(judge, collator, [m.w])
      output = m(Datum())
      loss.forward(output)
      with pytest.raises(RuntimeError, match='collator broke'):
        loss.backward()

    _fresh_context_run(run)

  def test_backward_collator_returns_gradient_for_unknown_id(self):
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False
      m = _LeafModule()
      params = [m.w]
      gradients = {
        params[0].id: TextGradient(attribution='known'),
        'nonexistent-id': TextGradient(attribution='unknown'),
      }
      collator = MagicMock(spec=GradientCollator)
      collator.collate.return_value = CollationResult(context='c', gradients=cast(Any, gradients))
      judge = _mock_judge()
      loss = JudgeLoss(judge, collator, params)
      output = m(Datum())
      loss.forward(output)
      loss.backward()
      assert m.w.grad is not None

    _fresh_context_run(run)


class TestJudgeLossGradientsProperty:
  def test_gradients_none_before_backward(self):
    judge = _mock_judge()
    collator = MagicMock(spec=GradientCollator)
    loss = JudgeLoss(judge, collator)
    assert loss.gradients is None

  def test_gradients_property_after_backward(self):
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False
      m = _LeafModule()
      params = [m.w]
      collator = _collator_returning(params)
      judge = _mock_judge()
      loss = JudgeLoss(judge, collator, params)
      output = m(Datum())
      loss.forward(output)
      loss.backward()
      assert isinstance(loss.gradients, CollationResult)
      assert loss.gradients.context == 'ctx'

    _fresh_context_run(run)

  def test_gradients_cleared_after_reset(self):
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False
      m = _LeafModule()
      params = [m.w]
      collator = _collator_returning(params)
      judge = _mock_judge()
      loss = JudgeLoss(judge, collator, params)
      output = m(Datum())
      loss.forward(output)
      loss.backward()
      assert loss.gradients is not None
      loss.reset()
      assert loss._accumulated == []
      assert loss.gradients is None

    _fresh_context_run(run)


class TestJudgeLossReset:
  def test_reset_clears_accumulated(self):
    judge = _mock_judge()
    collator = MagicMock(spec=GradientCollator)
    loss = JudgeLoss(judge, collator)
    loss.forward(EvalDatum(feedback='f'))
    loss.forward(EvalDatum(feedback='g'))
    loss.forward(EvalDatum(feedback='h'))
    loss.reset()
    assert loss._accumulated == []
