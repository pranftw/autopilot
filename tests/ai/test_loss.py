"""Tests for JudgeLoss — graph-seeded backward (sub-plan 08).

Tests 12-15: JudgeLoss with params, without params, no forward raises, reset.
Uses mocked JudgeAgent and ConcatCollator to avoid real LLM calls.
"""

from autopilot.ai.gradient import (
  CollationResult,
  GradientCollator,
  TextGradient,
)
from autopilot.ai.loss import JudgeLoss
from autopilot.core.gradient import Gradient
from autopilot.core.graph import get_current_graph
from autopilot.core.module.module import Module
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from contextvars import copy_context
from typing import Any
from unittest.mock import MagicMock
import pytest


def _fresh_context_run(fn: Any) -> Any:
  """Run fn in a fresh contextvars context for graph isolation."""
  ctx = copy_context()
  return ctx.run(fn)


def _make_mock_judge() -> MagicMock:
  """Create a minimal mock JudgeAgent."""
  judge = MagicMock()
  judge.run = MagicMock(return_value=None)
  return judge


class _FixedCollator(GradientCollator):
  """Collator that returns a fixed CollationResult for testing."""

  def __init__(self, context: str, gradients: dict[str, Gradient] | None = None) -> None:
    self._context = context
    self._gradients = gradients or {}

  def collate(
    self,
    feedback: list[dict[str, Any]],
    parameters: list[Parameter],
  ) -> CollationResult:
    return CollationResult(context=self._context, gradients=self._gradients)


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


# Test 12: JudgeLoss backward with params — attributed seed


class TestJudgeLossBackwardWithParams:
  def test_backward_with_params_seeds_graph(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False

      m = _LeafModule()
      output = m(Datum())
      assert output.grad_fn is not None

      judge = _make_mock_judge()
      grad_for_param = TextGradient(
        text='improve accuracy',
        attribution='fix matching logic',
        severity=0.8,
      )
      collator = _FixedCollator(
        context='test context',
        gradients={m.w.id: grad_for_param},
      )
      loss = JudgeLoss(judge=judge, collator=collator, parameters=[m.w])
      loss.forward(output)
      loss.backward()

      assert m.w.grad is not None

    _fresh_context_run(run)

  def test_collation_result_stored(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False

      m = _LeafModule()
      output = m(Datum())

      judge = _make_mock_judge()
      collator = _FixedCollator(context='stored context')
      loss = JudgeLoss(judge=judge, collator=collator, parameters=[m.w])
      loss.forward(output)
      loss.backward()

      assert loss.gradients is not None
      assert isinstance(loss.gradients, CollationResult)
      assert loss.gradients.context == 'stored context'

    _fresh_context_run(run)


# Test 13: JudgeLoss backward no params — unattributed seed


class TestJudgeLossBackwardNoParams:
  def test_backward_no_params_broadcasts(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False

      m = _LeafModule()
      output = m(Datum())
      assert output.grad_fn is not None

      judge = _make_mock_judge()
      collator = _FixedCollator(context='unattributed feedback')
      loss = JudgeLoss(judge=judge, collator=collator, parameters=None)
      loss.forward(output)
      loss.backward()

      assert m.w.grad is not None

    _fresh_context_run(run)

  def test_backward_no_params_two_param_module(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False

      m = _TwoParamModule()
      output = m(Datum())

      judge = _make_mock_judge()
      collator = _FixedCollator(context='broadcast to all')
      loss = JudgeLoss(judge=judge, collator=collator, parameters=None)
      loss.forward(output)
      loss.backward()

      assert m.a.grad is not None
      assert m.b.grad is not None

    _fresh_context_run(run)


# Test 14: JudgeLoss backward without forward raises


class TestJudgeLossBackwardNoForwardRaises:
  def test_no_forward_raises(self) -> None:
    judge = _make_mock_judge()
    collator = _FixedCollator(context='unused')
    loss = JudgeLoss(judge=judge, collator=collator)
    with pytest.raises(RuntimeError, match=r'Loss.backward\(\) called without prior forward'):
      loss.backward()

  def test_after_reset_raises(self) -> None:
    judge = _make_mock_judge()
    collator = _FixedCollator(context='unused')
    loss = JudgeLoss(judge=judge, collator=collator)
    d = Datum()
    d.grad_fn = 'fake'
    loss.forward(d)
    loss.reset()
    with pytest.raises(RuntimeError, match=r'Loss.backward\(\) called without prior forward'):
      loss.backward()


# Test 15: JudgeLoss reset clears state


class TestJudgeLossReset:
  def test_reset_clears_all_state(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False

      m = _LeafModule()
      output = m(Datum())

      judge = _make_mock_judge()
      collator = _FixedCollator(context='will be cleared')
      loss = JudgeLoss(judge=judge, collator=collator, parameters=[m.w])
      loss.forward(output)
      loss.backward()

      assert loss.gradients is not None
      assert len(loss._accumulated) > 0

      loss.reset()
      assert len(loss._accumulated) == 0
      assert loss._last_data is None
      assert loss._last_collation is None
      assert loss.gradients is None

    _fresh_context_run(run)

  def test_reset_idempotent(self) -> None:
    judge = _make_mock_judge()
    collator = _FixedCollator(context='unused')
    loss = JudgeLoss(judge=judge, collator=collator)
    loss.reset()
    loss.reset()
    assert len(loss._accumulated) == 0
    assert loss._last_data is None
    assert loss.gradients is None
