"""Tests for Loss base class — graph-seeded backward (sub-plan 08).

Tests 1-7: core Loss unit tests
Tests 8-11: integration tests (module -> loss -> backward -> param.grad)
"""

from autopilot.core.gradient import NumericGradient
from autopilot.core.graph import get_current_graph, no_grad
from autopilot.core.loss import Loss
from autopilot.core.module.module import Module
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from contextvars import copy_context
from typing import Any
import pytest


class _ConcreteLoss(Loss):
  """Test subclass that produces a NumericGradient seed."""

  def __init__(
    self,
    seed_value: float = 1.0,
    parameters: list[Parameter] | None = None,
  ) -> None:
    super().__init__(parameters)
    self._seed_value = seed_value

  def compute_seed_gradient(self) -> NumericGradient:
    return NumericGradient(value=self._seed_value)


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


def _fresh_context_run(fn: Any) -> Any:
  """Run fn in a fresh contextvars context for graph isolation."""
  ctx = copy_context()
  return ctx.run(fn)


# Test 1: forward returns None


class TestLossForwardReturnsNone:
  def test_forward_returns_none(self) -> None:
    loss = _ConcreteLoss()
    result = loss.forward(Datum(), targets='batch')
    assert result is None

  def test_call_returns_none(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False
      loss = _ConcreteLoss()
      with no_grad():
        result = loss(Datum(), targets='batch')
      assert result is None

    _fresh_context_run(run)


# Test 2: forward accumulates


class TestLossForwardAccumulates:
  def test_single_forward_accumulates(self) -> None:
    loss = _ConcreteLoss()
    d = Datum()
    loss.forward(d, targets='t1')
    assert len(loss._accumulated) == 1
    assert loss._accumulated[0]['data'] is d
    assert loss._accumulated[0]['targets'] == 't1'
    assert loss._last_data is d

  def test_multiple_forwards_accumulate(self) -> None:
    loss = _ConcreteLoss()
    d1, d2, d3 = Datum(), Datum(), Datum()
    loss.forward(d1, targets='t1')
    loss.forward(d2, targets='t2')
    loss.forward(d3)
    assert len(loss._accumulated) == 3
    assert loss._last_data is d3

  def test_forward_default_targets_none(self) -> None:
    loss = _ConcreteLoss()
    loss.forward(Datum())
    assert loss._accumulated[0]['targets'] is None


# Test 2b: compute_seed_gradient base raises


class TestComputeSeedGradientBaseRaises:
  def test_base_loss_raises(self) -> None:
    loss = Loss()
    with pytest.raises(NotImplementedError):
      loss.compute_seed_gradient()


# Test 3: backward without forward raises


class TestLossBackwardNoForwardRaises:
  def test_no_forward_raises(self) -> None:
    loss = _ConcreteLoss()
    with pytest.raises(RuntimeError, match=r'Loss.backward\(\) called without prior forward'):
      loss.backward()

  def test_after_reset_raises(self) -> None:
    loss = _ConcreteLoss()
    d = Datum()
    d.grad_fn = 'fake'
    loss.forward(d)
    loss.reset()
    with pytest.raises(RuntimeError, match=r'Loss.backward\(\) called without prior forward'):
      loss.backward()


# Test 4: backward with no grad_fn raises


class TestLossBackwardNoGradFnRaises:
  def test_no_grad_fn_raises(self) -> None:
    loss = _ConcreteLoss()
    d = Datum()
    loss.forward(d)
    assert d.grad_fn is None
    with pytest.raises(RuntimeError, match='cannot backward: data has no grad_fn'):
      loss.backward()


# Test 5: reset clears state


class TestLossResetClears:
  def test_reset_clears_accumulated(self) -> None:
    loss = _ConcreteLoss()
    loss.forward(Datum())
    loss.forward(Datum())
    assert len(loss._accumulated) == 2
    assert loss._last_data is not None
    loss.reset()
    assert len(loss._accumulated) == 0
    assert loss._last_data is None

  def test_reset_idempotent(self) -> None:
    loss = _ConcreteLoss()
    loss.reset()
    loss.reset()
    assert len(loss._accumulated) == 0
    assert loss._last_data is None


# Test 6: loss is not a graph node


class TestLossNotGraphNode:
  def test_loss_call_does_not_add_graph_nodes(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False
      loss = _ConcreteLoss()
      initial_len = len(g)
      loss.forward(Datum(), targets='batch')
      assert len(g) == initial_len

    _fresh_context_run(run)

  def test_loss_output_has_no_grad_fn(self) -> None:
    loss = _ConcreteLoss()
    result = loss.forward(Datum())
    assert result is None


# Test 7: loss parameters not registered in Module._parameters


class TestLossParametersNotRegistered:
  def test_loss_params_not_in_module_parameters(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    loss = _ConcreteLoss(parameters=[p1, p2])
    assert loss._loss_parameters == [p1, p2]
    module_params = dict(loss.named_parameters(recurse=False))
    assert len(module_params) == 0

  def test_loss_params_not_in_recursive_parameters(self) -> None:
    p1 = Parameter()
    loss = _ConcreteLoss(parameters=[p1])
    all_params = list(loss.parameters())
    assert p1 not in all_params


# Test 8: loss.backward() seeds graph (unit integration)


class TestLossBackwardSeedsGraph:
  def test_backward_propagates_to_parameter(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False
      m = _LeafModule()
      d = Datum()
      output = m(d)
      assert output.grad_fn is not None

      loss = _ConcreteLoss(seed_value=5.0)
      loss.forward(output)
      loss.backward()
      grad = m.w.grad
      assert grad is not None
      assert isinstance(grad, NumericGradient)
      assert grad.value == 5.0

    _fresh_context_run(run)


# Test 9: module -> loss -> backward -> param.grad (single chain)


class TestLossBackwardChain:
  def test_single_chain(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False
      m = _LeafModule()
      output = m(Datum())
      loss = _ConcreteLoss(seed_value=3.0)
      loss.forward(output)
      loss.backward()
      grad = m.w.grad
      assert grad is not None
      assert isinstance(grad, NumericGradient)
      assert grad.value == 3.0

    _fresh_context_run(run)

  def test_chain_with_reset_and_second_pass(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False
      m = _LeafModule()
      output = m(Datum())
      loss = _ConcreteLoss(seed_value=2.0)
      loss.forward(output)
      loss.backward()
      grad = m.w.grad
      assert grad is not None
      assert isinstance(grad, NumericGradient)
      assert grad.value == 2.0

      m.w.grad = None
      loss.reset()
      g._freed = False
      output2 = m(Datum())
      loss.forward(output2)
      loss.backward()
      grad2 = m.w.grad
      assert grad2 is not None
      assert isinstance(grad2, NumericGradient)
      assert grad2.value == 2.0

    _fresh_context_run(run)


# Test 10: A -> B -> loss -> backward (multi-module chain)


class TestLossBackwardMultiModuleChain:
  def test_two_module_chain_both_receive_grads(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False

      class ModA(Module):
        def __init__(self):
          super().__init__()
          self.pa = Parameter(requires_grad=True)

        def forward(self, x):
          return Datum()

      class ModB(Module):
        def __init__(self):
          super().__init__()
          self.pb = Parameter(requires_grad=True)

        def forward(self, x):
          return Datum()

      a = ModA()
      b = ModB()
      out_a = a(Datum())
      out_b = b(out_a)

      loss = _ConcreteLoss(seed_value=4.0)
      loss.forward(out_b)
      loss.backward()

      assert a.pa.grad is not None
      assert b.pb.grad is not None
      assert isinstance(a.pa.grad, NumericGradient)
      assert isinstance(b.pb.grad, NumericGradient)

    _fresh_context_run(run)


# Test 11: diamond forward pattern + single loss


class TestLossBackwardDiamond:
  def test_diamond_accumulates_correctly(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False

      class Shared(Module):
        def __init__(self):
          super().__init__()
          self.ws = Parameter(requires_grad=True)

        def forward(self, x):
          return Datum()

      class Left(Module):
        def __init__(self):
          super().__init__()
          self.wl = Parameter(requires_grad=True)

        def forward(self, x):
          return Datum()

      class Right(Module):
        def __init__(self):
          super().__init__()
          self.wr = Parameter(requires_grad=True)

        def forward(self, x):
          return Datum()

      class Merge(Module):
        def __init__(self):
          super().__init__()
          self.wm = Parameter(requires_grad=True)

        def forward(self, left, right):
          return Datum()

      shared = Shared()
      left = Left()
      right = Right()
      merge = Merge()

      s_out = shared(Datum())
      l_out = left(s_out)
      r_out = right(s_out)
      m_out = merge(l_out, r_out)

      loss = _ConcreteLoss(seed_value=1.0)
      loss.forward(m_out)
      loss.backward()

      assert merge.wm.grad is not None
      assert left.wl.grad is not None
      assert right.wr.grad is not None
      assert shared.ws.grad is not None

    _fresh_context_run(run)


# Loss is a Module


class TestLossIsModule:
  def test_loss_is_module_subclass(self) -> None:
    loss = Loss()
    assert isinstance(loss, Module)

  def test_loss_auto_registers_as_child(self) -> None:
    parent = Module()
    parent.my_loss = _ConcreteLoss()
    children = dict(parent.named_children())
    assert 'my_loss' in children

  def test_loss_with_parameters(self) -> None:
    p = Parameter()
    loss = _ConcreteLoss(parameters=[p])
    assert loss._loss_parameters == [p]

  def test_loss_without_parameters(self) -> None:
    loss = _ConcreteLoss()
    assert loss._loss_parameters == []


class TestLossGradients:
  def test_gradients_is_none(self) -> None:
    loss = Loss()
    assert loss.gradients is None

  def test_concrete_loss_gradients_none(self) -> None:
    loss = _ConcreteLoss()
    assert loss.gradients is None


class TestLossForwardOptionalTargets:
  def test_loss_forward_accepts_optional_targets(self) -> None:
    loss = _ConcreteLoss()
    d = Datum()
    loss.forward(d)
    assert len(loss._accumulated) == 1
    assert loss._accumulated[0]['targets'] is None
    loss.forward(d, targets={'label': 'pos'})
    assert loss._accumulated[1]['targets'] == {'label': 'pos'}

  def test_loss_forward_accepts_explicit_none(self) -> None:
    loss = _ConcreteLoss()
    d = Datum()
    loss.forward(d, targets=None)
    assert loss._accumulated[0]['targets'] is None
