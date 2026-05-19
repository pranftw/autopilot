"""Tests for Loss and Optimizer base classes."""

from autopilot.core.gradient import NumericGradient
from autopilot.core.loss import Loss
from autopilot.core.module.module import Module
from autopilot.core.optimizer import Optimizer
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
import pytest


class TestLossForwardAccumulatesAndReturnsNone:
  def test_forward_returns_none(self) -> None:
    loss = Loss()
    result = loss.forward(Datum())
    assert result is None

  def test_forward_with_targets_returns_none(self) -> None:
    loss = Loss()
    result = loss.forward(Datum(), targets='some_target')
    assert result is None


class TestLossBackwardRaises:
  def test_backward_raises_without_forward(self) -> None:
    loss = Loss()
    with pytest.raises(RuntimeError, match=r'Loss.backward.*called without prior forward'):
      loss.backward()


class TestLossGradientsReturnsNone:
  def test_gradients_is_none(self) -> None:
    loss = Loss()
    assert loss.gradients is None

  def test_gradients_property_type(self) -> None:
    loss = Loss()
    result = loss.gradients
    assert result is None


class TestLossResetNoOp:
  def test_reset_does_not_raise(self) -> None:
    loss = Loss()
    loss.reset()

  def test_reset_returns_none(self) -> None:
    loss = Loss()
    result = loss.reset()
    assert result is None

  def test_multiple_resets(self) -> None:
    loss = Loss()
    loss.reset()
    loss.reset()
    loss.reset()


class TestLossIsModule:
  def test_loss_is_module_subclass(self) -> None:
    loss = Loss()
    assert isinstance(loss, Module)

  def test_loss_auto_registers_as_child(self) -> None:
    parent = Module()
    parent.my_loss = Loss()
    children = dict(parent.named_children())
    assert 'my_loss' in children

  def test_loss_with_parameters(self) -> None:
    p = Parameter()
    loss = Loss(parameters=[p])
    assert loss._loss_parameters == [p]

  def test_loss_without_parameters(self) -> None:
    loss = Loss()
    assert loss._loss_parameters == []


class TestOptimizerStepRaises:
  def test_step_raises_not_implemented(self) -> None:
    p = Parameter()
    opt = Optimizer([p])
    with pytest.raises(NotImplementedError):
      opt.step()


class TestOptimizerZeroGrad:
  def test_zero_grad_clears_gradients(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    p1.grad = NumericGradient(value=1.0)
    p2.grad = NumericGradient(value=2.0)
    opt = Optimizer([p1, p2])
    opt.zero_grad()
    assert p1.grad is None
    assert p2.grad is None

  def test_zero_grad_on_already_none_grads(self) -> None:
    p = Parameter()
    assert p.grad is None
    opt = Optimizer([p])
    opt.zero_grad()
    assert p.grad is None

  def test_zero_grad_clears_frozen_param_grads(self) -> None:
    p = Parameter(requires_grad=False)
    p.grad = NumericGradient(value=3.0)
    opt = Optimizer([p])
    opt.zero_grad()
    assert p.grad is None


class TestOptimizerNoParameters:
  def test_step_with_no_params_raises_not_implemented(self) -> None:
    opt = Optimizer([])
    with pytest.raises(NotImplementedError):
      opt.step()

  def test_zero_grad_with_no_params(self) -> None:
    opt = Optimizer([])
    opt.zero_grad()

  def test_subclass_step_with_no_params_is_noop(self) -> None:

    class SafeOptimizer(Optimizer):
      def step(self) -> None:
        for _param in self.parameters:
          pass

    opt = SafeOptimizer([])
    opt.step()


class TestOptimizerHasStateDict:
  def test_has_state_dict_method(self) -> None:
    p = Parameter()
    opt = Optimizer([p])
    assert hasattr(opt, 'state_dict')
    state = opt.state_dict()
    assert 'defaults' in state
    assert 'blocked_strategies' in state


class TestOptimizerStrategyBlocklist:
  def test_block_and_check(self) -> None:
    opt = Optimizer([])
    opt.block_strategy('rewrite')
    assert opt.is_strategy_blocked('rewrite') is True
    assert opt.is_strategy_blocked('other') is False

  def test_unblock(self) -> None:
    opt = Optimizer([])
    opt.block_strategy('rewrite')
    opt.unblock_strategy('rewrite')
    assert opt.is_strategy_blocked('rewrite') is False

  def test_unblock_nonexistent_no_error(self) -> None:
    opt = Optimizer([])
    opt.unblock_strategy('never-added')

  def test_blocked_strategies_property(self) -> None:
    opt = Optimizer([])
    opt.block_strategy('a')
    opt.block_strategy('b')
    assert opt.blocked_strategies == frozenset({'a', 'b'})

  def test_blocked_strategies_immutable(self) -> None:
    opt = Optimizer([])
    opt.block_strategy('x')
    bs = opt.blocked_strategies
    assert isinstance(bs, frozenset)
