"""Tests for Optimizer.zero_grad() autograd semantics (sub-plan 10).

Unit tests for zero_grad clearing both param.grad and param.grad_accumulator,
integration tests for full training cycle with graph propagation, and
AgentOptimizer inheritance parity.
"""

from autopilot.core.gradient import NumericGradient
from autopilot.core.graph import Graph, current_graph
from autopilot.core.operator import AccumulateGrad, Context, Operator, OperatorNode
from autopilot.core.optimizer import Optimizer
from autopilot.core.parameter import Parameter
from contextvars import copy_context
from tests.doubles import StepCountingOptimizer
from typing import Any, cast
import pytest


class _PassthroughOp(Operator):
  @staticmethod
  def forward(ctx, x):
    return x

  @staticmethod
  def backward(ctx, *grads):
    grad = grads[0] if grads else None
    return (grad,)


# Unit: Optimizer.zero_grad()


class TestZeroGradClearsParamGrad:
  """Test 1: zero_grad() sets param.grad to None."""

  def test_single_param(self) -> None:
    p = Parameter()
    p.grad = NumericGradient(value=1.0)
    opt = Optimizer([p])
    opt.zero_grad()
    assert p.grad is None

  def test_multiple_params(self) -> None:
    p1 = Parameter()
    p2 = Parameter()
    p1.grad = NumericGradient(value=1.0)
    p2.grad = NumericGradient(value=2.0)
    opt = Optimizer([p1, p2])
    opt.zero_grad()
    assert p1.grad is None
    assert p2.grad is None


class TestZeroGradClearsGradAccumulator:
  """Test 2: zero_grad() sets param.grad_accumulator to None."""

  def test_with_accumulate_grad_stub(self) -> None:
    p = Parameter()
    g = Graph()
    acc = AccumulateGrad(p, g.next_sequence_nr())
    p.grad_accumulator = acc
    assert p.grad_accumulator is acc

    opt = Optimizer([p])
    opt.zero_grad()
    assert p.grad_accumulator is None

  def test_with_arbitrary_object(self) -> None:
    p = Parameter()
    sentinel = object()
    p.grad_accumulator = sentinel
    opt = Optimizer([p])
    opt.zero_grad()
    assert p.grad_accumulator is None


class TestZeroGradBothCleared:
  """Test 3: Both grad and grad_accumulator cleared in one call."""

  def test_both_cleared_simultaneously(self) -> None:
    p = Parameter()
    p.grad = NumericGradient(value=5.0)
    g = Graph()
    acc = AccumulateGrad(p, g.next_sequence_nr())
    p.grad_accumulator = acc

    opt = Optimizer([p])
    opt.zero_grad()
    assert p.grad is None
    assert p.grad_accumulator is None

  def test_multiple_params_both_cleared(self) -> None:
    params = []
    g = Graph()
    for i in range(3):
      p = Parameter()
      p.grad = NumericGradient(value=float(i))
      acc = AccumulateGrad(p, g.next_sequence_nr())
      p.grad_accumulator = acc
      params.append(p)

    opt = Optimizer(params)
    opt.zero_grad()
    for p in params:
      assert p.grad is None
      assert p.grad_accumulator is None


class TestZeroGradFrozenParam:
  """Test 4: Frozen param (requires_grad=False) still cleared."""

  def test_frozen_grad_cleared(self) -> None:
    p = Parameter(requires_grad=False)
    p.grad = NumericGradient(value=3.0)
    opt = Optimizer([p])
    opt.zero_grad()
    assert p.grad is None

  def test_frozen_accumulator_cleared(self) -> None:
    p = Parameter(requires_grad=False)
    g = Graph()
    acc = AccumulateGrad(p, g.next_sequence_nr())
    p.grad_accumulator = acc
    opt = Optimizer([p])
    opt.zero_grad()
    assert p.grad_accumulator is None

  def test_frozen_both_cleared(self) -> None:
    p = Parameter(requires_grad=False)
    p.grad = NumericGradient(value=1.0)
    sentinel = object()
    p.grad_accumulator = sentinel
    opt = Optimizer([p])
    opt.zero_grad()
    assert p.grad is None
    assert p.grad_accumulator is None


class TestZeroGradAlreadyNone:
  """Test 5: Idempotent when both are already None."""

  def test_already_none(self) -> None:
    p = Parameter()
    assert p.grad is None
    assert p.grad_accumulator is None
    opt = Optimizer([p])
    opt.zero_grad()
    assert p.grad is None
    assert p.grad_accumulator is None

  def test_double_zero_grad(self) -> None:
    p = Parameter()
    p.grad = NumericGradient(value=1.0)
    g = Graph()
    acc = AccumulateGrad(p, g.next_sequence_nr())
    p.grad_accumulator = acc
    opt = Optimizer([p])
    opt.zero_grad()
    opt.zero_grad()
    assert p.grad is None
    assert p.grad_accumulator is None


# Integration: graph + optimizer


def _run_forward_backward_cycle(param, graph):
  """Build forward graph, run backward, return the AccumulateGrad created."""
  acc = AccumulateGrad.get_or_create(param, graph)
  ctx = Context()
  op_node = OperatorNode(
    operator_cls=_PassthroughOp,
    ctx=ctx,
    next_functions=((acc, 0),),
    sequence_nr=graph.next_sequence_nr(),
  )
  graph.add_node(op_node)
  graph.backward(op_node, NumericGradient(value=1.0), retain_graph=True)
  return acc


class TestFullCycleCreatesFreshAccumulateGrad:
  """Test 6: forward -> backward -> step -> zero_grad -> forward
  yields a new AccumulateGrad distinct from the first."""

  def test_fresh_accumulate_grad_after_zero_grad(self) -> None:
    def run():
      current_graph.set(None)
      p = Parameter(requires_grad=True)
      g = Graph()
      opt = StepCountingOptimizer([p])

      acc1 = _run_forward_backward_cycle(p, g)
      assert p.grad is not None
      assert p.grad_accumulator is acc1

      opt.step()
      opt.zero_grad()
      assert p.grad_accumulator is None

      g2 = Graph()
      acc2 = AccumulateGrad.get_or_create(p, g2)
      assert acc2 is not acc1

    ctx = copy_context()
    ctx.run(run)


class TestZeroGradAfterBackward:
  """Test 7: backward populates param.grad; zero_grad() clears it."""

  def test_backward_then_zero_grad(self) -> None:
    def run():
      current_graph.set(None)
      p = Parameter(requires_grad=True)
      g = Graph()
      opt = Optimizer([p])

      _run_forward_backward_cycle(p, g)
      grad = p.grad
      assert grad is not None
      assert isinstance(grad, NumericGradient)
      assert grad.value == 1.0

      opt.zero_grad()
      assert p.grad is None
      assert p.grad_accumulator is None

    ctx = copy_context()
    ctx.run(run)


class TestMultipleTrainingSteps:
  """Test 8: three consecutive (forward, backward, step, zero_grad) iterations;
  no growing leaks of graph nodes nor stale grad_accumulator references."""

  def test_three_iterations_no_leaks(self) -> None:
    def run():
      current_graph.set(None)
      p = Parameter(requires_grad=True)
      opt = StepCountingOptimizer([p])
      previous_accumulators: list = []

      for i in range(3):
        g = Graph()
        acc = AccumulateGrad.get_or_create(p, g)

        for prev_acc in previous_accumulators:
          assert acc is not prev_acc, f'iteration {i} reused stale AccumulateGrad'

        ctx = Context()
        op_node = OperatorNode(
          operator_cls=_PassthroughOp,
          ctx=ctx,
          next_functions=((acc, 0),),
          sequence_nr=g.next_sequence_nr(),
        )
        g.add_node(op_node)
        g.backward(op_node, NumericGradient(value=float(i + 1)))

        grad = p.grad
        assert grad is not None
        assert isinstance(grad, NumericGradient)
        assert grad.value == float(i + 1)

        opt.step()
        opt.zero_grad()

        assert p.grad is None
        assert p.grad_accumulator is None
        previous_accumulators.append(acc)

      assert opt.step_count == 3

    ctx = copy_context()
    ctx.run(run)


# Optimizer state_dict / load_state_dict


class TestOptimizerStateDictRoundtrip:
  """state_dict and load_state_dict preserve defaults and blocked_strategies."""

  def test_roundtrip_defaults(self) -> None:
    p = Parameter()
    opt = Optimizer([p])
    state = opt.state_dict()
    assert state['defaults'] == {'lr': 1.0}
    assert state['blocked_strategies'] == []
    assert len(state['param_groups']) == 1
    assert state['param_groups'][0]['params'] == [p.id]
    assert state['param_groups'][0]['lr'] == 1.0

    opt2 = Optimizer([p], lr=99.0)
    opt2.load_state_dict(state)
    assert opt2.defaults['lr'] == 1.0
    assert opt2.blocked_strategies == frozenset()

  def test_roundtrip_nondefault(self) -> None:
    p = Parameter()
    opt = Optimizer([p], lr=0.42)
    opt.block_strategy('greedy')
    opt.block_strategy('random')
    state = opt.state_dict()
    assert state['defaults']['lr'] == 0.42
    assert state['blocked_strategies'] == ['greedy', 'random']

    opt2 = Optimizer([p])
    opt2.load_state_dict(state)
    assert opt2.defaults['lr'] == 0.42
    assert opt2.blocked_strategies == frozenset({'greedy', 'random'})

  def test_load_state_dict_missing_blocked_strategies_raises(self) -> None:
    """Missing blocked_strategies key raises KeyError."""
    p = Parameter()
    opt = Optimizer([p], lr=0.1)
    with pytest.raises(KeyError, match='blocked_strategies'):
      opt.load_state_dict(
        {
          'defaults': {'lr': 0.5},
          'param_groups': [{'params': [p.id], 'lr': 0.5}],
        }
      )


class TestAgentOptimizerStateDictIncludesContext:
  """AgentOptimizer state_dict includes context; round-trip preserves it."""

  def test_context_in_state_dict(self) -> None:
    from autopilot.ai.optimizer import AgentOptimizer

    class _StubAgent:
      limiter = None

      def run(self, prompt, context=None):
        return None

    p = Parameter()
    agent = _StubAgent()
    opt = AgentOptimizer(
      agent=cast(Any, agent),
      params=[p],
      context={'epoch': 3, 'metrics': {'acc': 1.0}},
    )
    state = opt.state_dict()
    assert state['context'] == {'epoch': 3, 'metrics': {'acc': 1.0}}
    assert state['defaults']['lr'] == 1.0

  def test_context_roundtrip(self) -> None:
    from autopilot.ai.optimizer import AgentOptimizer

    class _StubAgent:
      limiter = None

      def run(self, prompt, context=None):
        return None

    p = Parameter()
    agent = _StubAgent()
    opt = AgentOptimizer(
      agent=cast(Any, agent),
      params=[p],
      context={'epoch': 5, 'tag': 'v1'},
    )
    state = opt.state_dict()

    opt2 = AgentOptimizer(agent=cast(Any, agent), params=[p])
    opt2.load_state_dict(state)
    ctx = opt2.build_context()
    assert ctx['epoch'] == 5
    assert ctx['tag'] == 'v1'
    assert ctx['allowed_paths'] == []
    assert ctx['forbidden_paths'] == []

  def test_load_without_context_key_preserves_default(self) -> None:
    from autopilot.ai.optimizer import AgentOptimizer

    class _StubAgent:
      limiter = None

      def run(self, prompt, context=None):
        return None

    p = Parameter()
    agent = _StubAgent()
    opt = AgentOptimizer(agent=cast(Any, agent), params=[p])
    state = opt.state_dict()
    state['defaults']['lr'] = 0.5
    opt.load_state_dict(state)
    ctx = opt.build_context()
    assert ctx['allowed_paths'] == []
    assert ctx['forbidden_paths'] == []
    assert opt.defaults['lr'] == 0.5


# AgentOptimizer parity


class TestAgentOptimizerZeroGradClearsAccumulator:
  """Test 9: AgentOptimizer inherits zero_grad that clears grad_accumulator."""

  def test_inherited_zero_grad(self) -> None:
    from autopilot.ai.optimizer import AgentOptimizer

    p = Parameter()
    p.grad = NumericGradient(value=1.0)
    g = Graph()
    acc = AccumulateGrad(p, g.next_sequence_nr())
    p.grad_accumulator = acc

    class _StubAgent:
      limiter = None

      def run(self, prompt, context=None):
        return None

    agent = _StubAgent()
    opt = AgentOptimizer(agent=cast(Any, agent), params=[p])
    opt.zero_grad()
    assert p.grad is None
    assert p.grad_accumulator is None

  def test_agent_optimizer_does_not_override_zero_grad(self) -> None:
    from autopilot.ai.optimizer import AgentOptimizer

    assert 'zero_grad' not in AgentOptimizer.__dict__
