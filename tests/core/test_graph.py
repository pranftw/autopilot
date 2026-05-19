"""Tests for computation graph: Graph, backward engine, context managers, RemovableHandle,
Module.__call__ via ModuleCallOperator.

Uses OperatorNode / AccumulateGrad from core/operator.py (plan 01).
Old Node / Graph.record / _create_gradient_edge tests are removed.
"""

from autopilot.core.gradient import NumericGradient
from autopilot.core.graph import (
  Graph,
  RemovableHandle,
  enable_grad,
  get_current_graph,
  is_grad_enabled,
  no_grad,
)
from autopilot.core.module.module import Module
from autopilot.core.operator import (
  AccumulateGrad,
  Context,
  Operator,
  OperatorNode,
)
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from collections import OrderedDict
from contextvars import copy_context
from tests.doubles import NoOpOptimizer
import pytest


class _PassthroughOp(Operator):
  """Operator that passes gradient through unchanged to all inputs."""

  @staticmethod
  def forward(ctx, *args):
    return args[0] if args else Datum()

  @staticmethod
  def backward(ctx, *grads):
    grad = grads[0] if grads else None
    return (grad,)


class _FanOutOp(Operator):
  """Operator that duplicates gradient to two inputs."""

  @staticmethod
  def forward(ctx, a, b):
    return a

  @staticmethod
  def backward(ctx, *grads):
    grad = grads[0] if grads else None
    return (grad, grad)


class _NoneGradOp(Operator):
  """Operator that returns None gradient for the second input."""

  @staticmethod
  def forward(ctx, a, b):
    return a

  @staticmethod
  def backward(ctx, *grads):
    grad = grads[0] if grads else None
    return (grad, None)


def _make_passthrough_node(
  graph,
  next_functions=(),
) -> OperatorNode:
  """Create a passthrough OperatorNode wired into the graph."""
  ctx = Context()
  node = OperatorNode(
    operator_cls=_PassthroughOp,
    ctx=ctx,
    next_functions=tuple(next_functions),
    sequence_nr=graph.next_sequence_nr(),
  )
  graph.add_node(node)
  return node


# Graph basics (tests 1-6)


class TestGraphBasics:
  def test_graph_init(self) -> None:
    g = Graph()
    assert len(g) == 0
    assert g._freed is False

  def test_add_node(self) -> None:
    g = Graph()
    ctx = Context()
    node = OperatorNode(operator_cls=_PassthroughOp, ctx=ctx, sequence_nr=0)
    g.add_node(node)
    assert len(g) == 1

  def test_sequence_nr_increments(self) -> None:
    g = Graph()
    assert g.next_sequence_nr() == 0
    assert g.next_sequence_nr() == 1

  def test_nodes_iterator(self) -> None:
    g = Graph()
    n1 = _make_passthrough_node(g)
    n2 = _make_passthrough_node(g)
    nodes = list(g.nodes())
    assert len(nodes) == 2
    assert n1 in nodes
    assert n2 in nodes

  def test_reset_clears(self) -> None:
    g = Graph()
    _make_passthrough_node(g)
    g.reset()
    assert len(g) == 0
    assert g._freed is True

  def test_repr(self) -> None:
    g = Graph()
    assert repr(g) == 'Graph(nodes=0)'
    _make_passthrough_node(g)
    assert repr(g) == 'Graph(nodes=1)'


# Backward engine (tests 7-12)


class TestBackwardEngine:
  def test_linear_chain(self) -> None:
    """AccumulateGrad <- Op1 <- Op2, gradient flows through."""
    g = Graph()
    p = Parameter(requires_grad=True)
    acc = AccumulateGrad(p, g.next_sequence_nr())
    g.add_node(acc)
    op1 = _make_passthrough_node(g, next_functions=[(acc, 0)])
    op2 = _make_passthrough_node(g, next_functions=[(op1, 0)])
    g.backward(op2, grad=NumericGradient(value=1.0))
    grad = p.grad
    assert grad is not None
    assert isinstance(grad, NumericGradient)
    assert grad.value == 1.0

  def test_diamond_fan_in(self) -> None:
    """Two paths into the same AccumulateGrad; accumulation 1.0 + 1.0 = 2.0."""
    g = Graph()
    p = Parameter(requires_grad=True)
    acc = AccumulateGrad(p, g.next_sequence_nr())
    g.add_node(acc)
    left = _make_passthrough_node(g, next_functions=[(acc, 0)])
    right = _make_passthrough_node(g, next_functions=[(acc, 0)])

    ctx = Context()
    merge = OperatorNode(
      operator_cls=_FanOutOp,
      ctx=ctx,
      next_functions=((left, 0), (right, 0)),
      sequence_nr=g.next_sequence_nr(),
    )
    g.add_node(merge)
    g.backward(merge, grad=NumericGradient(value=1.0))
    grad = p.grad
    assert grad is not None
    assert isinstance(grad, NumericGradient)
    assert grad.value == 2.0

  def test_backward_resets_by_default(self) -> None:
    g = Graph()
    node = _make_passthrough_node(g)
    g.backward(node, grad=NumericGradient(value=1.0))
    assert len(g) == 0

  def test_backward_retain_graph(self) -> None:
    g = Graph()
    node = _make_passthrough_node(g)
    g.backward(node, grad=NumericGradient(value=1.0), retain_graph=True)
    assert len(g) >= 1

  def test_fan_out(self) -> None:
    """One node feeds two downstream nodes."""
    g = Graph()
    p1 = Parameter(requires_grad=True)
    p2 = Parameter(requires_grad=True)
    acc1 = AccumulateGrad(p1, g.next_sequence_nr())
    g.add_node(acc1)
    acc2 = AccumulateGrad(p2, g.next_sequence_nr())
    g.add_node(acc2)

    ctx = Context()
    fan = OperatorNode(
      operator_cls=_FanOutOp,
      ctx=ctx,
      next_functions=((acc1, 0), (acc2, 0)),
      sequence_nr=g.next_sequence_nr(),
    )
    g.add_node(fan)
    g.backward(fan, grad=NumericGradient(value=3.0))
    g1 = p1.grad
    g2 = p2.grad
    assert g1 is not None
    assert g2 is not None
    assert isinstance(g1, NumericGradient)
    assert isinstance(g2, NumericGradient)
    assert g1.value == 3.0
    assert g2.value == 3.0

  def test_none_gradient_skip(self) -> None:
    """backward returns None for one input; that branch receives no gradient."""
    g = Graph()
    p1 = Parameter(requires_grad=True)
    p2 = Parameter(requires_grad=True)
    acc1 = AccumulateGrad(p1, g.next_sequence_nr())
    g.add_node(acc1)
    acc2 = AccumulateGrad(p2, g.next_sequence_nr())
    g.add_node(acc2)

    ctx = Context()
    node = OperatorNode(
      operator_cls=_NoneGradOp,
      ctx=ctx,
      next_functions=((acc1, 0), (acc2, 0)),
      sequence_nr=g.next_sequence_nr(),
    )
    g.add_node(node)
    g.backward(node, grad=NumericGradient(value=5.0))
    grad = p1.grad
    assert grad is not None
    assert isinstance(grad, NumericGradient)
    assert grad.value == 5.0
    assert p2.grad is None


# Cycle detection (tests 13-15)


class TestCycleDetection:
  def test_cycle_raises(self) -> None:
    """Artificial cycle raises RuntimeError."""
    g = Graph()
    ctx = Context()
    node_a = OperatorNode(
      operator_cls=_PassthroughOp,
      ctx=ctx,
      next_functions=(),
      sequence_nr=0,
    )
    node_b = OperatorNode(
      operator_cls=_PassthroughOp,
      ctx=ctx,
      next_functions=((node_a, 0),),
      sequence_nr=1,
    )
    node_a._next_functions = ((node_b, 0),)
    g.add_node(node_a)
    g.add_node(node_b)
    with pytest.raises(RuntimeError, match='cycle detected'):
      g.backward(node_b, grad=NumericGradient(value=1.0))

  def test_diamond_no_false_positive(self) -> None:
    """Diamond graph is not reported as a cycle."""
    g = Graph()
    p = Parameter(requires_grad=True)
    acc = AccumulateGrad(p, g.next_sequence_nr())
    g.add_node(acc)
    left = _make_passthrough_node(g, next_functions=[(acc, 0)])
    right = _make_passthrough_node(g, next_functions=[(acc, 0)])
    ctx = Context()
    merge = OperatorNode(
      operator_cls=_FanOutOp,
      ctx=ctx,
      next_functions=((left, 0), (right, 0)),
      sequence_nr=g.next_sequence_nr(),
    )
    g.add_node(merge)
    g.backward(merge, grad=NumericGradient(value=1.0))
    grad = p.grad
    assert grad is not None
    assert isinstance(grad, NumericGradient)
    assert grad.value == 2.0

  def test_deep_chain_no_false_positive(self) -> None:
    """Depth 10 chain is not a cycle."""
    g = Graph()
    p = Parameter(requires_grad=True)
    acc = AccumulateGrad(p, g.next_sequence_nr())
    g.add_node(acc)
    prev = acc
    for _ in range(10):
      prev = _make_passthrough_node(g, next_functions=[(prev, 0)])
    g.backward(prev, grad=NumericGradient(value=1.0))
    grad = p.grad
    assert grad is not None
    assert isinstance(grad, NumericGradient)
    assert grad.value == 1.0


# _freed flag (tests 16-19)


class TestFreedFlag:
  def test_second_backward_raises_after_reset(self) -> None:
    g = Graph()
    node = _make_passthrough_node(g)
    g.backward(node, grad=NumericGradient(value=1.0), retain_graph=False)
    with pytest.raises(RuntimeError, match='graph has been freed'):
      g.backward(node, grad=NumericGradient(value=1.0))

  def test_new_forward_after_reset_works(self) -> None:
    g = Graph()
    node = _make_passthrough_node(g)
    g.backward(node, grad=NumericGradient(value=1.0))
    assert g._freed is True
    p = Parameter(requires_grad=True)
    acc = AccumulateGrad(p, g.next_sequence_nr())
    g.add_node(acc)
    assert g._freed is False
    new_node = _make_passthrough_node(g, next_functions=[(acc, 0)])
    g.backward(new_node, grad=NumericGradient(value=2.0))
    grad = p.grad
    assert grad is not None
    assert isinstance(grad, NumericGradient)
    assert grad.value == 2.0

  def test_freed_flag_false_on_fresh_graph(self) -> None:
    g = Graph()
    assert g._freed is False

  def test_retain_graph_then_second_backward_works(self) -> None:
    g = Graph()
    p = Parameter(requires_grad=True)
    acc = AccumulateGrad(p, g.next_sequence_nr())
    g.add_node(acc)
    node = _make_passthrough_node(g, next_functions=[(acc, 0)])
    g.backward(node, grad=NumericGradient(value=1.0), retain_graph=True)
    grad = p.grad
    assert grad is not None
    assert isinstance(grad, NumericGradient)
    assert grad.value == 1.0
    p.grad = None
    g.backward(node, grad=NumericGradient(value=3.0))
    grad2 = p.grad
    assert grad2 is not None
    assert isinstance(grad2, NumericGradient)
    assert grad2.value == 3.0


# Boundary / edge cases (tests 32-33)


class TestBoundaryEdgeCases:
  def test_backward_single_accumulate_grad_only(self) -> None:
    """Graph with one AccumulateGrad node as root; gradient delivered directly."""
    g = Graph()
    p = Parameter(requires_grad=True)
    acc = AccumulateGrad(p, g.next_sequence_nr())
    g.add_node(acc)
    g.backward(acc, grad=NumericGradient(value=7.0))
    grad = p.grad
    assert grad is not None
    assert isinstance(grad, NumericGradient)
    assert grad.value == 7.0

  def test_backward_none_grad_skips_processing(self) -> None:
    """backward(root, grad=None) -- no parameter receives a gradient."""
    g = Graph()
    p = Parameter(requires_grad=True)
    acc = AccumulateGrad(p, g.next_sequence_nr())
    g.add_node(acc)
    node = _make_passthrough_node(g, next_functions=[(acc, 0)])
    g.backward(node, grad=None)
    assert p.grad is None


# Context managers (tests 34-38)


class TestContextManagers:
  def test_grad_enabled_by_default(self) -> None:
    assert is_grad_enabled() is True

  def test_no_grad_disables(self) -> None:
    with no_grad():
      assert is_grad_enabled() is False
    assert is_grad_enabled() is True

  def test_enable_grad_inside_no_grad(self) -> None:
    with no_grad():
      assert is_grad_enabled() is False
      with enable_grad():
        assert is_grad_enabled() is True
      assert is_grad_enabled() is False

  def test_triple_nesting(self) -> None:
    with no_grad():
      with enable_grad():
        with no_grad():
          assert is_grad_enabled() is False
        assert is_grad_enabled() is True
      assert is_grad_enabled() is False

  def test_contextvar_isolation(self) -> None:
    """Separate contexts isolate graphs."""
    from autopilot.core.graph import current_graph

    graphs: list[Graph] = []

    def worker():
      current_graph.set(None)
      g = get_current_graph()
      graphs.append(g)

    ctx1 = copy_context()
    ctx2 = copy_context()
    ctx1.run(worker)
    ctx2.run(worker)
    assert graphs[0] is not graphs[1]


# RemovableHandle (tests 39-40)


class TestRemovableHandle:
  def test_remove_from_dict(self) -> None:
    d: OrderedDict = OrderedDict()
    handle = RemovableHandle(d)
    d[handle.id] = lambda: None
    assert handle.id in d
    handle.remove()
    assert handle.id not in d

  def test_double_remove_idempotent(self) -> None:
    d: OrderedDict = OrderedDict()
    handle = RemovableHandle(d)
    d[handle.id] = lambda: None
    handle.remove()
    handle.remove()


# Module.__call__ via ModuleCallOperator (tests 41-42)


class TestModuleCallOperator:
  def test_module_call_runs_hooks(self) -> None:
    """Pre- and post-hooks still run."""
    calls: list[str] = []

    class M(Module):
      def forward(self, x):
        return Datum()

    m = M()
    m.register_forward_pre_hook(lambda mod, args, kwargs: calls.append('pre'))
    m.register_forward_hook(lambda mod, args, out: calls.append('post'))
    with no_grad():
      m(Datum())
    assert calls == ['pre', 'post']

  def test_module_call_records_graph(self) -> None:
    """Module.__call__ sets grad_fn on Datum outputs via ModuleCallOperator."""

    def run():
      from autopilot.core.parameter import Parameter

      class M(Module):
        def __init__(self):
          super().__init__()
          self.w = Parameter(requires_grad=True)

        def forward(self, x):
          return Datum()

      g = get_current_graph()
      g.reset()
      g._freed = False
      m = M()
      result = m(Datum())
      assert result.grad_fn is not None
      assert len(g) > 0

    ctx = copy_context()
    ctx.run(run)


# Non-Datum forward return (test 43)


class TestNonDatumForwardReturn:
  def test_non_datum_forward_creates_no_graph_edges(self) -> None:
    """Forward returning non-Datum does not create invalid graph edges."""

    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False
      initial_count = len(g)

      class M(Module):
        def __init__(self):
          super().__init__()
          self.w = Parameter(requires_grad=True)

        def forward(self, x):
          return 'not a datum'

      m = M()
      result = m(Datum())
      assert result == 'not a datum'
      assert len(g) == initial_count

    ctx = copy_context()
    ctx.run(run)

  def test_none_return_creates_no_graph_edges(self) -> None:
    """Forward returning None does not create invalid graph edges."""

    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False
      initial_count = len(g)

      class M(Module):
        def forward(self, x):
          return None

      m = M()
      result = m(Datum())
      assert result is None
      assert len(g) == initial_count

    ctx = copy_context()
    ctx.run(run)


# Multi-loss on same graph with retain_graph (test 6 from plan 13)


class TestMultiLossRetainGraph:
  def test_two_losses_retain_graph(self) -> None:
    """Two losses on the same graph with retain_graph=True both propagate."""
    from autopilot.core.loss import Loss

    class _NumLoss(Loss):
      def __init__(self, seed_value, **kw):
        super().__init__(**kw)
        self._seed_value = seed_value

      def compute_seed_gradient(self):
        return NumericGradient(value=self._seed_value)

    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False

      class M(Module):
        def __init__(self):
          super().__init__()
          self.w = Parameter(requires_grad=True)

        def forward(self, x):
          return Datum()

      m = M()
      output = m(Datum())
      assert output.grad_fn is not None

      loss1 = _NumLoss(seed_value=3.0)
      loss2 = _NumLoss(seed_value=5.0)

      loss1.forward(output)
      loss2.forward(output)

      g.backward(output.grad_fn, loss1.compute_seed_gradient(), retain_graph=True)
      grad1 = m.w.grad
      assert grad1 is not None
      assert isinstance(grad1, NumericGradient)
      assert grad1.value == 3.0

      g.backward(output.grad_fn, loss2.compute_seed_gradient())
      grad2 = m.w.grad
      assert grad2 is not None
      assert isinstance(grad2, NumericGradient)
      assert grad2.value == 8.0

    ctx = copy_context()
    ctx.run(run)


# Name disambiguation: OperatorNode vs Node (test 7 from plan 13)


class TestNameDisambiguation:
  def test_operatornode_and_node_coexist(self) -> None:
    """Both OperatorNode and Node can be imported in the same scope."""
    from autopilot.core.node import Node as TreeNode
    from autopilot.core.operator import OperatorNode as OpNode

    assert OpNode is not TreeNode
    assert OpNode.__name__ == 'OperatorNode'
    assert TreeNode.__name__ == 'Node'

  def test_graph_does_not_reexport_operatornode(self) -> None:
    """graph.py does not re-export OperatorNode, Node, or other symbols."""
    import autopilot.core.graph as graph_module

    public_names = [n for n in dir(graph_module) if not n.startswith('_')]
    assert 'OperatorNode' not in public_names
    assert 'Node' not in public_names
    assert 'AccumulateGrad' not in public_names


# zero_grad + grad_accumulator cleanup (test 5 from plan 13)


class TestZeroGradAccumulatorCleanup:
  def test_zero_grad_clears_grad_and_accumulator(self) -> None:
    """optimizer.zero_grad() clears param.grad AND grad_accumulator."""

    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False

      p = Parameter(requires_grad=True)
      opt = NoOpOptimizer([p])

      class M(Module):
        def __init__(self):
          super().__init__()
          self.w = p

        def forward(self, x):
          return Datum()

      m = M()
      output = m(Datum())
      output.backward(NumericGradient(value=1.0))
      assert p.grad is not None
      assert p.grad_accumulator is not None

      opt.zero_grad()
      assert p.grad is None
      assert p.grad_accumulator is None

    ctx = copy_context()
    ctx.run(run)

  def test_zero_grad_allows_fresh_forward_backward(self) -> None:
    """After zero_grad, a new forward+backward cycle works with fresh AccumulateGrad."""

    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False

      p = Parameter(requires_grad=True)
      opt = NoOpOptimizer([p])

      class M(Module):
        def __init__(self):
          super().__init__()
          self.w = p

        def forward(self, x):
          return Datum()

      m = M()
      output1 = m(Datum())
      output1.backward(NumericGradient(value=2.0))
      grad1 = p.grad
      assert grad1 is not None
      assert isinstance(grad1, NumericGradient)
      assert grad1.value == 2.0

      opt.zero_grad()
      g._freed = False

      output2 = m(Datum())
      output2.backward(NumericGradient(value=7.0))
      grad2 = p.grad
      assert grad2 is not None
      assert isinstance(grad2, NumericGradient)
      assert grad2.value == 7.0

    ctx = copy_context()
    ctx.run(run)

  def test_zero_grad_frozen_param_still_cleared(self) -> None:
    """zero_grad() clears accumulator even on frozen (requires_grad=False) params."""
    p = Parameter(requires_grad=False)
    p.grad_accumulator = 'stale_ref'
    opt = NoOpOptimizer([p])
    opt.zero_grad()
    assert p.grad_accumulator is None


# Datum subclass preservation through operator (test 8 from plan 13)


class TestDatumSubclassPreservation:
  def test_clone_operator_preserves_subclass(self) -> None:
    """CloneOperator preserves Datum subclass type."""
    from autopilot.core.ops import CloneOperator
    from dataclasses import dataclass

    @dataclass
    class AgentOutput(Datum):
      response: str | None = None

    original = AgentOutput(response='hello')
    cloned = CloneOperator.apply(original)
    assert type(cloned) is AgentOutput
    assert cloned.response == 'hello'

  def test_identity_operator_preserves_subclass(self) -> None:
    """IdentityOperator preserves Datum subclass type."""
    from autopilot.core.ops import IdentityOperator
    from dataclasses import dataclass

    @dataclass
    class AgentOutput(Datum):
      response: str | None = None

    original = AgentOutput(response='world')
    result = IdentityOperator.apply(original)
    assert type(result) is AgentOutput
    assert result.response == 'world'


# Loss.forward() returns None, no graph node (test 11 from plan 13)


# Serialization round-trip: grad_fn not serialized (test 4 from plan 13)


class TestSerializationRoundTripGradFn:
  def test_datum_with_grad_fn_roundtrip_loses_grad_fn(self) -> None:
    """Datum with grad_fn → to_dict → from_dict → grad_fn absent."""
    d = Datum()
    mock_node = _make_passthrough_node(Graph())
    d.grad_fn = mock_node
    assert d.grad_fn is not None

    data = d.to_dict()
    assert 'grad_fn' not in data

    restored = Datum.from_dict(data)
    assert restored.grad_fn is None
    assert restored.id == d.id

  def test_parameter_with_grad_fn_roundtrip_loses_grad_fn(self) -> None:
    """Parameter with grad_fn → to_dict → from_dict → grad_fn absent."""
    p = Parameter(requires_grad=True)
    mock_node = _make_passthrough_node(Graph())
    p.grad_fn = mock_node
    assert p.grad_fn is not None

    data = p.to_dict()
    assert 'grad_fn' not in data

    restored = Parameter.from_dict(data)
    assert restored.grad_fn is None
    assert restored.requires_grad is True


# Post-hook grad_fn backward (test 13 supplement from plan 13)


class TestPostHookGradFnBackward:
  def test_post_hook_replacement_backward_flows(self) -> None:
    """Post-hook replaces Datum; backward through replacement reaches param."""

    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False

      class M(Module):
        def __init__(self):
          super().__init__()
          self.w = Parameter(requires_grad=True)

        def forward(self, x):
          return Datum()

      m = M()

      def hook(mod, args, output):
        return Datum()

      m.register_forward_hook(hook)
      output = m(Datum())
      assert output.grad_fn is not None
      output.backward(NumericGradient(value=4.0))
      grad = m.w.grad
      assert grad is not None
      assert isinstance(grad, NumericGradient)
      assert grad.value == 4.0

    ctx = copy_context()
    ctx.run(run)


# Loss.forward() returns None, no graph node (test 11 from plan 13)


class TestLossForwardNoneNoGraphNode:
  def test_loss_forward_returns_none_and_no_node_added(self) -> None:
    """Loss.forward() returns None; no graph node is created."""
    from autopilot.core.loss import Loss

    class _SimpleLoss(Loss):
      def compute_seed_gradient(self):
        return NumericGradient(value=1.0)

    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False

      loss = _SimpleLoss()
      nodes_before = len(g)
      out = loss.forward(Datum(), targets=Datum())
      assert out is None
      assert len(g) == nodes_before

    ctx = copy_context()
    ctx.run(run)

  def test_loss_call_returns_none_and_no_node_added(self) -> None:
    """Loss.__call__() returns None; no graph node is created."""
    from autopilot.core.loss import Loss

    class _SimpleLoss(Loss):
      def compute_seed_gradient(self):
        return NumericGradient(value=1.0)

    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False

      loss = _SimpleLoss()
      nodes_before = len(g)
      out = loss(Datum(), targets=Datum())
      assert out is None
      assert len(g) == nodes_before

    ctx = copy_context()
    ctx.run(run)
