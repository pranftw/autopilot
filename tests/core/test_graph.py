"""Tests for computation graph: Node, Graph, AccumulateGrad, context managers."""

from autopilot.core.graph import (
  AccumulateGrad,
  Graph,
  Node,
  RemovableHandle,
  _collect_input_nodes,
  _create_gradient_edge,
  _flatten,
  enable_grad,
  get_current_graph,
  is_grad_enabled,
  no_grad,
)
from autopilot.core.models import Datum
from autopilot.core.module import Module
from autopilot.core.parameter import Parameter
from collections import OrderedDict
from contextvars import copy_context


class TestNodeBasics:
  def test_node_creation(self) -> None:
    node = Node(module=None, next_functions=(), sequence_nr=0)
    assert node.name() == 'Node'
    assert node.next_functions == ()
    assert node._sequence_nr == 0

  def test_node_with_module(self) -> None:
    class MyMod:
      pass

    mod = MyMod()
    node = Node(module=mod, sequence_nr=5)
    assert node.name() == 'MyMod'
    assert node._sequence_nr == 5

  def test_node_next_functions_wiring(self) -> None:
    parent = Node(sequence_nr=0)
    child = Node(next_functions=((parent, 0),), sequence_nr=1)
    assert child.next_functions == ((parent, 0),)

  def test_node_apply_passthrough(self) -> None:
    node = Node()
    result = node.apply(42)
    assert result == (42,)

  def test_node_call_runs_hooks(self) -> None:
    calls: list[str] = []
    node = Node()

    def prehook(n, grads):
      calls.append('pre')
      return None

    def posthook(n, grads, output):
      calls.append('post')
      return None

    node.register_prehook(prehook)
    node.register_hook(posthook)
    node(1.0)
    assert calls == ['pre', 'post']

  def test_node_repr(self) -> None:
    node = Node(sequence_nr=3)
    assert 'seq=3' in repr(node)


class TestAccumulateGrad:
  def test_accumulate_sets_grad(self) -> None:
    p = Parameter(requires_grad=True)
    acc = AccumulateGrad(p, sequence_nr=0)
    assert acc.name() == 'AccumulateGrad'
    acc.apply(10)
    assert p.grad == 10

  def test_accumulate_adds_to_existing(self) -> None:
    p = Parameter(requires_grad=True)
    p.grad = 5
    acc = AccumulateGrad(p, sequence_nr=0)
    acc.apply(3)
    assert p.grad == 8

  def test_accumulate_none_grad_noop(self) -> None:
    p = Parameter(requires_grad=True)
    acc = AccumulateGrad(p, sequence_nr=0)
    acc.apply(None)
    assert p.grad is None

  def test_multiple_accumulations(self) -> None:
    p = Parameter(requires_grad=True)
    acc = AccumulateGrad(p, sequence_nr=0)
    acc.apply(1)
    acc.apply(2)
    acc.apply(3)
    assert p.grad == 6

  def test_zero_grad_then_accumulate(self) -> None:
    p = Parameter(requires_grad=True)
    acc = AccumulateGrad(p, sequence_nr=0)
    acc.apply(5)
    assert p.grad == 5
    p.grad = None
    acc.apply(3)
    assert p.grad == 3


class TestGraph:
  def test_record_creates_node(self) -> None:
    g = Graph()
    node = g.record(module=None, inputs=(), output=None, prev_nodes=[])
    assert len(g) == 1
    assert node._sequence_nr == 0

  def test_sequence_nr_increments(self) -> None:
    g = Graph()
    n0 = g.record(None, (), None, [])
    n1 = g.record(None, (), None, [])
    assert n0._sequence_nr == 0
    assert n1._sequence_nr == 1

  def test_nodes_iterator(self) -> None:
    g = Graph()
    g.record(None, (), None, [])
    g.record(None, (), None, [])
    assert len(list(g.nodes())) == 2

  def test_reset_clears(self) -> None:
    g = Graph()
    g.record(None, (), None, [])
    g.reset()
    assert len(g) == 0

  def test_repr(self) -> None:
    g = Graph()
    assert 'nodes=0' in repr(g)


class TestBackward:
  def test_linear_chain(self) -> None:
    p = Parameter(requires_grad=True)
    g = Graph()
    acc = AccumulateGrad(p, g._next_sequence_nr())
    g._nodes.append(acc)
    n1 = g.record(None, (), None, [(acc, 0)])
    n2 = g.record(None, (), None, [(n1, 0)])
    g.backward(n2, grad=1.0)
    assert p.grad == 1.0

  def test_diamond_dag_accumulates(self) -> None:
    p = Parameter(requires_grad=True)
    g = Graph()
    acc = AccumulateGrad(p, g._next_sequence_nr())
    g._nodes.append(acc)
    left = g.record(None, (), None, [(acc, 0)])
    right = g.record(None, (), None, [(acc, 0)])
    merge = g.record(None, (), None, [(left, 0), (right, 0)])

    merge.apply = lambda *grads: (grads[0], grads[0]) if grads else ()
    g.backward(merge, grad=1.0)
    assert p.grad is not None

  def test_backward_resets_by_default(self) -> None:
    g = Graph()
    n = g.record(None, (), None, [])
    g.backward(n, grad=1.0)
    assert len(g) == 0

  def test_backward_retain_graph(self) -> None:
    g = Graph()
    n = g.record(None, (), None, [])
    g.backward(n, grad=1.0, retain_graph=True)
    assert len(g) == 1

  def test_empty_graph_backward(self) -> None:
    g = Graph()
    n = Node(sequence_nr=0)
    g.backward(n, grad=1.0)


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

  def test_no_grad_restores_previous(self) -> None:
    with no_grad():
      with no_grad():
        pass
      assert is_grad_enabled() is False


class TestGetCurrentGraph:
  def test_lazily_creates_graph(self) -> None:
    def run():
      g = get_current_graph()
      assert isinstance(g, Graph)

    ctx = copy_context()
    ctx.run(run)

  def test_same_graph_returned(self) -> None:
    def run():
      g1 = get_current_graph()
      g2 = get_current_graph()
      assert g1 is g2

    ctx = copy_context()
    ctx.run(run)


class TestContextVarIsolation:
  def test_separate_contexts_separate_graphs(self) -> None:
    from autopilot.core.graph import _current_graph

    graphs: list[Graph] = []

    def worker():
      _current_graph.set(None)
      g = get_current_graph()
      graphs.append(g)

    ctx1 = copy_context()
    ctx2 = copy_context()
    ctx1.run(worker)
    ctx2.run(worker)
    assert graphs[0] is not graphs[1]


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

  def test_remove_after_dict_gc(self) -> None:
    d: OrderedDict = OrderedDict()
    handle = RemovableHandle(d)
    d[handle.id] = lambda: None
    del d
    handle.remove()

  def test_hooks_fire_in_order(self) -> None:
    order: list[int] = []
    node = Node()
    node.register_hook(lambda n, g, o: order.append(1))
    node.register_hook(lambda n, g, o: order.append(2))
    node(1.0)
    assert order == [1, 2]


class TestDatumGradFn:
  def test_grad_fn_default_none(self) -> None:
    d = Datum()
    assert d.grad_fn is None

  def test_grad_fn_not_in_to_dict(self) -> None:
    d = Datum()
    assert 'grad_fn' not in d.to_dict()

  def test_grad_fn_roundtrip(self) -> None:
    d = Datum(success=True, metrics={'x': 1.0})
    data = d.to_dict()
    d2 = Datum.from_dict(data)
    assert d2.grad_fn is None

  def test_parameter_grad_fn_default_none(self) -> None:
    p = Parameter()
    assert p.grad_fn is None
    assert p._grad_accumulator is None

  def test_parameter_to_dict_no_grad_fn(self) -> None:
    p = Parameter()
    d = p.to_dict()
    assert 'grad_fn' not in d
    assert '_grad_accumulator' not in d


class TestCollectInputNodes:
  def test_skips_non_datum(self) -> None:
    g = Graph()
    result = _collect_input_nodes(('hello', 42, [1, 2]), {}, g)
    assert result == []

  def test_collects_datum_with_grad_fn(self) -> None:
    g = Graph()
    d = Datum()
    node = Node(sequence_nr=0)
    object.__setattr__(d, 'grad_fn', node)
    result = _collect_input_nodes((d,), {}, g)
    assert len(result) == 1
    assert result[0][0] is node

  def test_creates_accumulate_grad_for_parameter(self) -> None:
    g = Graph()
    p = Parameter(requires_grad=True)
    result = _collect_input_nodes((p,), {}, g)
    assert len(result) == 1
    assert isinstance(result[0][0], AccumulateGrad)
    assert p._grad_accumulator is result[0][0]

  def test_caches_accumulate_grad(self) -> None:
    g = Graph()
    p = Parameter(requires_grad=True)
    r1 = _collect_input_nodes((p,), {}, g)
    r2 = _collect_input_nodes((p,), {}, g)
    assert r1[0][0] is r2[0][0]

  def test_no_accumulate_for_requires_grad_false(self) -> None:
    g = Graph()
    p = Parameter(requires_grad=False)
    result = _collect_input_nodes((p,), {}, g)
    assert result == []


class TestCreateGradientEdge:
  def test_sets_grad_fn(self) -> None:
    d = Datum()
    node = Node()
    _create_gradient_edge(d, node)
    assert d.grad_fn is node

  def test_non_datum_noop(self) -> None:
    _create_gradient_edge('not a datum', Node())


class TestFlatten:
  def test_flat_args(self) -> None:
    assert list(_flatten((1, 2, 3), {})) == [1, 2, 3]

  def test_nested_list(self) -> None:
    result = list(_flatten(([1, 2],), {}))
    assert 1 in result and 2 in result

  def test_kwargs(self) -> None:
    result = list(_flatten((), {'a': 1, 'b': 2}))
    assert 1 in result and 2 in result


class TestModuleGraphIntegration:
  def test_module_call_records_node(self) -> None:
    def run():
      class M(Module):
        def forward(self, x):
          return Datum(success=True)

      g = get_current_graph()
      g.reset()
      m = M()
      result = m(Datum())
      assert result.grad_fn is not None
      assert len(g) >= 1

    ctx = copy_context()
    ctx.run(run)

  def test_no_grad_skips_recording(self) -> None:
    def run():
      class M(Module):
        def forward(self, x):
          return Datum(success=True)

      g = get_current_graph()
      g.reset()
      m = M()
      with no_grad():
        result = m(Datum())
      assert result.grad_fn is None
      assert len(g) == 0

    ctx = copy_context()
    ctx.run(run)

  def test_nested_modules_record_multiple_nodes(self) -> None:
    def run():
      class Inner(Module):
        def forward(self, x):
          return Datum(success=True)

      class Outer(Module):
        def __init__(self):
          super().__init__()
          self.inner = Inner()

        def forward(self, x):
          return self.inner(x)

      g = get_current_graph()
      g.reset()
      m = Outer()
      m(Datum())
      assert len(g) >= 2

    ctx = copy_context()
    ctx.run(run)

  def test_train_eval_does_not_affect_graph(self) -> None:
    def run():
      class M(Module):
        def forward(self, x):
          return Datum(success=True)

      g = get_current_graph()
      g.reset()
      m = M()
      m.eval()
      result = m(Datum())
      assert result.grad_fn is not None

    ctx = copy_context()
    ctx.run(run)

  def test_hooks_then_graph(self) -> None:
    def run():
      hook_calls: list[str] = []

      class M(Module):
        def forward(self, x):
          return Datum(success=True)

      g = get_current_graph()
      g.reset()
      m = M()
      m.register_forward_pre_hook(lambda mod, args, kwargs: hook_calls.append('pre'))
      m.register_forward_hook(lambda mod, args, out: hook_calls.append('post'))
      result = m(Datum())
      assert hook_calls == ['pre', 'post']
      assert result.grad_fn is not None

    ctx = copy_context()
    ctx.run(run)
