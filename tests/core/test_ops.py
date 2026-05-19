"""Tests for core/ops.py: builtin operators, Sequential, and functional API."""

from autopilot.core.gradient import Gradient, NumericGradient
from autopilot.core.graph import Graph, get_current_graph, no_grad
from autopilot.core.module.module import Module
from autopilot.core.operator import AccumulateGrad, Context, Operator, OperatorNode
from autopilot.core.ops import (
  AttributionOperator,
  BroadcastOperator,
  CloneOperator,
  DetachOperator,
  IdentityOperator,
  MergeOperator,
  ReduceOperator,
  ScaleGradOperator,
  SelectOperator,
  Sequential,
  TransformGradOperator,
  attribution,
  broadcast,
  clone,
  detach,
  identity,
  merge,
  reduce,
  scale_grad,
  select,
  transform_grad,
)
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from dataclasses import dataclass
from typing import Any, cast
import copy
import pytest

# Helpers


@dataclass
class _SubDatum(Datum):
  """Datum subclass for subclass preservation tests."""

  label: str = 'sub'


class _StubModule(Module):
  """Minimal module that adds a tag item to prove it was called."""

  def __init__(self, tag):
    super().__init__()
    self.p = Parameter(requires_grad=True)
    self._tag = tag

  def forward(self, x):
    new_items = [*list(x.items), Datum(items=[Datum(items=[])])]
    return Datum(items=new_items)


class _IdentityModule(Module):
  """Module that returns its input unchanged (for Sequential tests)."""

  def __init__(self):
    super().__init__()
    self.p = Parameter(requires_grad=True)

  def forward(self, x):
    return Datum(items=list(x.items))


# MergeOperator


class TestMergeOperator:
  def test_merge_two_datums(self) -> None:
    d1 = Datum(items=[Datum()])
    d2 = Datum(items=[Datum(), Datum()])
    result = MergeOperator.apply(d1, d2)
    assert len(result.items) == 3

  def test_merge_backward_broadcasts(self) -> None:
    ctx = Context()
    ctx.save_for_backward(3)
    grad = NumericGradient(value=1.0)
    grads = MergeOperator.backward(ctx, grad)
    assert len(grads) == 3
    assert all(g is grad for g in grads)

  def test_merge_zero_inputs_raises(self) -> None:
    with pytest.raises(TypeError):
      MergeOperator.apply()

  def test_merge_single_datum(self) -> None:
    d = Datum(items=[Datum(), Datum()])
    result = MergeOperator.apply(d)
    assert len(result.items) == 2

  def test_merge_empty_items(self) -> None:
    d1 = Datum(items=[])
    d2 = Datum(items=[])
    result = MergeOperator.apply(d1, d2)
    assert result.items == []


# SelectOperator


class TestSelectOperator:
  def test_select_picks_index(self) -> None:
    d0 = Datum(items=[Datum()])
    d1 = Datum(items=[Datum(), Datum()])
    merged = MergeOperator.apply(d0, d1)
    result = SelectOperator.apply(merged, 1)
    assert isinstance(result, Datum)

  def test_select_backward_passes_grad(self) -> None:
    ctx = Context()
    ctx.save_for_backward(1)
    grad = NumericGradient(value=5.0)
    grads = SelectOperator.backward(ctx, grad)
    assert len(grads) == 1
    assert grads[0] is grad

  def test_select_out_of_range_raises(self) -> None:
    d = Datum(items=[Datum(), Datum()])
    with pytest.raises(IndexError):
      SelectOperator.apply(d, 5)

  def test_select_preserves_subclass(self) -> None:
    child = _SubDatum(label='first')
    d = Datum(items=[child])
    result = SelectOperator.apply(d, 0)
    assert isinstance(result, _SubDatum)
    assert result.label == 'first'


# CloneOperator


class TestCloneOperator:
  def test_clone_identity_backward(self) -> None:
    ctx = Context()
    ctx.save_for_backward(())
    grad = NumericGradient(value=7.0)
    grads = CloneOperator.backward(ctx, grad)
    assert grads == (grad,)

  def test_clone_fan_out_accumulates(self) -> None:
    param = Parameter(requires_grad=True)
    graph = get_current_graph()
    graph.reset()
    graph._freed = False

    class _WireOp(Operator):
      @staticmethod
      def forward(ctx, p):
        return Datum(items=[])

      @staticmethod
      def backward(ctx, *grads):
        grad = grads[0] if grads else None
        return (grad,)

    out = _WireOp.apply(param)
    c1 = CloneOperator.apply(out)
    c2 = CloneOperator.apply(out)

    assert c1.grad_fn is not None
    assert c2.grad_fn is not None

    c1.backward(NumericGradient(value=3.0))

    graph.reset()
    graph._freed = False
    param.grad = None
    param.grad_accumulator = None

    out2 = _WireOp.apply(param)
    c1b = CloneOperator.apply(out2)
    CloneOperator.apply(out2)

    c1b.backward(NumericGradient(value=2.0))
    graph2 = get_current_graph()
    graph2.reset()
    graph2._freed = False
    assert param.grad is not None
    assert param.grad.value == 2.0

  def test_clone_preserves_subclass(self) -> None:
    d = _SubDatum(label='test')
    result = CloneOperator.apply(d)
    assert isinstance(result, _SubDatum)
    assert result.label == 'test'

  def test_clone_data_independence(self) -> None:
    child = Datum(items=[])
    original = Datum(items=[child])
    result = CloneOperator.apply(original)
    result.items.append(Datum())
    assert len(original.items) == 1
    assert len(result.items) == 2


# ScaleGradOperator


class TestScaleGradOperator:
  def test_scale_grad_numeric(self) -> None:
    ctx = Context()
    ctx.save_for_backward(2.0)
    grad = NumericGradient(value=3.0)
    grads = ScaleGradOperator.backward(ctx, grad)
    assert len(grads) == 1
    assert isinstance(grads[0], NumericGradient)
    assert grads[0].value == 6.0

  def test_scale_grad_non_numeric_passthrough(self) -> None:
    ctx = Context()
    ctx.save_for_backward(2.0)

    @dataclass
    class _CustomGrad(Gradient):
      text: str = 'hello'

      def accumulate(self, other):
        return self

      def render(self):
        return self.text

    grad = _CustomGrad(text='unchanged')
    grads = ScaleGradOperator.backward(ctx, grad)
    assert grads[0] is grad

  def test_scale_grad_forward_clones(self) -> None:
    d = _SubDatum(label='scaled')
    result = ScaleGradOperator.apply(d, 3.0)
    assert isinstance(result, _SubDatum)
    assert result is not d

  def test_scale_grad_zero(self) -> None:
    ctx = Context()
    ctx.save_for_backward(0.0)
    grad = NumericGradient(value=10.0)
    grads = ScaleGradOperator.backward(ctx, grad)
    assert grads[0].value == 0.0


# TransformGradOperator


class TestTransformGradOperator:
  def test_transform_grad_applies_fn(self) -> None:
    ctx = Context()

    def fn(g):
      return NumericGradient(value=g.value * 10)

    ctx.save_for_backward(fn)
    grad = NumericGradient(value=2.0)
    grads = TransformGradOperator.backward(ctx, grad)
    assert grads[0].value == 20.0

  def test_transform_grad_none_fn_passthrough(self) -> None:
    ctx = Context()
    ctx.save_for_backward(None)
    grad = NumericGradient(value=5.0)
    grads = TransformGradOperator.backward(ctx, grad)
    assert grads[0] is grad

  def test_transform_grad_forward_clones(self) -> None:
    d = _SubDatum(label='transformed')
    result = TransformGradOperator.apply(d, lambda g: g)
    assert isinstance(result, _SubDatum)
    assert result is not d


# DetachOperator


class TestDetachOperator:
  def test_detach_blocks_gradient(self) -> None:
    ctx = Context()
    grad = NumericGradient(value=1.0)
    grads = DetachOperator.backward(ctx, grad)
    assert grads == (None,)

  def test_detach_clears_grad_fn(self) -> None:
    d = Datum()
    sentinel = OperatorNode(operator_cls=type(None), ctx=Context())
    d.grad_fn = sentinel
    result = DetachOperator.apply(d)
    assert result.grad_fn is None

  def test_detach_preserves_subclass(self) -> None:
    d = _SubDatum(label='detached')
    d.grad_fn = OperatorNode(type(None), Context())
    result = DetachOperator.apply(d)
    assert isinstance(result, _SubDatum)
    assert result.label == 'detached'
    assert result.grad_fn is None


# IdentityOperator


class TestIdentityOperator:
  def test_identity_passthrough(self) -> None:
    d = _SubDatum(label='ident')
    result = IdentityOperator.apply(d)
    assert isinstance(result, _SubDatum)
    assert result.label == 'ident'
    assert result is not d

  def test_identity_backward_passthrough(self) -> None:
    ctx = Context()
    grad = NumericGradient(value=3.0)
    grads = IdentityOperator.backward(ctx, grad)
    assert grads == (grad,)


# BroadcastOperator


class TestBroadcastOperator:
  def test_broadcast_creates_n_clones(self) -> None:
    d = Datum(items=[Datum()])
    result = BroadcastOperator.apply(d, 3)
    assert len(result.items) == 3
    for item in result.items:
      assert len(item.items) == 1
      assert item is not d

  def test_broadcast_backward_passes_grad_none_for_n(self) -> None:
    ctx = Context()
    ctx.save_for_backward(3)
    grad = NumericGradient(value=1.0)
    grads = BroadcastOperator.backward(ctx, grad)
    assert grads == (grad, None)

  def test_broadcast_n_zero_empty_items(self) -> None:
    d = Datum(items=[Datum()])
    result = BroadcastOperator.apply(d, 0)
    assert result.items == []

  def test_broadcast_n_one(self) -> None:
    d = Datum(items=[Datum()])
    result = BroadcastOperator.apply(d, 1)
    assert len(result.items) == 1
    assert result.items[0] is not d
    assert len(result.items[0].items) == 1

  def test_broadcast_clones_are_independent(self) -> None:
    d = Datum(items=[Datum()])
    result = BroadcastOperator.apply(d, 2)
    result.items[0].items.append(Datum())
    assert len(result.items[1].items) == 1


# ReduceOperator


class TestReduceOperator:
  def test_reduce_concatenates_items(self) -> None:
    d1 = Datum(items=[Datum()])
    d2 = Datum(items=[Datum(), Datum()])
    result = ReduceOperator.apply(d1, d2)
    assert len(result.items) == 3

  def test_reduce_backward_broadcasts_to_all_inputs(self) -> None:
    ctx = Context()
    ctx.save_for_backward(2)
    grad = NumericGradient(value=1.0)
    grads = ReduceOperator.backward(ctx, grad)
    assert len(grads) == 2
    assert all(g is grad for g in grads)

  def test_reduce_is_merge_subclass(self) -> None:
    assert issubclass(ReduceOperator, MergeOperator)


# AttributionOperator


class TestAttributionOperator:
  def test_attribution_forward_attaches_label(self) -> None:
    d = Datum()
    ctx = Context()
    result = AttributionOperator.forward(ctx, d, module_name='encoder')
    assert isinstance(result, Datum)
    assert result is not d
    assert ctx.saved == ('encoder',)

  def test_attribution_backward_passthrough(self) -> None:
    ctx = Context()
    ctx.save_for_backward('encoder')
    grad = NumericGradient(value=4.0)
    grads = AttributionOperator.backward(ctx, grad)
    assert grads == (grad,)

  def test_attribution_preserves_subclass(self) -> None:
    d = _SubDatum(label='attributed')
    result = AttributionOperator.apply(d, module_name='dec')
    assert isinstance(result, _SubDatum)
    assert result.label == 'attributed'


# Sequential


class TestSequential:
  def test_sequential_chains_modules(self) -> None:
    m0 = _StubModule('a')
    m1 = _StubModule('b')
    m2 = _StubModule('c')
    seq = Sequential(m0, m1, m2)
    inp = Datum(items=[])
    out = seq(inp)
    assert len(out.items) == 3

  def test_sequential_gradient_flows(self) -> None:
    m0 = _IdentityModule()
    m1 = _IdentityModule()
    m2 = _IdentityModule()
    seq = Sequential(m0, m1, m2)

    graph = get_current_graph()
    graph.reset()
    graph._freed = False

    inp = Datum(items=[Datum()])
    out = seq(inp)
    assert out.grad_fn is not None

    grad = NumericGradient(value=1.0)
    out.backward(grad)

    for mod in [m0, m1, m2]:
      assert mod.p.grad is not None

  def test_sequential_no_direct_params(self) -> None:
    m0 = _StubModule('a')
    m1 = _StubModule('b')
    seq = Sequential(m0, m1)
    direct_params = list(seq._parameters.values())
    assert direct_params == []

  def test_sequential_repr(self) -> None:
    m0 = _StubModule('a')
    m1 = _StubModule('b')
    seq = Sequential(m0, m1)
    r = repr(seq)
    assert 'Sequential' in r
    assert 'module_0' in r
    assert 'module_1' in r

  def test_sequential_children(self) -> None:
    m0 = _StubModule('a')
    m1 = _StubModule('b')
    seq = Sequential(m0, m1)
    children = list(seq.children())
    assert len(children) == 2
    assert children[0] is m0
    assert children[1] is m1

  def test_sequential_parameters_recursive(self) -> None:
    m0 = _StubModule('a')
    m1 = _StubModule('b')
    seq = Sequential(m0, m1)
    params = list(seq.parameters())
    assert len(params) == 2
    assert m0.p in params
    assert m1.p in params


# CloneOperator graph-aware cloning


class TestCloneOperatorGraphAware:
  def test_clone_operator_creates_grad_fn(self) -> None:
    graph = get_current_graph()
    graph.reset()
    graph._freed = False

    d = Datum()
    sentinel = OperatorNode(operator_cls=type(None), ctx=Context(), sequence_nr=0)
    graph.add_node(sentinel)
    d.grad_fn = sentinel

    result = CloneOperator.apply(d)
    assert result.grad_fn is not None
    assert result.grad_fn.name() == 'CloneOperator'

  def test_datum_clone_always_deepcopy(self) -> None:
    d = Datum(items=[Datum()])
    sentinel = OperatorNode(operator_cls=type(None), ctx=Context())
    d.grad_fn = sentinel
    cloned = d.clone()
    assert cloned.grad_fn is None
    assert len(cloned.items) == 1

  def test_datum_clone_no_grad_context(self) -> None:
    d = Datum(items=[Datum()])
    with no_grad():
      result = CloneOperator.apply(d)
    assert result.grad_fn is None


# Functional API


class TestFunctionalMerge:
  def test_merge_delegates(self) -> None:
    d1 = Datum(items=[Datum()])
    d2 = Datum(items=[Datum()])
    result = merge(d1, d2)
    assert len(result.items) == 2


class TestFunctionalSelect:
  def test_select_delegates(self) -> None:
    d = Datum(items=[Datum(), Datum(items=[Datum(), Datum()])])
    result = select(d, 0)
    assert isinstance(result, Datum)


class TestFunctionalClone:
  def test_clone_delegates(self) -> None:
    d = _SubDatum(label='fn')
    result = clone(d)
    assert isinstance(result, _SubDatum)
    assert result.label == 'fn'
    assert result is not d


class TestFunctionalIdentity:
  def test_identity_delegates(self) -> None:
    d = _SubDatum(label='fn_ident')
    result = identity(d)
    assert isinstance(result, _SubDatum)
    assert result is not d


class TestFunctionalDetach:
  def test_detach_delegates(self) -> None:
    d = Datum()
    sentinel = OperatorNode(operator_cls=type(None), ctx=Context())
    d.grad_fn = sentinel
    result = detach(d)
    assert result.grad_fn is None


class TestFunctionalBroadcast:
  def test_broadcast_delegates(self) -> None:
    d = Datum(items=[Datum()])
    result = broadcast(d, 2)
    assert len(result.items) == 2


class TestFunctionalReduce:
  def test_reduce_delegates(self) -> None:
    d1 = Datum(items=[Datum()])
    d2 = Datum(items=[Datum()])
    result = reduce(d1, d2)
    assert len(result.items) == 2


class TestFunctionalScaleGrad:
  def test_scale_grad_delegates(self) -> None:
    d = Datum()
    result = scale_grad(d, 2.0)
    assert isinstance(result, Datum)
    assert result is not d


class TestFunctionalTransformGrad:
  def test_transform_grad_delegates(self) -> None:
    d = Datum()
    result = transform_grad(d, lambda g: g)
    assert isinstance(result, Datum)
    assert result is not d


class TestFunctionalAttribution:
  def test_attribution_delegates(self) -> None:
    d = Datum()
    result = attribution(d, module_name='enc')
    assert isinstance(result, Datum)
    assert result is not d


# Wired-parameter operator tests (BUG-001 regression)


class TestWiredParameterOperators:
  def test_identity_operator_on_wired_parameter(self) -> None:
    p = Parameter(requires_grad=True)
    g = Graph()
    AccumulateGrad.get_or_create(p, g)
    out = IdentityOperator.apply(p)
    assert isinstance(out, Datum)

  def test_clone_operator_on_wired_parameter(self) -> None:
    p = Parameter(requires_grad=True)
    g = Graph()
    AccumulateGrad.get_or_create(p, g)
    out = CloneOperator.apply(p)
    assert isinstance(out, Datum)

  def test_select_operator_on_wired_parameter(self) -> None:
    p = Parameter(requires_grad=True)
    g = Graph()
    AccumulateGrad.get_or_create(p, g)
    d = Datum(items=[p])
    out = SelectOperator.apply(d, 0)
    assert isinstance(out, Datum)

  def test_parameter_deepcopy_with_grad_and_accumulator(self) -> None:
    p = Parameter(requires_grad=True)
    g = Graph()
    AccumulateGrad.get_or_create(p, g)
    p.grad = NumericGradient(value=1.0)
    c = copy.deepcopy(p)
    assert c.grad is None
    assert c.grad_accumulator is None


class TestMergeReturnsDatum:
  def test_merge_returns_datum(self) -> None:
    d1 = Datum(items=cast(Any, ['a']))
    d2 = Datum(items=cast(Any, ['b']))
    out = merge(d1, d2)
    assert isinstance(out, Datum)
    assert len(out.items) == 2

  def test_merge_single_datum(self) -> None:
    d = Datum(items=cast(Any, ['x', 'y']))
    out = merge(d)
    assert isinstance(out, Datum)
    assert len(out.items) == 2

  def test_merge_empty_items(self) -> None:
    d1 = Datum(items=[])
    d2 = Datum(items=cast(Any, ['a']))
    out = merge(d1, d2)
    assert isinstance(out, Datum)
    assert len(out.items) == 1
