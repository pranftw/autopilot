"""Tests for core/operator.py: Context, Operator, OperatorNode, AccumulateGrad,
flatten_args, collect_input_nodes, AccumulateGrad.get_or_create.
"""

from autopilot.core.gradient import NumericGradient
from autopilot.core.graph import Graph
from autopilot.core.operator import (
  AccumulateGrad,
  Context,
  Operator,
  OperatorNode,
  collect_input_nodes,
  flatten_args,
)
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
import copy
import pytest


class _AddOp(Operator):
  @staticmethod
  def forward(ctx, a, b):
    ctx.save_for_backward(a, b)
    return a

  @staticmethod
  def backward(ctx, *grads):
    grad = grads[0] if grads else None
    return (grad, grad)


class _SingleReturnOp(Operator):
  @staticmethod
  def forward(ctx, x):
    return x

  @staticmethod
  def backward(ctx, *grads):
    return grads[0] if grads else None


class TestContext:
  def test_save_for_backward(self) -> None:
    ctx = Context()
    ctx.save_for_backward(1, 2, 3)
    assert ctx.saved == (1, 2, 3)

  def test_saved_empty_default(self) -> None:
    ctx = Context()
    assert ctx.saved == ()

  def test_needs_input_grad_default(self) -> None:
    ctx = Context()
    assert ctx.needs_input_grad == ()

  def test_custom_metadata_setattr(self) -> None:
    ctx = Context()
    ctx.my_val = 42
    assert ctx.my_val == 42

  def test_custom_metadata_getattr_missing(self) -> None:
    ctx = Context()
    with pytest.raises(AttributeError, match='Context has no attribute'):
      _ = ctx.nonexistent

  def test_private_attrs_bypass_metadata(self) -> None:
    ctx = Context()
    ctx._foo = 'bar'
    assert ctx._foo == 'bar'
    assert '_foo' not in ctx._metadata
    assert 'foo' not in ctx._metadata

  def test_multiple_save_for_backward_overwrites(self) -> None:
    ctx = Context()
    ctx.save_for_backward('a', 'b')
    ctx.save_for_backward('x', 'y', 'z')
    assert ctx.saved == ('x', 'y', 'z')

  def test_context_deepcopy(self) -> None:
    ctx = Context()
    c2 = copy.deepcopy(ctx)
    assert isinstance(c2, Context)
    assert c2 is not ctx
    assert c2.saved == ()
    assert c2._metadata == {}

  def test_context_deepcopy_with_metadata(self) -> None:
    ctx = Context()
    ctx.note = 'x'
    ctx.save_for_backward(1, 2)
    c2 = copy.deepcopy(ctx)
    assert c2.saved == (1, 2)
    assert c2.note == 'x'

  def test_context_deepcopy_with_needs_input_grad(self) -> None:
    ctx = Context()
    ctx._needs_input_grad = (True, False)
    c2 = copy.deepcopy(ctx)
    assert c2.needs_input_grad == (True, False)

  def test_context_getattr_missing_metadata(self) -> None:
    shell = object.__new__(Context)
    with pytest.raises(AttributeError, match='Context has no attribute'):
      _ = shell.anything

  def test_context_deepcopy_complex_saved(self) -> None:
    ctx = Context()
    ctx.save_for_backward({'key': [1, 2]}, [Datum()])
    c2 = copy.deepcopy(ctx)
    assert c2.saved[0] == {'key': [1, 2]}
    assert isinstance(c2.saved[1][0], Datum)
    c2.saved[0]['key'].append(3)
    assert ctx.saved[0]['key'] == [1, 2]


class TestOperatorNode:
  def test_operator_node_creation(self) -> None:
    ctx = Context()
    nf = ((None, 0),)
    node = OperatorNode(
      operator_cls=_AddOp,
      ctx=ctx,
      next_functions=nf,
      sequence_nr=7,
    )
    assert node._operator_cls is _AddOp
    assert node._ctx is ctx
    assert node.next_functions == nf
    assert node.sequence_nr == 7

  def test_operator_node_name(self) -> None:
    node = OperatorNode(operator_cls=_AddOp, ctx=Context())
    assert node.name() == '_AddOp'

  def test_operator_node_call_dispatches_backward(self) -> None:
    ctx = Context()
    ctx.save_for_backward(10, 20)
    node = OperatorNode(operator_cls=_AddOp, ctx=ctx)
    grad = NumericGradient(value=1.0)
    result = node(grad)
    assert isinstance(result, tuple)
    assert result == (grad, grad)

  def test_operator_node_wraps_non_tuple_in_tuple(self) -> None:
    ctx = Context()
    node = OperatorNode(operator_cls=_SingleReturnOp, ctx=ctx)
    grad = NumericGradient(value=5.0)
    result = node(grad)
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0] is grad

  def test_operator_node_repr(self) -> None:
    node = OperatorNode(operator_cls=_AddOp, ctx=Context(), sequence_nr=42)
    r = repr(node)
    assert '_AddOp' in r
    assert '42' in r


class TestAccumulateGrad:
  def test_accumulate_sets_grad(self) -> None:
    param = Parameter(requires_grad=True)
    node = AccumulateGrad(param, sequence_nr=0)
    grad = NumericGradient(value=3.0)
    node(grad)
    assert param.grad is grad

  def test_accumulate_adds_to_existing(self) -> None:
    param = Parameter(requires_grad=True)
    node = AccumulateGrad(param, sequence_nr=0)
    node(NumericGradient(value=2.0))
    node(NumericGradient(value=5.0))
    grad = param.grad
    assert grad is not None
    assert isinstance(grad, NumericGradient)
    assert grad.value == 7.0

  def test_accumulate_none_noop(self) -> None:
    param = Parameter(requires_grad=True)
    node = AccumulateGrad(param, sequence_nr=0)
    node(None)
    assert param.grad is None

  def test_accumulate_none_preserves_existing(self) -> None:
    param = Parameter(requires_grad=True)
    node = AccumulateGrad(param, sequence_nr=0)
    param.grad = NumericGradient(value=1.0)
    node(None)
    grad = param.grad
    assert grad is not None
    assert isinstance(grad, NumericGradient)
    assert grad.value == 1.0

  def test_accumulate_respects_requires_grad(self) -> None:
    param = Parameter(requires_grad=False)
    node = AccumulateGrad(param, sequence_nr=0)
    node(NumericGradient(value=10.0))
    assert param.grad is None

  def test_accumulate_name(self) -> None:
    param = Parameter()
    node = AccumulateGrad(param, sequence_nr=0)
    assert node.name() == 'AccumulateGrad'

  def test_accumulate_multiple_calls(self) -> None:
    param = Parameter(requires_grad=True)
    node = AccumulateGrad(param, sequence_nr=0)
    node(NumericGradient(value=1.0))
    node(NumericGradient(value=2.0))
    node(NumericGradient(value=3.0))
    grad = param.grad
    assert grad is not None
    assert isinstance(grad, NumericGradient)
    assert grad.value == 6.0

  def test_accumulate_zero_grad_then_reaccumulate(self) -> None:
    param = Parameter(requires_grad=True)
    node = AccumulateGrad(param, sequence_nr=0)
    node(NumericGradient(value=5.0))
    grad = param.grad
    assert grad is not None
    assert isinstance(grad, NumericGradient)
    assert grad.value == 5.0
    param.grad = None
    node(NumericGradient(value=2.0))
    grad2 = param.grad
    assert grad2 is not None
    assert isinstance(grad2, NumericGradient)
    assert grad2.value == 2.0

  def test_accumulate_empty_grads(self) -> None:
    param = Parameter(requires_grad=True)
    node = AccumulateGrad(param, sequence_nr=0)
    result = node()
    assert result == ()
    assert param.grad is None

  def test_accumulate_returns_empty_tuple(self) -> None:
    param = Parameter(requires_grad=True)
    node = AccumulateGrad(param, sequence_nr=0)
    result = node(NumericGradient(value=1.0))
    assert result == ()


class TestOperatorBase:
  def test_forward_not_implemented(self) -> None:
    with pytest.raises(NotImplementedError):
      Operator.forward(Context())

  def test_backward_not_implemented(self) -> None:
    with pytest.raises(NotImplementedError):
      Operator.backward(Context())


class TestFlattenArgs:
  def test_flat_args(self) -> None:
    assert flatten_args((1, 2, 3), {}) == [1, 2, 3]

  def test_nested_list(self) -> None:
    assert flatten_args(([1, 2],), {}) == [1, 2]

  def test_kwargs(self) -> None:
    assert flatten_args((), {'a': 1, 'b': 2}) == [1, 2]

  def test_mixed_nesting(self) -> None:
    assert flatten_args(((1, [2, 3]),), {'k': 4}) == [1, 2, 3, 4]

  def test_empty(self) -> None:
    assert flatten_args((), {}) == []

  def test_nested_dict(self) -> None:
    assert flatten_args(({'x': 10},), {}) == [10]

  def test_deeply_nested(self) -> None:
    assert flatten_args(([[[1]]],), {}) == [1]


class TestCollectInputNodes:
  def test_skips_non_datum(self) -> None:
    graph = Graph()
    nodes = collect_input_nodes(('hello', 42, [1, 2]), {}, graph)
    assert nodes == []

  def test_collects_datum_with_grad_fn(self) -> None:
    graph = Graph()
    d = Datum()
    sentinel = OperatorNode(operator_cls=_AddOp, ctx=Context(), sequence_nr=5)
    d.grad_fn = sentinel
    nodes = collect_input_nodes((d,), {}, graph)
    assert len(nodes) == 1
    assert nodes[0] == (sentinel, 0)

  def test_creates_accumulate_for_parameter(self) -> None:
    graph = Graph()
    param = Parameter(requires_grad=True)
    nodes = collect_input_nodes((param,), {}, graph)
    assert len(nodes) == 1
    node, idx = nodes[0]
    assert isinstance(node, AccumulateGrad)
    assert idx == 0

  def test_caches_accumulate(self) -> None:
    graph = Graph()
    param = Parameter(requires_grad=True)
    nodes1 = collect_input_nodes((param,), {}, graph)
    nodes2 = collect_input_nodes((param,), {}, graph)
    assert nodes1[0][0] is nodes2[0][0]

  def test_deduplicates_same_datum(self) -> None:
    graph = Graph()
    d = Datum()
    sentinel = OperatorNode(operator_cls=_AddOp, ctx=Context(), sequence_nr=1)
    d.grad_fn = sentinel
    nodes = collect_input_nodes((d, d), {}, graph)
    assert len(nodes) == 1

  def test_datum_without_grad_fn_skipped(self) -> None:
    graph = Graph()
    d = Datum()
    nodes = collect_input_nodes((d,), {}, graph)
    assert nodes == []

  def test_mixed_datum_and_non_datum(self) -> None:
    graph = Graph()
    param = Parameter(requires_grad=True)
    d = Datum()
    nodes = collect_input_nodes(('text', param, 42, d), {}, graph)
    assert len(nodes) == 1
    assert isinstance(nodes[0][0], AccumulateGrad)


class TestAccumulateGradGetOrCreate:
  def test_get_or_create_creates(self) -> None:
    graph = Graph()
    param = Parameter(requires_grad=True)
    node = AccumulateGrad.get_or_create(param, graph)
    assert isinstance(node, AccumulateGrad)
    assert param.grad_accumulator is node
    assert len(graph._nodes) == 1

  def test_get_or_create_caches(self) -> None:
    graph = Graph()
    param = Parameter(requires_grad=True)
    node1 = AccumulateGrad.get_or_create(param, graph)
    node2 = AccumulateGrad.get_or_create(param, graph)
    assert node1 is node2
    assert len(graph._nodes) == 1

  def test_get_or_create_frozen_param(self) -> None:
    graph = Graph()
    param = Parameter(requires_grad=False)
    node = AccumulateGrad.get_or_create(param, graph)
    node(NumericGradient(value=10.0))
    assert param.grad is None
