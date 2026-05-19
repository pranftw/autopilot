"""Tests for backward hooks on Datum and Parameter (Plan 25).

Covers registration, removal, FIFO order, observation-only semantics
(return value ignored), AccumulateGrad dispatch via dispatch_backward_hooks,
and interaction with the full graph backward pass.
"""

from autopilot.core.gradient import NumericGradient
from autopilot.core.graph import RemovableHandle, get_current_graph
from autopilot.core.module.module import Module
from autopilot.core.operator import AccumulateGrad
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
import pytest


class _SimpleModule(Module):
  """Leaf module returning a fresh Datum for graph wiring."""

  def __init__(self):
    super().__init__()
    self.p = Parameter()

  def forward(self, x):
    return Datum(items=[x])


# -- Datum.register_hook tests --


def test_datum_register_hook_returns_removable_handle():
  """register_hook returns a RemovableHandle."""
  d = Datum()
  handle = d.register_hook(lambda g: None)
  assert isinstance(handle, RemovableHandle)


def test_datum_backward_hook_receives_seed_gradient():
  """Hook on a datum receives the seed gradient passed to backward()."""
  module = _SimpleModule()
  inp = Datum()
  output = module(inp)

  captured = []
  output.register_hook(captured.append)

  grad = NumericGradient(value=42.0)
  output.backward(grad)

  assert len(captured) == 1
  assert isinstance(captured[0], NumericGradient)
  assert captured[0].value == 42.0


def test_datum_backward_hook_multiple_fifo_order():
  """Multiple hooks fire in FIFO registration order."""
  module = _SimpleModule()
  inp = Datum()
  output = module(inp)

  call_order = []
  output.register_hook(lambda g: call_order.append('first'))
  output.register_hook(lambda g: call_order.append('second'))
  output.register_hook(lambda g: call_order.append('third'))

  output.backward(NumericGradient(value=1.0))
  assert call_order == ['first', 'second', 'third']


def test_datum_backward_hook_return_value_ignored():
  """Hook return values are discarded (observation-only)."""
  module = _SimpleModule()
  inp = Datum()
  output = module(inp)

  different_grad = NumericGradient(value=999.0)
  output.register_hook(lambda g: different_grad)

  seed = NumericGradient(value=1.0)
  output.backward(seed)

  assert module.p.grad is not None
  assert isinstance(module.p.grad, NumericGradient)
  assert module.p.grad.value == 1.0


def test_datum_backward_hook_removable_handle():
  """handle.remove() silences the hook."""
  module = _SimpleModule()
  inp = Datum()
  output = module(inp)

  captured = []
  handle = output.register_hook(captured.append)
  handle.remove()

  output.backward(NumericGradient(value=1.0))
  assert captured == []


def test_datum_backward_hook_exception_propagates():
  """Exceptions from hooks propagate (fail-fast)."""
  module = _SimpleModule()
  inp = Datum()
  output = module(inp)

  def bad_hook(g):
    msg = 'hook error'
    raise ValueError(msg)

  output.register_hook(bad_hook)

  with pytest.raises(ValueError, match='hook error'):
    output.backward(NumericGradient(value=1.0))


def test_datum_deepcopy_clears_hooks():
  """Deep-copying a datum clears its backward hooks."""
  d = Datum()
  d.register_hook(lambda g: None)
  assert len(d._backward_hooks) == 1

  d2 = d.clone()
  assert len(d2._backward_hooks) == 0


# -- Parameter.dispatch_backward_hooks tests --


def test_parameter_dispatch_backward_hooks_fires():
  """dispatch_backward_hooks calls registered hooks on the parameter."""
  p = Parameter()
  captured = []
  p.register_hook(captured.append)

  grad = NumericGradient(value=5.0)
  p.dispatch_backward_hooks(grad)

  assert len(captured) == 1
  assert captured[0].value == 5.0


def test_parameter_dispatch_backward_hooks_fifo():
  """Parameter hooks fire in FIFO order."""
  p = Parameter()
  order = []
  p.register_hook(lambda g: order.append('a'))
  p.register_hook(lambda g: order.append('b'))

  p.dispatch_backward_hooks(NumericGradient(value=1.0))
  assert order == ['a', 'b']


def test_parameter_dispatch_backward_hooks_removable():
  """Removing a handle silences the parameter hook."""
  p = Parameter()
  captured = []
  handle = p.register_hook(captured.append)
  handle.remove()

  p.dispatch_backward_hooks(NumericGradient(value=1.0))
  assert captured == []


# -- AccumulateGrad + backward hook integration tests --


def test_parameter_accumulate_hook_runs_before_grad_assignment():
  """Hook on parameter fires before grad is accumulated by AccumulateGrad."""
  param = Parameter()

  grad_at_hook_time = []
  param_grad_at_hook_time = []

  def observe_hook(g):
    grad_at_hook_time.append(g)
    param_grad_at_hook_time.append(param.grad)

  param.register_hook(observe_hook)

  graph = get_current_graph()
  acc = AccumulateGrad.get_or_create(param, graph)
  incoming = NumericGradient(value=7.0)
  acc(incoming)

  assert len(grad_at_hook_time) == 1
  assert grad_at_hook_time[0].value == 7.0
  assert param_grad_at_hook_time[0] is None
  assert param.grad is not None
  assert isinstance(param.grad, NumericGradient)
  assert param.grad.value == 7.0

  graph.reset()
  param.grad = None
  param.grad_accumulator = None


def test_backward_hook_return_value_ignored_via_accumulate_grad():
  """AccumulateGrad ignores hook return values; param.grad reflects accumulation only."""
  param = Parameter()
  param.register_hook(lambda g: NumericGradient(value=999.0))

  graph = get_current_graph()
  acc = AccumulateGrad.get_or_create(param, graph)
  acc(NumericGradient(value=3.0))

  assert param.grad is not None
  assert isinstance(param.grad, NumericGradient)
  assert param.grad.value == 3.0

  graph.reset()
  param.grad = None
  param.grad_accumulator = None


def test_backward_hook_full_graph_integration():
  """Hooks fire during a full forward-backward pass through Module."""
  module = _SimpleModule()
  inp = Datum()
  output = module(inp)

  datum_hook_captured = []
  output.register_hook(datum_hook_captured.append)

  param_hook_captured = []
  module.p.register_hook(param_hook_captured.append)

  seed = NumericGradient(value=10.0)
  output.backward(seed)

  assert len(datum_hook_captured) == 1
  assert datum_hook_captured[0].value == 10.0

  assert len(param_hook_captured) == 1
  assert param_hook_captured[0].value == 10.0

  assert module.p.grad is not None
  assert isinstance(module.p.grad, NumericGradient)
  assert module.p.grad.value == 10.0


def test_backward_hook_no_hooks_no_error():
  """Backward works fine when no hooks are registered."""
  module = _SimpleModule()
  inp = Datum()
  output = module(inp)

  output.backward(NumericGradient(value=1.0))
  assert module.p.grad is not None
  assert isinstance(module.p.grad, NumericGradient)
  assert module.p.grad.value == 1.0


def test_parameter_hooks_not_called_when_requires_grad_false():
  """AccumulateGrad skips hooks on frozen parameters (requires_grad=False)."""
  param = Parameter(requires_grad=False)
  captured = []
  param.register_hook(captured.append)

  graph = get_current_graph()
  acc = AccumulateGrad.get_or_create(param, graph)
  acc(NumericGradient(value=1.0))

  assert captured == []
  assert param.grad is None

  graph.reset()
  param.grad_accumulator = None


def test_datum_no_hooks_attr_after_init():
  """Every Datum gets an empty _backward_hooks OrderedDict."""
  d = Datum()
  assert hasattr(d, '_backward_hooks')
  assert len(d._backward_hooks) == 0
