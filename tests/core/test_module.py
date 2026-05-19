"""Tests for Module base class (nn.Module pattern) and ModuleCallOperator."""

from autopilot.core.gradient import NumericGradient
from autopilot.core.graph import get_current_graph, no_grad
from autopilot.core.metric import Metric
from autopilot.core.module.module import Module
from autopilot.core.operator import Context
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from contextvars import copy_context
from typing import Any, cast
import pytest


class _ChildModule(Module):
  def forward(self, batch: object) -> Datum:
    return Datum()


class _ConcreteModule(Module):
  def forward(self, *args, **kwargs) -> Datum:
    return Datum()


class _StubMetric(Metric):
  def update(self, datum: Datum) -> None:
    pass

  def compute(self) -> dict[str, float]:
    return {}


class TestModuleForward:
  def test_forward_raises_not_implemented(self) -> None:
    mod = Module()
    with pytest.raises(NotImplementedError):
      mod.forward()

  def test_subclass_forward_returns_datum(self) -> None:
    mod = _ConcreteModule()
    result = mod.forward()
    assert isinstance(result, Datum)

  def test_call_delegates_to_forward(self) -> None:
    mod = _ConcreteModule()
    assert mod() == mod.forward()

  def test_subclass_with_constructor_kwargs(self) -> None:
    class Configured(Module):
      def __init__(self, host: str) -> None:
        super().__init__()
        self.host = host

      def forward(self, *args, **kwargs) -> Datum:
        return Datum()

    mod = Configured(host='myhost')
    result = mod.forward()
    assert isinstance(result, Datum)
    assert mod.host == 'myhost'


class TestModuleSetattrRegistration:
  def test_setattr_registers_child_module(self) -> None:
    mod = Module()
    child = _ChildModule()
    mod.backend = child
    assert mod._modules['backend'] is child
    assert mod.backend is child

  def test_setattr_registers_metric_as_module(self) -> None:
    mod = Module()
    m = _StubMetric()
    mod.my_metric = m
    assert mod._modules['my_metric'] is m
    assert mod.my_metric is m

  def test_setattr_registers_parameter(self) -> None:
    mod = Module()
    p = Parameter()
    mod.weight = p
    assert mod._parameters['weight'] is p
    assert mod.weight is p

  def test_setattr_stores_regular_attributes(self) -> None:
    mod = Module()
    mod.count = 7
    mod.label = 'x'
    assert mod.count == 7
    assert mod.label == 'x'
    assert 'count' not in mod._modules
    assert 'count' not in mod._parameters

  def test_non_module_attribute_not_in_modules(self) -> None:
    mod = Module()
    mod.name = 'test'
    assert len(mod._modules) == 0
    assert mod.name == 'test'

  def test_pre_init_assignment_raises(self) -> None:
    class BadModule(Module):
      def __init__(self) -> None:
        self.x = 5

    with pytest.raises(AttributeError, match=r'cannot assign before Module.__init__'):
      BadModule()


class TestModuleChildren:
  def test_children_returns_registered_modules(self) -> None:
    mod = Module()
    a = _ChildModule()
    b = _ChildModule()
    mod.deploy = a
    mod.validate = b
    children = list(mod.children())
    assert len(children) == 2
    assert a in children
    assert b in children

  def test_children_yields_only_immediate(self) -> None:
    parent = Module()
    child = Module()
    grandchild = _ChildModule()
    child.gc = grandchild
    parent.c = child
    children = list(parent.children())
    assert children == [child]
    assert grandchild not in children

  def test_children_count(self) -> None:
    mod = Module()
    assert len(list(mod.children())) == 0
    mod.a = _ChildModule()
    assert len(list(mod.children())) == 1

  def test_named_children_returns_name_pairs(self) -> None:
    mod = Module()
    a = _ChildModule()
    b = _ChildModule()
    mod.deploy = a
    mod.validate = b
    named = dict(mod.named_children())
    assert named == {'deploy': a, 'validate': b}

  def test_child_in_modules_dict(self) -> None:
    mod = Module()
    mod.x = _ChildModule()
    assert 'x' in mod._modules
    assert 'y' not in mod._modules


class TestModuleTree:
  def test_modules_yields_recursive_with_self(self) -> None:
    parent = Module()
    child = _ChildModule()
    parent.c = child
    all_mods = list(parent.modules())
    assert all_mods == [parent, child]

  def test_named_modules_dotted_prefix(self) -> None:
    parent = Module()
    child = Module()
    grandchild = _ChildModule()
    child.gc = grandchild
    parent.c = child
    named = dict(parent.named_modules())
    assert '' in named
    assert 'c' in named
    assert 'c.gc' in named
    assert named[''] is parent
    assert named['c'] is child
    assert named['c.gc'] is grandchild

  def test_deeply_nested_tree(self) -> None:
    root = Module()
    level1 = Module()
    level2 = Module()
    level3 = _ChildModule()
    level2.leaf = level3
    level1.mid = level2
    root.top = level1
    all_mods = list(root.modules())
    assert len(all_mods) == 4
    named = dict(root.named_modules())
    assert 'top.mid.leaf' in named


class TestModuleParameters:
  def test_parameters_recurse_true(self) -> None:
    parent = Module()
    child = Module()
    p1 = Parameter()
    p2 = Parameter()
    parent.p1 = p1
    child.p2 = p2
    parent.c = child
    params = list(parent.parameters(recurse=True))
    assert p1 in params
    assert p2 in params

  def test_parameters_recurse_false(self) -> None:
    parent = Module()
    child = Module()
    p1 = Parameter()
    p2 = Parameter()
    parent.p1 = p1
    child.p2 = p2
    parent.c = child
    params = list(parent.parameters(recurse=False))
    assert params == [p1]

  def test_named_parameters_dotted_prefix(self) -> None:
    parent = Module()
    child = Module()
    p = Parameter()
    child.weight = p
    parent.layer = child
    named = dict(parent.named_parameters())
    assert 'layer.weight' in named
    assert named['layer.weight'] is p

  def test_mixed_children_and_parameters(self) -> None:
    mod = Module()
    mod.child = _ChildModule()
    mod.param = Parameter()
    assert len(mod._modules) == 1
    assert len(mod._parameters) == 1
    assert mod._modules['child'] is mod.child
    assert mod._parameters['param'] is mod.param

  def test_register_parameter_via_setattr(self) -> None:
    mod = Module()
    p = Parameter(requires_grad=True)
    mod.w = p
    params = list(mod.parameters())
    assert p in params

  def test_parameter_not_in_modules(self) -> None:
    mod = Module()
    mod.p = Parameter()
    assert 'p' not in mod._modules


class TestModuleTrainEval:
  def test_train_propagates_to_children(self) -> None:
    parent = Module()
    child = _ChildModule()
    parent.c = child
    parent.eval()
    assert parent.training is False
    assert child.training is False
    parent.train()
    assert parent.training is True
    assert child.training is True

  def test_eval_sets_training_false(self) -> None:
    mod = Module()
    assert mod.training is True
    mod.eval()
    assert mod.training is False

  def test_train_returns_self(self) -> None:
    mod = Module()
    assert mod.train() is mod
    assert mod.eval() is mod


class TestModuleApply:
  def test_apply_post_order_traversal(self) -> None:
    visited: list[str] = []

    parent = Module()
    child = _ChildModule()
    parent.c = child

    def record(m: Module) -> None:
      visited.append(type(m).__name__)

    parent.apply(record)
    assert visited == ['_ChildModule', 'Module']


class TestModuleStateDict:
  def test_state_dict_empty_default(self) -> None:
    assert Module().state_dict() == {}

  def test_state_dict_recursive(self) -> None:
    parent = Module()
    child = Module()
    p = Parameter(requires_grad=True)
    child.w = p
    parent.layer = child
    state = parent.state_dict()
    assert 'layer.w' in state

  def test_load_state_dict_recursive(self) -> None:
    parent = Module()
    child = Module()
    p = Parameter(requires_grad=True)
    child.w = p
    parent.layer = child

    state = parent.state_dict()
    p.grad = NumericGradient(value=1.0)

    parent2 = Module()
    child2 = Module()
    p2 = Parameter(requires_grad=False)
    child2.w = p2
    parent2.layer = child2

    parent2.load_state_dict(state)
    assert p2.requires_grad is True


class TestModuleRepr:
  def test_module_repr_shows_classname(self) -> None:
    mod = Module()
    assert repr(mod) == 'Module()'

  def test_subclass_repr(self) -> None:
    mod = _ConcreteModule()
    assert repr(mod).startswith('_ConcreteModule')

  def test_repr_tree_includes_child_names(self) -> None:
    mod = Module()
    mod.deploy = _ChildModule()
    mod.validate = _ChildModule()
    r = repr(mod)
    assert 'deploy' in r
    assert 'validate' in r
    assert '_ChildModule' in r

  def test_repr_tree_structure(self) -> None:
    mod = Module()
    mod.a = _ChildModule()
    r = repr(mod)
    assert '(a):' in r

  def test_module_subclass_with_children(self) -> None:
    class MyModule(Module):
      def __init__(self) -> None:
        super().__init__()
        self.deploy = _ChildModule()
        self.validate = _ChildModule()

      def forward(self, *args, **kwargs) -> Datum:
        return Datum()

    mod = MyModule()
    assert len(list(mod.children())) == 2
    assert 'deploy' in repr(mod)
    assert 'validate' in repr(mod)


class TestModuleMetricAccess:
  def test_metric_registration_and_access(self) -> None:
    mod = Module()
    m = _StubMetric()
    mod.accuracy = m
    assert mod._modules['accuracy'] is m
    assert mod.accuracy is m


class TestModuleCompetingStore:
  def test_reassign_module_to_parameter(self) -> None:
    mod = Module()
    child = _ChildModule()
    mod.x = child
    assert 'x' in mod._modules
    p = Parameter()
    mod.x = p
    assert 'x' not in mod._modules
    assert 'x' in mod._parameters

  def test_reassign_parameter_to_module(self) -> None:
    mod = Module()
    mod.x = Parameter()
    assert 'x' in mod._parameters
    mod.x = _ChildModule()
    assert 'x' not in mod._parameters
    assert 'x' in mod._modules

  def test_reassign_metric_to_parameter(self) -> None:
    mod = Module()
    m = _StubMetric()
    mod.x = m
    assert 'x' in mod._modules
    p = Parameter()
    mod.x = p
    assert 'x' in mod._parameters
    assert 'x' not in mod._modules

  def test_reassign_parameter_to_metric(self) -> None:
    mod = Module()
    p = Parameter()
    mod.x = p
    assert 'x' in mod._parameters
    m = _StubMetric()
    mod.x = m
    assert 'x' in mod._modules
    assert 'x' not in mod._parameters

  def test_reassign_to_regular_cleans_all(self) -> None:
    mod = Module()
    mod.x = Parameter()
    mod.x = 42
    assert 'x' not in mod._parameters
    assert 'x' not in mod._modules
    assert mod.x == 42


class TestModuleReservedNames:
  def test_child_named_eval_raises_valueerror(self) -> None:
    mod = Module()
    with pytest.raises(ValueError, match='reserved name'):
      mod.eval = cast(Any, _ChildModule())

  def test_child_named_train_raises_valueerror(self) -> None:
    mod = Module()
    with pytest.raises(ValueError, match='reserved name'):
      mod.train = cast(Any, _ChildModule())

  def test_parameter_named_parameters_raises_valueerror(self) -> None:
    mod = Module()
    with pytest.raises(ValueError, match='reserved name'):
      mod.parameters = cast(Any, Parameter())

  def test_child_named_state_dict_raises_valueerror(self) -> None:
    mod = Module()
    with pytest.raises(ValueError, match='reserved name'):
      mod.state_dict = cast(Any, _ChildModule())

  def test_all_reserved_names_rejected_for_module(self) -> None:
    reserved = {
      'eval',
      'train',
      'parameters',
      'named_parameters',
      'modules',
      'named_modules',
      'children',
      'named_children',
      'state_dict',
      'load_state_dict',
      'to_dict',
      'from_dict',
    }
    mod = Module()
    for name in reserved:
      with pytest.raises(ValueError, match='reserved name'):
        setattr(mod, name, _ChildModule())

  def test_all_reserved_names_rejected_for_parameter(self) -> None:
    reserved = {
      'eval',
      'train',
      'parameters',
      'named_parameters',
      'modules',
      'named_modules',
      'children',
      'named_children',
      'state_dict',
      'load_state_dict',
      'to_dict',
      'from_dict',
    }
    mod = Module()
    for name in reserved:
      with pytest.raises(ValueError, match='reserved name'):
        setattr(mod, name, Parameter())

  def test_plain_attribute_with_reserved_name_works(self) -> None:
    mod = Module()
    mod.eval = cast(Any, 'some_string')
    assert mod.eval == 'some_string'
    mod.train = cast(Any, 42)
    assert mod.train == 42


class TestModuleGetattr:
  def test_getattr_finds_parameter(self) -> None:
    mod = Module()
    p = Parameter()
    mod._parameters['hidden'] = p
    assert mod.hidden is p

  def test_getattr_finds_module(self) -> None:
    mod = Module()
    child = _ChildModule()
    mod._modules['hidden'] = child
    assert mod.hidden is child

  def test_getattr_raises_for_missing(self) -> None:
    mod = Module()
    with pytest.raises(AttributeError, match='nonexistent'):
      _ = mod.nonexistent


class TestModuleHooks:
  def test_forward_pre_hook_called(self) -> None:
    calls: list[str] = []

    class M(Module):
      def forward(self, x):
        return Datum()

    m = M()
    m.register_forward_pre_hook(lambda mod, args, kwargs: calls.append('pre'))
    with no_grad():
      m(Datum())
    assert 'pre' in calls

  def test_forward_hook_called(self) -> None:
    calls: list[str] = []

    class M(Module):
      def forward(self, x):
        return Datum()

    m = M()
    m.register_forward_hook(lambda mod, args, out: calls.append('post'))
    with no_grad():
      m(Datum())
    assert 'post' in calls

  def test_hook_removal(self) -> None:
    calls: list[str] = []

    class M(Module):
      def forward(self, x):
        return Datum()

    m = M()
    handle = m.register_forward_hook(lambda mod, args, out: calls.append('hook'))
    with no_grad():
      m(Datum())
    assert len(calls) == 1
    handle.remove()
    with no_grad():
      m(Datum())
    assert len(calls) == 1


class TestModuleExtraRepr:
  def test_extra_repr_default_empty(self) -> None:
    assert not Module().extra_repr()

  def test_extra_repr_in_repr(self) -> None:
    class Custom(Module):
      def forward(self):
        return Datum()

      def extra_repr(self):
        return 'host=example.com'

    r = repr(Custom())
    assert 'host=example.com' in r


# Helper modules for ModuleCallOperator tests


class _LeafModule(Module):
  def __init__(self):
    super().__init__()
    self.weight = Parameter(requires_grad=True)

  def forward(self, x):
    return Datum()


class _TwoParamLeaf(Module):
  def __init__(self):
    super().__init__()
    self.w1 = Parameter(requires_grad=True)
    self.w2 = Parameter(requires_grad=True)

  def forward(self, x):
    return Datum()


class _InnerModule(Module):
  def __init__(self):
    super().__init__()
    self.inner_param = Parameter(requires_grad=True)

  def forward(self, x):
    return Datum()


class _ContainerNoDirectParams(Module):
  def __init__(self):
    super().__init__()
    self.inner = _InnerModule()

  def forward(self, x):
    return self.inner(x)


class _ContainerWithDirectParam(Module):
  def __init__(self):
    super().__init__()
    self.inner = _InnerModule()
    self.own_param = Parameter(requires_grad=True)

  def forward(self, x):
    return self.inner(x)


class _NoneOutputModule(Module):
  def forward(self, x):
    return None


class _TupleOutputModule(Module):
  def __init__(self):
    super().__init__()
    self.w = Parameter(requires_grad=True)

  def forward(self, x):
    return (Datum(), 'extra')


def _fresh_context_run(fn):
  """Run fn in a fresh contextvars context for graph isolation."""
  ctx = copy_context()
  return ctx.run(fn)


# ModuleCallOperator tests (1-9)


class TestModuleCallOperator:
  def test_leaf_module_creates_grad_fn(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False
      m = _LeafModule()
      output = m(Datum())
      assert output.grad_fn is not None

    _fresh_context_run(run)

  def test_leaf_module_wires_all_params(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False
      m = _TwoParamLeaf()
      output = m(Datum())
      assert output.grad_fn is not None
      output.backward(NumericGradient(value=1.0))
      g1 = m.w1.grad
      g2 = m.w2.grad
      assert g1 is not None
      assert g2 is not None
      assert isinstance(g1, NumericGradient)
      assert isinstance(g2, NumericGradient)
      assert g1.value == 1.0
      assert g2.value == 1.0

    _fresh_context_run(run)

  def test_container_preserves_inner_graph(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False
      m = _ContainerNoDirectParams()
      inner_direct = m.inner(Datum())
      inner_grad_fn = inner_direct.grad_fn
      assert inner_grad_fn is not None
      g.reset()
      g._freed = False
      output = m(Datum())
      assert output.grad_fn is not None
      assert output.grad_fn.name() == 'ModuleCallOperator'

    _fresh_context_run(run)

  def test_container_no_direct_params_no_wrapper(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False
      m = _ContainerNoDirectParams()
      output = m(Datum())
      assert output.grad_fn is not None
      inner_only = m.inner(Datum())
      inner_name = inner_only.grad_fn.name()
      assert output.grad_fn.name() == inner_name

    _fresh_context_run(run)

  def test_container_with_direct_params_wraps(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False
      m = _ContainerWithDirectParam()
      output = m(Datum())
      assert output.grad_fn is not None
      assert output.grad_fn.name() == 'ModuleCallOperator'
      nf = output.grad_fn.next_functions
      assert len(nf) >= 2

    _fresh_context_run(run)

  def test_no_grad_skips_all(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False
      m = _LeafModule()
      with no_grad():
        output = m(Datum())
      assert output.grad_fn is None

    _fresh_context_run(run)

  def test_non_datum_output_no_grad_fn(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False
      m = _NoneOutputModule()
      output = m(Datum())
      assert output is None

    _fresh_context_run(run)

  def test_tuple_output_no_grad_fn(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False
      m = _TupleOutputModule()
      output = m(Datum())
      assert isinstance(output, tuple)
      assert len(output) == 2

    _fresh_context_run(run)

  def test_backward_flows_to_leaf_param(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False
      m = _LeafModule()
      output = m(Datum())
      assert output.grad_fn is not None
      output.backward(NumericGradient(value=5.0))
      grad = m.weight.grad
      assert grad is not None
      assert isinstance(grad, NumericGradient)
      assert grad.value == 5.0

    _fresh_context_run(run)


# backward_transform tests (10-13)


class TestBackwardTransform:
  def test_backward_transform_default_none(self) -> None:
    ctx = Context()
    grad = NumericGradient(value=1.0)
    assert Module().backward_transform(ctx, grad) is None

  def test_backward_transform_single_broadcasts(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False
      custom_grad = NumericGradient(value=99.0)

      class BroadcastModule(Module):
        def __init__(self):
          super().__init__()
          self.w1 = Parameter(requires_grad=True)
          self.w2 = Parameter(requires_grad=True)

        def forward(self, x):
          return Datum()

        def backward_transform(self, ctx, grad_output):
          return custom_grad

      m = BroadcastModule()
      output = m(Datum())
      output.backward(NumericGradient(value=1.0))
      gw1 = m.w1.grad
      gw2 = m.w2.grad
      assert gw1 is not None
      assert gw2 is not None
      assert isinstance(gw1, NumericGradient)
      assert isinstance(gw2, NumericGradient)
      assert gw1.value == 99.0
      assert gw2.value == 99.0

    _fresh_context_run(run)

  def test_backward_transform_tuple_exact_match(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False

      class ExactTupleModule(Module):
        def __init__(self):
          super().__init__()
          self.w1 = Parameter(requires_grad=True)
          self.w2 = Parameter(requires_grad=True)

        def forward(self, x):
          return Datum()

        def backward_transform(self, ctx, grad_output):
          return (NumericGradient(value=10.0), NumericGradient(value=20.0))

      m = ExactTupleModule()
      output = m(Datum())
      n_funcs = output.grad_fn._ctx.n_next_functions
      assert n_funcs == 2
      output.backward(NumericGradient(value=1.0))
      gw1 = m.w1.grad
      gw2 = m.w2.grad
      assert gw1 is not None
      assert gw2 is not None
      assert isinstance(gw1, NumericGradient)
      assert isinstance(gw2, NumericGradient)
      assert gw1.value == 10.0
      assert gw2.value == 20.0

    _fresh_context_run(run)

  def test_backward_transform_tuple_mismatch_broadcasts_first(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False

      class MismatchTupleModule(Module):
        def __init__(self):
          super().__init__()
          self.w1 = Parameter(requires_grad=True)
          self.w2 = Parameter(requires_grad=True)

        def forward(self, x):
          return Datum()

        def backward_transform(self, ctx, grad_output):
          return (NumericGradient(value=42.0),)

      m = MismatchTupleModule()
      output = m(Datum())
      n_funcs = output.grad_fn._ctx.n_next_functions
      assert n_funcs == 2
      output.backward(NumericGradient(value=1.0))
      gw1 = m.w1.grad
      gw2 = m.w2.grad
      assert gw1 is not None
      assert gw2 is not None
      assert isinstance(gw1, NumericGradient)
      assert isinstance(gw2, NumericGradient)
      assert gw1.value == 42.0
      assert gw2.value == 42.0

    _fresh_context_run(run)


# Post-hook grad_fn transfer tests (14-16)


class TestPostHookGradFnTransfer:
  def test_post_hook_new_datum_gets_grad_fn(self) -> None:
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

    _fresh_context_run(run)

  def test_post_hook_returns_none_preserves_original(self) -> None:
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
        return None

      m.register_forward_hook(hook)
      output = m(Datum())
      assert output.grad_fn is not None

    _fresh_context_run(run)

  def test_post_hook_non_datum_loses_grad_fn(self) -> None:
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
        return 'not a datum'

      m.register_forward_hook(hook)
      output = m(Datum())
      assert output == 'not a datum'
      assert not hasattr(output, 'grad_fn')

    _fresh_context_run(run)


# training_step contract tests (17-18)


class TestTrainingStepContract:
  def test_self_call_creates_grad_fn(self) -> None:
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

    _fresh_context_run(run)

  def test_self_forward_no_grad_fn(self) -> None:
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
      output = m.forward(Datum())
      assert output.grad_fn is None

    _fresh_context_run(run)


# Multi-module composition tests (19-22)


class TestMultiModuleComposition:
  def test_two_module_chain(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False

      class A(Module):
        def __init__(self):
          super().__init__()
          self.w = Parameter(requires_grad=True)

        def forward(self, x):
          return Datum()

      class B(Module):
        def __init__(self):
          super().__init__()
          self.w = Parameter(requires_grad=True)

        def forward(self, x):
          return Datum()

      a = A()
      b = B()
      out_a = a(Datum())
      out_b = b(out_a)
      assert out_b.grad_fn is not None
      out_b.backward(NumericGradient(value=1.0))
      gb = b.w.grad
      ga = a.w.grad
      assert gb is not None
      assert ga is not None
      assert isinstance(gb, NumericGradient)
      assert isinstance(ga, NumericGradient)
      assert gb.value == 1.0
      assert ga.value == 1.0

    _fresh_context_run(run)

  def test_three_module_explicit_wiring(self) -> None:
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

      a, b, c = M(), M(), M()
      out = a(Datum())
      out = b(out)
      out = c(out)
      assert out.grad_fn is not None
      out.backward(NumericGradient(value=1.0))
      gc = c.w.grad
      gb = b.w.grad
      ga = a.w.grad
      assert gc is not None
      assert gb is not None
      assert ga is not None
      assert isinstance(gc, NumericGradient)
      assert isinstance(gb, NumericGradient)
      assert isinstance(ga, NumericGradient)
      assert gc.value == 1.0
      assert gb.value == 1.0
      assert ga.value == 1.0

    _fresh_context_run(run)

  def test_nested_module_hierarchy(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False

      class Child(Module):
        def __init__(self):
          super().__init__()
          self.w = Parameter(requires_grad=True)

        def forward(self, x):
          return Datum()

      class Parent(Module):
        def __init__(self):
          super().__init__()
          self.child_a = Child()
          self.child_b = Child()

        def forward(self, x):
          a_out = self.child_a(x)
          return self.child_b(a_out)

      p = Parent()
      output = p(Datum())
      assert output.grad_fn is not None
      output.backward(NumericGradient(value=1.0))
      gcb = p.child_b.w.grad
      gca = p.child_a.w.grad
      assert gcb is not None
      assert gca is not None
      assert isinstance(gcb, NumericGradient)
      assert isinstance(gca, NumericGradient)
      assert gcb.value == 1.0
      assert gca.value == 1.0

    _fresh_context_run(run)

  def test_diamond_accumulation(self) -> None:
    def run():
      g = get_current_graph()
      g.reset()
      g._freed = False

      class Shared(Module):
        def __init__(self):
          super().__init__()
          self.w = Parameter(requires_grad=True)

        def forward(self, x):
          return Datum()

      shared = Shared()
      shared(Datum())
      out2 = shared(Datum())
      assert out2.grad_fn is not None
      out2.backward(NumericGradient(value=1.0))
      grad = shared.w.grad
      assert grad is not None
      assert isinstance(grad, NumericGradient)
      assert grad.value >= 1.0

    _fresh_context_run(run)
