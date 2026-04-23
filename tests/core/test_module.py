"""Tests for Module base class (nn.Module pattern)."""

from autopilot.core.metric import Metric
from autopilot.core.module import Module
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from helpers import NumericGradient
import pytest


class _ChildModule(Module):
  def forward(self, batch: object) -> Datum:
    return Datum(success=True)


class _ConcreteModule(Module):
  def forward(self, *args, **kwargs) -> Datum:
    return Datum(success=True, metrics={'ok': 1.0})


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
    assert result.success is True

  def test_call_delegates_to_forward(self) -> None:
    mod = _ConcreteModule()
    assert mod() == mod.forward()

  def test_subclass_with_constructor_kwargs(self) -> None:
    class Configured(Module):
      def __init__(self, host: str) -> None:
        super().__init__()
        self.host = host

      def forward(self, *args, **kwargs) -> Datum:
        return Datum(success=True, metadata={'host': self.host})

    mod = Configured(host='myhost')
    result = mod.forward()
    assert result.metadata['host'] == 'myhost'


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

    with pytest.raises(AttributeError, match='cannot assign before Module.__init__'):
      BadModule()


class TestModuleChildren:
  def test_children_returns_registered_modules(self) -> None:
    mod = Module()
    a = _ChildModule()
    b = _ChildModule()
    mod.deploy = a
    mod.eval = b
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
    mod.eval = b
    named = dict(mod.named_children())
    assert named == {'deploy': a, 'eval': b}

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
    assert 'Module()' == repr(mod)

  def test_subclass_repr(self) -> None:
    mod = _ConcreteModule()
    assert repr(mod).startswith('_ConcreteModule')

  def test_repr_tree_includes_child_names(self) -> None:
    mod = Module()
    mod.deploy = _ChildModule()
    mod.eval = _ChildModule()
    r = repr(mod)
    assert 'deploy' in r
    assert 'eval' in r
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
        self.eval = _ChildModule()

      def forward(self, *args, **kwargs) -> Datum:
        return Datum(success=True)

    mod = MyModule()
    assert len(list(mod.children())) == 2
    assert 'deploy' in repr(mod)
    assert 'eval' in repr(mod)


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
        return Datum(success=True)

    m = M()
    m.register_forward_pre_hook(lambda mod, args, kwargs: calls.append('pre'))
    from autopilot.core.graph import no_grad

    with no_grad():
      m(Datum())
    assert 'pre' in calls

  def test_forward_hook_called(self) -> None:
    calls: list[str] = []

    class M(Module):
      def forward(self, x):
        return Datum(success=True)

    m = M()
    m.register_forward_hook(lambda mod, args, out: calls.append('post'))
    from autopilot.core.graph import no_grad

    with no_grad():
      m(Datum())
    assert 'post' in calls

  def test_hook_removal(self) -> None:
    calls: list[str] = []

    class M(Module):
      def forward(self, x):
        return Datum(success=True)

    m = M()
    handle = m.register_forward_hook(lambda mod, args, out: calls.append('hook'))
    from autopilot.core.graph import no_grad

    with no_grad():
      m(Datum())
    assert len(calls) == 1
    handle.remove()
    with no_grad():
      m(Datum())
    assert len(calls) == 1


class TestModuleExtraRepr:
  def test_extra_repr_default_empty(self) -> None:
    assert Module().extra_repr() == ''

  def test_extra_repr_in_repr(self) -> None:
    class Custom(Module):
      def forward(self):
        return Datum()

      def extra_repr(self):
        return 'host=example.com'

    r = repr(Custom())
    assert 'host=example.com' in r
