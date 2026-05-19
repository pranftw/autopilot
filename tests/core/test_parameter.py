"""Tests for Parameter base class."""

from autopilot.core.gradient import Gradient, NumericGradient
from autopilot.core.graph import Graph
from autopilot.core.module.module import Module
from autopilot.core.operator import AccumulateGrad
from autopilot.core.optimizer import Optimizer
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
import copy


class TestParameterBase:
  def test_parameter_is_datum_subclass(self) -> None:
    p = Parameter()
    assert isinstance(p, Datum)

  def test_parameter_requires_grad_default_true(self) -> None:
    p = Parameter()
    assert p.requires_grad is True

  def test_parameter_grad_default_none(self) -> None:
    p = Parameter()
    assert p.grad is None

  def test_parameter_set_grad(self) -> None:
    p = Parameter()
    p.grad = NumericGradient(value=1.0)
    assert isinstance(p.grad, Gradient)
    grad = p.grad
    assert isinstance(grad, NumericGradient)
    assert grad.value == 1.0

  def test_parameter_to_dict_includes_grad_fields(self) -> None:
    p = Parameter(requires_grad=True)
    d = p.to_dict()
    assert 'requires_grad' in d
    assert d['requires_grad'] is True

  def test_parameter_from_dict_round_trip(self) -> None:
    p = Parameter(requires_grad=True)
    d = p.to_dict()
    p2 = Parameter.from_dict(d)
    assert p2.requires_grad is True
    assert p2.id == p.id


class TestParameterFromDictHydration:
  def test_parameter_from_dict_preserves_requires_grad(self) -> None:
    for rg in (True, False):
      p = Parameter(requires_grad=rg)
      data = p.to_dict()
      restored = Parameter.from_dict(data)
      assert restored.requires_grad is rg
      assert restored.id == p.id

  def test_parameter_from_dict_does_not_pop_type(self) -> None:
    restored = Parameter.from_dict({'value': 'x', 'type': 'prompt'})
    assert isinstance(restored, Parameter)
    payload = restored.to_dict()
    assert 'type' in payload
    assert payload['type'].endswith('.Parameter')


class TestParameterToDictExcludesGrad:
  """Gradient is transient and excluded from serialization."""

  def test_to_dict_has_no_grad_key(self) -> None:
    p = Parameter()
    d = p.to_dict()
    assert 'grad' not in d

  def test_to_dict_has_no_grad_key_when_grad_set(self) -> None:
    p = Parameter()
    p.grad = NumericGradient(value=5.0)
    d = p.to_dict()
    assert 'grad' not in d

  def test_to_dict_has_no_grad_key_with_requires_grad_false(self) -> None:
    p = Parameter(requires_grad=False)
    p.grad = NumericGradient(value=1.0)
    d = p.to_dict()
    assert 'grad' not in d


class TestOptimizerZeroGrad:
  def test_zero_grad_clears_grad_on_trainable_param(self) -> None:
    p = Parameter(requires_grad=True)
    p.grad = NumericGradient(value=1.0)
    opt = Optimizer([p])
    opt.zero_grad()
    assert p.grad is None

  def test_zero_grad_clears_grad_on_frozen_param(self) -> None:
    p = Parameter(requires_grad=False)
    p.grad = NumericGradient(value=2.0)
    opt = Optimizer([p])
    opt.zero_grad()
    assert p.grad is None

  def test_zero_grad_clears_all_params_mixed(self) -> None:
    trainable = Parameter(requires_grad=True)
    frozen = Parameter(requires_grad=False)
    trainable.grad = NumericGradient(value=1.0)
    frozen.grad = NumericGradient(value=3.0)
    opt = Optimizer([trainable, frozen])
    opt.zero_grad()
    assert trainable.grad is None
    assert frozen.grad is None

  def test_zero_grad_noop_on_none_grads(self) -> None:
    p = Parameter()
    opt = Optimizer([p])
    opt.zero_grad()
    assert p.grad is None


class TestParameterModuleIntegration:
  def test_parameter_registered_in_module(self) -> None:
    mod = Module()
    p = Parameter()
    mod.weight = p
    assert p in list(mod.parameters())

  def test_multiple_parameters_registered(self) -> None:
    mod = Module()
    p1 = Parameter()
    p2 = Parameter()
    mod.w1 = p1
    mod.w2 = p2
    params = list(mod.parameters())
    assert len(params) == 2
    assert p1 in params
    assert p2 in params

  def test_parameter_not_in_modules(self) -> None:
    mod = Module()
    mod.p = Parameter()
    assert 'p' not in mod._modules
    assert len(list(mod.children())) == 0

  def test_parameter_in_named_parameters(self) -> None:
    mod = Module()
    p = Parameter()
    mod.weight = p
    named = dict(mod.named_parameters())
    assert 'weight' in named
    assert named['weight'] is p


class TestParameterDeepcopy:
  def test_parameter_deepcopy_withgrad_accumulator(self) -> None:
    p = Parameter(requires_grad=True)
    g = Graph()
    AccumulateGrad.get_or_create(p, g)
    assert p.grad_accumulator is not None
    c = copy.deepcopy(p)
    assert isinstance(c, Parameter)
    assert c.grad_accumulator is None
    assert c.requires_grad is True

  def test_parameter_clone_withgrad_accumulator(self) -> None:
    p = Parameter(requires_grad=True)
    g = Graph()
    AccumulateGrad.get_or_create(p, g)
    assert p.grad_accumulator is not None
    c = p.clone()
    assert isinstance(c, Parameter)
    assert c.grad_accumulator is None
    assert c.requires_grad is True
