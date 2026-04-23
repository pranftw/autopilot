"""Tests for Parameter base class."""

from autopilot.core.gradient import Gradient
from autopilot.core.module import Module
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from helpers import NumericGradient


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
    assert p.grad.value == 1.0

  def test_parameter_to_dict_includes_grad_fields(self) -> None:
    p = Parameter(requires_grad=True)
    d = p.to_dict()
    assert 'requires_grad' in d
    assert d['requires_grad'] is True

  def test_parameter_from_dict_round_trip(self) -> None:
    p = Parameter(requires_grad=True, metrics={'x': 1.0})
    d = p.to_dict()
    p2 = Parameter.from_dict(d)
    assert p2.requires_grad is True
    assert p2.metrics == {'x': 1.0}


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
