"""Tests for Gradient base class."""

from autopilot.core.gradient import Gradient
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from helpers import NumericGradient
import pytest


class TestGradientBase:
  def test_gradient_is_datum(self) -> None:
    assert isinstance(Gradient(), Datum)

  def test_accumulate_raises(self) -> None:
    with pytest.raises(NotImplementedError):
      Gradient().accumulate(Gradient())

  def test_render_raises(self) -> None:
    with pytest.raises(NotImplementedError):
      Gradient().render()

  def test_to_dict_from_dict_roundtrip(self) -> None:
    g = Gradient(metadata={'a': 1})
    d = g.to_dict()
    g2 = Gradient.from_dict(d)
    assert g2.metadata == {'a': 1}

  def test_gradient_with_items(self) -> None:
    g = Gradient(items=[Datum(feedback='x')])
    d = g.to_dict()
    g2 = Gradient.from_dict(d)
    assert len(g2.items) == 1
    assert g2.items[0].feedback == 'x'

  def test_gradient_default_fields(self) -> None:
    g = Gradient()
    assert g.success is True
    assert g.items == []
    assert g.metadata == {}
    assert isinstance(g.id, str)
    assert len(g.id) == 12


class TestConcreteGradient:
  def test_concrete_subclass_accumulate(self) -> None:
    result = NumericGradient(value=3).accumulate(NumericGradient(value=7))
    assert result.value == 10.0

  def test_concrete_subclass_render(self) -> None:
    assert NumericGradient(value=5.0).render() == 'gradient: 5.0'

  def test_accumulate_wrong_type_raises(self) -> None:
    with pytest.raises(AttributeError):
      NumericGradient(value=1).accumulate('not a gradient')

  def test_subclass_must_override_to_dict_for_extra_fields(self) -> None:
    d = NumericGradient(value=42).to_dict()
    assert 'value' not in d


class TestParameterNewMethods:
  def test_parameter_render_default_empty(self) -> None:
    assert Parameter().render() == ''

  def test_parameter_snapshot_default_empty(self) -> None:
    assert Parameter().snapshot() == {}

  def test_parameter_restore_default_noop(self) -> None:
    Parameter().restore({'k': 'v'})

  def test_parameter_custom_render(self) -> None:
    class ScopedParam(Parameter):
      def render(self) -> str:
        return 'my scope'

    assert ScopedParam().render() == 'my scope'

  def test_parameter_custom_snapshot_restore(self) -> None:
    class MemParam(Parameter):
      def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, '_store', {})

      def snapshot(self) -> dict[str, str]:
        return dict(self._store)

      def restore(self, content: dict[str, str]) -> None:
        object.__setattr__(self, '_store', dict(content))

    p = MemParam()
    p._store = {'key': 'value'}
    snap = p.snapshot()
    p._store = {}
    p.restore(snap)
    assert p._store == {'key': 'value'}
