"""Tests for Gradient base class and NumericGradient."""

from autopilot.ai.gradient import TextGradient
from autopilot.core.gradient import Gradient, NumericGradient
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from typing import Any, cast
import pytest


class TestGradientBase:
  def test_gradient_is_datum(self) -> None:
    assert isinstance(Gradient(), Datum)

  def test_gradient_accumulate_not_implemented(self) -> None:
    """Same-type accumulate passes isinstance then raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
      Gradient().accumulate(Gradient())

  def test_gradient_cross_type_raises(self) -> None:
    """Gradient().accumulate(NumericGradient()) raises TypeError."""
    with pytest.raises(TypeError, match='Cannot accumulate'):
      Gradient().accumulate(NumericGradient())

  def test_gradient_transform_default_returns_self(self) -> None:
    g = Gradient()
    assert g.transform(None) is g

  def test_gradient_transform_with_arbitrary_context(self) -> None:
    g = Gradient()
    assert g.transform({'key': 'value'}) is g
    assert g.transform(42) is g

  def test_gradient_render_raises_not_implemented(self) -> None:
    with pytest.raises(NotImplementedError):
      Gradient().render()

  def test_to_dict_from_dict_roundtrip(self) -> None:
    g = Gradient()
    d = g.to_dict()
    g2 = Gradient.from_dict(d)
    assert g2.id == g.id

  def test_gradient_with_items(self) -> None:
    g = Gradient(items=[Datum()])
    d = g.to_dict()
    g2 = Gradient.from_dict(d)
    assert len(g2.items) == 1

  def test_gradient_default_fields(self) -> None:
    g = Gradient()
    assert g.items == []
    assert isinstance(g.id, str)
    assert len(g.id) == 12


class TestNumericGradient:
  def test_numeric_gradient_accumulate(self) -> None:
    result = NumericGradient(value=3).accumulate(NumericGradient(value=5))
    assert result.value == 8.0

  def test_numeric_gradient_cross_type_raises(self) -> None:
    """NumericGradient().accumulate(Gradient()) raises TypeError with class names."""
    with pytest.raises(TypeError, match=r'NumericGradient.*Gradient') as exc_info:
      NumericGradient().accumulate(Gradient())
    assert 'NumericGradient' in str(exc_info.value)
    assert 'Gradient' in str(exc_info.value)

  def test_numeric_gradient_cross_type_with_text_gradient(self) -> None:
    with pytest.raises(TypeError, match='Cannot accumulate'):
      NumericGradient().accumulate(TextGradient())

  def test_numeric_gradient_render(self) -> None:
    assert NumericGradient(value=3.0).render() == 'gradient: 3.0'

  def test_numeric_gradient_render_zero(self) -> None:
    assert NumericGradient(value=0.0).render() == 'gradient: 0.0'

  def test_numeric_gradient_render_negative(self) -> None:
    assert NumericGradient(value=-2.5).render() == 'gradient: -2.5'

  def test_numeric_gradient_to_dict_roundtrip(self) -> None:
    g = NumericGradient(value=42.0)
    d = g.to_dict()
    assert d['value'] == 42.0
    g2 = NumericGradient.from_dict(d)
    assert g2.value == 42.0
    assert g2.id == g.id

  def test_numeric_gradient_to_dict_includes_value(self) -> None:
    d = NumericGradient(value=7.5).to_dict()
    assert 'value' in d
    assert d['value'] == 7.5

  def test_numeric_gradient_is_gradient(self) -> None:
    assert isinstance(NumericGradient(), Gradient)
    assert isinstance(NumericGradient(), Datum)

  def test_numeric_gradient_default_value(self) -> None:
    assert NumericGradient().value == 0.0

  def test_numeric_gradient_accumulate_multiple(self) -> None:
    a = NumericGradient(value=1.0)
    b = NumericGradient(value=2.0)
    c = NumericGradient(value=3.0)
    result = a.accumulate(b).accumulate(c)
    assert result.value == 6.0

  def test_numeric_gradient_accumulate_wrong_type_string(self) -> None:
    with pytest.raises(TypeError):
      NumericGradient(value=1).accumulate(cast(Any, 'not a gradient'))

  def test_numeric_gradient_transform_returns_self(self) -> None:
    g = NumericGradient(value=5.0)
    assert g.transform(None) is g


class TestNumericGradientFromDictPreservesValue:
  def test_numeric_gradient_from_dict_preserves_value(self) -> None:
    g = NumericGradient(value=42.5, items=[Datum()])
    data = g.to_dict()
    restored = NumericGradient.from_dict(data)
    assert restored.value == 42.5
    assert restored.id == g.id
    assert len(restored.items) == 1

  def test_numeric_gradient_from_dict_default_value(self) -> None:
    restored = NumericGradient.from_dict({'items': []})
    assert restored.value == 0.0


class TestTextGradientFromDictPreservesAttribution:
  def test_text_gradient_from_dict_preserves_attribution(self) -> None:
    g = TextGradient(
      text='improve accuracy',
      attribution='fix prompt template',
      severity=0.8,
    )
    data = g.to_dict()
    restored = TextGradient.from_dict(data)
    assert restored.text == 'improve accuracy'
    assert restored.attribution == 'fix prompt template'
    assert restored.severity == 0.8
    assert restored.id == g.id

  def test_text_gradient_from_dict_with_items(self) -> None:
    g = TextGradient(
      text='d',
      attribution='a',
      items=[Datum()],
    )
    data = g.to_dict()
    restored = TextGradient.from_dict(data)
    assert len(restored.items) == 1
    assert restored.items[0].id == g.items[0].id


class TestParameterNewMethods:
  def test_parameter_render_default_empty(self) -> None:
    assert not Parameter().render()

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
        object.__setattr__(self, 'store_data', {})

      def snapshot(self) -> dict[str, str]:
        return dict(self.store_data)

      def restore(self, content: dict[str, str]) -> None:
        self.store_data = dict(content)

    p = MemParam()
    p.store_data = {'key': 'value'}
    snap = p.snapshot()
    p.store_data = {}
    p.restore(snap)
    assert p.store_data == {'key': 'value'}
