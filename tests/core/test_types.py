"""Tests for core types: Datum and GateResult."""

from autopilot.core.gradient import Gradient
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum, GateResult
import pytest


class TestDatumId:
  def test_datum_id_auto_generated(self) -> None:
    d = Datum()
    assert isinstance(d.id, str)
    assert len(d.id) == 12

  def test_datum_id_unique(self) -> None:
    ids = {Datum().id for _ in range(1000)}
    assert len(ids) == 1000

  def test_datum_id_is_read_only(self) -> None:
    d = Datum()
    with pytest.raises(AttributeError):
      d.id = 'new'

  def test_datum_id_not_in_constructor(self) -> None:
    with pytest.raises(TypeError):
      Datum(id='foo')

  def test_datum_to_dict_includes_id(self) -> None:
    d = Datum()
    assert d.to_dict()['id'] == d.id

  def test_datum_to_dict_no_item_id(self) -> None:
    d = Datum()
    assert 'item_id' not in d.to_dict()

  def test_datum_from_dict_restores_id(self) -> None:
    d = Datum.from_dict({'id': 'abc123'})
    assert d.id == 'abc123'

  def test_datum_from_dict_unknown_keys_ignored(self) -> None:
    d = Datum.from_dict({'foo': 'bar'})
    assert isinstance(d.id, str)
    assert len(d.id) == 12

  def test_datum_from_dict_no_id_generates_new(self) -> None:
    d = Datum.from_dict({})
    assert isinstance(d.id, str)
    assert len(d.id) == 12

  def test_parameter_inherits_id(self) -> None:
    p = Parameter()
    assert isinstance(p.id, str)
    assert len(p.id) == 12

  def test_gradient_inherits_id(self) -> None:
    g = Gradient()
    assert isinstance(g.id, str)
    assert len(g.id) == 12

  def test_datum_id_stable_across_items(self) -> None:
    child = Datum()
    parent = Datum(items=[child])
    assert parent.id != child.id
    assert len(parent.id) == 12
    assert len(child.id) == 12


class TestDatumRoundTrip:
  def test_defaults(self):
    d = Datum()
    assert d.split is None
    assert d.epoch == 0
    assert d.metrics == {}
    assert d.success is True

  def test_to_dict_from_dict(self):
    d = Datum(split='train', epoch=3, metrics={'accuracy': 0.9}, success=True)
    data = d.to_dict()
    d2 = Datum.from_dict(data)
    assert d2.split == 'train'
    assert d2.epoch == 3
    assert d2.metrics == {'accuracy': 0.9}
    assert d2.success is True

  def test_nested_items(self):
    child = Datum(split='train')
    parent = Datum(items=[child])
    data = parent.to_dict()
    restored = Datum.from_dict(data)
    assert len(restored.items) == 1
    assert restored.items[0].split == 'train'

  def test_bool(self):
    assert bool(Datum(success=True)) is True
    assert bool(Datum(success=False)) is False


class TestGateResult:
  def test_values(self):
    assert GateResult.PASS == 'pass'
    assert GateResult.FAIL == 'fail'
    assert GateResult.WARN == 'warn'
    assert GateResult.SKIP == 'skip'

  def test_is_str_enum(self):
    assert isinstance(GateResult.PASS, str)

  def test_all_values(self):
    values = {g.value for g in GateResult}
    assert values == {'pass', 'fail', 'warn', 'skip'}
