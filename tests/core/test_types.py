"""Tests for core types: Datum, EvalDatum, and GateResult."""

from autopilot.core.gradient import Gradient
from autopilot.core.types import Datum, EvalDatum, GateResult
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import MagicMock, patch
import copy
import pytest

# --- Helpers ---


@dataclass
class CustomDatum(Datum):
  label: str | None = None


@dataclass
class CustomGradient(Gradient):
  message: str | None = None

  def accumulate(self, other: 'Gradient') -> 'Gradient':
    assert isinstance(other, CustomGradient)
    return CustomGradient(
      message=f'{self.message}; {other.message}' if self.message else other.message,
    )

  def render(self) -> str:
    return self.message or ''


# --- Slim Datum tests (1-10) ---


class TestSlimDatum:
  def test_datum_only_has_items(self) -> None:
    d = Datum()
    assert d.items == []
    assert not hasattr(d, 'success')
    assert not hasattr(d, 'metadata')
    assert not hasattr(d, 'split')
    assert not hasattr(d, 'epoch')
    assert not hasattr(d, 'metrics')
    assert not hasattr(d, 'error_message')
    assert not hasattr(d, 'feedback')

  def test_datum_id_auto_generated(self) -> None:
    d = Datum()
    assert isinstance(d.id, str)
    assert len(d.id) == 12
    assert all(c in '0123456789abcdef' for c in d.id)

  def test_datum_id_unique(self) -> None:
    ids = {Datum().id for _ in range(1000)}
    assert len(ids) == 1000

  def test_datum_grad_fn_default_none(self) -> None:
    d = Datum()
    assert d.grad_fn is None

  def test_datum_to_dict_includes_type(self) -> None:
    d = Datum()
    result = d.to_dict()
    assert result['type'] == 'autopilot.core.types.Datum'

  def test_datum_to_dict_excludes_grad_fn(self) -> None:
    d = Datum()
    d.grad_fn = 'something'
    result = d.to_dict()
    assert 'grad_fn' not in result

  def test_datum_from_dict_roundtrip(self) -> None:
    child = Datum()
    parent = Datum(items=[child])
    data = parent.to_dict()
    restored = Datum.from_dict(data)
    assert restored.id == parent.id
    assert len(restored.items) == 1
    assert restored.items[0].id == child.id

  def test_datum_no_success_attr(self) -> None:
    assert not hasattr(Datum(), 'success')

  def test_datum_no_metadata_attr(self) -> None:
    assert not hasattr(Datum(), 'metadata')

  def test_datum_no_bool(self) -> None:
    d = Datum()
    assert bool(d) is True


# --- __deepcopy__ tests (11-13) ---


class TestDatumDeepcopy:
  def test_deepcopy_clears_grad_fn(self) -> None:
    d = Datum()
    d.grad_fn = MagicMock()
    assert d.grad_fn is not None
    copied = copy.deepcopy(d)
    assert copied.grad_fn is None

  def test_deepcopy_independent_items(self) -> None:
    child = Datum()
    parent = Datum(items=[child])
    copied = copy.deepcopy(parent)
    copied.items.append(Datum())
    assert len(parent.items) == 1
    assert len(copied.items) == 2

  def test_deepcopy_preserves_subclass(self) -> None:
    d = CustomDatum(label='test')
    copied = copy.deepcopy(d)
    assert isinstance(copied, CustomDatum)
    assert copied.label == 'test'
    assert copied.id == d.id


# --- clone() tests (14-16) ---


class TestDatumClone:
  def test_clone_returns_independent_copy(self) -> None:
    child = Datum()
    parent = Datum(items=[child])
    cloned = parent.clone()
    cloned.items.append(Datum())
    assert len(parent.items) == 1
    assert len(cloned.items) == 2

  def test_clone_preserves_subclass_type(self) -> None:
    d = CustomDatum(label='hello')
    cloned = d.clone()
    assert isinstance(cloned, CustomDatum)
    assert cloned.label == 'hello'

  def test_clone_grad_fn_none(self) -> None:
    d = Datum()
    d.grad_fn = MagicMock()
    cloned = d.clone()
    assert cloned.grad_fn is None


# --- detach() tests (17-19) ---


class TestDatumDetach:
  def test_detach_clears_grad_fn(self) -> None:
    d = Datum()
    d.grad_fn = MagicMock()
    detached = d.detach()
    assert detached.grad_fn is None

  def test_detach_preserves_subclass(self) -> None:
    d = CustomDatum(label='world')
    detached = d.detach()
    assert isinstance(detached, CustomDatum)
    assert detached.label == 'world'

  def test_detach_independent_copy(self) -> None:
    child = Datum()
    parent = Datum(items=[child])
    detached = parent.detach()
    detached.items.append(Datum())
    assert len(parent.items) == 1
    assert len(detached.items) == 2


# --- backward() tests (20-21) ---


class TestDatumBackward:
  def test_backward_no_grad_fn_raises(self) -> None:
    d = Datum()
    grad = CustomGradient(message='fix it')
    with pytest.raises(RuntimeError, match='cannot backward through a datum without grad_fn'):
      d.backward(grad)

  def test_backward_with_grad_fn(self) -> None:
    mock_node = MagicMock()
    mock_node.sequence_nr = 0
    mock_node.next_functions = []
    mock_node.return_value = []

    d = Datum()
    d.grad_fn = mock_node

    grad = CustomGradient(message='optimize')

    with patch('autopilot.core.types.get_current_graph') as mock_get_graph:
      mock_graph = MagicMock()
      mock_get_graph.return_value = mock_graph
      d.backward(grad)
      mock_graph.backward.assert_called_once_with(mock_node, grad)


# --- EvalDatum tests (22-28) ---


class TestEvalDatum:
  def test_eval_datum_has_all_fields(self) -> None:
    d = EvalDatum()
    assert hasattr(d, 'split')
    assert hasattr(d, 'epoch')
    assert hasattr(d, 'metrics')
    assert hasattr(d, 'success')
    assert hasattr(d, 'error_message')
    assert hasattr(d, 'feedback')
    assert hasattr(d, 'metadata')

  def test_eval_datum_bool(self) -> None:
    assert bool(EvalDatum(success=True)) is True
    assert bool(EvalDatum(success=False)) is False

  def test_eval_datum_to_dict_roundtrip(self) -> None:
    d = EvalDatum(
      split='train',
      epoch=3,
      metrics={'accuracy': 0.9, 'labels': ['a', 'b']},
      success=True,
      error_message='some error',
      feedback='good job',
      metadata={'key': 'value'},
      items=[Datum()],
    )
    data = d.to_dict()
    restored = EvalDatum.from_dict(data)
    assert restored.split == 'train'
    assert restored.epoch == 3
    assert restored.metrics == {'accuracy': 0.9, 'labels': ['a', 'b']}
    assert restored.success is True
    assert restored.error_message == 'some error'
    assert restored.feedback == 'good job'
    assert restored.metadata == {'key': 'value'}
    assert len(restored.items) == 1
    assert restored.id == d.id

  def test_eval_datum_inherits_items(self) -> None:
    child = Datum()
    d = EvalDatum(items=[child])
    assert len(d.items) == 1
    assert d.items[0] is child

  def test_eval_datum_inherits_id(self) -> None:
    d = EvalDatum()
    assert isinstance(d.id, str)
    assert len(d.id) == 12

  def test_eval_datum_to_dict_includes_type(self) -> None:
    d = EvalDatum()
    result = d.to_dict()
    assert result['type'] == 'autopilot.core.types.EvalDatum'

  def test_eval_datum_defaults(self) -> None:
    d = EvalDatum()
    assert d.success is True
    assert d.epoch is None
    assert d.split is None
    assert d.metrics == {}
    assert d.metadata == {}
    assert d.error_message is None
    assert d.feedback is None


# --- GateResult tests (29-32) ---


class TestGateResult:
  def test_values(self) -> None:
    assert GateResult.PASSED == 'pass'
    assert GateResult.FAIL == 'fail'
    assert GateResult.WARN == 'warn'
    assert GateResult.SKIP == 'skip'

  def test_is_str_enum(self) -> None:
    assert isinstance(GateResult.PASSED, str)

  def test_all_values(self) -> None:
    values = {g.value for g in GateResult}
    assert values == {'pass', 'fail', 'warn', 'skip'}

  def test_membership(self) -> None:
    assert GateResult('pass') is GateResult.PASSED
    assert GateResult('fail') is GateResult.FAIL


# --- Additional edge-case tests ---


class TestDatumSerialization:
  def test_datum_from_dict_unknown_keys_ignored(self) -> None:
    d = Datum.from_dict({'foo': 'bar'})
    assert isinstance(d.id, str)
    assert len(d.id) == 12

  def test_datum_from_dict_restores_id(self) -> None:
    d = Datum.from_dict({'id': 'abc123def456'})
    assert d.id == 'abc123def456'

  def test_datum_from_dict_no_id_generates_new(self) -> None:
    d = Datum.from_dict({})
    assert isinstance(d.id, str)
    assert len(d.id) == 12

  def test_datum_id_is_read_only(self) -> None:
    d = Datum()
    d_any = cast(Any, d)
    with pytest.raises(AttributeError):
      d_any.id = 'new'

  def test_datum_to_dict_nested_items(self) -> None:
    grandchild = Datum()
    child = Datum(items=[grandchild])
    parent = Datum(items=[child])
    data = parent.to_dict()
    assert len(data['items']) == 1
    assert len(data['items'][0]['items']) == 1
    assert data['items'][0]['items'][0]['id'] == grandchild.id

  def test_custom_datum_to_dict_type(self) -> None:
    d = CustomDatum(label='test')
    result = d.to_dict()
    assert 'CustomDatum' in result['type']

  def test_eval_datum_from_dict_with_type_key(self) -> None:
    data = {
      'type': 'autopilot.core.types.EvalDatum',
      'id': 'test123',
      'items': [],
      'split': 'val',
      'epoch': 5,
      'metrics': {'f1': 0.85},
      'success': False,
      'error_message': 'timeout',
      'feedback': None,
      'metadata': {'run': 1},
    }
    d = EvalDatum.from_dict(data)
    assert d.id == 'test123'
    assert d.split == 'val'
    assert d.epoch == 5
    assert d.success is False


class TestDatumGradFn:
  def test_grad_fn_in_dataclass_fields_but_excluded_from_init_and_compare(self) -> None:
    from dataclasses import fields as dc_fields

    field_map = {f.name: f for f in dc_fields(Datum)}
    assert 'grad_fn' in field_map
    assert '_id' in field_map
    assert field_map['grad_fn'].init is False
    assert field_map['grad_fn'].compare is False
    assert field_map['_id'].init is False
    assert field_map['_id'].compare is False

  def test_grad_fn_survives_attribute_access(self) -> None:
    d = Datum()
    mock_fn = MagicMock()
    d.grad_fn = mock_fn
    assert d.grad_fn is mock_fn

  def test_eval_datum_grad_fn_default_none(self) -> None:
    d = EvalDatum()
    assert d.grad_fn is None

  def test_eval_datum_clone_clears_grad_fn(self) -> None:
    d = EvalDatum(success=True)
    d.grad_fn = MagicMock()
    cloned = d.clone()
    assert cloned.grad_fn is None
    assert isinstance(cloned, EvalDatum)
    assert cloned.success is True


# --- Plan 07 migration tests ---


class TestDatumSubclassesMigration:
  def test_base_datum_no_success(self) -> None:
    assert not hasattr(Datum(), 'success')

  def test_base_datum_no_metadata(self) -> None:
    assert not hasattr(Datum(), 'metadata')

  def test_base_datum_no_feedback(self) -> None:
    assert not hasattr(Datum(), 'feedback')

  def test_eval_datum_has_success(self) -> None:
    assert hasattr(EvalDatum(), 'success')

  def test_eval_datum_roundtrip(self) -> None:
    original = EvalDatum(success=True, metrics={'f1': 0.9})
    data = original.to_dict()
    restored = EvalDatum.from_dict(data)
    assert restored.success == original.success
    assert restored.metrics == original.metrics
    assert restored.id == original.id

  def test_eval_datum_is_datum(self) -> None:
    assert isinstance(EvalDatum(), Datum)

  def test_old_datum_constructor_with_success_fails(self) -> None:
    datum_ctor = cast(Any, Datum)
    with pytest.raises(TypeError):
      datum_ctor(success=True)


# --- Plan 02: EvalDatum nested serialization tests ---


class TestEvalDatumNestedSerialization:
  def test_evaldatum_nested_roundtrip(self) -> None:
    inner = EvalDatum(success=False, feedback='nested feedback')
    parent = EvalDatum(items=[inner])
    restored = EvalDatum.from_dict(parent.to_dict())
    assert type(restored.items[0]) is EvalDatum
    assert restored.items[0].feedback == 'nested feedback'
    assert restored.items[0].success is False

  def test_evaldatum_nested_mixed_items(self) -> None:
    parent = EvalDatum(items=[Datum(), EvalDatum(metrics={'f1': 0.9})])
    restored = EvalDatum.from_dict(parent.to_dict())
    assert type(restored.items[0]) is Datum
    assert type(restored.items[1]) is EvalDatum
    assert restored.items[1].metrics == {'f1': 0.9}

  def test_evaldatum_deeply_nested(self) -> None:
    deep = EvalDatum(items=[EvalDatum(items=[EvalDatum(success=False)])])
    restored = EvalDatum.from_dict(deep.to_dict())
    assert type(restored) is EvalDatum
    level1 = restored.items[0]
    assert type(level1) is EvalDatum
    level2 = level1.items[0]
    assert type(level2) is EvalDatum
    assert level2.success is False

  def test_datum_from_dict_still_returns_datum(self) -> None:
    d = Datum()
    restored = Datum.from_dict(d.to_dict())
    assert type(restored) is Datum
    assert restored.id == d.id

  def test_evaldatum_from_dict_no_type_field(self) -> None:
    restored = EvalDatum.from_dict(
      {
        'items': [],
        'success': True,
        'metrics': {},
      }
    )
    assert type(restored) is EvalDatum
    assert restored.success is True
    assert restored.items == []

  def test_evaldatum_unknown_type_in_items(self) -> None:
    parent = EvalDatum(items=[Datum()])
    data = parent.to_dict()
    data['items'][0]['type'] = 'some.Unknown'
    restored = EvalDatum.from_dict(data)
    assert type(restored.items[0]) is Datum

  def test_datum_from_dict_round_trip_nested(self) -> None:
    child = Datum()
    parent = Datum(items=[child])
    data = parent.to_dict()
    data['id'] = 'custom_parent'
    restored = Datum.from_dict(data)
    assert restored.id == 'custom_parent'
    assert len(restored.items) == 1
    assert restored.items[0].id == child.id

  def test_eval_datum_from_dict_hydrates_mixed_children(self) -> None:
    plain_child = {'type': 'autopilot.core.types.Datum', 'id': 'plain1', 'items': []}
    eval_child = {
      'type': 'autopilot.core.types.EvalDatum',
      'id': 'eval1',
      'items': [],
      'success': False,
      'metrics': {},
      'split': None,
      'epoch': None,
      'error_message': None,
      'feedback': None,
      'metadata': {},
    }
    data = {
      'type': 'autopilot.core.types.EvalDatum',
      'id': 'parent1',
      'items': [plain_child, eval_child],
      'success': True,
      'metrics': {},
      'split': None,
      'epoch': None,
      'error_message': None,
      'feedback': None,
      'metadata': {},
    }
    restored = EvalDatum.from_dict(data)
    assert type(restored.items[0]) is Datum
    assert type(restored.items[1]) is EvalDatum
    assert restored.items[1].success is False

  def test_hydrate_preserves_unknown_keys_drop(self) -> None:
    data = {'unknown_key': 'should_not_crash', 'extra': 42, 'items': []}
    d = Datum.from_dict(data)
    assert isinstance(d, Datum)
    assert not hasattr(d, 'unknown_key')
    assert not hasattr(d, 'extra')

  def test_eval_datum_nested_eval_datum(self) -> None:
    inner = EvalDatum(success=False, feedback='inner feedback', metrics={'f1': 0.5})
    outer = EvalDatum(items=[inner], success=True)
    restored = EvalDatum.from_dict(outer.to_dict())
    assert type(restored.items[0]) is EvalDatum
    assert restored.items[0].success is False
    assert restored.items[0].feedback == 'inner feedback'
    assert restored.items[0].metrics == {'f1': 0.5}

  def test_datum_from_dict_uses_cls_from_dict_for_children(self) -> None:
    """Datum.from_dict hydrates children via cls.from_dict (same class)."""

    @dataclass
    class TaggedDatum(Datum):
      tag: str | None = None

      def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload['tag'] = self.tag
        return payload

      @classmethod
      def from_dict(cls, data: dict[str, Any]) -> 'TaggedDatum':
        from autopilot.core.types import _hydrate_datum_base

        return _hydrate_datum_base(cls, data, hydrate_child=cls.from_dict, pop_type=True)

    parent = TaggedDatum(tag='parent', items=[TaggedDatum(tag='child')])
    restored = TaggedDatum.from_dict(parent.to_dict())
    assert type(restored) is TaggedDatum
    assert restored.tag == 'parent'
    assert type(restored.items[0]) is TaggedDatum
    assert restored.items[0].tag == 'child'

  def test_evaldatum_from_dict_uses_datum_from_dict_dispatch(self) -> None:
    """EvalDatum.from_dict uses _datum_from_dict for polymorphic children."""
    parent = EvalDatum(
      items=[Datum(), EvalDatum(success=False, feedback='child fb')],
    )
    restored = EvalDatum.from_dict(parent.to_dict())
    assert type(restored.items[0]) is Datum
    assert type(restored.items[1]) is EvalDatum
    child_eval = restored.items[1]
    assert isinstance(child_eval, EvalDatum)
    assert child_eval.success is False
    assert child_eval.feedback == 'child fb'

  def test_evaldatum_all_fields_roundtrip(self) -> None:
    original = EvalDatum(
      split='train',
      epoch=3,
      metrics={'f1': 0.9, 'acc': 0.8},
      success=False,
      error_message='err',
      feedback='fix it',
      metadata={'k': 'v'},
      items=[EvalDatum(feedback='nested')],
    )
    restored = EvalDatum.from_dict(original.to_dict())
    assert restored.split == 'train'
    assert restored.epoch == 3
    assert restored.metrics == {'f1': 0.9, 'acc': 0.8}
    assert restored.success is False
    assert restored.error_message == 'err'
    assert restored.feedback == 'fix it'
    assert restored.metadata == {'k': 'v'}
    assert restored.id == original.id
    nested = restored.items[0]
    assert type(nested) is EvalDatum
    assert nested.feedback == 'nested'
