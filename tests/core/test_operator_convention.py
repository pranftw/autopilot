"""Convention tests for operator argument order (data-first).

Validates that:
- select(datum, index) works correctly (data-first)
- Old argument order select(index, datum) raises helpful TypeError
- Out-of-range, empty datum, non-integer index raise appropriate errors
- broadcast(datum, n) is already data-first
- merge(d1, d2) accepts only datum operands
- Operator docstrings are present
- CLAUDE.md documents the convention
"""

from autopilot.core.ops import broadcast, merge, select
from autopilot.core.types import Datum
from pathlib import Path
import pytest


def test_select_datum_first() -> None:
  """select(datum, 0) returns the item at index 0."""
  child_a = Datum(items=[])
  child_b = Datum(items=[])
  d = Datum(items=[child_a, child_b])
  result = select(d, 0)
  assert isinstance(result, Datum)


def test_select_old_order_helpful_error() -> None:
  """Calling old pattern select(0, datum) raises TypeError with migration hint."""
  d = Datum(items=[Datum()])
  with pytest.raises(TypeError, match=r'(?i)argument order changed') as exc_info:
    select(0, d)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
  msg = str(exc_info.value).lower()
  assert 'argument order changed' in msg


def test_select_out_of_range() -> None:
  """select(datum, 99) raises IndexError when length is insufficient."""
  d = Datum(items=[Datum(), Datum()])
  with pytest.raises(IndexError, match='out of range'):
    select(d, 99)


def test_broadcast_datum_first() -> None:
  """broadcast(datum, 3) is already data-first; result has 3 items."""
  d = Datum(items=[Datum()])
  result = broadcast(d, 3)
  assert len(result.items) == 3


def test_merge_datums_only() -> None:
  """merge(d1, d2) accepts only datum operands and concatenates items."""
  d1 = Datum(items=[Datum()])
  d2 = Datum(items=[Datum(), Datum()])
  result = merge(d1, d2)
  assert len(result.items) == 3


def test_operator_convention_documented() -> None:
  """CLAUDE.md contains explicit argument order / data-first section."""
  path = Path(__file__).resolve().parents[2] / 'CLAUDE.md'
  content = path.read_text(encoding='utf-8')
  assert 'select(datum, index)' in content
  assert 'data-first' in content.lower() or 'Data-first' in content


def test_all_operators_have_docstring() -> None:
  """Every public functional operator from core/ops.py has a non-empty docstring."""
  from autopilot.core import ops

  public_functions = [
    'merge',
    'select',
    'clone',
    'identity',
    'detach',
    'broadcast',
    'reduce',
    'scale_grad',
    'transform_grad',
    'attribution',
  ]
  for name in public_functions:
    fn = getattr(ops, name)
    assert fn.__doc__ is not None, f'{name} has no docstring'
    assert len(fn.__doc__.strip()) > 0, f'{name} has empty docstring'


def test_select_non_integer_index_error() -> None:
  """select(datum, 'a') raises TypeError (not IndexError)."""
  d = Datum(items=[Datum()])
  with pytest.raises(TypeError, match='int as second argument'):
    select(d, 'a')  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_select_non_datum_first_arg_error() -> None:
  """select(42, 0) raises TypeError with guidance referencing datum-first usage."""
  with pytest.raises(TypeError, match='Datum as first argument') as exc_info:
    select(42, 0)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
  msg = str(exc_info.value)
  assert 'select(datum, index)' in msg


def test_select_empty_datum() -> None:
  """select(Datum(items=[]), 0) raises IndexError (empty items)."""
  d = Datum(items=[])
  with pytest.raises(IndexError, match='empty'):
    select(d, 0)


def test_merge_single_datum() -> None:
  """merge(d1) returns d1's items unchanged (singleton merge)."""
  d = Datum(items=[Datum(), Datum()])
  result = merge(d)
  assert len(result.items) == 2
