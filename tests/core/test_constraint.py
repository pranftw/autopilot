"""Tests for ConstraintResult, gate_to_constraint, and Result.gates type change.

Covers: round-trip, constructor validation, from_dict, Result integration,
computed passed property, and old-format migration error.
"""

from autopilot.core.constraint import ConstraintResult
from autopilot.core.models import Result
import pytest


class TestConstraintResultRoundTrip:
  """Test 1: to_dict / from_dict identity."""

  def test_constraint_result_roundtrip(self) -> None:
    cr = ConstraintResult(
      name='MinGate',
      passed=True,
      metric='accuracy',
      value=0.9,
      threshold='>= 0.8',
      message=None,
    )
    d = cr.to_dict()
    cr2 = ConstraintResult.from_dict(d)
    assert cr2.name == cr.name
    assert cr2.passed == cr.passed
    assert cr2.metric == cr.metric
    assert cr2.value == cr.value
    assert cr2.threshold == cr.threshold
    assert cr2.message == cr.message


class TestResultGatesListOfConstraintResults:
  """Test 2: Result accepts list[ConstraintResult]."""

  def test_result_gates_list_of_constraint_results(self) -> None:
    cr = ConstraintResult(
      name='MinGate', passed=True, metric='accuracy', value=0.9, threshold='>= 0.8'
    )
    r = Result(metrics={'accuracy': 0.9}, gates=[cr])
    assert len(r.gates) == 1
    assert r.gates[0].name == 'MinGate'


class TestResultPassedAllPass:
  """Test 3: passed property True when all gates pass."""

  def test_result_passed_all_pass(self) -> None:
    gates = [
      ConstraintResult(name='g1', passed=True, metric='a', value=1.0, threshold='>= 0'),
      ConstraintResult(name='g2', passed=True, metric='b', value=0.5, threshold='>= 0'),
    ]
    r = Result(metrics={'a': 1.0, 'b': 0.5}, gates=gates)
    assert r.passed is True


class TestResultPassedOneFail:
  """Test 4: passed property False when any gate fails."""

  def test_result_passed_one_fail(self) -> None:
    gates = [
      ConstraintResult(name='g1', passed=True, metric='a', value=1.0, threshold='>= 0'),
      ConstraintResult(name='g2', passed=False, metric='b', value=0.1, threshold='>= 0.5'),
    ]
    r = Result(metrics={'a': 1.0, 'b': 0.1}, gates=gates)
    assert r.passed is False


class TestConstraintResultWithMessage:
  """Test 5: optional message serialized."""

  def test_constraint_result_with_message(self) -> None:
    cr = ConstraintResult(
      name='MaxGate',
      passed=False,
      metric='latency',
      value=150.0,
      threshold='<= 100',
      message='latency too high',
    )
    d = cr.to_dict()
    assert d['message'] == 'latency too high'
    cr2 = ConstraintResult.from_dict(d)
    assert cr2.message == 'latency too high'


class TestConstraintResultNoneValue:
  """Test 6: value=None allowed."""

  def test_constraint_result_none_value(self) -> None:
    cr = ConstraintResult(
      name='MinGate', passed=False, metric='missing', value=None, threshold='>= 0.5'
    )
    assert cr.value is None
    d = cr.to_dict()
    assert d['value'] is None
    cr2 = ConstraintResult.from_dict(d)
    assert cr2.value is None


class TestConstraintResultConstructorInvalidType:
  """Test 7: TypeError on bad types."""

  def test_constraint_result_constructor_invalid_type(self) -> None:
    with pytest.raises(TypeError, match='name must be str'):
      ConstraintResult(name=42, passed=True, metric='x', value=1.0, threshold='t')  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    with pytest.raises(TypeError, match='passed must be bool'):
      ConstraintResult(name='g', passed='yes', metric='x', value=1.0, threshold='t')  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    with pytest.raises(TypeError, match='metric must be str'):
      ConstraintResult(name='g', passed=True, metric=99, value=1.0, threshold='t')  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    with pytest.raises(TypeError, match='threshold must be str'):
      ConstraintResult(name='g', passed=True, metric='x', value=1.0, threshold=0.8)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


class TestConstraintResultValueValidation:
  """Value field accepts int, float, None; rejects bool and other types."""

  def test_value_string_rejected(self) -> None:
    with pytest.raises(TypeError, match=r'value must be float, int, or None.*got str'):
      ConstraintResult(name='g', passed=True, metric='m', value='oops', threshold='t')  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

  def test_value_bool_rejected(self) -> None:
    with pytest.raises(TypeError, match='got bool'):
      ConstraintResult(name='g', passed=True, metric='m', value=True, threshold='t')  # type: ignore[arg-type]

  def test_value_list_rejected(self) -> None:
    with pytest.raises(TypeError, match='got list'):
      ConstraintResult(name='g', passed=True, metric='m', value=[1.0], threshold='t')  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

  def test_value_int_accepted(self) -> None:
    cr = ConstraintResult(name='g', passed=True, metric='m', value=42, threshold='t')
    assert cr.value == 42

  def test_value_float_accepted(self) -> None:
    cr = ConstraintResult(name='g', passed=True, metric='m', value=0.95, threshold='t')
    assert cr.value == 0.95

  def test_value_zero_accepted(self) -> None:
    cr = ConstraintResult(name='g', passed=True, metric='m', value=0, threshold='t')
    assert cr.value == 0

  def test_value_negative_accepted(self) -> None:
    cr = ConstraintResult(name='g', passed=True, metric='m', value=-1.5, threshold='t')
    assert cr.value == -1.5

  def test_value_none_accepted(self) -> None:
    cr = ConstraintResult(name='g', passed=True, metric='m', value=None, threshold='t')
    assert cr.value is None


class TestConstraintResultMessageValidation:
  """Message field accepts str or None; rejects other types."""

  def test_message_int_rejected(self) -> None:
    with pytest.raises(TypeError, match=r'message must be str or None.*got int'):
      ConstraintResult(name='g', passed=True, metric='m', value=1.0, threshold='t', message=123)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

  def test_message_list_rejected(self) -> None:
    with pytest.raises(TypeError, match='got list'):
      ConstraintResult(name='g', passed=True, metric='m', value=1.0, threshold='t', message=['bad'])  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

  def test_message_string_accepted(self) -> None:
    cr = ConstraintResult(name='g', passed=True, metric='m', value=1.0, threshold='t', message='ok')
    assert cr.message == 'ok'

  def test_message_none_accepted(self) -> None:
    cr = ConstraintResult(name='g', passed=True, metric='m', value=1.0, threshold='t')
    assert cr.message is None

  def test_message_empty_string_accepted(self) -> None:
    cr = ConstraintResult(name='g', passed=True, metric='m', value=1.0, threshold='t', message='')
    assert cr.message is not None
    assert len(cr.message) == 0


class TestConstraintResultFromDictMissingKey:
  """Test 8: raise on incomplete dict with all missing keys reported."""

  def test_from_dict_single_missing_key(self) -> None:
    with pytest.raises(KeyError, match='name'):
      ConstraintResult.from_dict({'passed': True, 'metric': 'x', 'threshold': 't'})

  def test_from_dict_all_missing_reports_all(self) -> None:
    with pytest.raises(KeyError, match='name') as exc_info:
      ConstraintResult.from_dict({})
    err = str(exc_info.value)
    assert 'name' in err
    assert 'passed' in err
    assert 'metric' in err
    assert 'threshold' in err

  def test_from_dict_two_missing_reports_both(self) -> None:
    with pytest.raises(KeyError) as exc_info:
      ConstraintResult.from_dict({'metric': 'm', 'threshold': 't'})
    err = str(exc_info.value)
    assert 'name' in err
    assert 'passed' in err

  def test_from_dict_shows_provided_keys(self) -> None:
    with pytest.raises(KeyError, match='Provided keys') as exc_info:
      ConstraintResult.from_dict({'metric': 'm'})
    err = str(exc_info.value)
    assert "['metric']" in err


class TestOldGatesDictMigrationError:
  """Test 9: dict old-style raises TypeError."""

  def test_old_gates_dict_migration_error(self) -> None:
    with pytest.raises(TypeError, match='list\\[ConstraintResult\\]'):
      Result(gates={'accuracy': 'pass'})  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

  def test_from_dict_rejects_nonempty_dict_gates(self) -> None:
    """from_dict raises TypeError for non-empty legacy dict gates."""
    with pytest.raises(TypeError, match=r'list\[ConstraintResult\]'):
      Result.from_dict({'metrics': {'x': 1.0}, 'gates': {'accuracy': 'pass'}})

  def test_from_dict_rejects_empty_dict_gates(self) -> None:
    """from_dict raises TypeError for empty dict gates (consistency with constructor)."""
    with pytest.raises(TypeError, match=r'list\[ConstraintResult\]'):
      Result.from_dict({'metrics': {'x': 1.0}, 'gates': {}})


class TestResultEmptyGatesPassed:
  """Test 10: Result(gates=[]) has passed=True (vacuous truth)."""

  def test_result_empty_gates_passed(self) -> None:
    r = Result(gates=[])
    assert r.passed is True
    r2 = Result()
    assert r2.passed is True
