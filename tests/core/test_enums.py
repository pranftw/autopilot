"""Tests for core/enums.py -- Status enum lifecycle states."""

from autopilot.core.enums import Status
from autopilot.core.serialization import DictMixin, serialize
from dataclasses import dataclass
import pytest


class TestStatus:
  def test_all_six_members_exist(self):
    assert hasattr(Status, 'pending')
    assert hasattr(Status, 'running')
    assert hasattr(Status, 'completed')
    assert hasattr(Status, 'failed')
    assert hasattr(Status, 'cancelled')
    assert hasattr(Status, 'invalidated')

  def test_member_count(self):
    assert len(Status) == 6

  def test_values_are_lowercase_strings(self):
    for member in Status:
      assert member.value == member.name
      assert member.value.islower()

  def test_construct_from_value(self):
    assert Status('pending') is Status.pending
    assert Status('running') is Status.running
    assert Status('completed') is Status.completed
    assert Status('failed') is Status.failed
    assert Status('cancelled') is Status.cancelled
    assert Status('invalidated') is Status.invalidated

  def test_invalid_value_raises(self):
    with pytest.raises(ValueError, match=r"'invalid' is not a valid Status"):
      Status('invalid')

  def test_invalid_value_unknown_raises(self):
    with pytest.raises(ValueError, match=r"'PENDING' is not a valid Status"):
      Status('PENDING')

  def test_iterate(self):
    members = list(Status)
    assert len(members) == 6
    assert Status.pending in members
    assert Status.cancelled in members
    assert Status.invalidated in members

  def test_compare_as_enum(self):
    assert Status.pending == Status.pending
    assert Status.pending != Status.running


class TestStatusProperties:
  @pytest.mark.parametrize(
    ('member', 'expected_terminal', 'expected_active'),
    [
      (Status.pending, False, False),
      (Status.running, False, True),
      (Status.completed, True, False),
      (Status.failed, True, False),
      (Status.cancelled, True, False),
      (Status.invalidated, True, False),
    ],
  )
  def test_is_terminal_and_is_active(self, member, expected_terminal, expected_active):
    assert member.is_terminal is expected_terminal
    assert member.is_active is expected_active

  def test_terminal_members_exhaustive(self):
    terminal = [s for s in Status if s.is_terminal]
    assert set(terminal) == {Status.completed, Status.failed, Status.cancelled, Status.invalidated}

  def test_active_members_exhaustive(self):
    active = [s for s in Status if s.is_active]
    assert active == [Status.running]


class TestEnumStrBehavior:
  def test_str_equality(self):
    assert Status.pending == 'pending'
    assert Status.running == 'running'
    assert Status.completed == 'completed'
    assert Status.failed == 'failed'
    assert Status.cancelled == 'cancelled'

  def test_str_function(self):
    result = str(Status.pending)
    assert 'pending' in result

  def test_fstring_interpolation(self):
    msg = f'status is {Status.running}'
    assert 'running' in msg

  def test_in_string_comparison(self):
    statuses = ['pending', 'running']
    assert Status.pending in statuses
    assert Status.completed not in statuses


class TestEnumSerialization:
  def testserialize_status_to_value(self):
    assert serialize(Status.pending) == 'pending'
    assert serialize(Status.completed) == 'completed'

  def testserialize_status_in_list(self):
    result = serialize([Status.pending, Status.running])
    assert result == ['pending', 'running']

  def testserialize_status_in_dict(self):
    result = serialize({'s': Status.failed})
    assert result == {'s': 'failed'}


class TestEnumInDictMixin:
  def test_dataclass_with_status_round_trip(self):
    @dataclass
    class Record(DictMixin):
      name: str
      status: Status

    rec = Record(name='exp-1', status=Status.running)
    d = rec.to_dict()
    assert d == {'name': 'exp-1', 'status': 'running'}

    rec2 = Record.from_dict({'name': 'exp-1', 'status': 'running'})
    assert rec2.status == 'running'

  def test_dataclass_with_status_none_default(self):
    @dataclass
    class NullableRecord(DictMixin):
      name: str
      status: Status | None = None

    rec = NullableRecord(name='test')
    d = rec.to_dict()
    assert d == {'name': 'test', 'status': None}
