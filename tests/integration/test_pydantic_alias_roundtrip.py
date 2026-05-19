"""Integration test for Pydantic alias serialization round-trip fidelity.

Verifies that models with Field(alias=...) and populate_by_name=True
correctly serialize and deserialize using both Python names and alias names
at JSON boundaries.
"""

from autopilot.ai.evaluation.schemas import (
  CheckpointEvent,
  CheckpointHeader,
  ConversationTurn,
  DataItem,
  JudgeInput,
  JudgeResult,
  JudgeVerdict,
)
from pydantic import BaseModel
import pytest


class DictCustom(BaseModel):
  """Concrete custom payload for generic schema tests."""

  extra: str = ''


class ConcreteDataItem(DataItem[DictCustom]):
  """Concrete DataItem for round-trip tests."""


class ConcreteJudgeInput(JudgeInput[DictCustom]):
  """Concrete JudgeInput for round-trip tests."""


class ConcreteJudgeResult(JudgeResult[DictCustom]):
  """Concrete JudgeResult for round-trip tests."""


class TestCheckpointEventAliasRoundtrip:
  """CheckpointEvent uses alias='type' and alias='id'."""

  def test_construct_with_alias_names(self) -> None:
    event = CheckpointEvent(type='result', id='item-1', timestamp='2024-01-01T00:00:00')
    assert event.event_kind == 'result'
    assert event.item_id == 'item-1'

  def test_construct_with_python_names(self) -> None:
    event = CheckpointEvent.model_validate(
      {
        'event_kind': 'error',
        'item_id': 'item-2',
        'timestamp': '2024-01-01T00:00:00',
      }
    )
    assert event.event_kind == 'error'
    assert event.item_id == 'item-2'

  def test_dump_by_alias_uses_wire_keys(self) -> None:
    event = CheckpointEvent(type='skip', id='item-3', timestamp='2024-01-01T00:00:00')
    dumped = event.model_dump(by_alias=True)
    assert 'type' in dumped
    assert 'id' in dumped
    assert dumped['type'] == 'skip'
    assert dumped['id'] == 'item-3'

  def test_roundtrip_alias_keys(self) -> None:
    original = CheckpointEvent(
      type='result', id='item-4', timestamp='2024-01-01T00:00:00', payload={'score': 0.95}
    )
    dumped = original.model_dump(by_alias=True)
    reconstructed = CheckpointEvent.model_validate(dumped)
    assert reconstructed.model_dump(by_alias=True) == dumped

  def test_roundtrip_python_keys(self) -> None:
    original = CheckpointEvent.model_validate(
      {
        'event_kind': 'result',
        'item_id': 'item-5',
        'timestamp': '2024-01-01T00:00:00',
      }
    )
    dumped = original.model_dump()
    reconstructed = CheckpointEvent.model_validate(dumped)
    assert reconstructed.event_kind == original.event_kind
    assert reconstructed.item_id == original.item_id


class TestCheckpointHeaderAliasRoundtrip:
  """CheckpointHeader uses alias='type' for checkpoint_type."""

  def test_construct_with_alias(self) -> None:
    header = CheckpointHeader(
      type='header', subsystem='generator', config_hash='abc123', created_at='2024-01-01'
    )
    assert header.checkpoint_type == 'header'

  def test_dump_by_alias(self) -> None:
    header = CheckpointHeader(subsystem='judge', config_hash='def456', created_at='2024-01-01')
    dumped = header.model_dump(by_alias=True)
    assert 'type' in dumped
    assert dumped['type'] == 'header'

  def test_roundtrip(self) -> None:
    original = CheckpointHeader(
      subsystem='generator',
      config_hash='xyz789',
      created_at='2024-06-15T10:00:00',
      args={'model': 'test'},
    )
    dumped = original.model_dump(by_alias=True)
    reconstructed = CheckpointHeader.model_validate(dumped)
    assert reconstructed.model_dump(by_alias=True) == dumped


class TestDataItemAliasRoundtrip:
  """DataItem uses alias='id' for item_id."""

  def test_construct_with_alias(self) -> None:
    item = ConcreteDataItem(id='d-1', turns=[], custom=DictCustom())
    assert item.item_id == 'd-1'

  def test_dump_by_alias(self) -> None:
    item = ConcreteDataItem(id='d-2', turns=[], custom=DictCustom())
    dumped = item.model_dump(by_alias=True)
    assert 'id' in dumped
    assert dumped['id'] == 'd-2'

  def test_roundtrip_nested(self) -> None:
    item = ConcreteDataItem(
      id='d-3',
      turns=[ConversationTurn(role='user', content='hello')],
      split='train',
      custom=DictCustom(extra='ground_truth'),
    )
    dumped = item.model_dump(by_alias=True)
    reconstructed = ConcreteDataItem.model_validate(dumped)
    assert reconstructed.model_dump(by_alias=True) == dumped


class TestJudgeInputAliasRoundtrip:
  """JudgeInput uses alias='id' for item_id."""

  def test_roundtrip(self) -> None:
    inp = ConcreteJudgeInput(
      id='j-1',
      turns=[ConversationTurn(role='assistant', content='response')],
      custom=DictCustom(),
    )
    dumped = inp.model_dump(by_alias=True)
    reconstructed = ConcreteJudgeInput.model_validate(dumped)
    assert reconstructed.model_dump(by_alias=True) == dumped


class TestJudgeResultAliasRoundtrip:
  """JudgeResult uses alias='id' for item_id."""

  def test_roundtrip_with_verdict(self) -> None:
    verdict = JudgeVerdict(category='correct', rationale='good answer', confidence=0.9)
    result = ConcreteJudgeResult(id='r-1', verdict=verdict, custom=DictCustom())
    dumped = result.model_dump(by_alias=True)
    reconstructed = ConcreteJudgeResult.model_validate(dumped)
    assert reconstructed.model_dump(by_alias=True) == dumped

  def test_roundtrip_without_verdict(self) -> None:
    result = ConcreteJudgeResult(id='r-2', verdict=None, custom=DictCustom(extra='data'))
    dumped = result.model_dump(by_alias=True)
    assert dumped['id'] == 'r-2'
    reconstructed = ConcreteJudgeResult.model_validate(dumped)
    assert reconstructed.item_id == 'r-2'


@pytest.mark.parametrize(
  ('model_cls', 'alias_kwargs', 'python_kwargs'),
  [
    (
      CheckpointEvent,
      {'type': 'result', 'id': 'p-1', 'timestamp': 'now'},
      {'event_kind': 'result', 'item_id': 'p-1', 'timestamp': 'now'},
    ),
    (
      CheckpointHeader,
      {'type': 'header', 'subsystem': 'gen', 'config_hash': 'h', 'created_at': 'now'},
      {'checkpoint_type': 'header', 'subsystem': 'gen', 'config_hash': 'h', 'created_at': 'now'},
    ),
  ],
)
def test_alias_and_python_name_equivalence(
  model_cls: type[BaseModel], alias_kwargs: dict, python_kwargs: dict
) -> None:
  """Both construction paths produce identical serialized output."""
  from_alias = model_cls.model_validate(alias_kwargs)
  from_python = model_cls.model_validate(python_kwargs)
  assert from_alias.model_dump(by_alias=True) == from_python.model_dump(by_alias=True)
