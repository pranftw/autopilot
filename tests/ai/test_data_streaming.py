"""Tests for StreamingDataset."""

from autopilot.ai.models import ConversationTurn, DataItem
from autopilot.data.dataloader import DataLoader
from autopilot.data.dataset import IterableDataset, StreamingDataset
from pydantic import BaseModel


class _Custom(BaseModel):
  value: int


def _make_item(id: str, value: int) -> DataItem[_Custom]:
  return DataItem(
    id=id,
    turns=[ConversationTurn(role='user', content='hi')],
    custom=_Custom(value=value),
  )


def _write_items(path, items):
  with path.open('w', encoding='utf-8') as f:
    for item in items:
      f.write(item.model_dump_json() + '\n')


def test_streaming_reads_lazily(tmp_path):
  path = tmp_path / 'data.jsonl'
  items = [_make_item(f'item_{i}', i) for i in range(10)]
  _write_items(path, items)

  ds = StreamingDataset(path, DataItem[_Custom])
  result = list(ds)
  assert len(result) == 10
  assert result[0].id == 'item_0'
  assert result[9].custom.value == 9


def test_streaming_is_iterable_dataset(tmp_path):
  path = tmp_path / 'data.jsonl'
  items = [_make_item('a', 1), _make_item('b', 2)]
  _write_items(path, items)

  ds = StreamingDataset(path, DataItem[_Custom])
  assert isinstance(ds, IterableDataset)
  result = list(ds)
  assert len(result) == 2


def test_streaming_with_dataloader(tmp_path):
  path = tmp_path / 'data.jsonl'
  items = [_make_item(f'i{i}', i) for i in range(6)]
  _write_items(path, items)

  ds = StreamingDataset(path, DataItem[_Custom])
  loader = DataLoader(ds, batch_size=2)
  batches = list(loader)
  assert len(batches) == 3


def test_streaming_empty_file(tmp_path):
  path = tmp_path / 'empty.jsonl'
  path.write_text('')

  ds = StreamingDataset(path, DataItem[_Custom])
  result = list(ds)
  assert result == []


def test_streaming_skips_blank_lines(tmp_path):
  path = tmp_path / 'data.jsonl'
  item = _make_item('x', 42)
  path.write_text(f'\n{item.model_dump_json()}\n\n')

  ds = StreamingDataset(path, DataItem[_Custom])
  result = list(ds)
  assert len(result) == 1
  assert result[0].id == 'x'


def test_streaming_multiple_iterations(tmp_path):
  path = tmp_path / 'data.jsonl'
  items = [_make_item('a', 1), _make_item('b', 2)]
  _write_items(path, items)

  ds = StreamingDataset(path, DataItem[_Custom])
  first = list(ds)
  second = list(ds)
  assert first == second
