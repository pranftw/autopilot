from autopilot.ai.evaluation.schemas import ConversationTurn, DataItem
from autopilot.core.types import Datum
from autopilot.data.dataloader import DataLoader
from autopilot.data.dataset import IterableDataset, ListDataset
from pydantic import BaseModel
import pytest
import random


class _StubCustom(BaseModel):
  x: str


def _make_eval_item(id: str) -> DataItem[_StubCustom]:
  return DataItem(
    id=id,
    turns=[ConversationTurn(role='user', content='hi')],
    custom=_StubCustom(x='y'),
  )


def _make_datums(n: int) -> list[Datum]:
  return [Datum(metadata={'idx': i}) for i in range(n)]


class _DatumIterable(IterableDataset[Datum]):
  def __init__(self, n: int) -> None:
    self._n = n

  def __iter__(self):
    for i in range(self._n):
      yield Datum(metadata={'idx': i})


def test_batch_size_one_each_yield_is_datum():
  loader = DataLoader(_make_datums(2), batch_size=1, shuffle=False)
  out = list(loader)
  assert len(out) == 2
  assert all(isinstance(d, Datum) for d in out)


def test_batch_size_three_seven_items():
  loader = DataLoader(_make_datums(7), batch_size=3, shuffle=False)
  batches = list(loader)
  assert len(batches) == 3
  assert [d.metadata['idx'] for d in batches[0].items] == [0, 1, 2]
  assert [d.metadata['idx'] for d in batches[1].items] == [3, 4, 5]
  assert batches[2].metadata['idx'] == 6


def test_batch_size_three_drop_last():
  loader = DataLoader(_make_datums(7), batch_size=3, shuffle=False, drop_last=True)
  batches = list(loader)
  assert len(batches) == 2


def test_batch_larger_than_dataset():
  loader = DataLoader(_make_datums(2), batch_size=10, shuffle=False)
  batches = list(loader)
  assert len(batches) == 1
  assert len(batches[0].items) == 2


def test_shuffle_true_reproducible():
  data = _make_datums(10)
  random.seed(123)
  first = [b.metadata['idx'] for b in DataLoader(data, batch_size=1, shuffle=True)]
  random.seed(123)
  second = [b.metadata['idx'] for b in DataLoader(data, batch_size=1, shuffle=True)]
  assert first == second
  assert first != list(range(10))


def test_shuffle_false_insertion_order():
  data = [Datum(metadata={'val': v}) for v in [3, 1, 4]]
  loader = DataLoader(data, batch_size=1, shuffle=False)
  vals = [b.metadata['val'] for b in loader]
  assert vals == [3, 1, 4]


def test_len_drop_last_false():
  assert len(DataLoader(_make_datums(7), batch_size=3, drop_last=False)) == 3


def test_len_drop_last_true():
  assert len(DataLoader(_make_datums(7), batch_size=3, drop_last=True)) == 2


def test_empty_dataset():
  loader = DataLoader([], batch_size=3, shuffle=False)
  assert len(loader) == 0
  assert list(loader) == []


def test_plain_list_wrapped():
  loader = DataLoader(_make_datums(2), batch_size=2, shuffle=False)
  batches = list(loader)
  assert len(batches) == 1
  assert [d.metadata['idx'] for d in batches[0].items] == [0, 1]


def test_list_of_datum_collated():
  a = Datum(split='train')
  b = Datum(split='val')
  loader = DataLoader([a, b], batch_size=2, shuffle=False)
  batches = list(loader)
  assert len(batches) == 1
  assert len(batches[0].items) == 2
  assert batches[0].items[0].split == 'train'


def test_list_of_dicts_collated():
  loader = DataLoader([{'split': 'x', 'epoch': 2}], batch_size=1, shuffle=False)
  d = next(iter(loader))
  assert d.split == 'x'
  assert d.epoch == 2


def test_iterable_dataset_batching():
  loader = DataLoader(_DatumIterable(7), batch_size=3, shuffle=False)
  batches = list(loader)
  assert len(batches) == 3
  assert [d.metadata['idx'] for d in batches[0].items] == [0, 1, 2]


def test_iterable_dataloader_len_raises():
  with pytest.raises(TypeError, match='IterableDataset'):
    len(DataLoader(_DatumIterable(5), batch_size=1))


def test_custom_collate_fn():
  seen: list[list] = []

  def collate(batch: list):
    seen.append(batch)
    return Datum(metadata={'n': len(batch)})

  loader = DataLoader([1, 2, 3], batch_size=2, shuffle=False, collate_fn=collate)
  batches = list(loader)
  assert seen == [[1, 2], [3]]
  assert [b.metadata['n'] for b in batches] == [2, 1]


def test_works_with_list_dataset():
  items = [_make_eval_item('a'), _make_eval_item('b')]
  ds = ListDataset(items)

  def collate(batch: list):
    datum_items = [Datum(metadata={'raw': item}) for item in batch]
    if len(datum_items) == 1:
      return datum_items[0]
    return Datum(items=datum_items)

  loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate)
  batches = list(loader)
  assert len(batches) == 1
  raw_items = [d.metadata['raw'] for d in batches[0].items]
  assert raw_items[0].id == 'a'
  assert raw_items[1].id == 'b'


def test_non_datum_non_dict_raises_type_error():
  loader = DataLoader([42], batch_size=1, shuffle=False)
  with pytest.raises(TypeError, match='expected Datum or dict'):
    list(loader)


def test_iterable_length_hint():
  loader = DataLoader(_DatumIterable(10), batch_size=3, length_hint=10)
  assert len(loader) == 4
