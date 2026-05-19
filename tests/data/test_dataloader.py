"""Tests for DataLoader with sampler-based ordering."""

from autopilot.ai.evaluation.schemas import ConversationTurn, DataItem
from autopilot.core.types import Datum, EvalDatum
from autopilot.data.dataloader import DataLoader
from autopilot.data.dataset import ListDataset
from autopilot.data.sampler import (
  BatchSampler,
  RandomSampler,
  SequentialSampler,
  SubsetSampler,
)
from pydantic import BaseModel
from tests.doubles import EvalDatumIterable
from typing import Any, cast
import inspect
import pytest
import random


class _StubCustom(BaseModel):
  x: str


def _make_eval_item(id_: str) -> DataItem[_StubCustom]:
  return DataItem(
    id=id_,
    turns=[ConversationTurn(role='user', content='hi')],
    custom=_StubCustom(x='y'),
  )


def _make_datums(n: int) -> list[EvalDatum]:
  return [EvalDatum(metadata={'idx': i}) for i in range(n)]


# 4.2 DataLoader sampler and batching


def test_dataloader_init_rejects_shuffle_kwarg():
  """DataLoader(ds, shuffle=True) raises TypeError (removed parameter)."""
  kwargs: dict[str, Any] = {'shuffle': True}
  with pytest.raises(TypeError, match='shuffle'):
    DataLoader(_make_datums(3), **kwargs)


def test_dataloader_batch_size_none_raises_value_error():
  """Explicit batch_size=None."""
  with pytest.raises(ValueError, match='batch_size'):
    DataLoader(_make_datums(3), batch_size=cast(Any, None))


def test_dataloader_batch_size_zero_raises_value_error():
  with pytest.raises(ValueError, match='batch_size'):
    DataLoader(_make_datums(3), batch_size=0)


def test_dataloader_batch_size_negative_raises_value_error():
  with pytest.raises(ValueError, match='batch_size'):
    DataLoader(_make_datums(3), batch_size=-1)


def test_dataloader_batch_size_non_int_raises_value_error():
  """e.g. batch_size='2' raises ValueError."""
  with pytest.raises(ValueError, match='batch_size'):
    DataLoader(_make_datums(3), batch_size=cast(Any, '2'))


def test_dataloader_default_sampler_is_sequential():
  """Map dataset order matches index order without passing sampler."""
  data = _make_datums(5)
  loader = DataLoader(data, batch_size=1)
  indices = []
  for batch in loader:
    assert isinstance(batch, Datum)
    assert len(batch.items) == 1
    item = batch.items[0]
    assert isinstance(item, EvalDatum)
    indices.append(item.metadata['idx'])
  assert indices == [0, 1, 2, 3, 4]


def test_dataloader_explicit_sequential_sampler():
  """Explicit SequentialSampler matches baseline order."""
  data = _make_datums(4)
  ds = ListDataset(cast(Any, data))
  sampler = SequentialSampler(ds)
  loader = DataLoader(ds, batch_size=1, sampler=sampler)
  indices = []
  for batch in loader:
    item = batch.items[0]
    assert isinstance(item, EvalDatum)
    indices.append(item.metadata['idx'])
  assert indices == [0, 1, 2, 3]


def test_dataloader_random_sampler_deterministic():
  """Same RandomSampler(generator=rng) + set_epoch yields reproducible batches."""
  data = _make_datums(10)
  ds = ListDataset(cast(Any, data))

  def run_with_seed(seed: int) -> list[int]:
    rng = random.Random(seed)
    sampler = RandomSampler(ds, generator=rng)
    sampler.set_epoch(0)
    loader = DataLoader(ds, batch_size=1, sampler=sampler)
    indices = []
    for batch in loader:
      item = batch.items[0]
      assert isinstance(item, EvalDatum)
      indices.append(item.metadata['idx'])
    return indices

  first = run_with_seed(42)
  second = run_with_seed(42)
  assert first == second
  assert first != list(range(10))


def test_dataloader_batch_sampler_yields_expected_cardinality():
  """Length and batch count match BatchSampler rules."""
  data = _make_datums(7)
  ds = ListDataset(cast(Any, data))
  sampler = SequentialSampler(ds)
  batch_sampler = BatchSampler(sampler, batch_size=3, drop_last=False)
  loader = DataLoader(ds, batch_sampler=batch_sampler)
  batches = list(loader)
  assert len(batches) == 3
  assert len(loader) == 3


def test_dataloader_batch_sampler_mutual_exclusion_ignore_drop_last():
  """With batch_sampler, batch formation follows the sampler, not loader's drop_last."""
  data = _make_datums(7)
  ds = ListDataset(cast(Any, data))
  sampler = SequentialSampler(ds)
  batch_sampler = BatchSampler(sampler, batch_size=3, drop_last=True)
  loader = DataLoader(ds, batch_sampler=batch_sampler, drop_last=False)
  batches = list(loader)
  assert len(batches) == 2


def test_dataloader_iterable_with_batch_sampler_raises_type_error():
  """batch_sampler + IterableDataset raises TypeError."""
  ds = EvalDatumIterable(5)
  sampler = SequentialSampler(ListDataset(cast(Any, _make_datums(5))))
  batch_sampler = BatchSampler(sampler, batch_size=2)
  with pytest.raises(TypeError, match='batch_sampler cannot be used with IterableDataset'):
    DataLoader(ds, batch_sampler=batch_sampler)


def test_dataloader_list_input_wrapped_dataset():
  """list still wraps to ListDataset and batches correctly."""
  loader = DataLoader(_make_datums(3), batch_size=2)
  batches = list(loader)
  assert len(batches) == 2
  assert len(batches[0].items) == 2
  assert len(batches[1].items) == 1


# 4.3 Collate


def test_default_collate_single_datum_wrapped():
  """One Datum input -> Datum(items=[that]), not bare object."""
  loader = DataLoader([EvalDatum(metadata={'k': 'v'})], batch_size=1)
  result = list(loader)
  assert len(result) == 1
  batch = result[0]
  assert isinstance(batch, Datum)
  assert len(batch.items) == 1
  inner = batch.items[0]
  assert isinstance(inner, EvalDatum)
  assert inner.metadata['k'] == 'v'


def test_default_collate_multi_items():
  """len(items) == batch_size."""
  loader = DataLoader(_make_datums(4), batch_size=2)
  batches = list(loader)
  assert len(batches) == 2
  assert all(len(b.items) == 2 for b in batches)


def test_default_collate_dict_row_promotes_to_datum():
  """Existing dict path still works inside wrapped batch."""
  loader = DataLoader([{'split': 'x', 'epoch': 2}], batch_size=1)
  batch = next(iter(loader))
  assert isinstance(batch, Datum)
  assert len(batch.items) == 1
  item = batch.items[0]
  assert isinstance(item, EvalDatum)
  assert item.split == 'x'
  assert item.epoch == 2


# 4.6 Regression / migration coverage


def test_iterable_dataloader_batches_without_shuffle_flag():
  """Iterable path still batches correctly after shuffle removal."""
  loader = DataLoader(EvalDatumIterable(5), batch_size=2)
  batches = list(loader)
  assert len(batches) == 3
  assert len(batches[0].items) == 2
  assert len(batches[2].items) == 1


def test_signature_inspection_no_length_hint():
  """length_hint absent from DataLoader.__init__ signature."""
  sig = inspect.signature(DataLoader.__init__)
  assert 'length_hint' not in sig.parameters


def test_map_dataloader_len_matches_math():
  """Map-style length matches ceil(n/batch) or floor when drop_last."""
  data = _make_datums(7)
  loader_no_drop = DataLoader(data, batch_size=3)
  assert len(loader_no_drop) == 3  # ceil(7/3) = 3

  loader_drop = DataLoader(data, batch_size=3, drop_last=True)
  assert len(loader_drop) == 2  # 7 // 3 = 2


def test_subset_sampler_integration_dataloader():
  """SubsetSampler + DataLoader batches expected indices."""
  data = _make_datums(10)
  ds = ListDataset(cast(Any, data))
  subset = SubsetSampler([2, 5, 8])
  loader = DataLoader(ds, batch_size=2, sampler=subset)
  batches = list(loader)
  assert len(batches) == 2

  indices = []
  for batch in batches:
    for item in batch.items:
      assert isinstance(item, EvalDatum)
      indices.append(item.metadata['idx'])
  assert indices == [2, 5, 8]


# Existing tests (updated for new API)


def test_batch_size_one_each_yield_is_datum():
  loader = DataLoader(_make_datums(2), batch_size=1)
  out = list(loader)
  assert len(out) == 2
  assert all(isinstance(d, Datum) for d in out)


def test_batch_size_three_seven_items():
  loader = DataLoader(_make_datums(7), batch_size=3)
  batches = list(loader)
  assert len(batches) == 3
  idxs0: list[int] = []
  for d in batches[0].items:
    assert isinstance(d, EvalDatum)
    idxs0.append(d.metadata['idx'])
  assert idxs0 == [0, 1, 2]
  idxs1: list[int] = []
  for d in batches[1].items:
    assert isinstance(d, EvalDatum)
    idxs1.append(d.metadata['idx'])
  assert idxs1 == [3, 4, 5]
  last_batch = batches[2]
  assert isinstance(last_batch, Datum)
  assert len(last_batch.items) == 1
  assert isinstance(last_batch.items[0], EvalDatum)
  assert last_batch.items[0].metadata['idx'] == 6


def test_batch_size_three_drop_last():
  loader = DataLoader(_make_datums(7), batch_size=3, drop_last=True)
  batches = list(loader)
  assert len(batches) == 2


def test_batch_larger_than_dataset():
  loader = DataLoader(_make_datums(2), batch_size=10)
  batches = list(loader)
  assert len(batches) == 1
  assert len(batches[0].items) == 2


def test_len_drop_last_false():
  assert len(DataLoader(_make_datums(7), batch_size=3, drop_last=False)) == 3


def test_len_drop_last_true():
  assert len(DataLoader(_make_datums(7), batch_size=3, drop_last=True)) == 2


def test_empty_dataset():
  loader = DataLoader([], batch_size=3)
  assert len(loader) == 0
  assert list(loader) == []


def test_plain_list_wrapped():
  loader = DataLoader(_make_datums(2), batch_size=2)
  batches = list(loader)
  assert len(batches) == 1
  idxs: list[int] = []
  for d in batches[0].items:
    assert isinstance(d, EvalDatum)
    idxs.append(d.metadata['idx'])
  assert idxs == [0, 1]


def test_list_of_datum_collated():
  a = EvalDatum(split='train')
  b = EvalDatum(split='val')
  loader = DataLoader([a, b], batch_size=2)
  batches = list(loader)
  assert len(batches) == 1
  assert len(batches[0].items) == 2
  first_item = batches[0].items[0]
  assert isinstance(first_item, EvalDatum)
  assert first_item.split == 'train'


def test_iterable_dataset_batching():
  loader = DataLoader(EvalDatumIterable(7), batch_size=3)
  batches = list(loader)
  assert len(batches) == 3
  idxs: list[int] = []
  for d in batches[0].items:
    assert isinstance(d, EvalDatum)
    idxs.append(d.metadata['idx'])
  assert idxs == [0, 1, 2]


def test_iterable_dataloader_len_raises():
  with pytest.raises(TypeError, match='IterableDataset'):
    len(DataLoader(EvalDatumIterable(5), batch_size=1))


def test_custom_collate_fn():
  seen: list[list] = []

  def collate(batch: list):
    seen.append(batch)
    return EvalDatum(metadata={'n': len(batch)})

  loader = DataLoader([1, 2, 3], batch_size=2, collate_fn=collate)
  batches = list(loader)
  assert seen == [[1, 2], [3]]
  ns: list[int] = []
  for b in batches:
    assert isinstance(b, EvalDatum)
    ns.append(b.metadata['n'])
  assert ns == [2, 1]


def test_works_with_list_dataset():
  items = [_make_eval_item('a'), _make_eval_item('b')]
  ds = ListDataset(items)

  def collate(batch: list):
    datum_items = [EvalDatum(metadata={'raw': item}) for item in batch]
    if len(datum_items) == 1:
      return datum_items[0]
    return Datum(items=cast(Any, datum_items))

  loader = DataLoader(ds, batch_size=2, collate_fn=collate)
  batches = list(loader)
  assert len(batches) == 1
  raw_items: list = []
  for d in batches[0].items:
    assert isinstance(d, EvalDatum)
    raw_items.append(d.metadata['raw'])
  assert raw_items[0].item_id == 'a'
  assert raw_items[1].item_id == 'b'


def test_non_datum_non_dict_raises_type_error():
  loader = DataLoader([42], batch_size=1)
  with pytest.raises(TypeError, match='expected Datum or dict'):
    list(loader)


def test_tuple_dataset_raises_type_error():
  with pytest.raises(TypeError, match='dataset must be a Dataset or list, got tuple'):
    DataLoader(cast(Any, tuple(_make_datums(3))), batch_size=1)


def test_range_dataset_raises_type_error():
  with pytest.raises(TypeError, match='dataset must be a Dataset or list, got range'):
    DataLoader(cast(Any, range(5)), batch_size=1)


def test_collate_fn_typed_invoke():
  loader = DataLoader(ListDataset([Datum(), Datum()]), batch_size=2)
  batch = next(iter(loader))
  assert isinstance(batch, Datum)
  assert len(batch.items) == 2


class TestCollateEvalDatumVsDatum:
  """Verify _default_collate classifies dicts into EvalDatum vs Datum correctly."""

  def test_dict_with_success_key_becomes_eval_datum(self) -> None:
    loader = DataLoader([{'success': True}], batch_size=1)
    batch = next(iter(loader))
    assert isinstance(batch.items[0], EvalDatum)
    assert batch.items[0].success is True

  def test_dict_with_metrics_key_becomes_eval_datum(self) -> None:
    loader = DataLoader([{'metrics': {'acc': 0.9}}], batch_size=1)
    batch = next(iter(loader))
    assert isinstance(batch.items[0], EvalDatum)

  def test_dict_with_feedback_key_becomes_eval_datum(self) -> None:
    loader = DataLoader([{'feedback': 'good'}], batch_size=1)
    batch = next(iter(loader))
    assert isinstance(batch.items[0], EvalDatum)

  def test_dict_with_error_message_key_becomes_eval_datum(self) -> None:
    loader = DataLoader([{'error_message': 'fail'}], batch_size=1)
    batch = next(iter(loader))
    assert isinstance(batch.items[0], EvalDatum)

  def test_dict_with_only_epoch_becomes_eval_datum(self) -> None:
    loader = DataLoader([{'epoch': 3}], batch_size=1)
    batch = next(iter(loader))
    assert isinstance(batch.items[0], EvalDatum)
    assert batch.items[0].epoch == 3

  def test_dict_with_only_metadata_becomes_eval_datum(self) -> None:
    loader = DataLoader([{'metadata': {'k': 'v'}}], batch_size=1)
    batch = next(iter(loader))
    assert isinstance(batch.items[0], EvalDatum)
    assert batch.items[0].metadata == {'k': 'v'}

  def test_dict_with_only_split_becomes_eval_datum(self) -> None:
    loader = DataLoader([{'split': 'train'}], batch_size=1)
    batch = next(iter(loader))
    assert isinstance(batch.items[0], EvalDatum)
    assert batch.items[0].split == 'train'

  def test_dict_without_eval_keys_becomes_datum(self) -> None:
    loader = DataLoader([{'items': []}], batch_size=1)
    batch = next(iter(loader))
    item = batch.items[0]
    assert type(item) is Datum

  def test_empty_dict_becomes_datum(self) -> None:
    loader = DataLoader([{}], batch_size=1)
    batch = next(iter(loader))
    item = batch.items[0]
    assert type(item) is Datum

  def test_dict_with_multiple_discriminator_keys(self) -> None:
    loader = DataLoader([{'success': True, 'metrics': {'x': 1.0}, 'epoch': 2}], batch_size=1)
    batch = next(iter(loader))
    assert isinstance(batch.items[0], EvalDatum)
    assert batch.items[0].success is True
    assert batch.items[0].epoch == 2

  def test_collate_eval_datum_vs_datum_parity(self) -> None:
    """Batch of mixed dicts: eval-keyed dicts become EvalDatum, others become Datum."""
    batch_input = [
      {'success': True, 'epoch': 1},
      {'items': []},
      {'metadata': {'key': 'val'}},
    ]
    loader = DataLoader(batch_input, batch_size=3)
    batch = next(iter(loader))
    assert isinstance(batch.items[0], EvalDatum)
    assert type(batch.items[1]) is Datum
    assert isinstance(batch.items[2], EvalDatum)

  def test_dict_with_none_value_for_eval_key_becomes_eval_datum(self) -> None:
    loader = DataLoader([{'success': None}], batch_size=1)
    batch = next(iter(loader))
    assert isinstance(batch.items[0], EvalDatum)


class TestDataLoaderRejectsNonDatum:
  def test_string_items_raise_type_error(self) -> None:
    loader = DataLoader(
      dataset=[Datum(), Datum()],
      batch_size=1,
    )
    batches = list(loader)
    assert len(batches) == 2

    loader_bad = DataLoader(
      dataset=['not a datum', 'also not'],
      batch_size=1,
    )
    with pytest.raises(TypeError, match='expected Datum'):
      list(loader_bad)

  def test_int_items_raise_type_error(self) -> None:
    loader = DataLoader(dataset=[1, 2, 3], batch_size=1)
    with pytest.raises(TypeError, match='expected Datum'):
      list(loader)


def test_dataloader_sampler_property_reflects_constructed_sampler():
  """Expose ``sampler`` so loops can invoke ``EpochAwareSamplerMixin.set_epoch``.

  Comprehensive property tests live in ``test_dataloader_properties.py``.
  This test is kept for backward compatibility.
  """
  data = _make_datums(3)
  ds = ListDataset(cast(Any, data))
  sampler = RandomSampler(ds)
  loader = DataLoader(ds, batch_size=2, sampler=sampler)
  assert loader.sampler is sampler
  assert loader.batch_sampler is None
