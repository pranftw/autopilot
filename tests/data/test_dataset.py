from autopilot.data.dataloader import DataLoader
from autopilot.data.dataset import (
  ConcatDataset,
  Dataset,
  IterableDataset,
  ListDataset,
  StreamingDataset,
)
from pydantic import BaseModel, ValidationError
import operator
import pytest


class _MapDataset(Dataset[int]):
  def __init__(self, values: list[int]) -> None:
    self._values = values

  def __getitem__(self, index: int) -> int:
    return self._values[index]

  def __len__(self) -> int:
    return len(self._values)


class _StreamDataset(IterableDataset[int]):
  def __init__(self, n: int) -> None:
    self._n = n

  def __iter__(self):
    yield from range(self._n)


def test_base_dataset_getitem_raises():
  with pytest.raises(NotImplementedError):
    _ = Dataset()[0]


def test_map_style_subclass():
  ds = _MapDataset([10, 20, 30])
  assert ds[1] == 20
  assert len(ds) == 3


def test_iterable_base_iter_raises():
  with pytest.raises(NotImplementedError):
    next(iter(IterableDataset()))


def test_iterable_getitem_raises():
  with pytest.raises(TypeError, match='IterableDataset'):
    _ = IterableDataset()[0]


def test_iterable_subclass_iter():
  ds = _StreamDataset(3)
  assert list(ds) == [0, 1, 2]


def test_concat_dataset_two():
  a = _MapDataset([1, 2])
  b = _MapDataset([3, 4, 5])
  c = ConcatDataset([a, b])
  assert len(c) == 5
  assert [c[i] for i in range(5)] == [1, 2, 3, 4, 5]


def test_dataset_add_returns_concat():
  a = _MapDataset([1])
  b = _MapDataset([2, 3])
  c = a + b
  assert isinstance(c, ConcatDataset)
  assert len(c) == 3
  assert c[2] == 3


def test_concat_dataset_slice():
  a = _MapDataset([10, 20])
  b = _MapDataset([30, 40, 50])
  c = ConcatDataset([a, b])
  assert c[1:4] == [20, 30, 40]


def test_concat_dataset_slice_full():
  a = _MapDataset([1, 2, 3])
  b = _MapDataset([4, 5])
  c = ConcatDataset([a, b])
  assert c[:] == [1, 2, 3, 4, 5]


def test_concat_dataset_slice_with_step():
  a = _MapDataset([0, 1, 2, 3, 4])
  c = ConcatDataset([a])
  assert c[::2] == [0, 2, 4]


def test_concat_dataset_slice_empty():
  a = _MapDataset([1, 2, 3])
  c = ConcatDataset([a])
  assert c[5:10] == []


def test_concat_dataset_with_iterable_raises_type_error():
  stream = _StreamDataset(3)
  with pytest.raises(TypeError, match='IterableDataset'):
    ConcatDataset([stream])


def test_concat_dataset_with_iterable_at_second_position():
  a = _MapDataset([1, 2])
  stream = _StreamDataset(3)
  with pytest.raises(TypeError, match='dataset at index 1 is an IterableDataset'):
    ConcatDataset([a, stream])


def test_dataset_add_with_iterable_raises_type_error():
  a = _MapDataset([1])
  stream = _StreamDataset(3)
  with pytest.raises(TypeError, match='IterableDataset'):
    a + stream


def test_concat_dataset_negative_index_raises():
  a = _MapDataset([1, 2, 3])
  c = ConcatDataset([a])
  with pytest.raises(IndexError, match='index -1 out of range'):
    c[-1]


def test_base_dataset_len_raises():
  with pytest.raises(NotImplementedError):
    len(Dataset())


def test_iterable_dataset_len_raises():
  with pytest.raises(TypeError, match='does not support __len__'):
    len(IterableDataset())


def test_iterable_subclass_len_raises():
  ds = _StreamDataset(3)
  with pytest.raises(TypeError, match='does not support __len__'):
    len(ds)


class _StubModel(BaseModel):
  name: str
  value: float


def test_list_dataset_from_jsonl_to_jsonl_roundtrip(tmp_path):
  items = [_StubModel(name='a', value=1.0), _StubModel(name='b', value=2.5)]
  ds = ListDataset(items)
  path = tmp_path / 'data.jsonl'
  ds.to_jsonl(path)
  loaded = ListDataset.from_jsonl(path, _StubModel)
  assert len(loaded) == 2
  assert loaded[0].name == 'a'
  assert loaded[0].value == 1.0
  assert loaded[1].name == 'b'
  assert loaded[1].value == 2.5


def test_list_dataset_from_jsonl_skips_blank_lines(tmp_path):
  path = tmp_path / 'data.jsonl'
  path.write_text('{"name":"x","value":1.0}\n\n{"name":"y","value":2.0}\n')
  ds = ListDataset.from_jsonl(path, _StubModel)
  assert len(ds) == 2


# Plan 05 §4.4 — from_jsonl max_rows


def test_list_dataset_from_jsonl_max_rows_truncates(tmp_path):
  """File has N lines; max_rows=k yields k items with k < N."""
  path = tmp_path / 'data.jsonl'
  lines = [f'{{"name":"row{i}","value":{float(i)}}}' for i in range(10)]
  path.write_text('\n'.join(lines) + '\n')
  ds = ListDataset.from_jsonl(path, _StubModel, max_rows=3)
  assert len(ds) == 3
  assert ds[0].name == 'row0'
  assert ds[2].name == 'row2'


def test_list_dataset_from_jsonl_max_rows_none_loads_all(tmp_path):
  """max_rows=None loads all rows."""
  path = tmp_path / 'data.jsonl'
  lines = [f'{{"name":"r{i}","value":{float(i)}}}' for i in range(5)]
  path.write_text('\n'.join(lines) + '\n')
  ds = ListDataset.from_jsonl(path, _StubModel, max_rows=None)
  assert len(ds) == 5


def test_list_dataset_from_jsonl_max_rows_invalid_raises(tmp_path):
  """max_rows=0 raises ValueError."""
  path = tmp_path / 'data.jsonl'
  path.write_text('{"name":"a","value":1.0}\n')
  with pytest.raises(ValueError, match='max_rows must be >= 1'):
    ListDataset.from_jsonl(path, _StubModel, max_rows=0)


def test_streaming_dataset_yields_correct_type(tmp_path):
  path = tmp_path / 'stream.jsonl'
  path.write_text('{"name":"a","value":1.0}\n{"name":"b","value":2.0}\n')
  ds = StreamingDataset(path, _StubModel)
  items = list(ds)
  assert len(items) == 2
  assert all(isinstance(item, _StubModel) for item in items)
  assert items[0].name == 'a'
  assert items[1].value == 2.0


def test_streaming_dataset_skips_blank_lines(tmp_path):
  path = tmp_path / 'stream.jsonl'
  path.write_text('{"name":"x","value":0.0}\n\n\n{"name":"y","value":1.0}\n')
  ds = StreamingDataset(path, _StubModel)
  assert len(list(ds)) == 2


def test_list_dataset_len_and_indexing():
  items = [_StubModel(name='a', value=0.0), _StubModel(name='b', value=1.0)]
  ds = ListDataset(items)
  assert len(ds) == 2
  assert ds[0].name == 'a'
  assert ds[1].name == 'b'


def test_list_dataset_subset():
  items = [
    _StubModel(name='a', value=0.0),
    _StubModel(name='b', value=1.0),
    _StubModel(name='c', value=2.0),
  ]
  ds = ListDataset(items)
  sub = ds.subset([0, 2])
  assert len(sub) == 2
  assert sub[0].name == 'a'
  assert sub[1].name == 'c'


def test_concat_dataset_with_list_datasets():
  a = ListDataset([_StubModel(name='a', value=1.0)])
  b = ListDataset([_StubModel(name='b', value=2.0), _StubModel(name='c', value=3.0)])
  c = ConcatDataset([a, b])
  assert len(c) == 3
  assert c[0].name == 'a'
  assert c[2].name == 'c'


def test_dataloader_list_dataset_len_and_iteration():
  items = [_StubModel(name='a', value=1.0), _StubModel(name='b', value=2.0)]
  ds = ListDataset(items)
  loader = DataLoader(ds, batch_size=1, collate_fn=operator.itemgetter(0))
  assert len(loader) == 2
  results = list(loader)
  assert len(results) == 2


class TestIterJsonlLinesBadJsonRaises:
  """Verify bad JSON lines propagate parse errors through iter_jsonl_lines routing."""

  def test_from_jsonl_bad_json_raises(self, tmp_path):
    path = tmp_path / 'bad.jsonl'
    path.write_text('{"name":"ok","value":1.0}\nnot valid json\n')
    with pytest.raises(ValidationError):
      ListDataset.from_jsonl(path, _StubModel)

  def test_streaming_bad_json_raises(self, tmp_path):
    path = tmp_path / 'bad_stream.jsonl'
    path.write_text('{"name":"ok","value":1.0}\n{broken\n')
    ds = StreamingDataset(path, _StubModel)
    with pytest.raises(ValidationError):
      list(ds)


class TestListDatasetJsonlViaIterJsonlLines:
  """Verify from_jsonl routes through iter_jsonl_lines correctly."""

  def test_blank_lines_and_trailing_newline(self, tmp_path):
    path = tmp_path / 'data.jsonl'
    path.write_text(
      '{"name":"a","value":1.0}\n\n  \n{"name":"b","value":2.0}\n{"name":"c","value":3.0}\n\n',
    )
    ds = ListDataset.from_jsonl(path, _StubModel)
    assert len(ds) == 3
    assert ds[0].name == 'a'
    assert ds[1].name == 'b'
    assert ds[2].name == 'c'

  def test_utf8_content(self, tmp_path):
    path = tmp_path / 'data.jsonl'
    path.write_text(
      '{"name":"caf\\u00e9","value":1.0}\n{"name":"\\u00fcber","value":2.0}\n',
      encoding='utf-8',
    )
    ds = ListDataset.from_jsonl(path, _StubModel)
    assert len(ds) == 2
    assert ds[0].name == 'caf\u00e9'
    assert ds[1].name == '\u00fcber'

  def test_max_rows_with_blank_lines(self, tmp_path):
    path = tmp_path / 'data.jsonl'
    path.write_text(
      '{"name":"a","value":1.0}\n\n{"name":"b","value":2.0}\n{"name":"c","value":3.0}\n',
    )
    ds = ListDataset.from_jsonl(path, _StubModel, max_rows=2)
    assert len(ds) == 2
    assert ds[0].name == 'a'
    assert ds[1].name == 'b'


class TestStreamingDatasetViaIterJsonlLines:
  """Verify StreamingDataset routes through iter_jsonl_lines correctly."""

  def test_blank_lines_and_trailing_newline(self, tmp_path):
    path = tmp_path / 'stream.jsonl'
    path.write_text(
      '{"name":"x","value":0.0}\n\n{"name":"y","value":1.0}\n\n',
    )
    ds = StreamingDataset(path, _StubModel)
    items = list(ds)
    assert len(items) == 2
    assert items[0].name == 'x'
    assert items[1].name == 'y'

  def test_utf8_content(self, tmp_path):
    path = tmp_path / 'stream.jsonl'
    path.write_text(
      '{"name":"caf\\u00e9","value":1.0}\n',
      encoding='utf-8',
    )
    ds = StreamingDataset(path, _StubModel)
    items = list(ds)
    assert len(items) == 1
    assert items[0].name == 'caf\u00e9'
