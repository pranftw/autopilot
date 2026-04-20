from autopilot.data.dataset import ConcatDataset, Dataset, IterableDataset
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
    for i in range(self._n):
      yield i


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
