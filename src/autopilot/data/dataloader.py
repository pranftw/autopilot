"""DataLoader. Mirrors torch.utils.data.DataLoader."""

from autopilot.core.models import Datum
from autopilot.data.dataset import Dataset, IterableDataset, ListDataset
from typing import Any, Callable, Iterator
import math
import random


def _default_collate(batch: list[Any]) -> Datum:
  items = []
  for item in batch:
    if isinstance(item, Datum):
      items.append(item)
    elif isinstance(item, dict):
      items.append(Datum(**{k: v for k, v in item.items() if k in Datum.__dataclass_fields__}))
    else:
      items.append(Datum(metadata={'raw': item}))
  if len(items) == 1:
    return items[0]
  return Datum(items=items)


class DataLoader:
  """Yields Datum batches from a dataset."""

  def __init__(
    self,
    dataset: Dataset | list,
    batch_size: int = 1,
    shuffle: bool = False,
    collate_fn: Callable | None = None,
    drop_last: bool = False,
    length_hint: int | None = None,
  ) -> None:
    if isinstance(dataset, list):
      dataset = ListDataset(dataset)
    self._dataset = dataset
    self._batch_size = batch_size
    self._shuffle = shuffle
    self._collate_fn = collate_fn or _default_collate
    self._drop_last = drop_last
    self._length_hint = length_hint

  def __iter__(self) -> Iterator[Datum]:
    if isinstance(self._dataset, IterableDataset):
      yield from self._iter_iterable()
    else:
      yield from self._iter_map()

  def _iter_map(self) -> Iterator[Datum]:
    indices = list(range(len(self._dataset)))
    if self._shuffle:
      random.shuffle(indices)
    batch: list[Any] = []
    for idx in indices:
      batch.append(self._dataset[idx])
      if len(batch) == self._batch_size:
        yield self._collate_fn(batch)
        batch = []
    if batch and not self._drop_last:
      yield self._collate_fn(batch)

  def _iter_iterable(self) -> Iterator[Datum]:
    batch: list[Any] = []
    for item in self._dataset:
      batch.append(item)
      if len(batch) == self._batch_size:
        yield self._collate_fn(batch)
        batch = []
    if batch and not self._drop_last:
      yield self._collate_fn(batch)

  def __len__(self) -> int:
    if isinstance(self._dataset, IterableDataset):
      if self._length_hint is not None:
        n = self._length_hint
      else:
        raise TypeError('IterableDataset does not support __len__ on DataLoader')
    else:
      n = len(self._dataset)
    if self._drop_last:
      return n // self._batch_size
    return math.ceil(n / self._batch_size) if self._batch_size > 0 else 0
