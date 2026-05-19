"""DataLoader with sampler-based ordering and strict batch_size validation.

Ordering is decoupled from the loader via the ``Sampler`` hierarchy
(``data/sampler.py``).  The ``shuffle`` and ``length_hint`` parameters are
removed; use ``RandomSampler`` for shuffled iteration and ``BatchSampler``
for custom batching policies.

Mutual-exclusion rules:

- When ``batch_sampler`` is provided, ``batch_size`` and ``drop_last`` are
  ignored for iteration and length.
- ``IterableDataset`` does not support ``batch_sampler`` (raises ``TypeError``).

``_default_collate`` always returns ``Datum(items=[...])`` -- single-item
batches are **not** unwrapped (BUG-033 fix).
"""

from autopilot.core.types import Datum, EvalDatum
from autopilot.data.dataset import Dataset, IterableDataset, ListDataset
from autopilot.data.sampler import BatchSampler, Sampler, SequentialSampler
from collections.abc import Callable, Iterator
from typing import Any
import math
import random

_EVAL_DATUM_KEYS = ('success', 'metrics', 'feedback', 'error_message', 'split', 'epoch', 'metadata')


def _default_collate(batch: list[Any]) -> Datum:
  """Collate a list of items into a single batched ``Datum``.

  Every non-empty batch is wrapped as ``Datum(items=[...])``, including
  single-element batches.

  Args:
    batch: List of ``Datum`` instances or compatible dict rows.

  Returns:
    A ``Datum`` whose ``items`` list contains the collated elements.

  Raises:
    TypeError: If an entry is neither ``Datum``-like dict nor ``Datum``.
  """
  items = []
  for item in batch:
    if isinstance(item, Datum):
      items.append(item)
    elif isinstance(item, dict):
      if any(k in item for k in _EVAL_DATUM_KEYS):
        items.append(
          EvalDatum(**{k: v for k, v in item.items() if k in EvalDatum.__dataclass_fields__})
        )
      else:
        items.append(Datum(**{k: v for k, v in item.items() if k in Datum.__dataclass_fields__}))
    else:
      msg = (
        f'DataLoader: expected Datum or dict, got {type(item).__name__}. '
        f'Provide a custom collate_fn for non-Datum types.'
      )
      raise TypeError(msg)
  return Datum(items=items)


class DataLoader:
  """Yields Datum batches from a dataset using sampler-based ordering.

  All index ordering is delegated to ``Sampler`` / ``BatchSampler`` instances.
  For map-style datasets, the default sampler is ``SequentialSampler``
  (deterministic sequential access).  Use ``RandomSampler`` with a seeded
  ``random.Random`` generator for reproducible shuffled iteration.

  ``batch_size`` must be ``int >= 1``; ``None``, non-int, zero, and negative
  values raise ``ValueError``.

  ``IterableDataset`` batches items in stream order; index-based
  ``batch_sampler`` is not supported (raises ``TypeError``).

  Properties:
    ``sampler`` and ``batch_sampler`` expose construction-time wiring only --
    they do **not** materialize the implicit ``SequentialSampler`` used during
    ``__iter__`` when neither is provided.  Training loops should prefer
    ``batch_sampler`` when present for ``set_epoch`` wiring.

  Note:
    The ``shuffle`` parameter is removed. Use ``sampler=RandomSampler(dataset)``
    from ``autopilot.data.sampler`` for shuffled iteration.

  Example:
    >>> from autopilot.data.dataset import Dataset
    >>> from autopilot.data.dataloader import DataLoader
    >>> from autopilot.data.sampler import RandomSampler
    >>> import random
    >>>
    >>> class TinyRows(Dataset[dict]):
    ...   def __getitem__(self, index):
    ...     return {'success': True}
    ...
    ...   def __len__(self):
    ...     return 4
    >>>
    >>> ds = TinyRows()
    >>> loader = DataLoader(
    ...   ds,
    ...   batch_size=2,
    ...   sampler=RandomSampler(ds, generator=random.Random(0)),
    ... )
    >>> batch = next(iter(loader))
    >>> len(batch.items)
    2
  """

  def __init__(
    self,
    dataset: Dataset | list,
    batch_size: int = 1,
    sampler: Sampler | None = None,
    batch_sampler: BatchSampler | None = None,
    collate_fn: Callable[[list[Any]], Datum] | None = None,
    drop_last: bool = False,
    generator: random.Random | None = None,
  ) -> None:
    """Configure batching, sampling, and collation for a dataset.

    Args:
      dataset: Map-style ``Dataset``, iterable dataset, or plain list (wrapped).
      batch_size: Number of items per yielded batch (must be ``int >= 1``).
      sampler: Optional index sampler for map-style datasets.
      batch_sampler: Optional ``BatchSampler`` (overrides ``batch_size`` /
        ``drop_last`` for iteration).  Incompatible with ``IterableDataset``.
      collate_fn: Optional batch collator (defaults to ``_default_collate``).
      drop_last: Whether to drop a final partial batch.
      generator: Optional ``random.Random`` stored for API parity; callers
        wire it into ``RandomSampler`` themselves for deterministic draws.

    Raises:
      ValueError: If ``batch_size`` is not ``int >= 1``.
      TypeError: If ``dataset`` is neither ``Dataset`` nor ``list``.
      TypeError: If ``batch_sampler`` is used with an ``IterableDataset``.
    """
    if type(batch_size) is not int or batch_size < 1:
      msg = f'batch_size must be an int >= 1, got {batch_size!r}'
      raise ValueError(msg)
    if not isinstance(dataset, (Dataset, list)):
      msg = f'dataset must be a Dataset or list, got {type(dataset).__name__}'
      raise TypeError(msg)
    if isinstance(dataset, list):
      dataset = ListDataset(dataset)
    if batch_sampler is not None and isinstance(dataset, IterableDataset):
      msg = (
        'batch_sampler cannot be used with IterableDataset because iterable '
        'datasets do not support index-based access'
      )
      raise TypeError(msg)
    self._dataset = dataset
    self._batch_size = batch_size
    self._sampler = sampler
    self._batch_sampler = batch_sampler
    self._collate_fn = collate_fn or _default_collate
    self._drop_last = drop_last
    self._generator = generator

  def __iter__(self) -> Iterator[Datum]:
    """Iterate batches produced from the underlying dataset.

    Yields:
      Batched ``Datum`` instances according to ``batch_size`` and ``collate_fn``.
    """
    if isinstance(self._dataset, IterableDataset):
      yield from self._iter_iterable()
    else:
      yield from self._iter_map()

  def _iter_map(self) -> Iterator[Datum]:
    """Map-style iteration using sampler or batch_sampler.

    Yields:
      Collated ``Datum`` batches from indexed access.
    """
    if self._batch_sampler is not None:
      for index_batch in self._batch_sampler:
        batch = [self._dataset[i] for i in index_batch]
        yield self._collate_fn(batch)
      return

    sampler = self._sampler if self._sampler is not None else SequentialSampler(self._dataset)
    batch: list[Any] = []
    for idx in sampler:
      batch.append(self._dataset[idx])
      if len(batch) == self._batch_size:
        yield self._collate_fn(batch)
        batch = []
    if batch and not self._drop_last:
      yield self._collate_fn(batch)

  def _iter_iterable(self) -> Iterator[Datum]:
    """Iterable-style iteration batched by batch_size.

    Yields:
      Collated ``Datum`` batches from stream iteration.
    """
    batch: list[Any] = []
    for item in self._dataset:
      batch.append(item)
      if len(batch) == self._batch_size:
        yield self._collate_fn(batch)
        batch = []
    if batch and not self._drop_last:
      yield self._collate_fn(batch)

  def __len__(self) -> int:
    """Return the number of batches per epoch.

    Raises:
      TypeError: For iterable datasets that do not implement ``__len__``.
    """
    if isinstance(self._dataset, IterableDataset):
      msg = 'IterableDataset does not support __len__ on DataLoader'
      raise TypeError(msg)

    if self._batch_sampler is not None:
      return len(self._batch_sampler)

    sampler = self._sampler if self._sampler is not None else SequentialSampler(self._dataset)
    dataset_len = len(sampler)
    if self._drop_last:
      return dataset_len // self._batch_size
    return math.ceil(dataset_len / self._batch_size)

  @property
  def sampler(self) -> Sampler | None:
    """Return the sampler passed at construction, or ``None``.

    When ``batch_sampler`` is the sole ordering source, this is ``None`` even
    though iteration uses the batch sampler.  Training loops should prefer
    ``batch_sampler`` when present for ``set_epoch`` wiring.

    Returns:
      Explicit ``Sampler`` from ``__init__``, or ``None``.
    """
    return self._sampler

  @property
  def batch_sampler(self) -> BatchSampler | None:
    """Return the ``BatchSampler`` passed at construction, if any.

    Returns:
      Explicit ``BatchSampler`` from ``__init__``, or ``None``.
    """
    return self._batch_sampler
