"""Dataset base classes. Mirrors torch.utils.data.Dataset."""

from autopilot.tracking.io import iter_jsonl_lines
from collections.abc import Iterator, Sequence
from pathlib import Path
from pydantic import BaseModel
from typing import Generic, TypeVar

T_co = TypeVar('T_co', covariant=True)
M = TypeVar('M', bound=BaseModel)


class Dataset(Generic[T_co]):
  """Map-style dataset. Same contract as torch.utils.data.Dataset.

  Subclasses must implement ``__getitem__`` and ``__len__``.
  ``IterableDataset`` overrides ``__len__`` to raise ``TypeError``.

  Example:
    >>> from autopilot.data.dataset import Dataset
    >>>
    >>> class NumbersDataset(Dataset[int]):
    ...   def __init__(self, n):
    ...     self._n = n
    ...
    ...   def __getitem__(self, index):
    ...     return index
    ...
    ...   def __len__(self):
    ...     return self._n
    >>>
    >>> data = NumbersDataset(3)
    >>> len(data)
    3
  """

  def __getitem__(self, index: int) -> T_co:
    """Return the item at ``index``; subclasses must implement.

    Raises:
      NotImplementedError: On the base class.
    """
    raise NotImplementedError

  def __len__(self) -> int:
    """Return the number of items. Must be implemented by map-style subclasses."""
    raise NotImplementedError

  def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
    """Concatenate two datasets. Both must be map-style (have __len__).

    Args:
      other: Second dataset to append after ``self``.

    Returns:
      ``ConcatDataset`` spanning both operands.
    """
    return ConcatDataset([self, other])


class IterableDataset(Dataset[T_co]):
  """Iterable dataset. Same contract as torch.utils.data.IterableDataset."""

  def __iter__(self):
    """Iterate dataset samples; subclasses must implement.

    Raises:
      NotImplementedError: On the base class.
    """
    raise NotImplementedError

  def __getitem__(self, index: int):
    """Disable random access for iterable datasets.

    Raises:
      TypeError: Always.
    """
    msg = f'{type(self).__name__} does not support __getitem__'
    raise TypeError(
      msg,
    )

  def __len__(self) -> int:
    """Disable length queries for iterable datasets.

    Raises:
      TypeError: Always.
    """
    msg = f'{type(self).__name__} does not support __len__'
    raise TypeError(
      msg,
    )


class ConcatDataset(Dataset[T_co]):
  """Concatenation of multiple map-style datasets.

  All children must implement __len__. IterableDataset children are rejected
  at construction with TypeError.
  """

  def __init__(self, datasets: Sequence[Dataset[T_co]]) -> None:
    """Validate children are map-style and compute cumulative lengths.

    Args:
      datasets: Ordered map-style datasets to concatenate.

    Raises:
      TypeError: If any child is an ``IterableDataset``.
    """
    for i, ds in enumerate(datasets):
      if isinstance(ds, IterableDataset):
        msg = (
          f'ConcatDataset requires map-style datasets with __len__, '
          f'but dataset at index {i} is an IterableDataset'
        )
        raise TypeError(msg)
    self._datasets = datasets
    self._cumulative_sizes: list[int] = []
    cumsum = 0
    for ds in datasets:
      cumsum += len(ds)
      self._cumulative_sizes.append(cumsum)

  def __getitem__(self, index: int | slice):
    """Index into the concatenated virtual sequence (scalar index or slice).

    Returns:
      A single element for integer ``index``, or a list for a ``slice``.
    """
    if isinstance(index, slice):
      indices = range(*index.indices(len(self)))
      return [self._get_single(i) for i in indices]
    return self._get_single(index)

  def _get_single(self, index: int):
    """Resolve a flat index to the correct child dataset and local offset.

    Performs bounds checking, then walks cumulative sizes to find the owning
    child dataset and delegates with the adjusted local index.

    Args:
      index: Non-negative flat index into the concatenated virtual sequence.

    Returns:
      The item from the child dataset at the resolved local position.

    Raises:
      IndexError: When ``index`` is out of bounds (negative or >= total length).
    """
    if index < 0 or index >= len(self):
      msg = f'index {index} out of range for ConcatDataset of size {len(self)}'
      raise IndexError(msg)
    for i, size in enumerate(self._cumulative_sizes):
      if index < size:
        offset = self._cumulative_sizes[i - 1] if i > 0 else 0
        return self._datasets[i][index - offset]
    raise IndexError(index)

  def __len__(self) -> int:
    """Total number of samples across all child datasets.

    Returns:
      Sum of child dataset lengths (0 when empty).
    """
    return self._cumulative_sizes[-1] if self._cumulative_sizes else 0


class ListDataset(Dataset[M]):
  """Map-style dataset backed by an in-memory list.

  Items must be ``pydantic.BaseModel`` subclasses so that ``from_jsonl`` /
  ``to_jsonl`` can call ``model_validate_json`` / ``model_dump_json``.
  """

  def __init__(self, items: list[M]) -> None:
    """Store pydantic model instances for map-style access.

    Args:
      items: In-memory list of homogeneous ``BaseModel`` rows.
    """
    self._items = items

  def __getitem__(self, index: int) -> M:
    """Return the model instance at ``index``."""
    return self._items[index]

  def __len__(self) -> int:
    """Return the number of stored items."""
    return len(self._items)

  def subset(self, indices: list[int]) -> 'ListDataset[M]':
    """Return a new ``ListDataset`` containing only the items at the given indices.

    Args:
      indices: Positional indices into the current items list. Out-of-bounds
        indices raise ``IndexError`` from the underlying list access.

    Returns:
      New ``ListDataset`` sharing the selected model instances (shallow copy).
    """
    return ListDataset([self._items[i] for i in indices])

  @classmethod
  def from_jsonl(
    cls,
    path: Path | str,
    item_type: type[M],
    max_rows: int | None = None,
  ) -> 'ListDataset[M]':
    """Deserialize a JSONL file into a ListDataset of the given Pydantic model type.

    Reads line-by-line.  When ``max_rows`` is ``None`` (default), all rows are
    loaded.  When ``max_rows >= 1``, loading stops after that many successfully
    parsed rows -- a simple OOM guard for large files when only a prefix is
    needed.

    Args:
      path: Path to a JSONL file (one JSON object per line).
      item_type: A ``pydantic.BaseModel`` subclass used to parse each line.
      max_rows: Optional cap on the number of rows to load.  ``None`` loads all.

    Returns:
      A new ListDataset containing the parsed model instances.

    Raises:
      ValueError: If ``max_rows`` is not ``None`` and is less than 1.
    """
    if max_rows is not None and max_rows < 1:
      msg = f'max_rows must be >= 1 or None, got {max_rows}'
      raise ValueError(msg)
    filepath = Path(path)
    items: list[M] = []
    for line in iter_jsonl_lines(filepath):
      items.append(item_type.model_validate_json(line))
      if max_rows is not None and len(items) >= max_rows:
        break
    return cls(items)

  def to_jsonl(self, path: Path) -> None:
    """Serialize items to a JSONL file (one JSON object per line).

    Args:
      path: Destination file path. Parent directories are created if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
      for item in self._items:
        f.write(item.model_dump_json())
        f.write('\n')


class StreamingDataset(IterableDataset[M]):
  """Lazily reads Pydantic model instances line-by-line from a JSONL file.

  Items must be ``pydantic.BaseModel`` subclasses so that
  ``model_validate_json`` resolves on each line.

  Ordering is determined entirely by the file's line order.  Index-based
  samplers (``SequentialSampler``, ``RandomSampler``, ``BatchSampler``) do not
  apply to iterable datasets.  Reordering requires dataset-side logic
  (e.g. pre-shuffle the file or implement a buffered-shuffle wrapper).
  ``DataLoader`` will raise ``TypeError`` if a ``batch_sampler`` is passed
  alongside an iterable dataset.
  """

  def __init__(self, path: Path, item_type: type[M]) -> None:
    """Open a JSONL path lazily without loading all rows.

    Args:
      path: JSONL file to stream line-by-line.
      item_type: ``BaseModel`` subtype used to validate each line.
    """
    self._path = path
    self._item_type = item_type

  def __iter__(self) -> Iterator[M]:
    """Yield models parsed from each non-empty JSONL line.

    Yields:
      Validated instances of ``item_type``.
    """
    for line in iter_jsonl_lines(self._path):
      yield self._item_type.model_validate_json(line)
