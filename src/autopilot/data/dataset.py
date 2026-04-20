"""Dataset base classes. Mirrors torch.utils.data.Dataset."""

from pathlib import Path
from typing import Generic, Iterator, TypeVar

T_co = TypeVar('T_co', covariant=True)


class Dataset(Generic[T_co]):
  """Map-style dataset. Same contract as torch.utils.data.Dataset."""

  def __getitem__(self, index: int) -> T_co:
    raise NotImplementedError

  def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
    return ConcatDataset([self, other])


class IterableDataset(Dataset[T_co]):
  """Iterable dataset. Same contract as torch.utils.data.IterableDataset."""

  def __iter__(self):
    raise NotImplementedError

  def __getitem__(self, index: int):
    raise TypeError('IterableDataset does not support __getitem__')


class ConcatDataset(Dataset[T_co]):
  """Concatenation of multiple datasets."""

  def __init__(self, datasets: list[Dataset[T_co]]) -> None:
    self._datasets = datasets
    self._cumulative_sizes: list[int] = []
    cumsum = 0
    for ds in datasets:
      cumsum += len(ds)
      self._cumulative_sizes.append(cumsum)

  def __getitem__(self, index: int):
    if index < 0 or index >= len(self):
      raise IndexError(f'index {index} out of range for ConcatDataset of size {len(self)}')
    for i, size in enumerate(self._cumulative_sizes):
      if index < size:
        offset = self._cumulative_sizes[i - 1] if i > 0 else 0
        return self._datasets[i][index - offset]
    raise IndexError(index)

  def __len__(self) -> int:
    return self._cumulative_sizes[-1] if self._cumulative_sizes else 0


class ListDataset(Dataset[T_co]):
  """Map-style dataset backed by an in-memory list."""

  def __init__(self, items: list[T_co]) -> None:
    self._items = items

  def __getitem__(self, index: int) -> T_co:
    return self._items[index]

  def __len__(self) -> int:
    return len(self._items)

  def subset(self, indices: list[int]) -> 'ListDataset[T_co]':
    return ListDataset([self._items[i] for i in indices])

  @classmethod
  def from_jsonl(cls, path: Path, item_type: type[T_co]) -> 'ListDataset[T_co]':
    lines = path.read_text(encoding='utf-8').splitlines()
    items = [item_type.model_validate_json(line) for line in lines if line.strip()]
    return cls(items)

  def to_jsonl(self, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
      for item in self._items:
        f.write(item.model_dump_json())
        f.write('\n')


class StreamingDataset(IterableDataset[T_co]):
  """Lazily reads Pydantic model instances line-by-line from a JSONL file."""

  def __init__(self, path: Path, item_type: type[T_co]) -> None:
    self._path = path
    self._item_type = item_type

  def __iter__(self) -> Iterator[T_co]:
    with self._path.open('r', encoding='utf-8') as f:
      for line in f:
        stripped = line.strip()
        if not stripped:
          continue
        yield self._item_type.model_validate_json(stripped)
