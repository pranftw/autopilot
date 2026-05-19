"""Data sampler hierarchy for index ordering.

Decouples index ordering from ``DataLoader`` via a PyTorch-aligned sampler
hierarchy.  Concrete samplers yield dataset indices (``int``).  Compose
``BatchSampler`` around any base sampler to produce batched index lists.

Deterministic RNG: ``RandomSampler`` and ``WeightedSampler`` capture a base
seed from the parent ``random.Random`` generator at construction and
XOR-combine it with the epoch set via ``set_epoch()``, so training order is
reproducible across restarts for any given ``(generator seed, epoch)`` pair.

``ValueError`` conditions:

- ``RandomSampler``: raised when ``replacement=False`` and ``num_samples``
  exceeds ``len(data_source)``.
- ``WeightedSampler``: raised when ``len(weights) != len(data_source)``,
  any weight is non-positive (``<= 0``), any weight is non-finite
  (``NaN`` / ``inf``), or the dataset is empty.
- ``BatchSampler``: raised when ``batch_size < 1``.

Intentional divergence from PyTorch: ``__len__`` is required on all concrete
samplers (master plan section 22).
"""

from autopilot.data.dataset import Dataset
from collections.abc import Iterator, Sequence
import math
import random


class Sampler:
  """Base sampler protocol.  Yields integer dataset indices.

  Concrete subclasses must implement ``__iter__`` and ``__len__``.
  ``BatchSampler`` yields ``list[int]`` batches instead of scalar indices;
  it subclasses ``Sampler`` for compositional parity with PyTorch.
  """

  def __iter__(self) -> Iterator[int]:
    """Yield dataset indices.

    Raises:
      NotImplementedError: On the abstract base.
    """
    raise NotImplementedError

  def __len__(self) -> int:
    """Return the number of indices this sampler will yield.

    Raises:
      NotImplementedError: On the abstract base.
    """
    raise NotImplementedError


class SequentialSampler(Sampler):
  """Samples elements sequentially from ``0`` to ``len(data_source) - 1``.

  Args:
    data_source: Map-style ``Dataset`` providing ``__len__``.
  """

  def __init__(self, data_source: Dataset) -> None:
    """Initialize with a map-style dataset.

    Args:
      data_source: Dataset providing ``__len__`` for index range.
    """
    self._data_source = data_source

  def __iter__(self) -> Iterator[int]:
    """Yield indices ``0, 1, ..., len(data_source) - 1``.

    Returns:
      Lazy iterator over the sequential index range.
    """
    return iter(range(len(self._data_source)))

  def __len__(self) -> int:
    """Return ``len(data_source)``."""
    return len(self._data_source)


class EpochAwareSamplerMixin:
  """Mixin providing shared epoch-aware RNG and length for samplers.

  Consolidates ``set_epoch``, ``__len__``, and seed/epoch RNG initialization
  that ``RandomSampler`` and ``WeightedSampler`` previously duplicated.
  Concrete samplers inherit this mixin alongside ``Sampler`` and implement
  only their divergent ``__iter__``.
  """

  _data_source: Dataset
  _num_samples: int | None
  _base_seed: int
  _epoch: int

  def _init_epoch_aware(
    self,
    data_source: Dataset,
    num_samples: int | None,
    generator: random.Random | None,
  ) -> None:
    """Initialize epoch-aware sampler state.

    Args:
      data_source: Map-style ``Dataset`` providing ``__len__``.
      num_samples: Override for the number of indices per pass; ``None``
        defaults to ``len(data_source)`` at iteration time.
      generator: ``random.Random`` for deterministic base-seed extraction.
        A fresh unseeded instance is created when ``None``.
    """
    self._data_source = data_source
    self._num_samples = num_samples
    gen = generator if generator is not None else random.Random()
    self._base_seed = gen.getrandbits(64)
    self._epoch = 0

  def set_epoch(self, epoch: int) -> None:
    """Set the epoch for deterministic per-epoch shuffling.

    Args:
      epoch: Current epoch number (0-based). Must be non-negative.

    Raises:
      ValueError: When ``epoch`` is negative.
    """
    if epoch < 0:
      msg = f'epoch must be >= 0, got {epoch}'
      raise ValueError(msg)
    self._epoch = epoch

  def __len__(self) -> int:
    """Return ``num_samples`` if set, otherwise ``len(data_source)``."""
    if self._num_samples is not None:
      return self._num_samples
    return len(self._data_source)


class RandomSampler(EpochAwareSamplerMixin, Sampler):
  """Samples elements randomly with deterministic, epoch-aware RNG.

  Each call to ``__iter__`` derives a fresh ``random.Random`` from the parent
  ``generator`` state XOR-ed with the current epoch, producing a reproducible
  per-epoch shuffle while advancing the parent generator across iterations.

  Args:
    data_source: Map-style ``Dataset`` providing ``__len__``.
    replacement: If ``True``, sample with replacement.
    num_samples: Number of indices to draw.  Defaults to ``len(data_source)``.
      When ``replacement`` is ``False``, must not exceed ``len(data_source)``.
    generator: ``random.Random`` instance for deterministic seeding.
      Defaults to a new ``random.Random()`` when ``None``.
  """

  def __init__(
    self,
    data_source: Dataset,
    replacement: bool = False,
    num_samples: int | None = None,
    generator: random.Random | None = None,
  ) -> None:
    """Initialize with a dataset and optional RNG configuration.

    Args:
      data_source: Map-style ``Dataset`` providing ``__len__``.
      replacement: If ``True``, sample with replacement.
      num_samples: Number of indices to draw; defaults to ``len(data_source)``.
      generator: Deterministic ``random.Random`` instance.  A fresh unseeded
        instance is created when ``None``.
    """
    self._replacement = replacement
    self._init_epoch_aware(data_source, num_samples, generator)

  def __iter__(self) -> Iterator[int]:
    """Yield randomly ordered dataset indices.

    Creates a child RNG seeded with ``base_seed ^ epoch`` so that the same
    ``(initial generator seed, epoch)`` pair always produces the same sequence.

    Returns:
      Iterator of ``int`` indices.

    Raises:
      ValueError: When ``replacement`` is ``False`` and ``num_samples``
        exceeds ``len(data_source)``.
    """
    n = len(self._data_source)
    if n == 0:
      return iter([])
    rng = random.Random(self._base_seed ^ self._epoch)
    count = self._num_samples if self._num_samples is not None else n
    if self._replacement:
      return (rng.randint(0, n - 1) for _ in range(count))
    if count > n:
      msg = (
        f'cannot sample {count} indices without replacement from a dataset '
        f'of size {n}; set replacement=True or reduce num_samples'
      )
      raise ValueError(msg)
    indices = list(range(n))
    rng.shuffle(indices)
    return iter(indices[:count])


class BatchSampler(Sampler):
  """Wraps a base sampler to yield batched index lists.

  Accumulates consecutive indices from the underlying sampler into lists of
  ``batch_size``.  A final partial batch is yielded unless ``drop_last`` is
  ``True``.

  Note: ``__iter__`` returns ``Iterator[list[int]]``, not ``Iterator[int]``.
  Subtyping ``Sampler`` is intentional to mirror PyTorch composition.

  Args:
    sampler: Underlying sampler providing scalar indices.
    batch_size: Number of indices per batch (must be >= 1).
    drop_last: If ``True``, discard a final batch smaller than ``batch_size``.

  Raises:
    ValueError: If ``batch_size < 1``.
  """

  def __init__(
    self,
    sampler: Sampler,
    batch_size: int,
    drop_last: bool = False,
  ) -> None:
    """Initialize with a base sampler and batching parameters.

    Args:
      sampler: Underlying sampler yielding scalar indices.
      batch_size: Number of indices per batch (must be >= 1).
      drop_last: If ``True``, discard partial final batch.

    Raises:
      ValueError: If ``batch_size < 1``.
    """
    if batch_size < 1:
      msg = f'batch_size must be >= 1, got {batch_size}'
      raise ValueError(msg)
    self._sampler = sampler
    self._batch_size = batch_size
    self._drop_last = drop_last

  @property
  def sampler(self) -> Sampler:
    """The underlying sampler."""
    return self._sampler

  def __iter__(self) -> Iterator[list[int]]:  # ty: ignore[invalid-method-override]  # intentional: BatchSampler yields list[int] batches (mirrors PyTorch composition)
    """Yield batched index lists from the underlying sampler.

    Yields:
      Batches of ``list[int]`` indices.
    """
    batch: list[int] = []
    for idx in self._sampler:
      batch.append(idx)
      if len(batch) == self._batch_size:
        yield batch
        batch = []
    if batch and not self._drop_last:
      yield batch

  def __len__(self) -> int:
    """Return the number of batches.

    Uses floor division when ``drop_last`` is ``True``, otherwise ceiling
    division.
    """
    num = len(self._sampler)
    if self._drop_last:
      return num // self._batch_size
    return (num + self._batch_size - 1) // self._batch_size


class SubsetSampler(Sampler):
  """Samples from a fixed list of indices.

  Yields indices in stored order when no ``generator`` is provided, or in
  shuffled order when a ``random.Random`` generator is supplied.

  Args:
    indices: Explicit index list (shallow-copied to prevent aliasing).
    generator: Optional ``random.Random`` for shuffled iteration.
  """

  def __init__(
    self,
    indices: list[int],
    generator: random.Random | None = None,
  ) -> None:
    """Initialize with an explicit index list.

    Args:
      indices: Indices to yield (shallow-copied to prevent aliasing).
      generator: Optional ``random.Random`` for shuffled iteration.
    """
    self._indices = list(indices)
    self._generator = generator

  def __iter__(self) -> Iterator[int]:
    """Yield stored indices, optionally shuffled.

    Returns:
      Iterator of ``int`` indices.
    """
    if self._generator is None:
      return iter(self._indices)
    order = list(self._indices)
    self._generator.shuffle(order)
    return iter(order)

  def __len__(self) -> int:
    """Return the number of indices."""
    return len(self._indices)


class WeightedSampler(EpochAwareSamplerMixin, Sampler):
  """Yields indices drawn according to ``weights`` (one weight per dataset index).

  Mirrors PyTorch's ``WeightedRandomSampler``: each dataset index has an
  associated weight controlling its sampling probability. Supports both
  with-replacement and without-replacement modes, optional ``num_samples``
  override, and deterministic per-epoch seeding via ``set_epoch()``.

  Composition example::

    sampler = WeightedSampler(dataset, weights=[1.0, 2.0, 3.0])
    batched = BatchSampler(sampler, batch_size=2)
    for batch in batched:
      items = [dataset[i] for i in batch]

  Args:
    data_source: Map-style ``Dataset`` providing ``__len__``.
    weights: One positive finite float per dataset index controlling
      sampling probability. ``len(weights)`` must equal ``len(data_source)``.
    replacement: If ``True`` (default), sample with replacement.
    num_samples: Number of indices to draw per full iterator pass.
      Defaults to ``len(data_source)`` when ``None``.
    generator: ``random.Random`` instance for deterministic seeding.
      Defaults to a new ``random.Random()`` when ``None``.

  Raises:
    ValueError: If ``data_source`` is empty, ``len(weights)`` mismatches,
      or any weight is non-positive or non-finite.
  """

  def __init__(
    self,
    data_source: Dataset,
    weights: Sequence[float],
    replacement: bool = True,
    num_samples: int | None = None,
    generator: random.Random | None = None,
  ) -> None:
    """Initialize with a dataset and per-index weights.

    Args:
      data_source: Map-style ``Dataset`` providing ``__len__``.
      weights: Positive finite floats, one per dataset index.
      replacement: If ``True``, sample with replacement (default).
      num_samples: Number of indices per pass; defaults to dataset length.
      generator: Deterministic ``random.Random`` instance.

    Raises:
      ValueError: On empty dataset, weight length mismatch, non-positive
        weight, or non-finite weight (names the offending index).
    """
    n = len(data_source)
    if n == 0:
      msg = 'WeightedSampler requires a non-empty dataset'
      raise ValueError(msg)
    if len(weights) != n:
      msg = (
        f'len(weights)={len(weights)} does not match '
        f'len(data_source)={n}; provide exactly one weight per index'
      )
      raise ValueError(msg)
    for i, w in enumerate(weights):
      if not math.isfinite(w):
        msg = f'weight at index {i} is non-finite ({w!r}); all weights must be finite and positive'
        raise ValueError(msg)
      if w <= 0:
        msg = f'weight at index {i} is non-positive ({w!r}); all weights must be strictly positive'
        raise ValueError(msg)

    self._weights = list(weights)
    self._replacement = replacement
    self._init_epoch_aware(data_source, num_samples, generator)

  def __iter__(self) -> Iterator[int]:
    """Yield dataset indices drawn according to weights.

    Each call creates a child RNG seeded with ``base_seed ^ epoch``
    for deterministic, epoch-aware sequences.

    Returns:
      Iterator of ``int`` indices.
    """
    n = len(self._data_source)
    rng = random.Random(self._base_seed ^ self._epoch)
    count = self._num_samples if self._num_samples is not None else n

    if self._replacement:
      cum = []
      total = 0.0
      for w in self._weights:
        total += w
        cum.append(total)

      def _sample_one() -> int:
        r = rng.random() * total
        lo, hi = 0, n - 1
        while lo < hi:
          mid = (lo + hi) // 2
          if cum[mid] < r:
            lo = mid + 1
          else:
            hi = mid
        return lo

      return (_sample_one() for _ in range(count))

    keys = [-math.log(rng.random()) / w for w in self._weights]
    order = sorted(range(n), key=lambda i: keys[i])
    return iter(order[:count])
