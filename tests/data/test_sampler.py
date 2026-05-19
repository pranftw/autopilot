"""Tests for data sampler hierarchy.

Covers SequentialSampler, RandomSampler, BatchSampler, SubsetSampler, and
their composition.  Uses ``SizedDataset`` and fixtures from
``tests/data/conftest.py``.
"""

from autopilot.data.sampler import (
  BatchSampler,
  RandomSampler,
  SequentialSampler,
  SubsetSampler,
)
from tests.data.conftest import SizedDataset
import pytest
import random

# 4.1 SequentialSampler


class TestSequentialSampler:
  """SequentialSampler tests (plan items 1-5)."""

  def test_sequential_yields_ordered(self, ds5: SizedDataset) -> None:
    assert list(SequentialSampler(ds5)) == [0, 1, 2, 3, 4]

  def test_sequential_len(self, ds5: SizedDataset) -> None:
    assert len(SequentialSampler(ds5)) == 5

  def test_sequential_empty(self, empty_ds: SizedDataset) -> None:
    sampler = SequentialSampler(empty_ds)
    assert list(sampler) == []
    assert len(sampler) == 0

  def test_sequential_single(self) -> None:
    ds = SizedDataset(1)
    assert list(SequentialSampler(ds)) == [0]

  def test_sequential_multiple_iterations(self, ds5: SizedDataset) -> None:
    sampler = SequentialSampler(ds5)
    first = list(sampler)
    second = list(sampler)
    assert first == [0, 1, 2, 3, 4]
    assert second == [0, 1, 2, 3, 4]


# 4.2 RandomSampler


class TestRandomSampler:
  """RandomSampler tests (plan items 6-14 + ValueError edge case)."""

  def test_random_deterministic_with_seed(self, ds5: SizedDataset) -> None:
    g1 = random.Random(12345)
    g2 = random.Random(12345)
    s1 = RandomSampler(ds5, generator=g1)
    s2 = RandomSampler(ds5, generator=g2)
    assert list(s1) == list(s2)

  def test_random_different_seeds(self, ds5: SizedDataset) -> None:
    s1 = RandomSampler(ds5, generator=random.Random(1))
    s2 = RandomSampler(ds5, generator=random.Random(2))
    assert list(s1) != list(s2), 'different seeds should produce different orderings for 5 items'

  def test_random_set_epoch_changes_order(self, ds5: SizedDataset) -> None:
    gen = random.Random(42)
    sampler = RandomSampler(ds5, generator=gen)
    sampler.set_epoch(0)
    order0 = list(sampler)
    sampler.set_epoch(1)
    order1 = list(sampler)
    assert order0 != order1, 'different epochs should produce different orderings'

  def test_random_set_epoch_reproducible(self, ds5: SizedDataset) -> None:
    gen = random.Random(99)
    sampler = RandomSampler(ds5, generator=gen)
    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(0)
    second = list(sampler)
    assert first == second

  def test_random_replacement(self, ds5: SizedDataset) -> None:
    sampler = RandomSampler(ds5, replacement=True, num_samples=10)
    assert len(sampler) == 10
    drawn = list(sampler)
    assert len(drawn) == 10
    assert all(0 <= idx < 5 for idx in drawn)

  def test_random_len_default(self, ds5: SizedDataset) -> None:
    assert len(RandomSampler(ds5)) == 5

  def test_random_len_custom(self, ds5: SizedDataset) -> None:
    sampler = RandomSampler(ds5, num_samples=3, generator=random.Random(7))
    assert len(sampler) == 3
    drawn = list(sampler)
    assert len(drawn) == 3

  def test_random_all_indices_covered(self, ds5: SizedDataset) -> None:
    order = list(RandomSampler(ds5, generator=random.Random(42)))
    assert sorted(order) == [0, 1, 2, 3, 4]
    assert len(set(order)) == 5

  def test_random_empty_dataset(self, empty_ds: SizedDataset) -> None:
    assert list(RandomSampler(empty_ds)) == []

  def test_random_num_samples_exceeds_raises(self, ds5: SizedDataset) -> None:
    sampler = RandomSampler(ds5, num_samples=10, replacement=False)
    with pytest.raises(ValueError, match='cannot sample 10'):
      list(sampler)


# 4.3 BatchSampler


class TestBatchSampler:
  """BatchSampler tests (plan items 15-24)."""

  def test_batch_basic(self, ds10: SizedDataset) -> None:
    batches = list(BatchSampler(SequentialSampler(ds10), batch_size=3, drop_last=False))
    assert batches[0] == [0, 1, 2]
    assert batches[1] == [3, 4, 5]
    assert batches[2] == [6, 7, 8]
    assert batches[3] == [9]
    assert len(batches) == 4

  def test_batch_drop_last(self, ds10: SizedDataset) -> None:
    batches = list(BatchSampler(SequentialSampler(ds10), batch_size=3, drop_last=True))
    assert len(batches) == 3
    assert all(len(b) == 3 for b in batches)

  def test_batch_exact_division(self, ds9: SizedDataset) -> None:
    batches = list(BatchSampler(SequentialSampler(ds9), batch_size=3, drop_last=False))
    assert len(batches) == 3
    assert all(len(b) == 3 for b in batches)

  def test_batch_len_no_drop(self, ds10: SizedDataset) -> None:
    assert len(BatchSampler(SequentialSampler(ds10), 3, drop_last=False)) == 4

  def test_batch_len_drop(self, ds10: SizedDataset) -> None:
    assert len(BatchSampler(SequentialSampler(ds10), 3, drop_last=True)) == 3

  def test_batch_size_one(self, ds5: SizedDataset) -> None:
    batches = list(BatchSampler(SequentialSampler(ds5), batch_size=1))
    assert batches == [[0], [1], [2], [3], [4]]

  def test_batch_invalid_size(self) -> None:
    with pytest.raises(ValueError, match='batch_size must be >= 1'):
      BatchSampler(SequentialSampler(SizedDataset(3)), batch_size=0)

  def test_batch_negative_size(self) -> None:
    with pytest.raises(ValueError, match='batch_size must be >= 1'):
      BatchSampler(SequentialSampler(SizedDataset(3)), batch_size=-1)

  def test_batch_composes_random(self, ds10: SizedDataset) -> None:
    def make_batches() -> list[list[int]]:
      return list(BatchSampler(RandomSampler(ds10, generator=random.Random(42)), 4))

    assert make_batches() == make_batches()

  def test_batch_empty(self, empty_ds: SizedDataset) -> None:
    bs = BatchSampler(SequentialSampler(empty_ds), batch_size=3)
    assert list(bs) == []
    assert len(bs) == 0


# 4.4 SubsetSampler


class TestSubsetSampler:
  """SubsetSampler tests (plan items 25-30)."""

  def test_subset_yields_specified(self) -> None:
    assert list(SubsetSampler([2, 5, 7])) == [2, 5, 7]

  def test_subset_with_generator_shuffles(self) -> None:
    ordered = [1, 2, 3, 4, 5]
    shuffled = list(SubsetSampler(ordered, generator=random.Random(42)))
    assert shuffled != [1, 2, 3, 4, 5], 'shuffled order should differ from input for this seed'

  def test_subset_without_generator_ordered(self) -> None:
    assert list(SubsetSampler([9, 7, 5, 3, 1])) == [9, 7, 5, 3, 1]

  def test_subset_len(self) -> None:
    assert len(SubsetSampler([0, 1, 2])) == 3

  def test_subset_empty(self) -> None:
    assert list(SubsetSampler([])) == []

  def test_subset_single(self) -> None:
    assert list(SubsetSampler([3])) == [3]


# 4.5 Composition


class TestComposition:
  """Composition tests (plan items 31-32)."""

  def test_subset_into_batch(self) -> None:
    subset = SubsetSampler([1, 3, 5, 7, 9])
    bs = BatchSampler(subset, batch_size=2, drop_last=False)
    batches = list(bs)
    assert batches == [[1, 3], [5, 7], [9]]
    assert len(bs) == 3

  def test_random_into_batch(self, ds9: SizedDataset) -> None:
    def make_batches() -> list[list[int]]:
      return list(BatchSampler(RandomSampler(ds9, generator=random.Random(99)), batch_size=3))

    assert make_batches() == make_batches()
