"""Tests for WeightedSampler.

Covers length, index range, weight validation, replacement modes,
set_epoch determinism, composition with BatchSampler, and edge cases.
"""

from autopilot.data.sampler import BatchSampler, WeightedSampler
from tests.data.conftest import SizedDataset
import math
import pytest
import random


class TestWeightedSamplerLen:
  def test_len_default(self, ds5: SizedDataset) -> None:
    sampler = WeightedSampler(ds5, weights=[1.0] * 5)
    assert len(sampler) == 5

  def test_len_custom_num_samples(self, ds5: SizedDataset) -> None:
    sampler = WeightedSampler(ds5, weights=[1.0] * 5, num_samples=20)
    assert len(sampler) == 20


class TestWeightedSamplerIndicesInRange:
  def test_indices_in_range(self, ds10: SizedDataset) -> None:
    sampler = WeightedSampler(ds10, weights=[1.0] * 10, generator=random.Random(42))
    indices = list(sampler)
    assert all(0 <= idx < 10 for idx in indices)

  def test_indices_in_range_without_replacement(self, ds10: SizedDataset) -> None:
    sampler = WeightedSampler(
      ds10, weights=[1.0] * 10, replacement=False, generator=random.Random(42)
    )
    indices = list(sampler)
    assert all(0 <= idx < 10 for idx in indices)


class TestWeightedSamplerWeightsLengthMismatch:
  def test_raises_on_mismatch(self, ds5: SizedDataset) -> None:
    with pytest.raises(ValueError, match=r'len\(weights\)=3.*len\(data_source\)=5'):
      WeightedSampler(ds5, weights=[1.0, 2.0, 3.0])


class TestWeightedSamplerNonPositiveWeight:
  def test_zero_weight_raises(self, ds5: SizedDataset) -> None:
    with pytest.raises(ValueError, match=r'index 2.*non-positive'):
      WeightedSampler(ds5, weights=[1.0, 2.0, 0.0, 1.0, 1.0])

  def test_negative_weight_raises(self, ds5: SizedDataset) -> None:
    with pytest.raises(ValueError, match=r'index 0.*non-positive'):
      WeightedSampler(ds5, weights=[-1.0, 2.0, 3.0, 1.0, 1.0])


class TestWeightedSamplerReplacementFalse:
  def test_consumes_all_once(self, ds5: SizedDataset) -> None:
    sampler = WeightedSampler(
      ds5, weights=[1.0] * 5, replacement=False, generator=random.Random(99)
    )
    indices = list(sampler)
    assert sorted(indices) == [0, 1, 2, 3, 4]
    assert len(set(indices)) == 5


class TestWeightedSamplerSetEpochDeterminism:
  def test_same_epoch_same_order(self, ds10: SizedDataset) -> None:
    gen = random.Random(42)
    sampler = WeightedSampler(ds10, weights=[1.0] * 10, generator=gen)
    sampler.set_epoch(0)
    order_a = list(sampler)
    sampler.set_epoch(0)
    order_b = list(sampler)
    assert order_a == order_b

  def test_different_epoch_different_order(self, ds10: SizedDataset) -> None:
    gen = random.Random(42)
    sampler = WeightedSampler(ds10, weights=[1.0] * 10, generator=gen)
    sampler.set_epoch(0)
    order_a = list(sampler)
    sampler.set_epoch(1)
    order_b = list(sampler)
    assert order_a != order_b


class TestWeightedSamplerWithBatchSampler:
  def test_batched_indices_valid(self, ds10: SizedDataset) -> None:
    sampler = WeightedSampler(ds10, weights=[1.0] * 10, generator=random.Random(42))
    batched = BatchSampler(sampler, batch_size=3)
    all_indices = []
    for batch in batched:
      all_indices.extend(batch)
    assert all(0 <= idx < 10 for idx in all_indices)


class TestWeightedSamplerEmptyDataset:
  def test_empty_dataset_raises(self) -> None:
    ds = SizedDataset(0)
    with pytest.raises(ValueError, match='non-empty dataset'):
      WeightedSampler(ds, weights=[])


class TestWeightedSamplerNanWeight:
  def test_nan_raises(self, ds5: SizedDataset) -> None:
    with pytest.raises(ValueError, match=r'index 1.*non-finite'):
      WeightedSampler(ds5, weights=[1.0, math.nan, 1.0, 1.0, 1.0])


class TestWeightedSamplerInfWeight:
  def test_inf_raises(self, ds5: SizedDataset) -> None:
    with pytest.raises(ValueError, match=r'index 3.*non-finite'):
      WeightedSampler(ds5, weights=[1.0, 1.0, 1.0, math.inf, 1.0])

  def test_neg_inf_raises(self, ds5: SizedDataset) -> None:
    with pytest.raises(ValueError, match=r'index 0.*non-finite'):
      WeightedSampler(ds5, weights=[float('-inf'), 1.0, 1.0, 1.0, 1.0])


class TestWeightedSamplerDataLoaderIntegration:
  def test_dataloader_integration(self, ds10: SizedDataset) -> None:
    """BatchSampler(WeightedSampler(...)) yields batches without crashing."""
    sampler = WeightedSampler(ds10, weights=[1.0] * 10, generator=random.Random(7))
    batched = BatchSampler(sampler, batch_size=4)
    batches = list(batched)
    assert len(batches) == 3
    flat = []
    for batch in batches:
      flat.extend(batch)
    assert len(flat) == 10
    assert all(0 <= idx < 10 for idx in flat)
