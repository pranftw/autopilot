"""Tests for sub-plan 08 section 2.2: EpochAwareSamplerMixin.

Covers:
- set_epoch round-trip, length, deterministic behavior across epochs.
- Negative epoch raises ValueError.
- Repeated set_epoch to same value is idempotent.
- Length matches dataset for both RandomSampler and WeightedSampler.
- Mixin provides single implementation of set_epoch and __len__.
"""

from autopilot.data.sampler import (
  EpochAwareSamplerMixin,
  RandomSampler,
  WeightedSampler,
)
from tests.data.conftest import SizedDataset
import pytest
import random


class TestMixinInheritance:
  """Both samplers inherit from EpochAwareSamplerMixin."""

  def test_random_sampler_is_mixin(self, ds5: SizedDataset) -> None:
    sampler = RandomSampler(ds5)
    assert isinstance(sampler, EpochAwareSamplerMixin)

  def test_weighted_sampler_is_mixin(self, ds5: SizedDataset) -> None:
    sampler = WeightedSampler(ds5, weights=[1.0] * 5)
    assert isinstance(sampler, EpochAwareSamplerMixin)


class TestMixinSetEpochRoundTrip:
  """set_epoch stores the epoch and affects iteration order."""

  def test_random_set_epoch_round_trip(self, ds5: SizedDataset) -> None:
    sampler = RandomSampler(ds5, generator=random.Random(42))
    sampler.set_epoch(3)
    assert sampler._epoch == 3

  def test_weighted_set_epoch_round_trip(self, ds5: SizedDataset) -> None:
    sampler = WeightedSampler(ds5, weights=[1.0] * 5, generator=random.Random(42))
    sampler.set_epoch(7)
    assert sampler._epoch == 7


class TestMixinNegativeEpochRejected:
  """set_epoch(-1) raises ValueError."""

  def test_random_sampler_negative_epoch(self, ds5: SizedDataset) -> None:
    sampler = RandomSampler(ds5)
    with pytest.raises(ValueError, match='epoch must be >= 0'):
      sampler.set_epoch(-1)

  def test_weighted_sampler_negative_epoch(self, ds5: SizedDataset) -> None:
    sampler = WeightedSampler(ds5, weights=[1.0] * 5)
    with pytest.raises(ValueError, match='epoch must be >= 0'):
      sampler.set_epoch(-1)

  def test_random_sampler_large_negative(self, ds5: SizedDataset) -> None:
    sampler = RandomSampler(ds5)
    with pytest.raises(ValueError, match='epoch must be >= 0'):
      sampler.set_epoch(-100)


class TestMixinEpochZero:
  """Epoch 0 is valid and produces deterministic results."""

  def test_random_epoch_zero(self, ds5: SizedDataset) -> None:
    sampler = RandomSampler(ds5, generator=random.Random(42))
    sampler.set_epoch(0)
    order1 = list(sampler)
    sampler.set_epoch(0)
    order2 = list(sampler)
    assert order1 == order2

  def test_weighted_epoch_zero(self, ds10: SizedDataset) -> None:
    sampler = WeightedSampler(
      ds10, weights=[1.0] * 10, replacement=False, generator=random.Random(42)
    )
    sampler.set_epoch(0)
    order1 = list(sampler)
    sampler.set_epoch(0)
    order2 = list(sampler)
    assert order1 == order2


class TestMixinRepeatedSetEpoch:
  """Repeated set_epoch to the same value is idempotent."""

  def test_random_repeated_set_epoch(self, ds5: SizedDataset) -> None:
    sampler = RandomSampler(ds5, generator=random.Random(42))
    sampler.set_epoch(5)
    order1 = list(sampler)
    sampler.set_epoch(5)
    order2 = list(sampler)
    assert order1 == order2

  def test_weighted_repeated_set_epoch(self, ds5: SizedDataset) -> None:
    sampler = WeightedSampler(
      ds5, weights=[1.0] * 5, replacement=False, generator=random.Random(42)
    )
    sampler.set_epoch(5)
    order1 = list(sampler)
    sampler.set_epoch(5)
    order2 = list(sampler)
    assert order1 == order2


class TestMixinLengthMatchesDataset:
  """__len__ returns dataset length when num_samples is not set."""

  def test_random_len_default(self, ds5: SizedDataset) -> None:
    assert len(RandomSampler(ds5)) == 5

  def test_random_len_custom(self, ds5: SizedDataset) -> None:
    assert len(RandomSampler(ds5, num_samples=3)) == 3

  def test_weighted_len_default(self, ds5: SizedDataset) -> None:
    assert len(WeightedSampler(ds5, weights=[1.0] * 5)) == 5

  def test_weighted_len_custom(self, ds5: SizedDataset) -> None:
    assert len(WeightedSampler(ds5, weights=[1.0] * 5, num_samples=20)) == 20


class TestMixinSingleImplementation:
  """set_epoch and __len__ resolve from the mixin, not duplicated per sampler."""

  def test_set_epoch_is_mixin_method(self) -> None:
    assert RandomSampler.set_epoch is EpochAwareSamplerMixin.set_epoch
    assert WeightedSampler.set_epoch is EpochAwareSamplerMixin.set_epoch

  def test_len_is_mixin_method(self) -> None:
    assert RandomSampler.__len__ is EpochAwareSamplerMixin.__len__
    assert WeightedSampler.__len__ is EpochAwareSamplerMixin.__len__


class TestMixinDeterminismAcrossEpochs:
  """Different epochs produce different orderings."""

  def test_random_different_epochs(self, ds10: SizedDataset) -> None:
    sampler = RandomSampler(ds10, generator=random.Random(42))
    sampler.set_epoch(0)
    order0 = list(sampler)
    sampler.set_epoch(1)
    order1 = list(sampler)
    assert order0 != order1

  def test_weighted_different_epochs(self, ds10: SizedDataset) -> None:
    sampler = WeightedSampler(
      ds10, weights=[1.0] * 10, replacement=False, generator=random.Random(42)
    )
    sampler.set_epoch(0)
    order0 = list(sampler)
    sampler.set_epoch(1)
    order1 = list(sampler)
    assert order0 != order1
