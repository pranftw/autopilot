"""Tests for DataLoader.sampler and DataLoader.batch_sampler property contract."""

from autopilot.data.dataloader import DataLoader
from autopilot.data.sampler import BatchSampler, RandomSampler, SequentialSampler
from tests.data.conftest import SizedDataset
import random


def test_sampler_returns_explicit_sampler() -> None:
  """DataLoader(ds, sampler=s, batch_size=2) -> loader.sampler is s."""
  ds = SizedDataset(6)
  sampler = RandomSampler(ds, generator=random.Random(42))
  loader = DataLoader(ds, batch_size=2, sampler=sampler)
  assert loader.sampler is sampler


def test_sampler_none_when_only_batch_sampler() -> None:
  """DataLoader(ds, batch_sampler=bs) -> loader.sampler is None."""
  ds = SizedDataset(6)
  inner = SequentialSampler(ds)
  bs = BatchSampler(inner, batch_size=2)
  loader = DataLoader(ds, batch_sampler=bs)
  assert loader.sampler is None


def test_sampler_none_when_implicit_sequential() -> None:
  """DataLoader(ds, batch_size=2) with no sampler args -> loader.sampler is None."""
  ds = SizedDataset(6)
  loader = DataLoader(ds, batch_size=2)
  assert loader.sampler is None


def test_batch_sampler_returns_explicit_batch_sampler() -> None:
  """batch_sampler=bs -> loader.batch_sampler is bs."""
  ds = SizedDataset(6)
  inner = SequentialSampler(ds)
  bs = BatchSampler(inner, batch_size=2)
  loader = DataLoader(ds, batch_sampler=bs)
  assert loader.batch_sampler is bs


def test_batch_sampler_none_when_not_provided() -> None:
  """Default loader -> loader.batch_sampler is None."""
  ds = SizedDataset(6)
  loader = DataLoader(ds, batch_size=2)
  assert loader.batch_sampler is None


def test_both_properties_reflect_construction_when_both_set() -> None:
  """Pass both sampler=s1 and batch_sampler=bs; both are reflected."""
  ds = SizedDataset(6)
  s1 = RandomSampler(ds, generator=random.Random(0))
  inner = SequentialSampler(ds)
  bs = BatchSampler(inner, batch_size=2)
  loader = DataLoader(ds, sampler=s1, batch_sampler=bs)
  assert loader.sampler is s1
  assert loader.batch_sampler is bs


def test_sampler_property_object_identity() -> None:
  """Repeated access returns the same object."""
  ds = SizedDataset(6)
  sampler = RandomSampler(ds, generator=random.Random(42))
  loader = DataLoader(ds, batch_size=2, sampler=sampler)
  first = loader.sampler
  second = loader.sampler
  assert first is sampler
  assert first is second


def test_batch_sampler_property_object_identity() -> None:
  """Repeated access returns the same BatchSampler instance."""
  ds = SizedDataset(6)
  inner = SequentialSampler(ds)
  bs = BatchSampler(inner, batch_size=2)
  loader = DataLoader(ds, batch_sampler=bs)
  first = loader.batch_sampler
  second = loader.batch_sampler
  assert first is bs
  assert first is second
