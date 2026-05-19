"""Shared data test doubles and fixtures for sampler/dataloader tests."""

from autopilot.data.dataset import Dataset
import pytest


class SizedDataset(Dataset[int]):
  """Tiny map-style dataset returning its index as the item."""

  def __init__(self, size: int) -> None:
    self._size = size

  def __getitem__(self, index: int) -> int:
    if index < 0 or index >= self._size:
      msg = f'index {index} out of range for SizedDataset of size {self._size}'
      raise IndexError(msg)
    return index

  def __len__(self) -> int:
    return self._size


@pytest.fixture
def ds5() -> SizedDataset:
  """5-item dataset."""
  return SizedDataset(5)


@pytest.fixture
def ds9() -> SizedDataset:
  """9-item dataset."""
  return SizedDataset(9)


@pytest.fixture
def ds10() -> SizedDataset:
  """10-item dataset."""
  return SizedDataset(10)


@pytest.fixture
def empty_ds() -> SizedDataset:
  """0-item dataset."""
  return SizedDataset(0)
