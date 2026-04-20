"""Tests for IterableDataset + EpochLoop integration."""

from autopilot.core.loops import EpochLoop, LoopConfig
from autopilot.core.models import Datum
from autopilot.data.dataloader import DataLoader
from autopilot.data.dataset import IterableDataset
from unittest.mock import MagicMock
import pytest


class _CountingIterable(IterableDataset):
  def __init__(self, n: int) -> None:
    self._n = n

  def __iter__(self):
    for i in range(self._n):
      yield Datum(metadata={'idx': i})


class _EmptyIterable(IterableDataset):
  def __iter__(self):
    return iter([])


def _make_trainer(module=None):
  trainer = MagicMock()
  m = module or MagicMock()
  m.return_value = Datum()
  trainer.module = m
  trainer._store = None
  trainer._policy = None
  trainer._best_epoch = 0
  trainer._last_val_metrics = None
  trainer.regression_detected = False
  trainer.should_stop_at = MagicMock(return_value=False)
  trainer._dispatch = MagicMock()
  return trainer


def test_iterable_dataset_runs_without_len():
  loader = DataLoader(_CountingIterable(5), batch_size=2)
  config = LoopConfig(max_epochs=1, train_loader=loader)
  trainer = _make_trainer()
  loop = EpochLoop()
  result = loop.run(trainer, config)
  assert result['total_epochs'] == 1
  assert trainer.module.call_count == 3


def test_empty_iterable_dataset():
  loader = DataLoader(_EmptyIterable(), batch_size=2)
  config = LoopConfig(max_epochs=1, train_loader=loader)
  trainer = _make_trainer()
  loop = EpochLoop()
  result = loop.run(trainer, config)
  assert result['total_epochs'] == 1
  assert trainer.module.call_count == 0


def test_accumulation_with_iterable_dataset():
  loader = DataLoader(_CountingIterable(6), batch_size=2)
  config = LoopConfig(max_epochs=1, train_loader=loader, accumulate_grad_batches=2)
  optimizer = MagicMock()
  config.optimizer = optimizer
  trainer = _make_trainer()
  loop = EpochLoop()
  loop.run(trainer, config)
  assert optimizer.step.call_count == 2


def test_length_hint_enables_len():
  loader = DataLoader(_CountingIterable(10), batch_size=5, length_hint=10)
  assert len(loader) == 2


def test_length_hint_drop_last():
  loader = DataLoader(_CountingIterable(7), batch_size=3, length_hint=7, drop_last=True)
  assert len(loader) == 2


def test_no_length_hint_raises():
  loader = DataLoader(_CountingIterable(5), batch_size=1)
  with pytest.raises(TypeError, match='IterableDataset'):
    len(loader)
