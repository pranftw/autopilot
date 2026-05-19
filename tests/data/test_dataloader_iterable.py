"""Tests for IterableDataset + EpochLoop integration."""

from autopilot.core.loops.epoch import EpochLoop
from autopilot.core.loops.loop import LoopConfig
from autopilot.core.types import Datum
from autopilot.data.dataloader import DataLoader
from autopilot.data.dataset import IterableDataset
from tests.doubles import EvalDatumIterable
from unittest.mock import MagicMock
import pytest


class _EmptyIterable(IterableDataset):
  def __iter__(self):
    return iter([])


def _make_trainer(module=None):
  trainer = MagicMock()
  m = module or MagicMock()
  m.return_value = Datum()
  trainer.module = m
  trainer.policy = None
  trainer.experiment = None
  trainer.should_stop_at = MagicMock(return_value=False)
  trainer.dispatch_callbacks = MagicMock()
  return trainer


def test_iterable_dataset_runs_without_len():
  loader = DataLoader(EvalDatumIterable(5), batch_size=2)
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
  loader = DataLoader(EvalDatumIterable(6), batch_size=2)
  config = LoopConfig(max_epochs=1, train_loader=loader, accumulate_grad_batches=2)
  optimizer = MagicMock()
  config.optimizer = optimizer
  trainer = _make_trainer()
  loop = EpochLoop()
  loop.run(trainer, config)
  assert optimizer.step.call_count == 2


def test_no_length_hint_raises():
  """Iterable datasets without __len__ raise TypeError on len()."""
  loader = DataLoader(EvalDatumIterable(5), batch_size=1)
  with pytest.raises(TypeError, match='IterableDataset'):
    len(loader)
