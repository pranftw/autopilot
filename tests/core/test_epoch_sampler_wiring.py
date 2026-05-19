"""Tests for EpochLoop sampler epoch wiring via _set_sampler_epoch_for_loader."""

from autopilot.core.loops.epoch import EpochLoop, _set_sampler_epoch_for_loader
from autopilot.core.loops.loop import LoopConfig
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from autopilot.data.dataset import Dataset
from autopilot.data.sampler import (
  BatchSampler,
  RandomSampler,
  Sampler,
  SequentialSampler,
  WeightedSampler,
)
from typing import Any
from unittest.mock import MagicMock
import random


class _SizedDataset(Dataset[int]):
  """Tiny dataset for sampler tests."""

  def __init__(self, size: int) -> None:
    self._size = size

  def __getitem__(self, index: int) -> int:
    return index

  def __len__(self) -> int:
    return self._size


class _MinimalModule(AutoPilotModule):
  """Module that does nothing for loop tests."""

  def __init__(self) -> None:
    super().__init__()
    self.p = Parameter(requires_grad=False)

  def forward(self, batch: Any) -> Datum:
    return Datum()

  def training_step(self, batch: Any, batch_idx: int) -> Datum:
    return self(batch)

  def configure_optimizers(self) -> None:
    return None


class _TrainerShim:
  """Minimal trainer for EpochLoop.run."""

  def __init__(self, module: AutoPilotModule) -> None:
    self.module = module
    self.policy = None
    self.store = None
    self.logger = None
    self.scheduler = None
    self.datamodule = None
    self.current_epoch: int = 0
    self.callbacks: list[Any] = []
    self._optimizer: Any = None
    self._cached_grad_summaries: list[dict[str, str]] = []

  def dispatch_callbacks(self, *args: object, **kwargs: object) -> list[object]:
    return []

  def on_epoch_start(self, epoch: int) -> list[object]:
    return []

  def on_epoch_end(self, epoch: int, result: Any = None) -> list[object]:
    return []

  def should_stop_at(self, hook_method: object, **kwargs: object) -> bool:
    return False

  def emit_context(self, reason: str, **kwargs: object) -> None:
    pass

  def capture_gradient_summaries(self) -> None:
    pass

  def run_eval_phase(self, module: Any, dataloader: Any, **kwargs: Any) -> dict[str, float]:
    return {}


class _SimpleLoader:
  """Loader that wraps a sampler and yields items."""

  def __init__(self, sampler: Sampler) -> None:
    self.sampler = sampler
    self._items = list(range(5))

  def __iter__(self):
    return iter([Datum() for _ in range(3)])


class _BatchSamplerLoader:
  """Loader exposing batch_sampler but not sampler (mirrors DataLoader construction)."""

  def __init__(self, bs: BatchSampler) -> None:
    self.sampler: Sampler | None = None
    self.batch_sampler: BatchSampler = bs

  def __iter__(self):
    return iter([Datum() for _ in range(3)])


class TestSetSamplerEpochForLoader:
  """Unit tests for the _set_sampler_epoch_for_loader helper."""

  def test_epoch_aware_sampler_receives_epoch(self) -> None:
    ds = _SizedDataset(10)
    sampler = RandomSampler(ds, generator=random.Random(42))
    loader = _SimpleLoader(sampler)
    _set_sampler_epoch_for_loader(loader, 5)
    assert sampler._epoch == 5

  def test_non_epoch_aware_sampler_skipped(self) -> None:
    ds = _SizedDataset(10)
    sampler = SequentialSampler(ds)
    loader = _SimpleLoader(sampler)
    _set_sampler_epoch_for_loader(loader, 3)

  def test_batch_sampler_inner_receives_epoch(self) -> None:
    ds = _SizedDataset(10)
    inner = RandomSampler(ds, generator=random.Random(42))
    batch_sampler = BatchSampler(inner, batch_size=2)
    loader = MagicMock()
    loader.sampler = batch_sampler
    loader.batch_sampler = None
    _set_sampler_epoch_for_loader(loader, 7)
    assert inner._epoch == 7

  def test_batch_sampler_non_epoch_aware_inner(self) -> None:
    ds = _SizedDataset(10)
    inner = SequentialSampler(ds)
    batch_sampler = BatchSampler(inner, batch_size=2)
    loader = MagicMock()
    loader.sampler = batch_sampler
    loader.batch_sampler = None
    _set_sampler_epoch_for_loader(loader, 3)

  def test_loader_without_sampler(self) -> None:
    loader = object()
    _set_sampler_epoch_for_loader(loader, 5)

  def test_batch_sampler_property_inner_receives_epoch(self) -> None:
    """loader.sampler is None, loader.batch_sampler=BatchSampler(RandomSampler)."""
    ds = _SizedDataset(10)
    inner = RandomSampler(ds, generator=random.Random(42))
    bs = BatchSampler(inner, batch_size=2)
    loader = _BatchSamplerLoader(bs)
    _set_sampler_epoch_for_loader(loader, 7)
    assert inner._epoch == 7

  def test_batch_sampler_precedence_over_sampler(self) -> None:
    """Both attributes set; only batch_sampler inner gets epoch."""
    ds = _SizedDataset(10)
    batch_inner = RandomSampler(ds, generator=random.Random(1))
    sampler_direct = RandomSampler(ds, generator=random.Random(2))
    bs = BatchSampler(batch_inner, batch_size=2)

    class _DualLoader:
      def __init__(self) -> None:
        self.sampler = sampler_direct
        self.batch_sampler = bs

      def __iter__(self):
        return iter([Datum() for _ in range(3)])

    loader = _DualLoader()
    _set_sampler_epoch_for_loader(loader, 9)
    assert batch_inner._epoch == 9
    assert sampler_direct._epoch == 0

  def test_batch_sampler_non_epoch_aware_inner_no_op(self) -> None:
    """batch_sampler=BatchSampler(SequentialSampler); no error, no mutation."""
    ds = _SizedDataset(10)
    inner = SequentialSampler(ds)
    bs = BatchSampler(inner, batch_size=2)
    loader = _BatchSamplerLoader(bs)
    _set_sampler_epoch_for_loader(loader, 3)

  def test_loader_without_sampler_attributes_no_op(self) -> None:
    """Plain object with no sampler/batch_sampler; no error."""
    _set_sampler_epoch_for_loader(object(), 5)


class TestEpochLoopSamplerWiring:
  """Integration tests verifying EpochLoop.run() calls set_epoch."""

  def test_epoch_loop_calls_set_epoch(self) -> None:
    ds = _SizedDataset(3)
    sampler = RandomSampler(ds, generator=random.Random(42))
    loader = _SimpleLoader(sampler)
    module = _MinimalModule()
    trainer = _TrainerShim(module)
    module.trainer = trainer

    loop = EpochLoop()
    config = LoopConfig(
      max_epochs=3,
      train_loader=loader,
    )
    loop.run(trainer, config)
    assert sampler._epoch == 2

  def test_epoch_loop_skips_non_epoch_aware_sampler(self) -> None:
    ds = _SizedDataset(3)
    sampler = SequentialSampler(ds)
    loader = _SimpleLoader(sampler)
    module = _MinimalModule()
    trainer = _TrainerShim(module)
    module.trainer = trainer

    loop = EpochLoop()
    config = LoopConfig(
      max_epochs=2,
      train_loader=loader,
    )
    loop.run(trainer, config)

  def test_weighted_sampler_different_per_epoch(self) -> None:
    ds = _SizedDataset(5)
    gen = random.Random(123)
    sampler = WeightedSampler(ds, weights=[1.0, 2.0, 3.0, 4.0, 5.0], generator=gen)

    sampler.set_epoch(0)
    epoch0_indices = list(sampler)
    sampler.set_epoch(1)
    epoch1_indices = list(sampler)
    assert epoch0_indices != epoch1_indices

  def test_set_epoch_no_val_loader(self) -> None:
    ds = _SizedDataset(3)
    sampler = RandomSampler(ds, generator=random.Random(42))
    loader = _SimpleLoader(sampler)
    module = _MinimalModule()
    trainer = _TrainerShim(module)
    module.trainer = trainer

    loop = EpochLoop()
    config = LoopConfig(
      max_epochs=2,
      train_loader=loader,
      val_loader=None,
    )
    loop.run(trainer, config)
    assert sampler._epoch == 1

  def test_set_epoch_batch_sampler_inner(self) -> None:
    ds = _SizedDataset(6)
    inner = RandomSampler(ds, generator=random.Random(99))
    batch_sampler = BatchSampler(inner, batch_size=2)

    class _BatchLoader:
      def __init__(self, bs: BatchSampler) -> None:
        self.sampler = bs

      def __iter__(self):
        return iter([Datum() for _ in range(3)])

    loader = _BatchLoader(batch_sampler)
    module = _MinimalModule()
    trainer = _TrainerShim(module)
    module.trainer = trainer

    loop = EpochLoop()
    config = LoopConfig(
      max_epochs=4,
      train_loader=loader,
    )
    loop.run(trainer, config)
    assert inner._epoch == 3
