"""E2E tests: real DataLoader + EpochLoop batch sampler epoch wiring.

Verifies that ``_set_sampler_epoch_for_loader`` correctly threads through
real ``DataLoader`` instances with ``batch_sampler`` construction paths,
using ``EpochLoop.run()`` with minimal module/trainer shims.
"""

from autopilot.core.loops.epoch import EpochLoop
from autopilot.core.loops.loop import LoopConfig
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum, EvalDatum
from autopilot.data.dataloader import DataLoader
from autopilot.data.dataset import Dataset
from autopilot.data.sampler import BatchSampler, RandomSampler, WeightedSampler
from tests.data.conftest import SizedDataset
from typing import Any
import random


class _EvalDatumDataset(Dataset[EvalDatum]):
  """Dataset yielding EvalDatum items with index metadata."""

  def __init__(self, size: int) -> None:
    self._size = size

  def __getitem__(self, index: int) -> EvalDatum:
    return EvalDatum(metadata={'idx': index})

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


def test_real_dataloader_batch_sampler_epoch_wiring() -> None:
  """DataLoader(batch_sampler=BatchSampler(RandomSampler)) gets set_epoch each epoch."""
  ds = _EvalDatumDataset(6)
  inner = RandomSampler(ds, generator=random.Random(99))
  batch_sampler = BatchSampler(inner, batch_size=2)
  loader = DataLoader(ds, batch_sampler=batch_sampler)
  assert loader.sampler is None
  assert loader.batch_sampler is batch_sampler

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


def test_real_dataloader_direct_sampler_epoch_wiring() -> None:
  """DataLoader(ds, sampler=RandomSampler, batch_size=2) regression guard."""
  ds = _EvalDatumDataset(6)
  sampler = RandomSampler(ds, generator=random.Random(42))
  loader = DataLoader(ds, batch_size=2, sampler=sampler)

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


def test_real_dataloader_batch_sampler_shuffles_differently_per_epoch() -> None:
  """After set_epoch wiring, epoch 0 and epoch 1 produce different index orders."""
  ds = SizedDataset(8)
  inner = RandomSampler(ds, generator=random.Random(42))
  batch_sampler = BatchSampler(inner, batch_size=2)

  def _int_collate(batch: list[Any]) -> Datum:
    return Datum(items=batch)

  loader = DataLoader(ds, batch_sampler=batch_sampler, collate_fn=_int_collate)

  def collect_indices(epoch: int) -> list[Any]:
    inner.set_epoch(epoch)
    indices: list[Any] = []
    for batch in loader:
      indices.extend(batch.items)
    return indices

  epoch0 = collect_indices(0)
  epoch1 = collect_indices(1)
  assert epoch0 != epoch1


def test_epoch_loop_real_dataloader_train_and_val_batch_samplers() -> None:
  """Both train_loader and val_loader with batch_sampler paths receive set_epoch."""
  train_ds = _EvalDatumDataset(6)
  train_inner = RandomSampler(train_ds, generator=random.Random(10))
  train_bs = BatchSampler(train_inner, batch_size=2)
  train_loader = DataLoader(train_ds, batch_sampler=train_bs)

  val_ds = _EvalDatumDataset(4)
  val_inner = RandomSampler(val_ds, generator=random.Random(20))
  val_bs = BatchSampler(val_inner, batch_size=2)
  val_loader = DataLoader(val_ds, batch_sampler=val_bs)

  module = _MinimalModule()
  trainer = _TrainerShim(module)
  module.trainer = trainer

  loop = EpochLoop()
  config = LoopConfig(
    max_epochs=3,
    train_loader=train_loader,
    val_loader=val_loader,
  )
  loop.run(trainer, config)
  assert train_inner._epoch == 2
  assert val_inner._epoch == 2


def test_epoch_loop_real_dataloader_weighted_batch_sampler() -> None:
  """WeightedSampler inside BatchSampler gets set_epoch via batch_sampler path."""
  ds = _EvalDatumDataset(6)
  weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  inner = WeightedSampler(ds, weights=weights, generator=random.Random(55))
  batch_sampler = BatchSampler(inner, batch_size=2)
  loader = DataLoader(ds, batch_sampler=batch_sampler)

  module = _MinimalModule()
  trainer = _TrainerShim(module)
  module.trainer = trainer

  loop = EpochLoop()
  config = LoopConfig(
    max_epochs=3,
    train_loader=loader,
  )
  loop.run(trainer, config)
  assert inner._epoch == 2
