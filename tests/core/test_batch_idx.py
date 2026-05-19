"""Tests for training_step(batch, batch_idx) and validation_step(batch, batch_idx) signatures.

Verifies that EpochLoop passes batch_idx to AutoPilotModule step methods
unconditionally, that batch_idx is 0-based and resets per epoch, and that
legacy (batch)-only signatures raise TypeError.
"""

from autopilot.core.graph import no_grad
from autopilot.core.loops.epoch import EpochLoop
from autopilot.core.loops.loop import LoopConfig
from autopilot.core.metric import Metric, MetricCollection
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum, EvalDatum
from autopilot.data.dataloader import DataLoader
from autopilot.data.dataset import Dataset
from tests.doubles import NoOpOptimizer
from typing import Any
import pytest


class _SizedDataset(Dataset[EvalDatum]):
  """Fixed-size dataset returning EvalDatum items."""

  def __init__(self, n: int) -> None:
    self._n = n

  def __len__(self) -> int:
    return self._n

  def __getitem__(self, index: int) -> EvalDatum:
    return EvalDatum(metadata={'idx': index})


class _TrainerShim:
  """Minimal trainer surface for EpochLoop without constructing Trainer."""

  def __init__(self, module: Any) -> None:
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

  def on_epoch_end(self, epoch: int, result: dict | None = None) -> list[object]:
    return []

  def should_stop_at(self, hook_method: object, **kwargs: object) -> bool:
    return False

  def emit_context(self, reason: str, **kwargs: object) -> None:
    pass

  def capture_gradient_summaries(self) -> None:
    pass

  def run_eval_phase(
    self,
    module: Any,
    dataloader: Any,
    *,
    step_method: str = 'validation_step',
    hook_prefix: str = 'validation',
    max_batches: int | None = None,
    epoch_arg: int = 0,
  ) -> dict[str, float]:
    """Eval phase for shim: discover metrics from module, run step_fn per batch."""
    step_fn = getattr(module, step_method)
    all_metrics = {
      name: m
      for name, m in module.named_modules()
      if isinstance(m, Metric) and not isinstance(m, MetricCollection)
    }
    for m in all_metrics.values():
      m.reset()
    module.eval()
    try:
      for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
          break
        with no_grad():
          try:
            step_output = step_fn(batch, batch_idx)
          except TypeError as exc:
            if 'argument' in str(exc) and 'positional' in str(exc):
              method_name = step_fn.__name__
              msg = (
                f'{type(module).__name__}.{method_name}() signature error: '
                f'expected {method_name}(self, batch, batch_idx) but got a TypeError. '
                f'Add batch_idx: int as the second parameter.'
              )
              raise TypeError(msg) from exc
            raise
        for m in all_metrics.values():
          m.update(step_output)
      result: dict[str, float] = {}
      for m in all_metrics.values():
        result.update(m.compute())
      return result
    finally:
      module.train()


class _RecordingModule(AutoPilotModule):
  """Records (batch, batch_idx) pairs from training and validation steps."""

  def __init__(self) -> None:
    super().__init__()
    self.p = Parameter()
    self.train_calls: list[tuple[Any, int]] = []
    self.val_calls: list[tuple[Any, int]] = []

  def forward(self, *args: Any, **kwargs: Any) -> Datum:
    return EvalDatum(success=True)

  def training_step(self, batch: Any, batch_idx: int) -> EvalDatum:
    self.train_calls.append((batch, batch_idx))
    return EvalDatum(success=True)

  def validation_step(self, batch: Any, batch_idx: int) -> EvalDatum:
    self.val_calls.append((batch, batch_idx))
    return EvalDatum(success=True)

  def configure_optimizers(self) -> Any:
    return NoOpOptimizer([self.p])


def _make_config(
  *,
  train_size: int = 3,
  val_size: int | None = None,
  max_epochs: int = 1,
) -> LoopConfig:
  """Build a LoopConfig with simple sized datasets."""
  train_loader = DataLoader(_SizedDataset(train_size), batch_size=1)
  val_loader = DataLoader(_SizedDataset(val_size), batch_size=1) if val_size is not None else None
  return LoopConfig(
    max_epochs=max_epochs,
    train_loader=train_loader,
    val_loader=val_loader,
  )


def test_training_step_receives_batch_idx():
  """Stub module records (batch, batch_idx) on each call; after one epoch
  with 3 batches, recorded indices are [0, 1, 2]."""
  module = _RecordingModule()
  trainer = _TrainerShim(module)
  config = _make_config(train_size=3)
  loop = EpochLoop()
  loop.run(trainer, config)

  assert len(module.train_calls) == 3
  indices = [idx for _, idx in module.train_calls]
  assert indices == [0, 1, 2]


def test_validation_step_receives_batch_idx():
  """Stub + val loader with 2 batches; validation captures [0, 1]."""
  module = _RecordingModule()
  trainer = _TrainerShim(module)
  config = _make_config(train_size=1, val_size=2)
  loop = EpochLoop()
  loop.run(trainer, config)

  assert len(module.val_calls) == 2
  indices = [idx for _, idx in module.val_calls]
  assert indices == [0, 1]


def test_batch_idx_is_zero_based():
  """First train batch receives batch_idx == 0."""
  module = _RecordingModule()
  trainer = _TrainerShim(module)
  config = _make_config(train_size=1)
  loop = EpochLoop()
  loop.run(trainer, config)

  assert len(module.train_calls) == 1
  assert module.train_calls[0][1] == 0


def test_batch_idx_resets_per_epoch():
  """Two epochs x 2 batches produces sequence 0, 1, 0, 1."""
  module = _RecordingModule()
  trainer = _TrainerShim(module)
  config = _make_config(train_size=2, max_epochs=2)
  loop = EpochLoop()
  loop.run(trainer, config)

  assert len(module.train_calls) == 4
  indices = [idx for _, idx in module.train_calls]
  assert indices == [0, 1, 0, 1]


def test_old_signature_raises_type_error():
  """Module defining training_step(self, batch) only raises TypeError
  when fit() is attempted -- Python's message mentions positional arguments."""

  class _LegacyModule(AutoPilotModule):
    def __init__(self) -> None:
      super().__init__()
      self.p = Parameter()

    def forward(self, *args: Any, **kwargs: Any) -> Datum:
      return EvalDatum(success=True)

    def training_step(self, batch: Any) -> EvalDatum:  # ty: ignore[invalid-method-override]
      return EvalDatum(success=True)

    def validation_step(self, batch: Any, batch_idx: int) -> EvalDatum:
      return EvalDatum(success=True)

    def configure_optimizers(self) -> Any:
      return NoOpOptimizer([self.p])

  module = _LegacyModule()
  trainer = _TrainerShim(module)
  config = _make_config(train_size=1)
  loop = EpochLoop()

  with pytest.raises(TypeError, match='batch_idx') as exc_info:
    loop.run(trainer, config)
  assert '_LegacyModule' in str(exc_info.value)
  assert 'training_step' in str(exc_info.value)


def test_old_validation_signature_raises_type_error():
  """Module defining validation_step(self, batch) only raises TypeError
  when validation is attempted -- improved message mentions batch_idx."""

  class _LegacyValModule(AutoPilotModule):
    def __init__(self) -> None:
      super().__init__()
      self.p = Parameter()

    def forward(self, *args: Any, **kwargs: Any) -> Datum:
      return EvalDatum(success=True)

    def training_step(self, batch: Any, batch_idx: int) -> EvalDatum:
      return EvalDatum(success=True)

    def validation_step(self, batch: Any) -> EvalDatum:  # ty: ignore[invalid-method-override]
      return EvalDatum(success=True)

    def configure_optimizers(self) -> Any:
      return NoOpOptimizer([self.p])

  module = _LegacyValModule()
  trainer = _TrainerShim(module)
  config = _make_config(train_size=1, val_size=1)
  loop = EpochLoop()

  with pytest.raises(TypeError, match='batch_idx') as exc_info:
    loop.run(trainer, config)
  assert '_LegacyValModule' in str(exc_info.value)
  assert 'validation_step' in str(exc_info.value)


def test_plain_module_path_unaffected():
  """Plain Module (not AutoPilotModule) path remains module(batch),
  unaffected by the batch_idx change."""
  from autopilot.core.module.module import Module

  class _PlainModule(Module):
    def __init__(self) -> None:
      super().__init__()
      self.call_count = 0

    def forward(self, *args: Any, **kwargs: Any) -> Datum:
      self.call_count += 1
      return EvalDatum(success=True)

  module = _PlainModule()
  trainer = _TrainerShim(_RecordingModule())
  trainer.module = module
  config = _make_config(train_size=2)
  loop = EpochLoop()
  loop.run(trainer, config)

  assert module.call_count == 2


def test_validation_batch_idx_resets_per_epoch():
  """Validation batch_idx resets to 0 each epoch."""
  module = _RecordingModule()
  trainer = _TrainerShim(module)
  config = _make_config(train_size=1, val_size=2, max_epochs=2)
  loop = EpochLoop()
  loop.run(trainer, config)

  assert len(module.val_calls) == 4
  val_indices = [idx for _, idx in module.val_calls]
  assert val_indices == [0, 1, 0, 1]


def test_legacy_predict_step_error_mentions_batch_idx():
  """Module with predict_step(self, batch) raises TypeError mentioning batch_idx."""
  from autopilot.core.trainer.trainer import Trainer

  class _LegacyPredictModule(AutoPilotModule):
    def __init__(self) -> None:
      super().__init__()
      self.p = Parameter()

    def forward(self, *args: Any, **kwargs: Any) -> Datum:
      return EvalDatum(success=True)

    def training_step(self, batch: Any, batch_idx: int) -> EvalDatum:
      return EvalDatum(success=True)

    def predict_step(self, batch: Any) -> Any:  # ty: ignore[invalid-method-override]
      return batch

    def configure_optimizers(self) -> Any:
      return NoOpOptimizer([self.p])

  module = _LegacyPredictModule()
  loader = DataLoader(_SizedDataset(1), batch_size=1)
  trainer = Trainer()

  with pytest.raises(TypeError, match='batch_idx') as exc_info:
    trainer.predict(module, dataloaders=loader)
  assert '_LegacyPredictModule' in str(exc_info.value)
  assert 'predict_step' in str(exc_info.value)


def test_non_batch_idx_type_error_propagates():
  """TypeError from inside step body (not signature) propagates unchanged."""

  class _InternalErrorModule(AutoPilotModule):
    def __init__(self) -> None:
      super().__init__()
      self.p = Parameter()

    def forward(self, *args: Any, **kwargs: Any) -> Datum:
      return EvalDatum(success=True)

    def training_step(self, batch: Any, batch_idx: int) -> EvalDatum:
      msg = 'unrelated internal error'
      raise TypeError(msg)

    def configure_optimizers(self) -> Any:
      return NoOpOptimizer([self.p])

  module = _InternalErrorModule()
  trainer = _TrainerShim(module)
  config = _make_config(train_size=1)
  loop = EpochLoop()

  with pytest.raises(TypeError, match='unrelated internal error'):
    loop.run(trainer, config)


# test_step and predict_step batch_idx tests (BUG-010)


class _TestStepRecordingModule(AutoPilotModule):
  """Records batch_idx values from test_step calls."""

  def __init__(self) -> None:
    super().__init__()
    self.p = Parameter()
    self.test_indices: list[int] = []

  def forward(self, *args: Any, **kwargs: Any) -> Datum:
    return EvalDatum(success=True)

  def training_step(self, batch: Any, batch_idx: int) -> EvalDatum:
    return EvalDatum(success=True)

  def test_step(self, batch: Any, batch_idx: int) -> EvalDatum:
    self.test_indices.append(batch_idx)
    return EvalDatum(success=True)

  def configure_optimizers(self) -> Any:
    return NoOpOptimizer([self.p])


class _PredictStepRecordingModule(AutoPilotModule):
  """Records batch_idx values from predict_step calls and returns them."""

  def __init__(self) -> None:
    super().__init__()
    self.p = Parameter()
    self.predict_indices: list[int] = []

  def forward(self, *args: Any, **kwargs: Any) -> Datum:
    return EvalDatum(success=True)

  def training_step(self, batch: Any, batch_idx: int) -> EvalDatum:
    return EvalDatum(success=True)

  def predict_step(self, batch: Any, batch_idx: int) -> int:
    self.predict_indices.append(batch_idx)
    return batch_idx

  def configure_optimizers(self) -> Any:
    return NoOpOptimizer([self.p])


def test_test_step_receives_batch_idx():
  """Trainer.test() passes correct 0-based batch_idx to test_step."""
  from autopilot.core.trainer.trainer import Trainer

  module = _TestStepRecordingModule()
  loader = DataLoader(_SizedDataset(3), batch_size=1)
  trainer = Trainer()
  trainer.test(module, dataloaders=loader)

  assert module.test_indices == [0, 1, 2]

  module.test_indices.clear()
  trainer.test(module, dataloaders=loader)
  assert module.test_indices == [0, 1, 2]


def test_predict_step_receives_batch_idx():
  """Trainer.predict() passes correct 0-based batch_idx to predict_step."""
  from autopilot.core.trainer.trainer import Trainer

  module = _PredictStepRecordingModule()
  loader = DataLoader(_SizedDataset(3), batch_size=1)
  trainer = Trainer()
  predictions = trainer.predict(module, dataloaders=loader)

  assert module.predict_indices == [0, 1, 2]
  assert predictions == [0, 1, 2]


def test_test_step_batch_idx_resets_per_invocation():
  """batch_idx restarts at 0 for each Trainer.test() invocation."""
  from autopilot.core.trainer.trainer import Trainer

  module = _TestStepRecordingModule()
  loader = DataLoader(_SizedDataset(3), batch_size=1)
  trainer = Trainer()

  trainer.test(module, dataloaders=loader)
  assert module.test_indices == [0, 1, 2]

  module.test_indices.clear()
  trainer.test(module, dataloaders=loader)
  assert module.test_indices == [0, 1, 2]
