"""Tests for the training contract: batched EvalDatum shape and loss lifecycle errors.

Plan 11 regression tests verifying:
  - batch_size > 1 yields Datum(items=[EvalDatum, ...]) shape
  - loss.backward() graph-freed error wraps with actionable guidance
  - normal training_step with Trainer-managed loss completes cleanly
  - training_step returning plain Datum (not EvalDatum) completes cleanly
"""

from autopilot.core.gradient import NumericGradient
from autopilot.core.loss import Loss
from autopilot.core.metric import Metric
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.parameter import Parameter
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import Datum, EvalDatum
from autopilot.data.dataloader import DataLoader
from autopilot.data.datamodule import DataModule
from tests.doubles import DirectNumericLoss, NoOpOptimizer
import pytest


class _PerItemCountingMetric(Metric):
  """Metric that iterates datum.items and counts per item, persisting across resets."""

  def __init__(self) -> None:
    super().__init__()
    self.add_state('_count', 0)
    self.total_item_updates = 0

  def update(self, datum: Datum) -> None:
    for _item in datum.items:
      self._count += 1
      self.total_item_updates += 1

  def compute(self) -> dict[str, float]:
    return {'count': float(self._count)}


class _BatchInspectModule(AutoPilotModule):
  """Module that records the batch shape seen in training_step."""

  def __init__(self) -> None:
    super().__init__()
    self.param = Parameter()
    self.loss = DirectNumericLoss([self.param])
    self.count_metric = _PerItemCountingMetric()
    self.seen_batches: list[Datum] = []

  def forward(self, batch: Datum) -> Datum:
    return batch

  def training_step(self, batch: Datum, batch_idx: int) -> Datum:
    self.seen_batches.append(batch)
    return self(batch)

  def configure_optimizers(self):
    return NoOpOptimizer([self.param])


class _DoubleBackwardModule(AutoPilotModule):
  """Module whose training_step manually calls loss.forward()/backward().

  This consumes the autograd graph before the Trainer's backward call,
  triggering the graph-freed RuntimeError that _process_batch should wrap.
  """

  def __init__(self) -> None:
    super().__init__()
    self.param = Parameter()
    self._loss = _GraphFreedLoss([self.param])
    self._opt = NoOpOptimizer([self.param])

  def forward(self, batch: Datum) -> Datum:
    return batch

  def training_step(self, batch: Datum, batch_idx: int) -> Datum:
    data = self(batch)
    self._loss.forward(data, batch)
    self._loss.backward()
    return data

  def configure_optimizers(self):
    return self._opt


class _GraphFreedLoss(Loss):
  """Loss that simulates graph-freed RuntimeError on second backward call.

  First backward succeeds (consuming the graph). The Trainer's subsequent
  backward call raises the graph-freed error since the graph is already consumed.
  """

  def __init__(self, params: list[Parameter] | None = None) -> None:
    super().__init__(params)
    self._backward_count = 0

  def forward(self, data: Datum, targets: Datum | None = None) -> None:
    super().forward(data, targets)

  def compute_seed_gradient(self) -> NumericGradient:
    return NumericGradient(value=1.0)

  def backward(self) -> None:
    self._backward_count += 1
    if self._backward_count > 1:
      msg = 'Trying to backward through the graph a second time, but the graph has been freed.'
      raise RuntimeError(msg)
    for p in self._loss_parameters:
      if p.requires_grad:
        p.grad = NumericGradient(value=1.0)

  def reset(self) -> None:
    super().reset()


class _NormalModule(AutoPilotModule):
  """Module that returns EvalDatum without manual loss calls."""

  def __init__(self) -> None:
    super().__init__()
    self.param = Parameter()
    self.loss = DirectNumericLoss([self.param])

  def forward(self, batch: Datum) -> Datum:
    return batch

  def training_step(self, batch: Datum, batch_idx: int) -> EvalDatum:
    return EvalDatum(success=True)

  def configure_optimizers(self):
    return NoOpOptimizer([self.param])


class _PlainDatumModule(AutoPilotModule):
  """Module that returns a plain Datum (not EvalDatum) from training_step."""

  def __init__(self) -> None:
    super().__init__()
    self.param = Parameter()
    self.loss = DirectNumericLoss([self.param])

  def forward(self, batch: Datum) -> Datum:
    return batch

  def training_step(self, batch: Datum, batch_idx: int) -> Datum:
    return Datum(items=[Datum()])

  def configure_optimizers(self):
    return NoOpOptimizer([self.param])


class _TwoSampleDataModule(DataModule):
  """DataModule yielding 2 EvalDatum samples batched together."""

  def train_dataloader(self) -> DataLoader:
    samples = [
      EvalDatum(metadata={'idx': 0}, success=True),
      EvalDatum(metadata={'idx': 1}, success=True),
    ]
    return DataLoader(samples, batch_size=2)


class _SingleSampleDataModule(DataModule):
  """DataModule yielding 1 EvalDatum sample per batch."""

  def train_dataloader(self) -> DataLoader:
    return DataLoader(
      [EvalDatum(metadata={'idx': 0}, success=True)],
      batch_size=1,
    )


class TestBatchedEvalDatumContract:
  def test_batched_evaldatum_metric_contract(self) -> None:
    """batch_size=2 yields Datum(items=[EvalDatum, EvalDatum])."""
    module = _BatchInspectModule()
    trainer = Trainer()
    trainer.fit(module, datamodule=_TwoSampleDataModule(), max_epochs=1)

    assert len(module.seen_batches) == 1
    batch = module.seen_batches[0]
    assert isinstance(batch, Datum)
    assert len(batch.items) == 2
    for item in batch.items:
      assert isinstance(item, EvalDatum)

    assert module.count_metric.total_item_updates == 2


class TestLossLifecycle:
  def test_loss_double_backward_error_message(self) -> None:
    """Manual loss calls in training_step produce actionable RuntimeError."""
    module = _DoubleBackwardModule()
    trainer = Trainer()
    with pytest.raises(RuntimeError, match='the Trainer manages the loss lifecycle') as exc_info:
      trainer.fit(module, datamodule=_SingleSampleDataModule(), max_epochs=1)

    assert exc_info.value.__cause__ is not None
    assert 'graph has been freed' in str(exc_info.value.__cause__).lower()

  def test_training_step_with_trainer_managed_loss(self) -> None:
    """Normal path: no manual loss calls, fit completes without error."""
    module = _NormalModule()
    trainer = Trainer()
    trainer.fit(module, datamodule=_SingleSampleDataModule(), max_epochs=1)

  def test_training_step_returns_datum(self) -> None:
    """Returning plain Datum (not EvalDatum) from training_step is allowed."""
    module = _PlainDatumModule()
    trainer = Trainer()
    trainer.fit(module, datamodule=_SingleSampleDataModule(), max_epochs=1)
