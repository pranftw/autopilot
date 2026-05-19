"""Tests for sanity check (pre-flight validation) in Trainer (Plan 16).

Covers:
  - Sanity check catches broken validation_step before training starts
  - num_sanity_val_steps=0 disables sanity check entirely
  - Batch count is capped to min(num_sanity_val_steps, dataset batches)
  - No val loader silently skips sanity check
  - Sanity batches run before first training_step
  - Metrics reset after sanity so epoch 0 train is uncontaminated
  - on_sanity_check_start/end dispatched to callbacks
  - trainer.sanity_checking flag is True during sanity phase
  - Sanity check skipped during dry_run
  - Empty val loader (zero batches) still runs without error
  - Sanity check resets _sanity_checking on exception
"""

from autopilot.core.callbacks.callback import Callback
from autopilot.core.metric import Metric
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import Datum, EvalDatum
from tests.doubles import NoOpOptimizer
from typing import Any
import pytest


class CountingMetric(Metric):
  """Metric that counts update calls."""

  higher_is_better = True

  def __init__(self) -> None:
    super().__init__()
    self.add_state('count', 0)

  def update(self, datum: Datum) -> None:
    self.count += 1

  def compute(self) -> dict[str, float]:
    return {'count': float(self.count)}

  def reset(self) -> None:
    super().reset()


class SanityModule(AutoPilotModule):
  """Module that tracks call order of training_step vs validation_step."""

  def __init__(self) -> None:
    super().__init__()
    self.metric = CountingMetric()
    self.call_log: list[str] = []

  def forward(self, *args: Any, **kwargs: Any) -> Datum:
    return EvalDatum(success=True)

  def training_step(self, batch: Any, batch_idx: int) -> Any:
    self.call_log.append(f'train:{batch_idx}')
    return EvalDatum(success=True)

  def validation_step(self, batch: Any, batch_idx: int) -> Any:
    self.call_log.append(f'val:{batch_idx}')
    return EvalDatum(success=True)

  def configure_optimizers(self) -> Any:
    return NoOpOptimizer(list(self.parameters()))


class BrokenValModule(AutoPilotModule):
  """Module whose validation_step always raises."""

  def __init__(self) -> None:
    super().__init__()
    self.train_count = 0

  def forward(self, *args: Any, **kwargs: Any) -> Datum:
    return EvalDatum(success=True)

  def training_step(self, batch: Any, batch_idx: int) -> Any:
    self.train_count += 1
    return EvalDatum(success=True)

  def validation_step(self, batch: Any, batch_idx: int) -> Any:
    msg = 'validation pipeline is broken'
    raise RuntimeError(msg)

  def configure_optimizers(self) -> Any:
    return NoOpOptimizer(list(self.parameters()))


class SanityCheckRecorder(Callback):
  """Records sanity check start/end hooks and trainer.sanity_checking state."""

  def __init__(self) -> None:
    self.hooks: list[str] = []
    self.sanity_checking_during_start: bool | None = None
    self.sanity_checking_during_end: bool | None = None

  def on_sanity_check_start(self, trainer: Any, module: Any) -> None:
    self.hooks.append('on_sanity_check_start')
    self.sanity_checking_during_start = trainer.sanity_checking

  def on_sanity_check_end(self, trainer: Any, module: Any) -> None:
    self.hooks.append('on_sanity_check_end')
    self.sanity_checking_during_end = trainer.sanity_checking

  def on_fit_start(self, trainer: Any, module: Any) -> None:
    self.hooks.append('on_fit_start')


TRAIN_BATCHES = [EvalDatum(metadata={'idx': i}) for i in range(3)]
VAL_BATCHES = [EvalDatum(metadata={'idx': i}) for i in range(5)]


class TestSanityCheckCatchesBrokenVal:
  """Sanity check detects broken validation_step before training starts."""

  def test_broken_val_raises_before_training(self) -> None:
    module = BrokenValModule()
    trainer = Trainer(num_sanity_val_steps=2)
    with pytest.raises(RuntimeError, match='validation pipeline is broken'):
      trainer.fit(
        module,
        train_dataloaders=TRAIN_BATCHES,
        val_dataloaders=VAL_BATCHES,
        max_epochs=1,
      )
    assert module.train_count == 0

  def test_exception_propagates_with_correct_type(self) -> None:
    module = BrokenValModule()
    trainer = Trainer(num_sanity_val_steps=1)
    with pytest.raises(RuntimeError):
      trainer.fit(
        module,
        train_dataloaders=TRAIN_BATCHES,
        val_dataloaders=VAL_BATCHES,
        max_epochs=1,
      )


class TestSanityCheckZeroSkips:
  """num_sanity_val_steps=0 never calls validation_step."""

  def test_zero_skips_sanity(self) -> None:
    module = SanityModule()
    trainer = Trainer(num_sanity_val_steps=0)
    trainer.fit(
      module,
      train_dataloaders=TRAIN_BATCHES,
      val_dataloaders=VAL_BATCHES,
      max_epochs=1,
    )
    val_before_train = []
    first_train_seen = False
    for entry in module.call_log:
      if entry.startswith('train:'):
        first_train_seen = True
      if entry.startswith('val:') and not first_train_seen:
        val_before_train.append(entry)
    assert len(val_before_train) == 0

  def test_zero_does_not_dispatch_sanity_callbacks(self) -> None:
    cb = SanityCheckRecorder()
    module = SanityModule()
    trainer = Trainer(callbacks=[cb], num_sanity_val_steps=0)
    trainer.fit(
      module,
      train_dataloaders=TRAIN_BATCHES,
      val_dataloaders=VAL_BATCHES,
      max_epochs=1,
    )
    assert 'on_sanity_check_start' not in cb.hooks
    assert 'on_sanity_check_end' not in cb.hooks

  def test_zero_with_no_val_loader_ok(self) -> None:
    module = SanityModule()
    trainer = Trainer(num_sanity_val_steps=0)
    trainer.fit(
      module,
      train_dataloaders=TRAIN_BATCHES,
      max_epochs=1,
    )


class TestSanityCheckLimitedBatches:
  """Sanity check processes min(num_sanity_val_steps, total batches)."""

  def test_caps_at_num_sanity_val_steps(self) -> None:
    module = SanityModule()
    trainer = Trainer(num_sanity_val_steps=2)
    trainer.fit(
      module,
      train_dataloaders=TRAIN_BATCHES,
      val_dataloaders=VAL_BATCHES,
      max_epochs=1,
    )
    sanity_vals = []
    for entry in module.call_log:
      if entry.startswith('val:'):
        sanity_vals.append(entry)
      if entry.startswith('train:'):
        break
    assert len(sanity_vals) == 2

  def test_caps_at_dataset_size_when_smaller(self) -> None:
    module = SanityModule()
    small_val = [EvalDatum() for _ in range(1)]
    trainer = Trainer(num_sanity_val_steps=10)
    trainer.fit(
      module,
      train_dataloaders=TRAIN_BATCHES,
      val_dataloaders=small_val,
      max_epochs=1,
    )
    sanity_vals = []
    for entry in module.call_log:
      if entry.startswith('val:'):
        sanity_vals.append(entry)
      if entry.startswith('train:'):
        break
    assert len(sanity_vals) == 1


class TestSanityCheckNoValLoader:
  """Sanity check is silently skipped when no val loader is configured."""

  def test_no_val_loader_skips_sanity(self) -> None:
    module = SanityModule()
    cb = SanityCheckRecorder()
    trainer = Trainer(callbacks=[cb], num_sanity_val_steps=2)
    trainer.fit(
      module,
      train_dataloaders=TRAIN_BATCHES,
      max_epochs=1,
    )
    assert 'on_sanity_check_start' not in cb.hooks
    assert 'on_sanity_check_end' not in cb.hooks

  def test_no_val_loader_still_trains(self) -> None:
    module = SanityModule()
    trainer = Trainer(num_sanity_val_steps=5)
    result = trainer.fit(
      module,
      train_dataloaders=TRAIN_BATCHES,
      max_epochs=1,
    )
    assert len(result['epochs']) == 1


class TestSanityCheckBeforeTraining:
  """Sanity validation batches occur before the first training_step."""

  def test_val_before_train_in_call_log(self) -> None:
    module = SanityModule()
    trainer = Trainer(num_sanity_val_steps=2)
    trainer.fit(
      module,
      train_dataloaders=TRAIN_BATCHES,
      val_dataloaders=VAL_BATCHES,
      max_epochs=1,
    )
    first_train_idx = None
    first_val_idx = None
    for i, entry in enumerate(module.call_log):
      if entry.startswith('train:') and first_train_idx is None:
        first_train_idx = i
      if entry.startswith('val:') and first_val_idx is None:
        first_val_idx = i
    assert first_val_idx is not None
    assert first_train_idx is not None
    assert first_val_idx < first_train_idx


class TestSanityCheckDoesNotCountMetrics:
  """Metric value after epoch 0 train excludes sanity batches."""

  def test_metrics_reset_after_sanity(self) -> None:
    module = SanityModule()
    trainer = Trainer(num_sanity_val_steps=3)
    result = trainer.fit(
      module,
      train_dataloaders=TRAIN_BATCHES,
      val_dataloaders=VAL_BATCHES,
      max_epochs=1,
    )
    epochs = result.get('epochs', [])
    assert len(epochs) == 1
    accepted_metrics = epochs[0].get('metrics', {})
    assert accepted_metrics['train_count'] == 3.0

  def test_sanity_metric_not_in_val_metrics(self) -> None:
    module = SanityModule()
    trainer = Trainer(num_sanity_val_steps=2)
    result = trainer.fit(
      module,
      train_dataloaders=TRAIN_BATCHES,
      val_dataloaders=VAL_BATCHES,
      max_epochs=1,
    )
    epochs = result.get('epochs', [])
    assert len(epochs) == 1
    val_metrics = epochs[0].get('val_metrics', {})
    assert val_metrics['count'] == 5.0


class TestSanityCheckDispatchesCallbacks:
  """Recording callback sees on_sanity_check_start/end."""

  def test_sanity_callbacks_dispatched(self) -> None:
    cb = SanityCheckRecorder()
    module = SanityModule()
    trainer = Trainer(callbacks=[cb], num_sanity_val_steps=2)
    trainer.fit(
      module,
      train_dataloaders=TRAIN_BATCHES,
      val_dataloaders=VAL_BATCHES,
      max_epochs=1,
    )
    assert 'on_sanity_check_start' in cb.hooks
    assert 'on_sanity_check_end' in cb.hooks

  def test_sanity_callbacks_before_fit_loop(self) -> None:
    cb = SanityCheckRecorder()
    module = SanityModule()
    trainer = Trainer(callbacks=[cb], num_sanity_val_steps=2)
    trainer.fit(
      module,
      train_dataloaders=TRAIN_BATCHES,
      val_dataloaders=VAL_BATCHES,
      max_epochs=1,
    )
    fit_start_idx = cb.hooks.index('on_fit_start')
    sanity_start_idx = cb.hooks.index('on_sanity_check_start')
    assert sanity_start_idx > fit_start_idx

  def test_sanity_checking_flag_true_during_callbacks(self) -> None:
    cb = SanityCheckRecorder()
    module = SanityModule()
    trainer = Trainer(callbacks=[cb], num_sanity_val_steps=2)
    trainer.fit(
      module,
      train_dataloaders=TRAIN_BATCHES,
      val_dataloaders=VAL_BATCHES,
      max_epochs=1,
    )
    assert cb.sanity_checking_during_start is True
    assert cb.sanity_checking_during_end is True

  def test_sanity_checking_flag_false_after_sanity(self) -> None:
    module = SanityModule()
    trainer = Trainer(num_sanity_val_steps=2)
    trainer.fit(
      module,
      train_dataloaders=TRAIN_BATCHES,
      val_dataloaders=VAL_BATCHES,
      max_epochs=1,
    )
    assert trainer.sanity_checking is False


class TestSanityCheckDryRun:
  """Sanity check is skipped in dry_run mode."""

  def test_dry_run_skips_sanity(self) -> None:
    cb = SanityCheckRecorder()
    module = SanityModule()
    trainer = Trainer(callbacks=[cb], dry_run=True, num_sanity_val_steps=2)
    trainer.fit(
      module,
      train_dataloaders=TRAIN_BATCHES,
      val_dataloaders=VAL_BATCHES,
      max_epochs=1,
    )
    assert 'on_sanity_check_start' not in cb.hooks
    assert 'on_sanity_check_end' not in cb.hooks


class TestSanityCheckEmptyValLoader:
  """Empty val loader (zero batches) with sanity steps > 0."""

  def test_empty_val_loader_runs_without_error(self) -> None:
    module = SanityModule()
    trainer = Trainer(num_sanity_val_steps=2)
    trainer.fit(
      module,
      train_dataloaders=TRAIN_BATCHES,
      val_dataloaders=[],
      max_epochs=1,
    )


class TestSanityCheckFlagReset:
  """_sanity_checking resets to False even if sanity phase raises."""

  def test_flag_reset_on_exception(self) -> None:
    module = BrokenValModule()
    trainer = Trainer(num_sanity_val_steps=1)
    with pytest.raises(RuntimeError):
      trainer.fit(
        module,
        train_dataloaders=TRAIN_BATCHES,
        val_dataloaders=VAL_BATCHES,
        max_epochs=1,
      )
    assert trainer.sanity_checking is False


class TestSanityCheckDefaultValue:
  """Default num_sanity_val_steps=2 when not specified."""

  def test_default_is_two(self) -> None:
    trainer = Trainer()
    assert trainer.num_sanity_val_steps == 2

  def test_custom_value(self) -> None:
    trainer = Trainer(num_sanity_val_steps=5)
    assert trainer.num_sanity_val_steps == 5

  def test_zero_value(self) -> None:
    trainer = Trainer(num_sanity_val_steps=0)
    assert trainer.num_sanity_val_steps == 0
