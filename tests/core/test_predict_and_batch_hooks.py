"""Tests for Trainer.predict() and batch hook dispatch consolidation (Plan 24).

Covers:
  - validate dispatches batch hooks with incrementing batch_idx
  - test dispatches batch hooks with incrementing batch_idx
  - predict collects outputs from predict_step
  - predict dispatches full lifecycle hooks in correct order
  - predict does not call configure_optimizers
  - predict_step NotImplementedError default on base AutoPilotModule
  - predict with DataModule triggers setup(Stage.predict) / teardown
  - predict no loader raises ConfigError
  - predict plain Module raises TypeError
  - predict restores module/trainer refs
  - predict respects eval mode
  - fit-loop validation dispatches batch hooks via consolidated path
  - predict with empty loader returns empty list
"""

from autopilot.core.callbacks.callback import Callback
from autopilot.core.errors import ConfigError
from autopilot.core.metric import Metric
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import Datum, EvalDatum
from autopilot.data.dataloader import DataLoader
from autopilot.data.datamodule import DataModule, Stage
from autopilot.data.dataset import Dataset
from tests.doubles import NoOpOptimizer, PlainModule
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


class PredictModule(AutoPilotModule):
  """Module with predict_step that echoes batch data."""

  def __init__(self) -> None:
    super().__init__()
    self.metric = CountingMetric()
    self.predict_calls: list[int] = []

  def forward(self, *args: Any, **kwargs: Any) -> Datum:
    return EvalDatum(success=True)

  def training_step(self, batch: Any, batch_idx: int) -> Any:
    return EvalDatum(success=True)

  def validation_step(self, batch: Any, batch_idx: int) -> Any:
    return EvalDatum(success=True)

  def test_step(self, batch: Any, batch_idx: int) -> Any:
    return EvalDatum(success=True)

  def predict_step(self, batch: Any, batch_idx: int) -> Any:
    self.predict_calls.append(batch_idx)
    return batch

  def configure_optimizers(self) -> Any:
    return NoOpOptimizer(list(self.parameters()))


class RaiseOnConfigureModule(AutoPilotModule):
  """Module whose configure_optimizers raises if called."""

  def __init__(self) -> None:
    super().__init__()
    self.metric = CountingMetric()

  def forward(self, *args: Any, **kwargs: Any) -> Datum:
    return EvalDatum(success=True)

  def training_step(self, batch: Any, batch_idx: int) -> Any:
    return EvalDatum(success=True)

  def validation_step(self, batch: Any, batch_idx: int) -> Any:
    return EvalDatum(success=True)

  def test_step(self, batch: Any, batch_idx: int) -> Any:
    return EvalDatum(success=True)

  def predict_step(self, batch: Any, batch_idx: int) -> Any:
    return batch

  def configure_optimizers(self) -> Any:
    msg = 'configure_optimizers should not be called during predict'
    raise RuntimeError(msg)


class FullRecordingCallback(Callback):
  """Records all hook names and arguments for verification."""

  def __init__(self) -> None:
    self.hooks: list[str] = []

  def on_validation_batch_start(
    self,
    trainer: Any,
    module: Any,
    batch: Any,
    batch_idx: int,
  ) -> None:
    self.hooks.append(f'on_validation_batch_start:{batch_idx}')

  def on_validation_batch_end(
    self,
    trainer: Any,
    module: Any,
    batch: Any,
    batch_idx: int,
  ) -> None:
    self.hooks.append(f'on_validation_batch_end:{batch_idx}')

  def on_test_batch_start(
    self,
    trainer: Any,
    module: Any,
    batch: Any,
    batch_idx: int,
  ) -> None:
    self.hooks.append(f'on_test_batch_start:{batch_idx}')

  def on_test_batch_end(
    self,
    trainer: Any,
    module: Any,
    batch: Any,
    batch_idx: int,
  ) -> None:
    self.hooks.append(f'on_test_batch_end:{batch_idx}')

  def on_predict_start(self, trainer: Any, module: Any) -> None:
    self.hooks.append('on_predict_start')

  def on_predict_end(self, trainer: Any, module: Any) -> None:
    self.hooks.append('on_predict_end')

  def on_predict_batch_start(
    self,
    trainer: Any,
    module: Any,
    batch: Any,
    batch_idx: int,
  ) -> None:
    self.hooks.append(f'on_predict_batch_start:{batch_idx}')

  def on_predict_batch_end(
    self,
    trainer: Any,
    module: Any,
    batch: Any,
    batch_idx: int,
  ) -> None:
    self.hooks.append(f'on_predict_batch_end:{batch_idx}')

  def on_validation_epoch_start(self, trainer: Any, module: Any, epoch: int) -> None:
    self.hooks.append('on_validation_epoch_start')

  def on_validation_epoch_end(self, trainer: Any, module: Any, epoch: int) -> None:
    self.hooks.append('on_validation_epoch_end')

  def on_test_epoch_start(self, trainer: Any, module: Any, epoch: int) -> None:
    self.hooks.append('on_test_epoch_start')

  def on_test_epoch_end(self, trainer: Any, module: Any, epoch: int) -> None:
    self.hooks.append('on_test_epoch_end')

  def setup(self, trainer: Any, module: Any, stage: Stage) -> None:
    self.hooks.append(f'setup:{stage.value}')

  def teardown(self, trainer: Any, module: Any, stage: Stage) -> None:
    self.hooks.append(f'teardown:{stage.value}')


class _SimpleDataset(Dataset):
  """Map-style dataset wrapping a list."""

  def __init__(self, items: list[Any]) -> None:
    self._items = items

  def __getitem__(self, index: int) -> Any:
    return self._items[index]

  def __len__(self) -> int:
    return len(self._items)


class PredictDataModule(DataModule):
  """DataModule with predict_dataloader that records setup/teardown."""

  def __init__(self, batches: list[Any]) -> None:
    self._batches = batches
    self.setup_calls: list[Stage] = []
    self.teardown_calls: list[Stage] = []

  def setup(self, stage: Stage) -> None:
    self.setup_calls.append(stage)

  def teardown(self, stage: Stage) -> None:
    self.teardown_calls.append(stage)

  def predict_dataloader(self) -> DataLoader:
    return DataLoader(_SimpleDataset(self._batches), batch_size=1)

  def val_dataloader(self) -> DataLoader:
    return DataLoader(_SimpleDataset(self._batches), batch_size=1)

  def test_dataloader(self) -> DataLoader:
    return DataLoader(_SimpleDataset(self._batches), batch_size=1)


BATCHES = [EvalDatum(metadata={'idx': i}) for i in range(3)]


class TestValidateDispatchesBatchHooks:
  """Recording callback sees start/end per batch with incrementing batch_idx."""

  def test_batch_hooks_with_incrementing_idx(self) -> None:
    module = PredictModule()
    cb = FullRecordingCallback()
    trainer = Trainer(callbacks=[cb])
    trainer.validate(module, dataloaders=BATCHES)
    batch_hooks = [h for h in cb.hooks if 'batch' in h]
    assert batch_hooks == [
      'on_validation_batch_start:0',
      'on_validation_batch_end:0',
      'on_validation_batch_start:1',
      'on_validation_batch_end:1',
      'on_validation_batch_start:2',
      'on_validation_batch_end:2',
    ]

  def test_batch_hooks_alternate_start_end(self) -> None:
    module = PredictModule()
    cb = FullRecordingCallback()
    trainer = Trainer(callbacks=[cb])
    trainer.validate(module, dataloaders=BATCHES)
    batch_hooks = [h for h in cb.hooks if 'batch' in h]
    for i in range(0, len(batch_hooks), 2):
      assert 'start' in batch_hooks[i]
      assert 'end' in batch_hooks[i + 1]


class TestTestDispatchesBatchHooks:
  """Same for test prefix."""

  def test_batch_hooks_with_incrementing_idx(self) -> None:
    module = PredictModule()
    cb = FullRecordingCallback()
    trainer = Trainer(callbacks=[cb])
    trainer.test(module, dataloaders=BATCHES)
    batch_hooks = [h for h in cb.hooks if 'batch' in h]
    assert batch_hooks == [
      'on_test_batch_start:0',
      'on_test_batch_end:0',
      'on_test_batch_start:1',
      'on_test_batch_end:1',
      'on_test_batch_start:2',
      'on_test_batch_end:2',
    ]


class TestPredictCollectsOutputs:
  """Toy dataloader length N -> list len N."""

  def test_output_length_matches_batch_count(self) -> None:
    module = PredictModule()
    trainer = Trainer()
    outputs = trainer.predict(module, dataloaders=BATCHES)
    assert len(outputs) == 3

  def test_outputs_are_batch_echoes(self) -> None:
    module = PredictModule()
    trainer = Trainer()
    outputs = trainer.predict(module, dataloaders=BATCHES)
    for i, out in enumerate(outputs):
      assert out.metadata['idx'] == i

  def test_predict_step_batch_idx_sequence(self) -> None:
    module = PredictModule()
    trainer = Trainer()
    trainer.predict(module, dataloaders=BATCHES)
    assert module.predict_calls == [0, 1, 2]

  def test_empty_loader_returns_empty_list(self) -> None:
    module = PredictModule()
    trainer = Trainer()
    outputs = trainer.predict(module, dataloaders=[])
    assert outputs == []


class TestPredictDispatchesLifecycleHooks:
  """Order: predict_start, (batch_start, batch_end)*, predict_end."""

  def test_hook_order(self) -> None:
    module = PredictModule()
    cb = FullRecordingCallback()
    trainer = Trainer(callbacks=[cb])
    trainer.predict(module, dataloaders=BATCHES)
    predict_hooks = [
      h for h in cb.hooks if 'predict' in h and 'setup' not in h and 'teardown' not in h
    ]
    assert predict_hooks == [
      'on_predict_start',
      'on_predict_batch_start:0',
      'on_predict_batch_end:0',
      'on_predict_batch_start:1',
      'on_predict_batch_end:1',
      'on_predict_batch_start:2',
      'on_predict_batch_end:2',
      'on_predict_end',
    ]

  def test_setup_teardown_dispatched(self) -> None:
    module = PredictModule()
    cb = FullRecordingCallback()
    trainer = Trainer(callbacks=[cb])
    trainer.predict(module, dataloaders=BATCHES)
    assert 'setup:predict' in cb.hooks
    assert 'teardown:predict' in cb.hooks

  def test_setup_before_predict_start(self) -> None:
    module = PredictModule()
    cb = FullRecordingCallback()
    trainer = Trainer(callbacks=[cb])
    trainer.predict(module, dataloaders=BATCHES)
    setup_idx = cb.hooks.index('setup:predict')
    predict_start_idx = cb.hooks.index('on_predict_start')
    assert setup_idx < predict_start_idx


class TestPredictNoConfigureOptimizers:
  """Module.configure_optimizers raises if called."""

  def test_no_configure_optimizers_call(self) -> None:
    module = RaiseOnConfigureModule()
    trainer = Trainer()
    outputs = trainer.predict(module, dataloaders=BATCHES)
    assert len(outputs) == 3


class TestPredictStepNotImplementedDefault:
  """Base AutoPilotModule raises helpful error."""

  def test_base_raises_not_implemented(self) -> None:
    class BareModule(AutoPilotModule):
      def __init__(self) -> None:
        super().__init__()

      def forward(self, *a: Any, **kw: Any) -> Datum:
        return EvalDatum(success=True)

      def training_step(self, batch: Any, batch_idx: int) -> Any:
        return EvalDatum(success=True)

      def configure_optimizers(self) -> Any:
        return None

    module = BareModule()
    trainer = Trainer()
    with pytest.raises(NotImplementedError, match='does not implement predict_step'):
      trainer.predict(module, dataloaders=BATCHES)

  def test_error_message_includes_class_name(self) -> None:
    class CustomModule(AutoPilotModule):
      def __init__(self) -> None:
        super().__init__()

      def forward(self, *a: Any, **kw: Any) -> Datum:
        return EvalDatum(success=True)

      def training_step(self, batch: Any, batch_idx: int) -> Any:
        return EvalDatum(success=True)

      def configure_optimizers(self) -> Any:
        return None

    module = CustomModule()
    trainer = Trainer()
    with pytest.raises(NotImplementedError, match='CustomModule'):
      trainer.predict(module, dataloaders=BATCHES)


class TestStandaloneDefaults:
  """Standalone tests for default NotImplementedError on base classes."""

  def test_predict_step_raises_not_implemented(self) -> None:
    module = AutoPilotModule()
    with pytest.raises(NotImplementedError, match='predict_step'):
      module.predict_step(Datum(), 0)

  def test_predict_dataloader_raises_not_implemented(self) -> None:
    dm = DataModule()
    with pytest.raises(NotImplementedError, match='predict_dataloader'):
      dm.predict_dataloader()


class TestPredictWithDataModule:
  """DataModule lifecycle for predict."""

  def test_datamodule_setup_predict(self) -> None:
    module = PredictModule()
    dm = PredictDataModule(BATCHES)
    trainer = Trainer()
    outputs = trainer.predict(module, datamodule=dm)
    assert Stage.predict in dm.setup_calls
    assert len(outputs) == 3

  def test_datamodule_teardown_predict(self) -> None:
    module = PredictModule()
    dm = PredictDataModule(BATCHES)
    trainer = Trainer()
    trainer.predict(module, datamodule=dm)
    assert Stage.predict in dm.teardown_calls

  def test_explicit_loader_wins_over_datamodule(self) -> None:
    module = PredictModule()
    custom_batches = [EvalDatum(metadata={'idx': i}) for i in range(7)]
    dm = PredictDataModule(BATCHES)
    trainer = Trainer()
    outputs = trainer.predict(module, dataloaders=custom_batches, datamodule=dm)
    assert len(outputs) == 7


class TestPredictErrorPaths:
  """ConfigError and TypeError for bad inputs."""

  def test_no_loader_raises_config_error(self) -> None:
    module = PredictModule()
    trainer = Trainer()
    with pytest.raises(ConfigError, match='No predict dataloader'):
      trainer.predict(module)

  def test_plain_module_raises_type_error(self) -> None:
    module = PlainModule()
    trainer = Trainer()
    with pytest.raises(TypeError, match='requires AutoPilotModule'):
      trainer.predict(module, dataloaders=BATCHES)  # type: ignore[ty:invalid-argument-type]


class TestPredictContextPreservation:
  """predict saves and restores _module, _datamodule, and module.trainer refs."""

  def test_restores_module_ref(self) -> None:
    module = PredictModule()
    trainer = Trainer()
    assert trainer._module is None
    trainer.predict(module, dataloaders=BATCHES)
    assert trainer._module is None

  def test_restores_module_trainer_ref(self) -> None:
    module = PredictModule()
    trainer = Trainer()
    assert module.trainer is None
    trainer.predict(module, dataloaders=BATCHES)
    assert module.trainer is None

  def test_restores_datamodule_ref(self) -> None:
    module = PredictModule()
    dm = PredictDataModule(BATCHES)
    trainer = Trainer()
    assert trainer._datamodule is None
    trainer.predict(module, datamodule=dm)
    assert trainer._datamodule is None


class TestPredictEvalMode:
  """module.training is False during predict; restored after."""

  def test_eval_mode_during_predict(self) -> None:
    training_flags: list[bool] = []

    class TrackingModule(AutoPilotModule):
      def __init__(self) -> None:
        super().__init__()

      def forward(self, *a: Any, **kw: Any) -> Datum:
        return EvalDatum(success=True)

      def training_step(self, batch: Any, batch_idx: int) -> Any:
        return EvalDatum(success=True)

      def predict_step(self, batch: Any, batch_idx: int) -> Any:
        training_flags.append(self.training)
        return batch

      def configure_optimizers(self) -> Any:
        return None

    module = TrackingModule()
    module.train()
    assert module.training is True
    trainer = Trainer()
    trainer.predict(module, dataloaders=BATCHES)
    assert all(f is False for f in training_flags)
    assert module.training is True


class TestFitLoopValidationBatchHooks:
  """Fit-loop validation dispatches batch hooks via consolidated path."""

  def test_fit_validation_dispatches_batch_hooks(self) -> None:
    module = PredictModule()
    cb = FullRecordingCallback()
    trainer = Trainer(callbacks=[cb], num_sanity_val_steps=0)
    trainer.fit(
      module,
      train_dataloaders=[EvalDatum()],
      val_dataloaders=BATCHES,
      max_epochs=1,
    )
    batch_hooks = [h for h in cb.hooks if 'validation_batch' in h]
    assert batch_hooks == [
      'on_validation_batch_start:0',
      'on_validation_batch_end:0',
      'on_validation_batch_start:1',
      'on_validation_batch_end:1',
      'on_validation_batch_start:2',
      'on_validation_batch_end:2',
    ]

  def test_single_dispatch_path_for_validation(self) -> None:
    """Fit-loop validation uses same _run_eval_phase as standalone validate."""
    module = PredictModule()
    standalone_cb = FullRecordingCallback()
    fit_cb = FullRecordingCallback()

    standalone_trainer = Trainer(callbacks=[standalone_cb])
    standalone_trainer.validate(module, dataloaders=BATCHES)

    fit_trainer = Trainer(callbacks=[fit_cb], num_sanity_val_steps=0)
    fit_trainer.fit(
      module,
      train_dataloaders=[EvalDatum()],
      val_dataloaders=BATCHES,
      max_epochs=1,
    )

    standalone_val_batch = [h for h in standalone_cb.hooks if 'validation_batch' in h]
    fit_val_batch = [h for h in fit_cb.hooks if 'validation_batch' in h]
    assert standalone_val_batch == fit_val_batch


class TestPredictTeardownOnError:
  """Teardown runs even when predict_step raises."""

  def test_teardown_on_error(self) -> None:
    class FailingPredictModule(AutoPilotModule):
      def __init__(self) -> None:
        super().__init__()

      def forward(self, *a: Any, **kw: Any) -> Datum:
        return EvalDatum(success=True)

      def training_step(self, batch: Any, batch_idx: int) -> Any:
        return EvalDatum(success=True)

      def predict_step(self, batch: Any, batch_idx: int) -> Any:
        msg = 'predict boom'
        raise RuntimeError(msg)

      def configure_optimizers(self) -> Any:
        return None

    module = FailingPredictModule()
    cb = FullRecordingCallback()
    trainer = Trainer(callbacks=[cb])
    with pytest.raises(RuntimeError, match='predict boom'):
      trainer.predict(module, dataloaders=BATCHES)
    assert 'teardown:predict' in cb.hooks

  def test_eval_mode_restored_on_error(self) -> None:
    class ErrorModule(AutoPilotModule):
      def __init__(self) -> None:
        super().__init__()

      def forward(self, *a: Any, **kw: Any) -> Datum:
        return EvalDatum(success=True)

      def training_step(self, batch: Any, batch_idx: int) -> Any:
        return EvalDatum(success=True)

      def predict_step(self, batch: Any, batch_idx: int) -> Any:
        msg = 'boom'
        raise ValueError(msg)

      def configure_optimizers(self) -> Any:
        return None

    module = ErrorModule()
    module.train()
    assert module.training is True
    trainer = Trainer()
    with pytest.raises(ValueError, match='boom'):
      trainer.predict(module, dataloaders=BATCHES)
    assert module.training is True
