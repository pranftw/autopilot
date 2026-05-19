"""Tests for standalone Trainer.validate() and Trainer.test() (Plan 13).

Covers:
  - validate returns metrics from a toy Metric
  - validate does not call configure_optimizers
  - validate dispatches epoch-level callbacks
  - validate with DataModule triggers setup(Stage.validate)
  - test calls test_step with correct batch_idx sequence
  - test returns metrics matching controlled batch loop
  - test dispatches epoch-level callbacks
  - validate respects eval mode during _run_eval_phase
  - fit test phase uses test_step via _run_eval_phase
  - on_test_start / on_test_end module hooks fire during test
  - batch-level callback hooks fire during eval phases
  - max_batches caps iteration in _run_eval_phase
  - ConfigError when no dataloader is available
  - TypeError when module is not AutoPilotModule
  - _collect_module_metrics matches previous behavior
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
from tests.doubles import NoopEvalModule, NoOpOptimizer, PlainModule
from typing import Any
import pytest


class CountingMetric(Metric):
  """Metric that counts update calls and returns the count."""

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


class EvalModule(AutoPilotModule):
  """Module with validation_step, test_step, and a counting metric."""

  def __init__(self) -> None:
    super().__init__()
    self.metric = CountingMetric()
    self.val_calls: list[int] = []
    self.test_calls: list[int] = []
    self.training_during_eval: list[bool] = []

  def forward(self, *args: Any, **kwargs: Any) -> Datum:
    return EvalDatum(success=True)

  def training_step(self, batch: Any, batch_idx: int) -> Any:
    return EvalDatum(success=True)

  def validation_step(self, batch: Any, batch_idx: int) -> Any:
    self.val_calls.append(batch_idx)
    self.training_during_eval.append(self.training)
    return EvalDatum(success=True)

  def test_step(self, batch: Any, batch_idx: int) -> Any:
    self.test_calls.append(batch_idx)
    self.training_during_eval.append(self.training)
    return EvalDatum(success=True)

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

  def configure_optimizers(self) -> Any:
    msg = 'configure_optimizers should not be called during eval'
    raise RuntimeError(msg)


class RecordingCallback(Callback):
  """Records which hooks were called."""

  def __init__(self) -> None:
    self.hooks: list[str] = []

  def on_validation_epoch_start(self, trainer: Any, module: Any, epoch: int) -> None:
    self.hooks.append('on_validation_epoch_start')

  def on_validation_epoch_end(self, trainer: Any, module: Any, epoch: int) -> None:
    self.hooks.append('on_validation_epoch_end')

  def on_test_epoch_start(self, trainer: Any, module: Any, epoch: int) -> None:
    self.hooks.append('on_test_epoch_start')

  def on_test_epoch_end(self, trainer: Any, module: Any, epoch: int) -> None:
    self.hooks.append('on_test_epoch_end')

  def on_validation_batch_start(
    self, trainer: Any, module: Any, batch: Any, batch_idx: int
  ) -> None:
    self.hooks.append(f'on_validation_batch_start:{batch_idx}')

  def on_validation_batch_end(self, trainer: Any, module: Any, batch: Any, batch_idx: int) -> None:
    self.hooks.append(f'on_validation_batch_end:{batch_idx}')

  def on_test_batch_start(self, trainer: Any, module: Any, batch: Any, batch_idx: int) -> None:
    self.hooks.append(f'on_test_batch_start:{batch_idx}')

  def on_test_batch_end(self, trainer: Any, module: Any, batch: Any, batch_idx: int) -> None:
    self.hooks.append(f'on_test_batch_end:{batch_idx}')

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


class StubDataModule(DataModule):
  """DataModule that records setup/teardown calls."""

  def __init__(self, batches: list[Any]) -> None:
    self._batches = batches
    self.setup_calls: list[Stage] = []
    self.teardown_calls: list[Stage] = []

  def setup(self, stage: Stage) -> None:
    self.setup_calls.append(stage)

  def teardown(self, stage: Stage) -> None:
    self.teardown_calls.append(stage)

  def val_dataloader(self) -> DataLoader:
    return DataLoader(_SimpleDataset(self._batches), batch_size=1)

  def test_dataloader(self) -> DataLoader:
    return DataLoader(_SimpleDataset(self._batches), batch_size=1)


class ModuleHookTracker(AutoPilotModule):
  """Module that records on_validation_start/end and on_test_start/end."""

  def __init__(self) -> None:
    super().__init__()
    self.metric = CountingMetric()
    self.hooks: list[str] = []

  def forward(self, *args: Any, **kwargs: Any) -> Datum:
    return EvalDatum(success=True)

  def training_step(self, batch: Any, batch_idx: int) -> Any:
    return EvalDatum(success=True)

  def validation_step(self, batch: Any, batch_idx: int) -> Any:
    return EvalDatum(success=True)

  def test_step(self, batch: Any, batch_idx: int) -> Any:
    return EvalDatum(success=True)

  def configure_optimizers(self) -> Any:
    return NoOpOptimizer(list(self.parameters()))

  def on_validation_start(self) -> None:
    self.hooks.append('on_validation_start')

  def on_validation_end(self) -> None:
    self.hooks.append('on_validation_end')

  def on_test_start(self) -> None:
    self.hooks.append('on_test_start')

  def on_test_end(self) -> None:
    self.hooks.append('on_test_end')


BATCHES = [EvalDatum(metadata={'idx': i}) for i in range(3)]


class TestValidateReturnsMetrics:
  """validate() returns dict[str, float] with expected key from a toy Metric."""

  def test_returns_metric_dict(self) -> None:
    module = EvalModule()
    trainer = Trainer()
    result = trainer.validate(module, dataloaders=BATCHES)
    assert isinstance(result, dict)
    assert 'count' in result
    assert result['count'] == 3.0

  def test_returns_correct_count_for_different_batch_sizes(self) -> None:
    module = EvalModule()
    trainer = Trainer()
    batches = [EvalDatum() for _ in range(5)]
    result = trainer.validate(module, dataloaders=batches)
    assert result['count'] == 5.0

  def test_returns_empty_metrics_for_empty_loader(self) -> None:
    module = EvalModule()
    trainer = Trainer()
    result = trainer.validate(module, dataloaders=[])
    assert result == {}


class TestValidateNoOptimizerNeeded:
  """Module whose configure_optimizers raises -- validate must not call it."""

  def test_no_configure_optimizers_call(self) -> None:
    module = RaiseOnConfigureModule()
    trainer = Trainer()
    result = trainer.validate(module, dataloaders=BATCHES)
    assert result['count'] == 3.0


class TestValidateDispatchesCallbacks:
  """Recording callback observes on_validation_epoch_start/end."""

  def test_epoch_callbacks_dispatched(self) -> None:
    module = EvalModule()
    cb = RecordingCallback()
    trainer = Trainer(callbacks=[cb])
    trainer.validate(module, dataloaders=BATCHES)
    assert 'on_validation_epoch_start' in cb.hooks
    assert 'on_validation_epoch_end' in cb.hooks

  def test_batch_callbacks_dispatched(self) -> None:
    module = EvalModule()
    cb = RecordingCallback()
    trainer = Trainer(callbacks=[cb])
    trainer.validate(module, dataloaders=BATCHES)
    assert 'on_validation_batch_start:0' in cb.hooks
    assert 'on_validation_batch_end:0' in cb.hooks
    assert 'on_validation_batch_start:2' in cb.hooks
    assert 'on_validation_batch_end:2' in cb.hooks

  def test_setup_teardown_callbacks_dispatched(self) -> None:
    module = EvalModule()
    cb = RecordingCallback()
    trainer = Trainer(callbacks=[cb])
    trainer.validate(module, dataloaders=BATCHES)
    assert 'setup:validate' in cb.hooks
    assert 'teardown:validate' in cb.hooks


class TestValidateWithDatamodule:
  """Stub datamodule records setup with Stage.validate."""

  def test_datamodule_setup_called(self) -> None:
    module = EvalModule()
    dm = StubDataModule(BATCHES)
    trainer = Trainer()
    result = trainer.validate(module, datamodule=dm)
    assert Stage.validate in dm.setup_calls
    assert result['count'] == 3.0

  def test_datamodule_teardown_called(self) -> None:
    module = EvalModule()
    dm = StubDataModule(BATCHES)
    trainer = Trainer()
    trainer.validate(module, datamodule=dm)
    assert Stage.validate in dm.teardown_calls

  def test_explicit_loader_wins_over_datamodule(self) -> None:
    module = EvalModule()
    custom_batches = [EvalDatum() for _ in range(7)]
    dm = StubDataModule(BATCHES)
    trainer = Trainer()
    result = trainer.validate(module, dataloaders=custom_batches, datamodule=dm)
    assert result['count'] == 7.0


class TestTestCallsTestStep:
  """test_step invoked with correct batch_idx sequence."""

  def test_batch_idx_sequence(self) -> None:
    module = EvalModule()
    trainer = Trainer()
    trainer.test(module, dataloaders=BATCHES)
    assert module.test_calls == [0, 1, 2]

  def test_test_returns_metrics(self) -> None:
    module = EvalModule()
    trainer = Trainer()
    result = trainer.test(module, dataloaders=BATCHES)
    assert result['count'] == 3.0


class TestTestDispatchesCallbacks:
  """Recording callback observes on_test_epoch_start/end."""

  def test_epoch_callbacks_dispatched(self) -> None:
    module = EvalModule()
    cb = RecordingCallback()
    trainer = Trainer(callbacks=[cb])
    trainer.test(module, dataloaders=BATCHES)
    assert 'on_test_epoch_start' in cb.hooks
    assert 'on_test_epoch_end' in cb.hooks

  def test_batch_callbacks_dispatched(self) -> None:
    module = EvalModule()
    cb = RecordingCallback()
    trainer = Trainer(callbacks=[cb])
    trainer.test(module, dataloaders=BATCHES)
    assert 'on_test_batch_start:0' in cb.hooks
    assert 'on_test_batch_end:0' in cb.hooks

  def test_setup_teardown_callbacks_dispatched(self) -> None:
    module = EvalModule()
    cb = RecordingCallback()
    trainer = Trainer(callbacks=[cb])
    trainer.test(module, dataloaders=BATCHES)
    assert 'setup:test' in cb.hooks
    assert 'teardown:test' in cb.hooks


class TestValidateRespectsEvalMode:
  """module.training is False during _run_eval_phase batches; restored after."""

  def test_eval_mode_during_validation(self) -> None:
    module = EvalModule()
    trainer = Trainer()
    module.train()
    assert module.training is True
    trainer.validate(module, dataloaders=BATCHES)
    assert all(t is False for t in module.training_during_eval)
    assert module.training is True

  def test_eval_mode_during_test(self) -> None:
    module = EvalModule()
    trainer = Trainer()
    module.train()
    trainer.test(module, dataloaders=BATCHES)
    assert all(t is False for t in module.training_during_eval)
    assert module.training is True


class TestFitTestPhaseUsesTestStep:
  """Short fit with test_dataloaders asserts test stub saw test_step."""

  def test_fit_tail_calls_test_step(self) -> None:
    module = EvalModule()
    trainer = Trainer()
    result = trainer.fit(
      module,
      train_dataloaders=[EvalDatum()],
      test_dataloaders=BATCHES,
      max_epochs=1,
    )
    assert module.test_calls == [0, 1, 2]
    assert 'test_results' in result
    assert result['test_results']['count'] == 3.0

  def test_fit_tail_dispatches_test_callbacks(self) -> None:
    module = EvalModule()
    cb = RecordingCallback()
    trainer = Trainer(callbacks=[cb])
    trainer.fit(
      module,
      train_dataloaders=[EvalDatum()],
      test_dataloaders=BATCHES,
      max_epochs=1,
    )
    assert 'on_test_epoch_start' in cb.hooks
    assert 'on_test_epoch_end' in cb.hooks


class TestModuleHooks:
  """on_validation_start/end and on_test_start/end fire during eval."""

  def test_validation_hooks(self) -> None:
    module = ModuleHookTracker()
    trainer = Trainer()
    trainer.validate(module, dataloaders=BATCHES)
    assert module.hooks == ['on_validation_start', 'on_validation_end']

  def test_test_hooks(self) -> None:
    module = ModuleHookTracker()
    trainer = Trainer()
    trainer.test(module, dataloaders=BATCHES)
    assert module.hooks == ['on_test_start', 'on_test_end']


class TestMaxBatches:
  """max_batches caps iteration in _run_eval_phase."""

  def test_max_batches_limits_iteration(self) -> None:
    module = EvalModule()
    trainer = Trainer()
    trainer._module = module
    result = trainer.run_eval_phase(
      module,
      BATCHES,
      step_method='validation_step',
      hook_prefix='validation',
      max_batches=2,
    )
    assert result['count'] == 2.0
    assert len(module.val_calls) == 2


class TestErrorPaths:
  """ConfigError and TypeError for bad inputs."""

  def test_validate_no_loader_raises(self) -> None:
    module = EvalModule()
    trainer = Trainer()
    with pytest.raises(ConfigError, match='No validation dataloader'):
      trainer.validate(module)

  def test_test_no_loader_raises(self) -> None:
    module = EvalModule()
    trainer = Trainer()
    with pytest.raises(ConfigError, match='No test dataloader'):
      trainer.test(module)

  def test_validate_plain_module_raises(self) -> None:
    module = PlainModule()
    trainer = Trainer()
    with pytest.raises(TypeError, match='requires AutoPilotModule'):
      trainer.validate(module, dataloaders=BATCHES)  # type: ignore[ty:invalid-argument-type]

  def test_test_plain_module_raises(self) -> None:
    module = PlainModule()
    trainer = Trainer()
    with pytest.raises(TypeError, match='requires AutoPilotModule'):
      trainer.test(module, dataloaders=BATCHES)  # type: ignore[ty:invalid-argument-type]

  def test_missing_step_method_raises(self) -> None:
    module = NoopEvalModule()
    trainer = Trainer()
    trainer._module = module
    with pytest.raises(ConfigError, match='no callable'):
      trainer.run_eval_phase(
        module,
        BATCHES,
        step_method='nonexistent_step',
        hook_prefix='test',
      )


class TestCollectModuleMetrics:
  """_collect_module_metrics matches previous behavior."""

  def test_collects_metrics(self) -> None:
    module = EvalModule()
    trainer = Trainer()
    metrics, metadata = trainer._collect_module_metrics(module)
    assert 'metric' in metrics
    assert isinstance(metrics['metric'], CountingMetric)
    assert metadata == {'metric': True}

  def test_excludes_loss(self) -> None:
    from autopilot.core.loss import Loss

    class ModuleWithLoss(AutoPilotModule):
      def __init__(self) -> None:
        super().__init__()
        self.loss = Loss()
        self.metric = CountingMetric()

      def forward(self, *a: Any, **kw: Any) -> Datum:
        return EvalDatum(success=True)

      def training_step(self, batch: Any, batch_idx: int) -> Any:
        return EvalDatum(success=True)

      def validation_step(self, batch: Any, batch_idx: int) -> Any:
        return EvalDatum(success=True)

      def configure_optimizers(self) -> Any:
        return None

    module = ModuleWithLoss()
    trainer = Trainer()
    metrics, _ = trainer._collect_module_metrics(module)
    for name in metrics:
      assert not isinstance(metrics[name], Loss)


class TestDatamoduleTest:
  """Datamodule lifecycle for test()."""

  def test_datamodule_setup_stage_test(self) -> None:
    module = EvalModule()
    dm = StubDataModule(BATCHES)
    trainer = Trainer()
    trainer.test(module, datamodule=dm)
    assert Stage.test in dm.setup_calls

  def test_datamodule_teardown_stage_test(self) -> None:
    module = EvalModule()
    dm = StubDataModule(BATCHES)
    trainer = Trainer()
    trainer.test(module, datamodule=dm)
    assert Stage.test in dm.teardown_calls


class TestContextPreservation:
  """validate/test save and restore _module, _datamodule, and module.trainer refs."""

  def test_validate_restores_module_ref(self) -> None:
    module = EvalModule()
    trainer = Trainer()
    assert trainer._module is None
    trainer.validate(module, dataloaders=BATCHES)
    assert trainer._module is None

  def test_test_restores_module_ref(self) -> None:
    module = EvalModule()
    trainer = Trainer()
    assert trainer._module is None
    trainer.test(module, dataloaders=BATCHES)
    assert trainer._module is None

  def test_validate_restores_module_trainer_ref(self) -> None:
    module = EvalModule()
    trainer = Trainer()
    assert module.trainer is None
    trainer.validate(module, dataloaders=BATCHES)
    assert module.trainer is None

  def test_test_restores_module_trainer_ref(self) -> None:
    module = EvalModule()
    trainer = Trainer()
    assert module.trainer is None
    trainer.test(module, dataloaders=BATCHES)
    assert module.trainer is None

  def test_teardown_runs_on_error(self) -> None:
    """Teardown callbacks fire even when eval phase errors out."""

    class FailingModule(AutoPilotModule):
      def __init__(self) -> None:
        super().__init__()
        self.metric = CountingMetric()

      def forward(self, *a: Any, **kw: Any) -> Datum:
        return EvalDatum(success=True)

      def training_step(self, batch: Any, batch_idx: int) -> Any:
        return EvalDatum(success=True)

      def validation_step(self, batch: Any, batch_idx: int) -> Any:
        msg = 'forced failure'
        raise RuntimeError(msg)

      def configure_optimizers(self) -> Any:
        return None

    module = FailingModule()
    cb = RecordingCallback()
    trainer = Trainer(callbacks=[cb])
    with pytest.raises(RuntimeError, match='forced failure'):
      trainer.validate(module, dataloaders=BATCHES)
    assert 'teardown:validate' in cb.hooks

  def test_eval_mode_restored_on_error(self) -> None:
    """module.train() is restored even when step raises."""

    class ErrorModule(AutoPilotModule):
      def __init__(self) -> None:
        super().__init__()
        self.metric = CountingMetric()

      def forward(self, *a: Any, **kw: Any) -> Datum:
        return EvalDatum(success=True)

      def training_step(self, batch: Any, batch_idx: int) -> Any:
        return EvalDatum(success=True)

      def validation_step(self, batch: Any, batch_idx: int) -> Any:
        msg = 'boom'
        raise ValueError(msg)

      def configure_optimizers(self) -> Any:
        return None

    module = ErrorModule()
    module.train()
    assert module.training is True
    trainer = Trainer()
    with pytest.raises(ValueError, match='boom'):
      trainer.validate(module, dataloaders=BATCHES)
    assert module.training is True
