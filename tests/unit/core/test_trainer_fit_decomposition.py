"""Tests for Trainer.fit decomposition helpers."""

from autopilot.core.experiment import Experiment
from autopilot.core.loops.loop import LoopConfig
from autopilot.core.metric import Metric, MetricCollection
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.parameter import Parameter
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.tree import Tree
from autopilot.core.types import Datum, EvalDatum
from autopilot.data.dataloader import DataLoader
from autopilot.data.datamodule import DataModule, Stage
from autopilot.data.dataset import Dataset
from tests.doubles import NoopEvalModule, NoOpOptimizer
from typing import Any, cast
from unittest.mock import Mock, patch
import pytest


class TestPrepareDatamoduleFit:
  def test_noop_when_datamodule_is_none(self) -> None:
    trainer = Trainer()
    trainer._prepare_datamodule_fit(None)

  def test_calls_prepare_data_and_setup(self) -> None:
    trainer = Trainer()
    dm = Mock()
    dm.prepare_data = Mock()
    dm.setup = Mock()
    trainer._prepare_datamodule_fit(dm)
    dm.prepare_data.assert_called_once()
    dm.setup.assert_called_once_with(Stage.fit)

  def test_always_calls_prepare_data(self) -> None:
    trainer = Trainer()
    dm = Mock()
    dm.prepare_data = Mock()
    dm.setup = Mock()
    trainer._prepare_datamodule_fit(dm)
    dm.prepare_data.assert_called_once()

  def test_always_calls_setup(self) -> None:
    trainer = Trainer()
    dm = Mock()
    dm.prepare_data = Mock()
    dm.setup = Mock()
    trainer._prepare_datamodule_fit(dm)
    dm.setup.assert_called_once_with(Stage.fit)


class TestResolveFitLoaders:
  def test_explicit_loaders_win(self) -> None:
    trainer = Trainer()
    dm = Mock()
    dm.train_dataloader = Mock(return_value='dm_train')
    dm.val_dataloader = Mock(return_value='dm_val')
    train, val, _test = trainer._resolve_fit_loaders('explicit_train', 'explicit_val', dm)
    assert train == 'explicit_train'
    assert val == 'explicit_val'
    dm.train_dataloader.assert_not_called()
    dm.val_dataloader.assert_not_called()

  def test_none_loaders_fall_back_to_datamodule(self) -> None:
    trainer = Trainer()
    dm = Mock()
    dm.train_dataloader = Mock(return_value='dm_train')
    dm.val_dataloader = Mock(return_value='dm_val')
    train, val, _test = trainer._resolve_fit_loaders(None, None, dm)
    assert train == 'dm_train'
    assert val == 'dm_val'

  def test_none_datamodule_returns_none_loaders(self) -> None:
    trainer = Trainer()
    train, val, test = trainer._resolve_fit_loaders(None, None, None)
    assert train is None
    assert val is None
    assert test is None

  def test_partial_explicit_override(self) -> None:
    trainer = Trainer()
    dm = Mock()
    dm.train_dataloader = Mock(return_value='dm_train')
    dm.val_dataloader = Mock(return_value='dm_val')
    train, val, _test = trainer._resolve_fit_loaders('my_train', None, dm)
    assert train == 'my_train'
    assert val == 'dm_val'


class TestConfigureOptimizerAndMetrics:
  def test_basic_no_optimizer_no_metrics(self) -> None:
    trainer = Trainer()
    module = NoopEvalModule()
    optimizer, loss_fn, metrics, metadata = trainer._configure_optimizer_and_metrics(module)
    assert optimizer is None
    assert loss_fn is None
    assert metrics == {}
    assert metadata == {}

  def test_excludes_metric_collection_children(self) -> None:
    trainer = Trainer()

    class _MetricA(Metric):
      higher_is_better = True

      def update(self, datum: Datum) -> None:
        pass

      def _compute(self):
        return {'a': 1.0}

    class _MetricB(Metric):
      higher_is_better = False

      def update(self, datum: Datum) -> None:
        pass

      def _compute(self):
        return {'b': 0.5}

    class _ModWithCollection(AutoPilotModule):
      def __init__(self):
        super().__init__()
        self.collection = MetricCollection([_MetricA(), _MetricB()])

      def forward(self, *args, **kwargs):
        return EvalDatum(success=True)

      def training_step(self, batch, batch_idx):
        return EvalDatum(success=True)

      def configure_optimizers(self):
        return None

    module = _ModWithCollection()
    _, _, metrics, _metadata = trainer._configure_optimizer_and_metrics(module)
    assert 'collection' in metrics
    child_names = [k for k in metrics if k.startswith('collection.')]
    assert child_names == []

  def test_metric_metadata_includes_higher_is_better(self) -> None:
    trainer = Trainer()

    class _Metric(Metric):
      higher_is_better = True

      def update(self, datum: Datum) -> None:
        pass

      def _compute(self):
        return {'acc': 0.9}

    class _ModWithMetric(AutoPilotModule):
      def __init__(self):
        super().__init__()
        self.acc = _Metric()

      def forward(self, *args, **kwargs):
        return EvalDatum(success=True)

      def training_step(self, batch, batch_idx):
        return EvalDatum(success=True)

      def configure_optimizers(self):
        return None

    module = _ModWithMetric()
    _, _, metrics, metadata = trainer._configure_optimizer_and_metrics(module)
    assert 'acc' in metrics
    assert metadata['acc'] is True


class TestBuildLoopConfig:
  def test_returns_loop_config_with_all_fields(self) -> None:
    trainer = Trainer(dry_run=True, accumulate_grad_batches=4)
    module = NoopEvalModule()
    config = trainer._build_loop_config(
      module=module,
      train_loader=['batch1'],
      val_loader=['val_batch'],
      max_epochs=5,
      fit_ctx={'key': 'value'},
      optimizer='opt',
      loss_fn='loss',
      metrics=cast(Any, {'m': 'metric_obj'}),
      metric_metadata={'m': True},
    )
    assert isinstance(config, LoopConfig)
    assert config.max_epochs == 5
    assert config.dry_run is True
    assert config.train_loader == ['batch1']
    assert config.val_loader == ['val_batch']
    assert config.loss == 'loss'
    assert config.optimizer == 'opt'
    assert config.metrics == {'m': 'metric_obj'}
    assert config.accumulate_grad_batches == 4
    assert config.metric_metadata == {'m': True}
    assert config.ctx == {'key': 'value'}


class TestTeardownFit:
  def test_calls_module_teardown(self) -> None:
    trainer = Trainer()
    module = Mock(spec=AutoPilotModule)
    trainer._teardown_fit(module, None)
    module.teardown.assert_called_once()

  def test_calls_datamodule_teardown_when_present(self) -> None:
    trainer = Trainer()
    module = Mock(spec=AutoPilotModule)
    dm = Mock()
    dm.teardown = Mock()
    trainer._teardown_fit(module, dm)
    module.teardown.assert_called_once()
    dm.teardown.assert_called_once_with(Stage.fit)

  def test_always_calls_datamodule_teardown(self) -> None:
    trainer = Trainer()
    module = Mock(spec=AutoPilotModule)
    dm = Mock()
    dm.teardown = Mock()
    trainer._teardown_fit(module, dm)
    module.teardown.assert_called_once()
    dm.teardown.assert_called_once_with(Stage.fit)


class TestFitTeardownRunsOnException:
  def test_teardown_runs_when_loop_raises(self) -> None:
    module = NoopEvalModule()
    module.teardown = cast(Any, Mock())
    dm = Mock()
    dm.prepare_data = Mock()
    dm.setup = Mock()
    dm.teardown = Mock()
    dm.train_dataloader = Mock(return_value=[])
    dm.val_dataloader = Mock(return_value=[])
    dm.test_dataloader = Mock(side_effect=NotImplementedError)

    trainer = Trainer()
    with patch.object(trainer, '_loop') as mock_loop:
      mock_loop.run = Mock(side_effect=RuntimeError('train boom'))
      with pytest.raises(RuntimeError, match='train boom'):
        trainer.fit(module, datamodule=dm, max_epochs=1)

    module.teardown.assert_called_once()
    dm.teardown.assert_called_once_with(Stage.fit)

  def test_teardown_runs_on_success(self) -> None:
    module = NoopEvalModule()
    module.teardown = cast(Any, Mock())
    dm = Mock()
    dm.prepare_data = Mock()
    dm.setup = Mock()
    dm.teardown = Mock()
    dm.train_dataloader = Mock(return_value=[])
    dm.val_dataloader = Mock(return_value=[])
    dm.test_dataloader = Mock(side_effect=NotImplementedError)

    trainer = Trainer()
    trainer.fit(module, datamodule=dm, max_epochs=1)

    module.teardown.assert_called_once()
    dm.teardown.assert_called_once_with(Stage.fit)


class TestTreeUpdateBeforeTeardown:
  def test_tree_update_before_teardown_on_success(self) -> None:
    call_order: list[str] = []

    module = NoopEvalModule()
    original_teardown = module.teardown

    def tracking_teardown():
      call_order.append('teardown')
      original_teardown()

    module.teardown = cast(Any, tracking_teardown)

    tree = Mock(spec=Tree)

    def tracking_update(*args, **kwargs):
      call_order.append('update')

    tree.update = tracking_update

    experiment = Experiment(experiment_id='exp-1')

    trainer = Trainer(tree=tree, experiment=experiment)
    trainer.fit(module, train_dataloaders=[], max_epochs=1)

    assert call_order.index('update') < call_order.index('teardown')


class TestFitUsesExperimentContextManager:
  def test_fit_uses_context_manager_for_experiment(self) -> None:
    """Verify fit() uses the experiment context manager (start via __enter__)."""
    experiment = Experiment(experiment_id='exp-cm')
    trainer = Trainer(experiment=experiment)
    trainer.fit(NoopEvalModule(), train_dataloaders=[], max_epochs=1)
    assert experiment.status.value == 'completed'
    assert experiment.started_at is not None
    assert experiment.completed_at is not None

  def test_fit_without_experiment_uses_nullcontext(self) -> None:
    """Verify fit() works without experiment (nullcontext path)."""
    trainer = Trainer()
    result = trainer.fit(NoopEvalModule(), train_dataloaders=[], max_epochs=1)
    assert result['total_epochs'] == 1


class TestCompleteExperimentSuccess:
  def test_noop_when_no_experiment(self) -> None:
    trainer = Trainer()
    trainer._complete_experiment_success({'epochs': [{'metrics': {'x': 1}}]})

  def test_calls_complete_with_last_epoch_metrics(self) -> None:
    experiment = Mock()
    trainer = Trainer(experiment=experiment)
    trainer._complete_experiment_success({'epochs': [{'metrics': {'a': 1}}, {'metrics': {'a': 2}}]})
    experiment.complete.assert_called_once_with({'a': 2})

  def test_complete_with_none_when_no_epochs(self) -> None:
    experiment = Mock()
    trainer = Trainer(experiment=experiment)
    trainer._complete_experiment_success({'epochs': []})
    experiment.complete.assert_called_once_with(None)


class _TinyDataset(Dataset[EvalDatum]):
  def __len__(self) -> int:
    return 1

  def __getitem__(self, index: int) -> EvalDatum:
    return EvalDatum(metadata={'idx': index})


class _MinimalValModule(AutoPilotModule):
  def __init__(self) -> None:
    super().__init__()
    self.p = Parameter()
    self.setup_called = False

  def forward(self, *args: Any, **kwargs: Any) -> Datum:
    return EvalDatum(success=True)

  def training_step(self, batch: Any, batch_idx: int) -> EvalDatum:
    return EvalDatum(success=True)

  def validation_step(self, batch: Any, batch_idx: int) -> EvalDatum:
    return EvalDatum(success=True)

  def predict_step(self, batch: Any, batch_idx: int) -> Any:
    return batch

  def configure_optimizers(self) -> Any:
    return NoOpOptimizer([self.p])


class TestHasattrRemovalRegression:
  """Verify direct calls to setup/teardown work after hasattr removal."""

  def test_validate_calls_setup_without_hasattr(self) -> None:
    dm = Mock(spec=DataModule)
    dm.setup = Mock()
    dm.teardown = Mock()
    dm.val_dataloader = Mock(return_value=DataLoader(_TinyDataset(), batch_size=1))
    module = _MinimalValModule()
    trainer = Trainer()
    trainer.validate(module, datamodule=dm)
    dm.setup.assert_called_once_with(Stage.validate)

  def test_predict_uses_direct_trainer_attr(self) -> None:
    module = _MinimalValModule()
    assert module.trainer is None
    loader = DataLoader(_TinyDataset(), batch_size=1)
    trainer = Trainer()
    trainer.predict(module, dataloaders=loader)
    assert module.trainer is None
