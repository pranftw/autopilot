"""Tests for integration test fixtures and doubles.

Verifies:
  - integration_workspace_with_store creates expected directory layout
  - minimal_trainer_stack produces correct metrics after Trainer.fit
  - Shared doubles subclass the correct base classes
"""

from autopilot.ai.experiment import AutoPilotExperiment
from autopilot.ai.parameter import PathParameter
from autopilot.core.callbacks.store import StoreCheckpointCallback
from autopilot.core.config import AutoPilotConfig
from autopilot.core.forest import Forest
from autopilot.core.loss import Loss
from autopilot.core.metric import Metric
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.optimizer import Optimizer
from autopilot.core.store.base import Store
from autopilot.core.trainer.trainer import Trainer
from autopilot.data.datamodule import DataModule
from tests.integration.doubles import (
  FixedAccuracyMetric,
  MinimalPathModule,
  NoOpOptimizer,
  NumericSeedLoss,
  TwoBatchTrainDatamodule,
  minimal_trainer_stack,
)
import pytest


def test_integration_workspace_creates_dirs(integration_workspace_with_store) -> None:
  """Fixture creates workspace/params dir, usable store and forest."""
  config, path_param, store, forest = integration_workspace_with_store

  params_dir = config.workspace / 'params'
  assert params_dir.is_dir()

  seed_file = params_dir / 'seed.txt'
  assert seed_file.exists()
  assert seed_file.read_text(encoding='utf-8') == 'seed'

  assert isinstance(config, AutoPilotConfig)
  assert isinstance(path_param, PathParameter)
  assert isinstance(store, Store)
  assert isinstance(forest, Forest)

  assert str(params_dir) == path_param.source
  assert forest.list_trees() == []


def test_minimal_trainer_stack_metric_requires_validation(
  integration_workspace_with_store,
) -> None:
  """Metric.compute() raises without validation data (no update calls)."""
  config, path_param, store, _ = integration_workspace_with_store
  module, dm = minimal_trainer_stack(path_param, accuracy=0.91)

  experiment = AutoPilotExperiment(experiment_id='fixture-metric-test')
  cb = StoreCheckpointCallback()
  trainer = Trainer(
    callbacks=[cb],
    experiment=experiment,
    store=store,
    config=config,
  )
  trainer.fit(module, datamodule=dm, max_epochs=1)

  with pytest.raises(RuntimeError, match='without prior update'):
    module.accuracy.compute()


def test_doubles_subclass_sanity() -> None:
  """Shared doubles inherit from the correct base classes."""
  assert issubclass(NumericSeedLoss, Loss)
  assert issubclass(NoOpOptimizer, Optimizer)
  assert issubclass(FixedAccuracyMetric, Metric)
  assert issubclass(MinimalPathModule, AutoPilotModule)
  assert issubclass(TwoBatchTrainDatamodule, DataModule)


def test_minimal_trainer_stack_default_accuracy(
  integration_workspace_with_store,
) -> None:
  """Default accuracy=0.5 when not specified."""
  _, path_param, _, _ = integration_workspace_with_store
  module, _ = minimal_trainer_stack(path_param)
  assert module.accuracy._value == pytest.approx(0.5)


def test_two_batch_datamodule_provides_batches() -> None:
  """TwoBatchTrainDatamodule provides exactly 2 train batches."""
  dm = TwoBatchTrainDatamodule()
  train_batches = list(dm.train_dataloader())
  assert len(train_batches) == 2


def test_fixture_forest_starts_empty(integration_workspace_with_store) -> None:
  """Forest from fixture has no trees before create_tree is called."""
  _, _, _, forest = integration_workspace_with_store
  assert forest.list_trees() == []
  assert forest.active is None
