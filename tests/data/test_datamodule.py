"""Tests for DataModule, Stage enum, and ensure_stage."""

from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import EvalDatum
from autopilot.data.dataloader import DataLoader
from autopilot.data.datamodule import DataModule, Stage, ensure_stage
from autopilot.data.dataset import ListDataset
from typing import Any
import pytest


class _LifecycleModule(DataModule):
  def __init__(self) -> None:
    self.events: list[str] = []

  def prepare_data(self) -> None:
    self.events.append('prepare_data')

  def setup(self, stage: Stage) -> None:
    self.events.append(f'setup:{stage.value}')

  def train_dataloader(self) -> DataLoader:
    self.events.append('train_dataloader')
    data = [EvalDatum(metadata={'v': 1}), EvalDatum(metadata={'v': 2})]
    return DataLoader(data, batch_size=1)

  def val_dataloader(self) -> DataLoader:
    return DataLoader([EvalDatum(metadata={'v': 3})], batch_size=1)

  def test_dataloader(self) -> DataLoader:
    return DataLoader([EvalDatum(metadata={'v': 4})], batch_size=1)

  def teardown(self, stage: Stage) -> None:
    self.events.append(f'teardown:{stage.value}')


# 4.1 Stage, validation, and DataModule hooks


def test_stage_enum_values():
  """Each Stage member's .value matches the expected string."""
  assert Stage.fit.value == 'fit'
  assert Stage.validate.value == 'validate'
  assert Stage.test.value == 'test'
  assert Stage.predict.value == 'predict'


def test_ensure_stage_accepts_stage_members():
  """ensure_stage(Stage.validate) returns Stage.validate."""
  for member in Stage:
    assert ensure_stage(member) is member


def test_ensure_stage_rejects_string():
  """ensure_stage('fit') raises TypeError mentioning Stage."""
  with pytest.raises(TypeError, match='Stage'):
    ensure_stage('fit')


def test_ensure_stage_rejects_arbitrary_object():
  """ensure_stage(object()) raises TypeError."""
  with pytest.raises(TypeError, match='Stage'):
    ensure_stage(object())


def test_datamodule_setup_signature_uses_stage():
  """Subclass setup invoked with Stage.fit executes body."""
  dm = _LifecycleModule()
  dm.setup(Stage.fit)
  assert 'setup:fit' in dm.events


def test_datamodule_teardown_signature_uses_stage():
  """teardown(Stage.fit) smoke test."""
  dm = _LifecycleModule()
  dm.teardown(Stage.fit)
  assert 'teardown:fit' in dm.events


def test_datamodule_state_dict_default_empty():
  """Fresh DataModule instance state_dict() == {}."""
  dm = DataModule()
  assert dm.state_dict() == {}


def test_datamodule_load_state_dict_default_noop():
  """load_state_dict({'noise': 1}) on base does not raise."""
  dm = DataModule()
  dm.load_state_dict({'noise': 1})


def test_datamodule_custom_state_round_trip():
  """Subclass stores counter; to/from dict equality."""

  class _StatefulDM(DataModule):
    def __init__(self) -> None:
      self.counter = 0
      self.indices: list[int] = []

    def state_dict(self) -> dict[str, Any]:
      return {'counter': self.counter, 'indices': self.indices}

    def load_state_dict(self, state: dict[str, Any]) -> None:
      self.counter = state['counter']
      self.indices = state['indices']

  dm = _StatefulDM()
  dm.counter = 42
  dm.indices = [1, 3, 5]
  saved = dm.state_dict()

  dm2 = _StatefulDM()
  dm2.load_state_dict(saved)
  assert dm2.counter == 42
  assert dm2.indices == [1, 3, 5]


# Base class NotImplementedError tests


def test_train_dataloader_not_implemented():
  with pytest.raises(NotImplementedError):
    DataModule().train_dataloader()


def test_val_dataloader_not_implemented():
  with pytest.raises(NotImplementedError):
    DataModule().val_dataloader()


def test_test_dataloader_not_implemented():
  with pytest.raises(NotImplementedError):
    DataModule().test_dataloader()


def test_base_prepare_and_setup_noop():
  dm = DataModule()
  dm.prepare_data()
  dm.setup(Stage.fit)


# Concrete subclass lifecycle


def test_concrete_subclass_lifecycle():
  dm = _LifecycleModule()
  dm.prepare_data()
  dm.setup(Stage.fit)
  train_batches = list(dm.train_dataloader())
  assert len(train_batches) == 2
  dm.teardown(Stage.fit)
  assert 'prepare_data' in dm.events
  assert 'setup:fit' in dm.events
  assert 'train_dataloader' in dm.events
  assert 'teardown:fit' in dm.events


class _ValTestModule(DataModule):
  def val_dataloader(self) -> DataLoader:
    return DataLoader(ListDataset([EvalDatum(metadata={'v': 10})]))

  def test_dataloader(self) -> DataLoader:
    return DataLoader(ListDataset([EvalDatum(metadata={'v': 20})]))


def test_custom_val_and_test_dataloaders():
  dm = _ValTestModule()
  val_dl = dm.val_dataloader()
  assert isinstance(val_dl, DataLoader)
  val_batches = list(val_dl)
  assert len(val_batches) >= 1

  test_dl = dm.test_dataloader()
  assert isinstance(test_dl, DataLoader)
  test_batches = list(test_dl)
  assert len(test_batches) >= 1


class _TrainWithEmptyVal(DataModule):
  def train_dataloader(self) -> DataLoader:
    return DataLoader([EvalDatum(success=True)], batch_size=1)

  def val_dataloader(self) -> DataLoader:
    return DataLoader(ListDataset([]), batch_size=1)


def test_trainer_fit_empty_val_loader_omits_val_metrics() -> None:
  class _Mod(AutoPilotModule):
    def forward(self, *args, **kwargs):
      return EvalDatum(success=True)

    def training_step(self, batch, batch_idx):
      return EvalDatum(success=True)

    def configure_optimizers(self):
      return None

  trainer = Trainer()
  result = trainer.fit(_Mod(), datamodule=_TrainWithEmptyVal(), max_epochs=1)
  assert 'val_metrics' not in result['epochs'][0]


def test_example_datamodule_setup_accepts_stage_enum() -> None:
  """Example/project DataModules inherit from DataModule with Stage-typed hooks."""
  dm = DataModule()
  dm.setup(Stage.fit)
  dm.setup(Stage.validate)
  dm.setup(Stage.test)
  dm.setup(Stage.predict)
  dm.teardown(Stage.fit)
  dm.teardown(Stage.test)


def test_example_datamodule_subclass_setup_accepts_stage() -> None:
  """Subclassing DataModule and overriding setup with Stage works."""
  calls: list[Stage] = []

  class _ExampleDM(DataModule):
    def setup(self, stage: Stage) -> None:
      calls.append(stage)

    def teardown(self, stage: Stage) -> None:
      calls.append(stage)

  dm = _ExampleDM()
  dm.setup(Stage.fit)
  dm.setup(Stage.validate)
  dm.teardown(Stage.test)
  assert calls == [Stage.fit, Stage.validate, Stage.test]
