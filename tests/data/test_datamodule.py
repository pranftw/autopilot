from autopilot.core.types import Datum
from autopilot.data.dataloader import DataLoader
from autopilot.data.datamodule import DataModule
import pytest


class _LifecycleModule(DataModule):
  def __init__(self) -> None:
    self.events: list[str] = []

  def prepare_data(self) -> None:
    self.events.append('prepare_data')

  def setup(self, stage: str) -> None:
    self.events.append(f'setup:{stage}')

  def train_dataloader(self) -> DataLoader:
    self.events.append('train_dataloader')
    data = [Datum(metadata={'v': 1}), Datum(metadata={'v': 2})]
    return DataLoader(data, batch_size=1, shuffle=False)

  def val_dataloader(self) -> DataLoader:
    return DataLoader([Datum(metadata={'v': 3})], batch_size=1, shuffle=False)

  def test_dataloader(self) -> DataLoader:
    return DataLoader([Datum(metadata={'v': 4})], batch_size=1, shuffle=False)

  def teardown(self, stage: str) -> None:
    self.events.append(f'teardown:{stage}')


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
  m = DataModule()
  m.prepare_data()
  m.setup('fit')


def test_concrete_subclass_lifecycle():
  m = _LifecycleModule()
  m.prepare_data()
  m.setup('fit')
  train_batches = list(m.train_dataloader())
  assert len(train_batches) == 2
  m.teardown('fit')
  assert 'prepare_data' in m.events
  assert 'setup:fit' in m.events
  assert 'train_dataloader' in m.events
  assert 'teardown:fit' in m.events
