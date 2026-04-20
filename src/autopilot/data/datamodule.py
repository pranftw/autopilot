"""DataModule. Mirrors Lightning's LightningDataModule."""

from autopilot.data.dataloader import DataLoader


class DataModule:
  """Lifecycle for data. Mirrors LightningDataModule."""

  def prepare_data(self) -> None:
    pass

  def setup(self, stage: str) -> None:
    pass

  def train_dataloader(self) -> DataLoader:
    raise NotImplementedError

  def val_dataloader(self) -> DataLoader:
    raise NotImplementedError

  def test_dataloader(self) -> DataLoader:
    raise NotImplementedError

  def teardown(self, stage: str) -> None:
    pass
