from autopilot.data.datamodule import DataModule, Stage


class ProjectDataModule(DataModule):
  """Project data module. Override setup and dataloader methods."""

  def setup(self, stage: Stage) -> None:
    pass

  def train_dataloader(self):
    raise NotImplementedError

  def val_dataloader(self):
    raise NotImplementedError
