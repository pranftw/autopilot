from autopilot.core.types import Datum
from autopilot.core.module.autopilot_module import AutoPilotModule


class ProjectModule(AutoPilotModule):
  """Project module. Override training_step and validation_step."""

  def training_step(self, batch: Datum, batch_idx: int) -> Datum:
    return batch

  def validation_step(self, batch: Datum, batch_idx: int) -> Datum:
    return batch

  def configure_optimizers(self):
    return None
