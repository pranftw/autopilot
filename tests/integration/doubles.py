"""Shared test doubles for integration tests.

Centralizes noop Loss / Optimizer / Metric / Module / DataModule stacks so
integration tests under tests/integration/ share one canonical implementation
instead of duplicating _BranchLoss / _ParLoss / _MTLoss etc.

NumericSeedLoss uses compute_seed_gradient() + graph-based backward (Loss.backward
seeds the computation graph via AccumulateGrad). MinimalPathModule.training_step
calls self(batch) so the output Datum carries grad_fn for the graph.
"""

from autopilot.ai.parameter import PathParameter
from autopilot.core.gradient import NumericGradient
from autopilot.core.loss import Loss
from autopilot.core.metric import Metric
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.types import Datum, EvalDatum
from autopilot.data.dataloader import DataLoader
from autopilot.data.datamodule import DataModule
from tests.doubles import NoOpOptimizer


class NumericSeedLoss(Loss):
  """Noop loss; NumericGradient seed flows through Loss.backward / AccumulateGrad."""

  def forward(self, data: Datum, targets: Datum | None = None) -> None:
    super().forward(data, targets)

  def compute_seed_gradient(self) -> NumericGradient:
    return NumericGradient(value=1.0)

  def reset(self) -> None:
    super().reset()


class FixedAccuracyMetric(Metric):
  """Metric that always returns a fixed accuracy value."""

  higher_is_better = True

  def __init__(self, value: float = 0.5) -> None:
    super().__init__()
    self._value = value
    self.add_state('_n', 0)

  def update(self, datum: Datum) -> None:
    self._n += 1

  def compute(self) -> dict[str, float]:
    return {'AccuracyMetric': self._value}


class MinimalPathModule(AutoPilotModule):
  """Minimal module wired with PathParameter for integration smoke paths.

  training_step calls self(batch) to route through Module.__call__ ->
  ModuleCallOperator, which sets grad_fn on the output Datum. This enables
  graph-based Loss.backward().
  """

  def __init__(self, path_param: PathParameter, accuracy: float = 0.5) -> None:
    super().__init__()
    self.param = path_param
    self.loss = NumericSeedLoss([path_param])
    self.accuracy = FixedAccuracyMetric(value=accuracy)
    self._opt = NoOpOptimizer([path_param])

  def forward(self, batch: Datum) -> Datum:
    return batch

  def training_step(self, batch: Datum, batch_idx: int) -> Datum:
    return self(batch)

  def configure_optimizers(self) -> NoOpOptimizer:
    return self._opt


class TwoBatchTrainDatamodule(DataModule):
  """DataModule providing 2 train batches (no validation data)."""

  def train_dataloader(self) -> DataLoader:
    return DataLoader(
      [EvalDatum(metadata={'i': i}, success=True) for i in range(2)],
      batch_size=1,
    )


def minimal_trainer_stack(
  path_param: PathParameter,
  accuracy: float = 0.5,
) -> tuple[MinimalPathModule, TwoBatchTrainDatamodule]:
  """Build minimal module + datamodule for one-epoch Trainer.fit smoke paths."""
  return MinimalPathModule(path_param, accuracy=accuracy), TwoBatchTrainDatamodule()
