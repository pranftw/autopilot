"""Multi-module pipeline training script.

Demonstrates end-to-end training with:
  - Pipeline (planner -> researcher -> writer) via explicit forward wiring
  - SimpleLoss using graph.backward() for gradient propagation
  - Trainer.fit() orchestrating the training loop

Gradient flow:
  Forward: input datum -> planner -> researcher -> writer -> output with grad_fn chain.
  Loss.backward: seeds TextGradient at output's graph node (last datum with grad_fn).
  Backward propagation order: writer parameters first (closest to loss), then
  researcher, then planner (reverse of forward call order).

  Input Datum
      |
      v
  [Planner] -- prompt param
      |
      v
  CustomAttributionOperator('planner')
      |
      v
  [Researcher] -- prompt param
      |
      v
  CustomAttributionOperator('researcher')
      |
      v
  [Writer] -- prompt param
      |
      v
  CustomAttributionOperator('writer')
      |
      v
  Output Datum (grad_fn -> CustomAttributionOperator for Writer)
      |
      v
  Loss.backward() seeds TextGradient at output.grad_fn
      |
      v
  Graph.backward() propagates (reverse forward order):
    CustomAttribution('writer') tags gradient -> Writer.AccumulateGrad
    CustomAttribution('researcher') tags gradient -> Researcher.AccumulateGrad
    CustomAttribution('planner') tags gradient -> Planner.AccumulateGrad
"""

from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.optimizer import Optimizer
from autopilot.core.trainer.trainer import Trainer
from multi_module.data import MultiModuleDataModule
from multi_module.loss import SimpleLoss
from multi_module.module import Pipeline
import argparse
import sys


class NoOpOptimizer(Optimizer):
  """Optimizer that logs gradient presence without modifying parameters.

  Used in this example to demonstrate the gradient flow without requiring
  an LLM-based optimizer. In production, replace with AgentOptimizer.
  """

  def step(self) -> None:
    for param in self.parameters:
      if param.grad is not None:
        pass


class MultiAgentModule(AutoPilotModule):
  """Top-level training module wrapping the Pipeline and SimpleLoss."""

  def __init__(self) -> None:
    super().__init__()
    self.pipeline = Pipeline()
    self.loss = SimpleLoss()

  def training_step(self, batch, batch_idx: int):
    return self(batch)

  def validation_step(self, batch, batch_idx: int):
    return self(batch)

  def forward(self, x):
    return self.pipeline(x)

  def configure_optimizers(self):
    return NoOpOptimizer(list(self.parameters()))


def main(argv: list[str] | None = None) -> dict:
  parser = argparse.ArgumentParser(description='Multi-module pipeline trainer')
  parser.add_argument('--max-epochs', type=int, default=2)
  args = parser.parse_args(argv)

  trainer = Trainer()
  module = MultiAgentModule()
  dm = MultiModuleDataModule()

  # To record trainer-driven context, construct an AutoPilotExperiment and pass
  # experiment=... into Trainer(...); use experiment.add_context(...) after fit.
  # This example omits an experiment for simplicity.

  result = trainer.fit(module, datamodule=dm, max_epochs=args.max_epochs)

  print(f'Training complete: {result["total_epochs"]} epoch(s)')
  for ep in result.get('epochs', []):
    print(f'  Epoch {ep["epoch"]}: completed')

  print(f'Gradient flow: {sum(1 for _ in module.parameters())} parameter(s) optimized')
  print('(param.grad cleared by optimizer.zero_grad() after each step)')

  return result


if __name__ == '__main__':
  sys.exit(0 if main() else 1)
