"""Manual PyTorch-style optimization loop for text classification rules.

Demonstrates: Module, Loss, Optimizer, Metric, DataLoader, DataModule --
the same forward -> loss -> backward -> optimizer.step() loop from PyTorch,
applied to optimizing regex classification rules.
"""

from pathlib import Path
from textmatch.data import TextMatchDataModule
from textmatch.module import TextMatchModule


def example_dir() -> Path:
  return Path(__file__).parent


def main():
  root = example_dir()
  module = TextMatchModule(str(root / 'rules'))
  loss = module.loss
  optimizer = module.configure_optimizers()
  metric = module.accuracy
  dm = TextMatchDataModule(str(root / 'datasets'))

  train_loader = dm.train_dataloader()
  val_loader = dm.val_dataloader()

  print('=== TextMatch: Manual Loop ===\n')

  module.train()
  for epoch in range(1, 6):
    metric.reset()
    loss.reset()

    for batch in train_loader:
      data = module(batch)
      loss(data, batch)
      metric.update(data)

    loss.backward()
    train_metrics = metric.compute()

    optimizer.step()
    optimizer.zero_grad()

    metric.reset()
    module.eval()
    for batch in val_loader:
      val_data = module(batch)
      metric.update(val_data)
    val_metrics = metric.compute()
    module.train()

    print(
      f'Epoch {epoch}: '
      f'train_acc={train_metrics["accuracy"]:.2%} '
      f'val_acc={val_metrics["accuracy"]:.2%}'
    )

  print('\nDone.')


if __name__ == '__main__':
  main()
