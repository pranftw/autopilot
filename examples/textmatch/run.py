"""Manual PyTorch-style optimization loop for text classification rules.

Accepts argparse flags for all tunable parameters. Use --json for
structured output suitable for agent consumption.
"""

from pathlib import Path
from textmatch.data import TextMatchDataModule
from textmatch.module import TextMatchModule
import argparse
import json


def example_dir() -> Path:
  return Path(__file__).parent


def main(argv: list[str] | None = None) -> dict:
  parser = argparse.ArgumentParser(description='TextMatch manual optimization loop')
  parser.add_argument('--rules-dir', default=None, metavar='PATH')
  parser.add_argument('--datasets-dir', default=None, metavar='PATH')
  parser.add_argument('--max-epochs', type=int, default=5)
  parser.add_argument('--json', action='store_true', dest='use_json')
  args = parser.parse_args(argv)

  root = example_dir()
  rules_dir = args.rules_dir or str(root / 'rules')
  datasets_dir = args.datasets_dir or str(root / 'datasets')

  module = TextMatchModule(rules_dir)
  loss = module.loss
  optimizer = module.configure_optimizers()
  metric = module.accuracy
  dm = TextMatchDataModule(datasets_dir)

  train_loader = dm.train_dataloader()
  val_loader = dm.val_dataloader()

  output = {'total_epochs': args.max_epochs, 'epochs': []}

  module.train()
  for epoch in range(args.max_epochs):
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

    output['epochs'].append({
      'epoch': epoch,
      'train_accuracy': train_metrics['accuracy'],
      'val_accuracy': val_metrics['accuracy'],
    })

  if output['epochs']:
    last = output['epochs'][-1]
    output['final_train_accuracy'] = last['train_accuracy']
    output['final_val_accuracy'] = last['val_accuracy']

  if args.use_json:
    print(json.dumps(output, indent=2))
  else:
    print('=== TextMatch: Manual Loop ===\n')
    for ep in output['epochs']:
      print(
        f'  Epoch {ep["epoch"]}: '
        f'train_acc={ep["train_accuracy"]:.2%} '
        f'val_acc={ep["val_accuracy"]:.2%}'
      )
    print(f'\nTotal epochs: {output["total_epochs"]}')
    print('\nDone.')

  return output


if __name__ == '__main__':
  main()
