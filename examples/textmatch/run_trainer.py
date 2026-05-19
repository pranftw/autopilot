"""Lightning-style Trainer.fit() for text classification rules.

Accepts argparse flags for all tunable parameters. Use --json for
structured output suitable for agent consumption.
"""

from pathlib import Path
from textmatch.data import TextMatchDataModule
from textmatch.module import TextMatchModule
from textmatch.trainer import build_trainer
import argparse
import json


def example_dir() -> Path:
  return Path(__file__).parent


def main(argv: list[str] | None = None) -> dict:
  parser = argparse.ArgumentParser(description='TextMatch Trainer.fit()')
  parser.add_argument('--rules-dir', default=None, metavar='PATH')
  parser.add_argument('--datasets-dir', default=None, metavar='PATH')
  parser.add_argument('--store-path', default=None, metavar='PATH')
  parser.add_argument('--max-epochs', type=int, default=5)
  parser.add_argument('--threshold', type=float, default=0.30)
  parser.add_argument('--accumulate-grad-batches', type=int, default=100)
  parser.add_argument('--experiment', default=None, metavar='SLUG')
  parser.add_argument('--json', action='store_true', dest='use_json')
  args = parser.parse_args(argv)

  root = example_dir()
  rules_dir = args.rules_dir or str(root / 'rules')
  datasets_dir = args.datasets_dir or str(root / 'datasets')
  store_path = Path(args.store_path) if args.store_path else root / '.store'

  module = TextMatchModule(rules_dir)
  dm = TextMatchDataModule(datasets_dir)
  trainer, store = build_trainer(
    module,
    store_path,
    threshold=args.threshold,
    accumulate_grad_batches=args.accumulate_grad_batches,
    experiment_slug=args.experiment,
  )

  experiment = trainer.experiment
  experiment.add_context(
    f'Starting rule optimization: max_epochs={args.max_epochs}, '
    f'threshold={args.threshold}, accumulate_grad_batches={args.accumulate_grad_batches}.',
    source='examples.textmatch.run_trainer',
    metadata={
      'argv_max_epochs': args.max_epochs,
      'argv_threshold': args.threshold,
      'argv_accumulate_grad_batches': args.accumulate_grad_batches,
      'rules_dir': rules_dir,
      'datasets_dir': datasets_dir,
    },
  )

  result = trainer.fit(module, datamodule=dm, max_epochs=args.max_epochs)
  experiment_id = experiment.id

  output = {
    'experiment': experiment_id,
    'total_epochs': result['total_epochs'],
    'epochs': [],
  }
  for ep in result['epochs']:
    train = ep['metrics']
    val = ep.get('val_metrics')
    acc = train.get('accuracy')
    train_acc = acc if acc is not None else train.get('train_accuracy', 0.0)
    val_acc = val['accuracy'] if val and 'accuracy' in val else None
    output['epochs'].append(
      {
        'epoch': ep['epoch'],
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
      }
    )

  if result['epochs']:
    last = output['epochs'][-1]
    output['final_train_accuracy'] = last['train_accuracy']
    output['final_val_accuracy'] = last['val_accuracy']

  if args.use_json:
    print(json.dumps(output, indent=2))
  else:
    print(f'=== TextMatch: Trainer.fit() [experiment: {experiment_id}] ===\n')
    for ep in output['epochs']:
      parts = [f'Epoch {ep["epoch"]}:']
      parts.append(f'train_acc={ep["train_accuracy"]:.2%}')
      if ep['val_accuracy'] is not None:
        parts.append(f'val_acc={ep["val_accuracy"]:.2%}')
      print(f'  {" ".join(parts)}')
    print(f'\nTotal epochs: {output["total_epochs"]}')
    print('\nDone.')

  return output


if __name__ == '__main__':
  main()
