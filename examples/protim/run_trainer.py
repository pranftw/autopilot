"""Lightning-style Trainer.fit() for agent-optimized prompt tuning.

Demonstrates: AutoPilotModule, Trainer, Policy, Store, StoreCheckpointCallback,
ClaudeCodeAgent -- same components as run.py but orchestrated by Trainer.
Requires: claude CLI installed.
"""

from pathlib import Path
from protim.data import QADataModule
from protim.module import PromptModule
from protim.trainer import build_trainer
import argparse


def example_dir() -> Path:
  return Path(__file__).parent


def main(argv: list[str] | None = None):
  parser = argparse.ArgumentParser(description='Protim Trainer.fit()')
  parser.add_argument('--prompts-dir', default=None, metavar='PATH')
  parser.add_argument('--datasets-dir', default=None, metavar='PATH')
  parser.add_argument('--store-path', default=None, metavar='PATH')
  parser.add_argument('--max-epochs', type=int, default=3)
  args = parser.parse_args(argv)

  root = example_dir()
  prompts_dir = args.prompts_dir or str(root / 'prompts')
  datasets_dir = args.datasets_dir or str(root / 'datasets')
  store_path = Path(args.store_path) if args.store_path else root / '.store'

  module = PromptModule(prompts_dir)
  dm = QADataModule(datasets_dir)
  trainer, store = build_trainer(module, store_path)

  experiment = trainer.experiment
  experiment_id = experiment.id
  experiment.add_context(
    f'Starting prompt optimization: max_epochs={args.max_epochs}, '
    f'prompts_dir={prompts_dir}.',
    source='examples.protim.run_trainer',
    metadata={
      'argv_max_epochs': args.max_epochs,
      'prompts_dir': prompts_dir,
      'datasets_dir': datasets_dir,
    },
  )

  print(f'=== Protim: Trainer.fit() [experiment: {experiment_id}] ===\n')
  result = trainer.fit(module, datamodule=dm, max_epochs=args.max_epochs)

  print(f'\nTotal epochs: {result["total_epochs"]}')
  for ep in result['epochs']:
    train = ep.get('metrics', {})
    val = ep.get('val_metrics', {})
    parts = [f'Epoch {ep["epoch"]}:']
    parts.append(f'train_acc={train.get("accuracy", 0):.2%}')
    if val:
      parts.append(f'val_acc={val.get("accuracy", 0):.2%}')
    print(f'  {" ".join(parts)}')

  print(f'\nStore history ({experiment_id}):')
  for entry in store.log(experiment_id):
    print(f'  epoch {entry.epoch}: {entry.file_count} files @ {entry.timestamp}')

  print('\nDone.')


if __name__ == '__main__':
  main()
