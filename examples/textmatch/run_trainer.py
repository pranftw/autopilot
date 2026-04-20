"""Lightning-style Trainer.fit() for text classification rules.

Demonstrates: AutoPilotModule, Trainer, Policy, Store, StoreCheckpoint --
same components as run.py but orchestrated by Trainer automatically.
"""

from pathlib import Path
from textmatch.data import TextMatchDataModule
from textmatch.module import TextMatchModule
from textmatch.trainer import build_trainer


def example_dir() -> Path:
  return Path(__file__).parent


def main():
  root = example_dir()
  store_path = root / '.store'

  module = TextMatchModule(str(root / 'rules'))
  dm = TextMatchDataModule(str(root / 'datasets'))
  trainer, store = build_trainer(module, store_path)

  print(f'=== TextMatch: Trainer.fit() [store slug: {store.slug}] ===\n')
  result = trainer.fit(module, datamodule=dm, max_epochs=5)

  print(f'\nTotal epochs: {result["total_epochs"]}')
  for ep in result['epochs']:
    train = ep.get('metrics', {})
    val = ep.get('val_metrics', {})
    parts = [f'Epoch {ep["epoch"]}:']
    parts.append(f'train_acc={train.get("accuracy", 0):.2%}')
    if val:
      parts.append(f'val_acc={val.get("accuracy", 0):.2%}')
    print(f'  {" ".join(parts)}')

  print(f'\nStore history ({store.slug}):')
  for entry in store.log():
    print(f'  epoch {entry.epoch}: {entry.file_count} files @ {entry.timestamp}')

  print('\nDone.')


if __name__ == '__main__':
  main()
