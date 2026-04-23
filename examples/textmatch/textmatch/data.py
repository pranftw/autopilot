from autopilot.core.types import Datum
from autopilot.data.dataloader import DataLoader
from autopilot.data.datamodule import DataModule
from autopilot.data.dataset import Dataset
from pathlib import Path
import json


class TextMatchDataset(Dataset):
  def __init__(self, path: Path):
    self._items = []
    if path.exists():
      with open(path) as f:
        for line in f:
          line = line.strip()
          if line:
            self._items.append(json.loads(line))

  def __getitem__(self, index: int) -> Datum:
    item = self._items[index]
    label = item.get('metadata', {}).get('label', '')
    return Datum(
      metadata={
        'text': item['text'],
        'expected': item['expected_category'],
        'label': label,
      },
    )

  def __len__(self) -> int:
    return len(self._items)


class TextMatchDataModule(DataModule):
  def __init__(self, datasets_dir: str, batch_size: int = 1):
    super().__init__()
    self._dir = Path(datasets_dir)
    self._batch_size = batch_size

  def train_dataloader(self) -> DataLoader:
    return DataLoader(
      TextMatchDataset(self._dir / 'train.jsonl'),
      batch_size=self._batch_size,
    )

  def val_dataloader(self) -> DataLoader:
    return DataLoader(
      TextMatchDataset(self._dir / 'val.jsonl'),
      batch_size=self._batch_size,
    )

  def test_dataloader(self) -> DataLoader:
    return DataLoader(
      TextMatchDataset(self._dir / 'test.jsonl'),
      batch_size=self._batch_size,
    )
