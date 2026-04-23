from autopilot.core.types import Datum
from autopilot.data.dataloader import DataLoader
from autopilot.data.datamodule import DataModule
from autopilot.data.dataset import Dataset
from pathlib import Path
import json


class QADataset(Dataset):
  def __init__(self, path: Path):
    self._items = []
    if path.exists():
      for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
          self._items.append(json.loads(line))

  def __getitem__(self, index: int) -> Datum:
    item = self._items[index]
    label = item.get('metadata', {}).get('label', '')
    return Datum(
      metadata={
        'question': item['question'],
        'expected': item['expected'],
        'label': label,
      },
    )

  def __len__(self) -> int:
    return len(self._items)


class QADataModule(DataModule):
  def __init__(self, datasets_dir: str, batch_size: int = 1):
    super().__init__()
    self._dir = Path(datasets_dir)
    self._batch_size = batch_size

  def train_dataloader(self) -> DataLoader:
    return DataLoader(
      QADataset(self._dir / 'train.jsonl'),
      batch_size=self._batch_size,
    )

  def val_dataloader(self) -> DataLoader:
    return DataLoader(
      QADataset(self._dir / 'val.jsonl'),
      batch_size=self._batch_size,
    )

  def test_dataloader(self) -> DataLoader:
    return DataLoader(
      QADataset(self._dir / 'test.jsonl'),
      batch_size=self._batch_size,
    )
