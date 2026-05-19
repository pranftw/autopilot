"""Minimal dataset and datamodule for the multi-module pipeline example.

Generates in-memory EvalDatum instances so the example is fully hermetic
(no external files required). Each item carries a task description in
metadata that flows through planner -> researcher -> writer.
"""

from autopilot.core.types import EvalDatum
from autopilot.data.dataloader import DataLoader
from autopilot.data.datamodule import DataModule
from autopilot.data.dataset import ListDataset

TRAIN_ITEMS = [
  EvalDatum(metadata={'task': 'Write a blog post about AI safety', 'expected': 'comprehensive'}),
  EvalDatum(metadata={'task': 'Summarize recent ML papers', 'expected': 'concise'}),
  EvalDatum(metadata={'task': 'Draft a product announcement', 'expected': 'engaging'}),
]


class MultiModuleDataModule(DataModule):
  """In-memory data for the multi-module pipeline example."""

  def __init__(self, train_items: list[EvalDatum] | None = None) -> None:
    self._train_items = train_items or TRAIN_ITEMS

  def train_dataloader(self) -> DataLoader:
    return DataLoader(ListDataset(self._train_items), batch_size=1)

  def val_dataloader(self) -> DataLoader:
    return DataLoader(ListDataset(self._train_items[:1]), batch_size=1)
