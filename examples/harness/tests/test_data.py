"""Tests for harness.data: HarnessDataset and HarnessDataModule."""

from autopilot.core.types import EvalDatum
from autopilot.data.dataloader import DataLoader
from autopilot.data.datamodule import Stage
from harness.data import HarnessDataModule, HarnessDataset
from pathlib import Path
import json

# ---------------------------------------------------------------------------
# 4.1  HarnessDataset
# ---------------------------------------------------------------------------


class TestHarnessDataset:
  """Unit tests for HarnessDataset."""

  def test_dataset_loads_jsonl(self, tmp_path: Path, sample_records: list[dict]) -> None:
    """Three-line JSONL loads three items with correct task_id."""
    path = tmp_path / 'data.jsonl'
    lines = [json.dumps(r, ensure_ascii=False) for r in sample_records]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    ds = HarnessDataset(path)
    assert len(ds) == 3
    assert ds[0].metadata['task_id'] == sample_records[0]['task_id']

  def test_dataset_returns_eval_datum(self, tmp_path: Path, sample_records: list[dict]) -> None:
    """Every index returns an EvalDatum."""
    path = tmp_path / 'data.jsonl'
    lines = [json.dumps(r, ensure_ascii=False) for r in sample_records]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    ds = HarnessDataset(path)
    assert isinstance(ds[0], EvalDatum)
    assert isinstance(ds[2], EvalDatum)

  def test_dataset_empty_file(self, tmp_path: Path) -> None:
    """Empty and whitespace-only files produce zero-length datasets."""
    empty = tmp_path / 'empty.jsonl'
    empty.write_text('', encoding='utf-8')
    assert len(HarnessDataset(empty)) == 0

    whitespace = tmp_path / 'ws.jsonl'
    whitespace.write_text('   \n  \n', encoding='utf-8')
    assert len(HarnessDataset(whitespace)) == 0

  def test_dataset_single_item(self, tmp_path: Path, sample_records: list[dict]) -> None:
    """Single-line JSONL with no trailing newline loads one record."""
    path = tmp_path / 'single.jsonl'
    path.write_text(json.dumps(sample_records[0], ensure_ascii=False), encoding='utf-8')

    ds = HarnessDataset(path)
    assert len(ds) == 1
    assert ds[0].metadata['task_id'] == sample_records[0]['task_id']
    assert ds[0].metadata['initial_message'] == sample_records[0]['initial_message']

  def test_dataset_metadata_structure(self, tmp_path: Path, sample_records: list[dict]) -> None:
    """Each loaded record has the full nested schema keys."""
    path = tmp_path / 'data.jsonl'
    lines = [json.dumps(r, ensure_ascii=False) for r in sample_records]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    ds = HarnessDataset(path)
    for idx in range(len(ds)):
      meta = ds[idx].metadata
      assert 'task_id' in meta
      assert 'initial_message' in meta
      assert 'user_instructions' in meta
      assert 'evaluation_criteria' in meta

      ui = meta['user_instructions']
      assert 'reason_for_call' in ui
      assert 'known_info' in ui
      assert 'task_instructions' in ui

      ec = meta['evaluation_criteria']
      assert 'expected_actions' in ec
      assert 'communicate_info' in ec
      assert 'nl_assertions' in ec


# ---------------------------------------------------------------------------
# 4.2  HarnessDataModule
# ---------------------------------------------------------------------------


class TestHarnessDataModule:
  """Unit tests for HarnessDataModule."""

  def test_datamodule_setup_noop(self, scenarios_dir: Path) -> None:
    """setup(Stage.fit) does not raise."""
    dm = HarnessDataModule(str(scenarios_dir))
    dm.setup(Stage.fit)

  def test_datamodule_train_dataloader(self, scenarios_dir: Path) -> None:
    """train_dataloader returns a DataLoader with iterable batches."""
    dm = HarnessDataModule(str(scenarios_dir))
    dl = dm.train_dataloader()
    assert isinstance(dl, DataLoader)
    batches = list(dl)
    assert len(batches) >= 1

  def test_datamodule_val_dataloader(self, scenarios_dir: Path) -> None:
    """val_dataloader returns a DataLoader with iterable batches."""
    dm = HarnessDataModule(str(scenarios_dir))
    dl = dm.val_dataloader()
    assert isinstance(dl, DataLoader)
    batches = list(dl)
    assert len(batches) >= 1

  def test_datamodule_test_dataloader(self, scenarios_dir: Path) -> None:
    """test_dataloader returns a DataLoader with iterable batches."""
    dm = HarnessDataModule(str(scenarios_dir))
    dl = dm.test_dataloader()
    assert isinstance(dl, DataLoader)
    batches = list(dl)
    assert len(batches) >= 1

  def test_datamodule_batch_items_are_eval_datum(self, scenarios_dir: Path) -> None:
    """Each batch.items entry is an EvalDatum with scenario metadata."""
    dm = HarnessDataModule(str(scenarios_dir))
    batch = next(iter(dm.train_dataloader()))
    assert len(batch.items) >= 1
    datum = batch.items[0]
    assert isinstance(datum, EvalDatum)
    assert 'task_id' in datum.metadata

  def test_datamodule_state_dict_roundtrip(self, scenarios_dir: Path) -> None:
    """state_dict / load_state_dict preserves scenarios_dir and batch_size."""
    dm1 = HarnessDataModule(str(scenarios_dir), batch_size=7)
    state = dm1.state_dict()

    dm2 = HarnessDataModule('/tmp/dummy', batch_size=1)
    dm2.load_state_dict(state)

    restored = dm2.state_dict()
    assert restored['scenarios_dir'] == state['scenarios_dir']
    assert restored['batch_size'] == 7

  def test_datamodule_custom_batch_size(self, scenarios_dir: Path) -> None:
    """batch_size=2 groups items into batches of 2."""
    dm = HarnessDataModule(str(scenarios_dir), batch_size=2)
    dl = dm.train_dataloader()
    batch = next(iter(dl))
    assert len(batch.items) == 2
