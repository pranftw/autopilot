"""Dogfood V7 regression tests for edge cases and behavioral contracts.

Covers: symlink boundary enforcement, store merge with empty branches,
deploy --replace no-prior semantics, combined query filters, performance
guards, checkpoint resume after invalidation, and large context logs.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.callbacks.checkpoint import CheckpointCallback
from autopilot.core.config import AutoPilotConfig
from autopilot.core.context import ContextLog
from autopilot.core.errors import ConfigError, StoreError
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from autopilot.core.trainer.trainer import Trainer
from autopilot.data.dataloader import DataLoader
from autopilot.data.datamodule import DataModule, Stage
from autopilot.data.dataset import Dataset
from pathlib import Path
from tests.doubles import NoopEvalModule
import json
import pytest
import time


class _SingleItemDataset(Dataset):
  """Dataset with one item for minimal training."""

  def __len__(self) -> int:
    return 1

  def __getitem__(self, index: int) -> dict:
    return {'x': 1}


class _MinimalDataModule(DataModule):
  """DataModule providing single-item train and val loaders."""

  def setup(self, stage: Stage) -> None:
    pass

  def train_dataloader(self) -> DataLoader:
    return DataLoader(_SingleItemDataset(), batch_size=1)

  def val_dataloader(self) -> DataLoader:
    return DataLoader(_SingleItemDataset(), batch_size=1)


def test_symlink_outside_root_excluded(tmp_path: Path) -> None:
  """Symlinks pointing outside PathParameter root are excluded from snapshot."""
  params_dir = tmp_path / 'params'
  params_dir.mkdir()
  (params_dir / 'legit.txt').write_text('ok', encoding='utf-8')

  outside_target = tmp_path / 'outside' / 'secret.txt'
  outside_target.parent.mkdir(parents=True)
  outside_target.write_text('sensitive data', encoding='utf-8')

  symlink = params_dir / 'escape.txt'
  symlink.symlink_to(outside_target)

  param = PathParameter(source=str(params_dir), pattern='*.txt')
  result = param.snapshot()

  assert 'legit.txt' in result
  assert 'escape.txt' not in result
  assert 'sensitive data' not in ''.join(result.values())


def test_store_merge_empty_branch(tmp_path: Path) -> None:
  """Merge analysis vs a branch at epoch -1 raises StoreError deterministically (no crash)."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)

  src_dir = ws / 'src'
  src_dir.mkdir()
  (src_dir / 'main.py').write_text('print("hi")\n', encoding='utf-8')

  param = PathParameter(source=str(src_dir), pattern='**/*.py')
  store = FileStore(config)
  store.register_parameters({'source': param})

  store.snapshot('root', 0)
  store.branch('exp-a')
  store.branch('exp-b')

  (src_dir / 'main.py').write_text('print("advanced")\n', encoding='utf-8')
  store.snapshot('exp-a', 1)

  store.reset_branch('exp-b')

  with pytest.raises(StoreError, match='snapshot not found'):
    store.merge_analysis('exp-a', 'exp-b')


def test_deploy_replace_no_prior(tmp_path: Path) -> None:
  """deploy --replace with no existing holder succeeds (no StoreError)."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')

  exp = Experiment(experiment_id='exp-1', hypothesis='test')
  exp.start()
  exp.complete(metrics={'score': 0.9})
  node = Node(experiment=exp)
  tree.add(node)
  forest.switch('main')

  prev = forest.deploy(node, 'production', replace=True)

  assert prev is None
  assert node.deployed_as == 'production'


def test_query_all_filters_combined(tmp_path: Path) -> None:
  """Combined --metric-gt + --metric-lt + --sort + limit narrows results."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')

  for i in range(20):
    exp = Experiment(experiment_id=f'exp-{i:03d}', hypothesis=f'h{i}')
    exp.start()
    exp.complete(metrics={'accuracy': i * 0.05, 'loss': 1.0 - i * 0.04})
    tree.add(Node(experiment=exp))

  forest.save()

  builder = forest.query()
  builder = builder.completed()
  baseline_count = len(builder.all())

  builder = forest.query()
  builder = builder.completed()
  builder = builder.metric_gt('accuracy', 0.3)
  builder = builder.metric_lt('accuracy', 0.8)
  builder = builder.order_by_metric('accuracy')
  filtered = builder.all()

  assert len(filtered) < baseline_count
  assert len(filtered) > 0

  accuracies = [n.experiment.metrics['accuracy'] for n in filtered]
  assert all(0.3 < a < 0.8 for a in accuracies)
  assert accuracies == sorted(accuracies, reverse=True)


def test_forest_50_experiments_perf(tmp_path: Path) -> None:
  """Forest.query() on 50+ experiments completes under 1.0s."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('perf')
  forest.switch('perf')

  for i in range(55):
    exp = Experiment(experiment_id=f'perf-{i:04d}', hypothesis=f'h{i}')
    exp.start()
    exp.complete(metrics={'accuracy': i * 0.01, 'loss': 1.0 - i * 0.01})
    tree.add(Node(experiment=exp))

  forest.save()

  start = time.perf_counter()
  builder = forest.query()
  builder = builder.completed()
  builder = builder.metric_gt('accuracy', 0.1)
  builder = builder.order_by_metric('accuracy')
  results = builder.all()
  elapsed = time.perf_counter() - start

  assert elapsed < 1.0
  assert len(results) > 0


def test_checkpoint_resume_after_invalidation(tmp_path: Path) -> None:
  """Resume training on an invalidated experiment raises ConfigError."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)

  ckpt_dir = tmp_path / 'checkpoints'
  ckpt_dir.mkdir()
  ckpt_file = ckpt_dir / 'epoch-0000.json'

  exp = Experiment(experiment_id='inv-exp', hypothesis='test')
  exp.start()
  exp.complete(metrics={'accuracy': 0.8})
  exp.invalidate('results were wrong')

  ckpt_data = {
    'experiment': exp.state_dict(),
    'module': {},
  }
  ckpt_file.write_text(json.dumps(ckpt_data), encoding='utf-8')

  module = NoopEvalModule()
  cb = CheckpointCallback(directory=ckpt_dir)

  trainer = Trainer(
    config=config,
    experiment=exp,
    callbacks=[cb],
  )

  with pytest.raises(ConfigError, match='invalidated') as exc_info:
    trainer.fit(module, datamodule=_MinimalDataModule(), max_epochs=2, ckpt_path='last')

  assert 'inv-exp' in str(exc_info.value)


def test_context_log_1000_entries() -> None:
  """Context log handles 1000+ entries and round-trips exactly."""
  exp = Experiment(experiment_id='big-log', hypothesis='scale test')
  exp.start()

  entry_count = 1000
  for i in range(entry_count):
    exp.add_context(f'entry {i}', source='test', metadata={'idx': i})

  assert len(exp.context_log) == entry_count

  serialized = exp.context_log.to_list()
  restored = ContextLog.from_list(serialized)

  assert len(restored) == entry_count
  assert restored.entries[0].reason == 'entry 0'
  assert restored.entries[999].reason == 'entry 999'
  assert restored.entries[500].metadata == {'idx': 500}
