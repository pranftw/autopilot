"""Full experiment lifecycle: Config -> FileStore -> FileForest -> Tree ->
Experiment -> Trainer.fit -> StoreCheckpointCallback -> QueryBuilder -> Stabilize.

Verifies the complete stack end-to-end using real objects and tmp_path.
"""

from autopilot.ai.experiment import AutoPilotExperiment
from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.callbacks.store import StoreCheckpointCallback
from autopilot.core.config import AutoPilotConfig
from autopilot.core.enums import Status
from autopilot.core.metric import Metric
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.node import Node
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import Datum, EvalDatum
from autopilot.data.dataloader import DataLoader
from autopilot.data.datamodule import DataModule
from pathlib import Path
from tests.doubles import DirectNumericLoss, NoOpOptimizer


class _AccuracyMetric(Metric):
  higher_is_better = True

  def __init__(self):
    super().__init__()
    self.add_state('_correct', 0)
    self.add_state('_total', 0)

  def update(self, datum):
    item = datum.items[0] if isinstance(datum, Datum) and datum.items else datum
    self._total += 1
    if item.success:
      self._correct += 1

  def compute(self):
    acc = self._correct / self._total if self._total > 0 else 0.0
    return {'AccuracyMetric': acc}


class _LifecycleModule(AutoPilotModule):
  def __init__(self, param):
    super().__init__()
    self.param = param
    self.loss = DirectNumericLoss([param])
    self.accuracy = _AccuracyMetric()
    self._opt = NoOpOptimizer([param])

  def forward(self, batch):
    return batch

  def training_step(self, batch, batch_idx):
    return batch

  def validation_step(self, batch, batch_idx):
    return batch

  def configure_optimizers(self):
    return self._opt


class _LifecycleDataModule(DataModule):
  def train_dataloader(self) -> DataLoader:
    return DataLoader(
      [EvalDatum(metadata={'i': i}, success=True) for i in range(4)],
      batch_size=1,
    )

  def val_dataloader(self) -> DataLoader:
    return DataLoader([], batch_size=1)


def _setup_workspace(tmp_path: Path) -> tuple[Path, Path]:
  """Create a minimal workspace with a test file for PathParameter."""
  workspace = tmp_path / 'workspace'
  workspace.mkdir()
  files_dir = workspace / 'params'
  files_dir.mkdir()
  test_file = files_dir / 'prompt.txt'
  test_file.write_text('initial content', encoding='utf-8')
  return workspace, files_dir


def _create_lifecycle_stack(tmp_path: Path):
  """Build the full Config/Store/Forest/Tree/Experiment/Trainer stack.

  Returns:
    Tuple of ``(config, store, tree, experiment, trainer, module)``.
  """
  workspace, files_dir = _setup_workspace(tmp_path)
  config = AutoPilotConfig(workspace=workspace)
  path_param = PathParameter(source=str(files_dir), pattern='*.txt')
  store = FileStore(config)
  store.register_parameters({'source': path_param})
  forest = FileForest(store=store)
  tree = forest.create_tree('main', description='primary exploration')
  experiment = AutoPilotExperiment(experiment_id='lifecycle-exp-1')
  tree.add(Node(experiment=experiment))
  module = _LifecycleModule(path_param)
  trainer = Trainer(
    callbacks=[StoreCheckpointCallback()],
    experiment=experiment,
    store=store,
    config=config,
    tree=tree,
    forest=forest,
  )
  return config, store, tree, experiment, trainer, module, path_param


def test_full_lifecycle(tmp_path: Path) -> None:
  """Complete lifecycle: Config -> Store -> Forest -> Tree -> Experiment ->
  Trainer.fit -> Checkpoint -> Query -> Stabilize -> Checkout.

  Steps 1-11 from the sub-plan.
  """
  config, store, tree, experiment, trainer, module, path_param = _create_lifecycle_stack(tmp_path)

  trainer.fit(module, datamodule=_LifecycleDataModule(), max_epochs=3)

  # 7. Verify experiment status and epoch
  assert experiment.status == Status.completed
  assert experiment.epoch == 2, (
    f'After 3 epochs [0,1,2], final advance_epoch() sets epoch to 2, got {experiment.epoch}'
  )

  # 8. Verify snapshots exist at epochs 0, 1, 2
  log_entries = store.log(experiment.id)
  assert len(log_entries) == 3
  log_epochs = [entry.epoch for entry in log_entries]
  assert log_epochs == [0, 1, 2]
  for entry in log_entries:
    assert entry.file_count > 0

  # 9. Query: completed experiments
  completed_nodes = tree.query().completed().all()
  assert len(completed_nodes) == 1
  assert completed_nodes[0].experiment.id == experiment.id

  # 10. Stabilize: copy from latest snapshot to workspace
  copied_paths = config.stabilize(experiment.id)
  assert len(copied_paths) > 0
  for p in copied_paths:
    assert p.exists(), f'stabilized path {p} does not exist'

  # 11. Checkout restores parameters; compare with stabilized
  store.checkout(experiment.id, 2)

  restored_content = path_param.snapshot()
  for rel_path, text in restored_content.items():
    for cp in copied_paths:
      if cp.name == rel_path:
        stabilized_text = cp.read_text(encoding='utf-8')
        assert text == stabilized_text, f'Restored and stabilized content differ for {rel_path}'


def test_lifecycle_result_structure(tmp_path: Path) -> None:
  """Verify the result dict from Trainer.fit has expected structure."""
  workspace, files_dir = _setup_workspace(tmp_path)
  config = AutoPilotConfig(workspace=workspace)
  path_param = PathParameter(source=str(files_dir), pattern='*.txt')
  store = FileStore(config)
  store.register_parameters({'source': path_param})
  experiment = AutoPilotExperiment(experiment_id='result-structure-exp')
  cb = StoreCheckpointCallback()
  module = _LifecycleModule(path_param)
  trainer = Trainer(
    callbacks=[cb],
    experiment=experiment,
    store=store,
    config=config,
  )
  result = trainer.fit(module, datamodule=_LifecycleDataModule(), max_epochs=2)

  assert 'total_epochs' in result
  assert result['total_epochs'] == 2
  assert 'epochs' in result
  assert len(result['epochs']) == 2
  for ep_result in result['epochs']:
    assert 'epoch' in ep_result
    assert 'metrics' in ep_result


def test_lifecycle_forest_persistence(tmp_path: Path) -> None:
  """Verify forest state persists after Trainer.fit completes."""
  workspace, files_dir = _setup_workspace(tmp_path)
  config = AutoPilotConfig(workspace=workspace)
  path_param = PathParameter(source=str(files_dir), pattern='*.txt')
  store = FileStore(config)
  store.register_parameters({'source': path_param})

  forest = FileForest(store=store)
  tree = forest.create_tree('persistence-tree')
  experiment = AutoPilotExperiment(experiment_id='persist-exp')
  node = Node(experiment=experiment)
  tree.add(node)

  module = _LifecycleModule(path_param)
  cb = StoreCheckpointCallback()
  trainer = Trainer(
    callbacks=[cb],
    experiment=experiment,
    store=store,
    config=config,
    tree=tree,
  )
  trainer.fit(module, datamodule=_LifecycleDataModule(), max_epochs=2)

  # Reload forest from disk and verify state survived
  forest2 = FileForest(store=store)
  tree2 = forest2.get_tree('persistence-tree')
  assert tree2 is not None
  node2 = tree2.get('persist-exp')
  assert node2 is not None
  assert node2.experiment.status == Status.completed


def test_lifecycle_store_log_file_counts(tmp_path: Path) -> None:
  """Verify store log entries have consistent file counts."""
  workspace, files_dir = _setup_workspace(tmp_path)
  config = AutoPilotConfig(workspace=workspace)
  path_param = PathParameter(source=str(files_dir), pattern='*.txt')
  store = FileStore(config)
  store.register_parameters({'source': path_param})
  experiment = AutoPilotExperiment(experiment_id='file-count-exp')
  cb = StoreCheckpointCallback()
  module = _LifecycleModule(path_param)
  trainer = Trainer(
    callbacks=[cb],
    experiment=experiment,
    store=store,
    config=config,
  )
  trainer.fit(module, datamodule=_LifecycleDataModule(), max_epochs=3)

  log_entries = store.log(experiment.id)
  file_counts = [entry.file_count for entry in log_entries]
  assert all(c > 0 for c in file_counts)
  assert all(c == file_counts[0] for c in file_counts), (
    f'Expected uniform file counts, got {file_counts}'
  )
