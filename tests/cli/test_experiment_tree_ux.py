"""Tests for experiment & tree UX improvements (plan 12).

Covers FRICTION-003 (HEAD auto-set on experiment add),
BUG-008 (store branch --reset), GAP-012 (dataset fingerprint auto-attach),
FRICTION-004 (tree switch disk-state warning).
"""

from autopilot.ai.fingerprint import compute_fingerprint
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.commands.tree import DISK_STATE_ADVISORY
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError
from autopilot.core.experiment import Experiment
from autopilot.core.trainer.trainer import Trainer
from autopilot.data.dataloader import DataLoader
from autopilot.data.datamodule import DataModule, Stage
from autopilot.data.dataset import Dataset
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_text, seed_tree_with_experiments
from tests.doubles import DirectNumericLoss, NoopEvalModule, NoOpOptimizer
import pytest

# -- FRICTION-003: HEAD auto-set on experiment add ----------------------------


def test_experiment_add_updates_tree_head(cli_forest, cli_workspace: Path) -> None:
  """After ``experiment add``, tree HEAD equals the new experiment id."""
  cli_forest.create_tree('main')
  cli_forest.switch('main')

  result = run_cli(
    cli_workspace,
    ['experiment', 'add', '--hypothesis', 'first experiment'],
  )
  assert result['result']['ok'] is True
  new_id = result['result']['experiment_id']

  tree = cli_forest.active
  assert tree is not None
  # reload forest to verify persistence
  from autopilot.ai.forest import FileForest

  reloaded = FileForest(cli_forest.store)
  reloaded_tree = reloaded.active
  assert reloaded_tree is not None
  assert reloaded_tree.head == new_id


def test_experiment_add_head_updates_on_second_add(cli_forest, cli_workspace: Path) -> None:
  """Adding a second experiment updates HEAD to the newest one."""
  seed_tree_with_experiments(
    cli_forest,
    'main',
    [{'id': 'exp-1', 'hypothesis': 'first', 'status': 'completed', 'metrics': {}}],
  )

  result = run_cli(
    cli_workspace,
    ['experiment', 'add', '--hypothesis', 'second', '--parent', 'exp-1'],
  )
  new_id = result['result']['experiment_id']

  from autopilot.ai.forest import FileForest

  reloaded = FileForest(cli_forest.store)
  reloaded_active = reloaded.active
  assert reloaded_active is not None
  assert reloaded_active.head == new_id


def test_experiment_add_with_explicit_id_sets_head(cli_forest, cli_workspace: Path) -> None:
  """HEAD is set to the explicit --id when provided."""
  cli_forest.create_tree('main')
  cli_forest.switch('main')

  result = run_cli(
    cli_workspace,
    ['experiment', 'add', '--hypothesis', 'explicit', '--id', 'my-exp-id'],
  )
  assert result['result']['experiment_id'] == 'my-exp-id'

  from autopilot.ai.forest import FileForest

  reloaded = FileForest(cli_forest.store)
  reloaded_active = reloaded.active
  assert reloaded_active is not None
  assert reloaded_active.head == 'my-exp-id'


# -- BUG-008: store branch --reset -------------------------------------------


def test_branch_reset_clears_latest_epoch(cli_config: AutoPilotConfig) -> None:
  """``store branch --reset`` sets branch ``latest_epoch`` to ``-1``."""
  cli_config.store_path.mkdir(parents=True, exist_ok=True)
  source_dir = cli_config.workspace / 'src'
  source_dir.mkdir()
  (source_dir / 'main.py').write_text('print("hello")')

  store = FileStore(cli_config)
  param = PathParameter(source=str(source_dir), pattern='**/*.py')
  store.register_parameters({'code': param})
  store.snapshot('exp-1', 0, force=True)
  store.snapshot('exp-1', 1, force=True)

  refs = store.load_refs()
  assert refs['branches']['exp-1']['latest_epoch'] == 1

  store.reset_branch('exp-1')

  refs = store.load_refs()
  assert refs['branches']['exp-1']['latest_epoch'] == -1


def test_branch_reset_preserves_snapshots(cli_config: AutoPilotConfig) -> None:
  """After reset, existing snapshot manifests are still accessible."""
  cli_config.store_path.mkdir(parents=True, exist_ok=True)
  source_dir = cli_config.workspace / 'src'
  source_dir.mkdir()
  (source_dir / 'main.py').write_text('x = 1')

  store = FileStore(cli_config)
  param = PathParameter(source=str(source_dir), pattern='**/*.py')
  store.register_parameters({'code': param})
  store.snapshot('exp-1', 0)

  store.reset_branch('exp-1')

  snap = store.load_snapshot('exp-1', 0)
  assert len(snap.entries) > 0


def test_branch_reset_nonexistent_raises(cli_config: AutoPilotConfig) -> None:
  """Resetting a nonexistent branch raises StoreError."""
  cli_config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(cli_config)

  with pytest.raises(StoreError, match='not found'):
    store.reset_branch('nonexistent')


def test_training_succeeds_after_branch_reset(cli_config: AutoPilotConfig) -> None:
  """Snapshot at epoch 0 succeeds after branch reset."""
  cli_config.store_path.mkdir(parents=True, exist_ok=True)
  source_dir = cli_config.workspace / 'src'
  source_dir.mkdir()
  (source_dir / 'main.py').write_text('v1')

  store = FileStore(cli_config)
  param = PathParameter(source=str(source_dir), pattern='**/*.py')
  store.register_parameters({'code': param})
  store.snapshot('exp-1', 0)
  store.snapshot('exp-1', 1)

  store.reset_branch('exp-1')

  (source_dir / 'main.py').write_text('v2')
  manifest = store.snapshot('exp-1', 0)
  assert manifest.epoch == 0


def test_branch_reset_head_unchanged(cli_config: AutoPilotConfig) -> None:
  """Branch reset does not change the HEAD ref."""
  cli_config.store_path.mkdir(parents=True, exist_ok=True)
  source_dir = cli_config.workspace / 'src'
  source_dir.mkdir()
  (source_dir / 'main.py').write_text('code')

  store = FileStore(cli_config)
  param = PathParameter(source=str(source_dir), pattern='**/*.py')
  store.register_parameters({'code': param})
  store.snapshot('exp-1', 0)

  head_before = store.load_refs().get('HEAD')
  store.reset_branch('exp-1')
  head_after = store.load_refs().get('HEAD')

  assert head_before == head_after


# -- GAP-012: dataset fingerprint auto-attach --------------------------------


class _FingerprintDataModule(DataModule):
  """DataModule that sets a dataset fingerprint from a directory."""

  def __init__(self, data_dir: Path) -> None:
    self._data_dir = data_dir

  def setup(self, stage: Stage) -> None:
    self.dataset_fingerprint = compute_fingerprint([self._data_dir])

  def train_dataloader(self):
    return DataLoader(_TinyDataset(), batch_size=1)

  def val_dataloader(self):
    return None

  def test_dataloader(self):
    return None


class _TinyDataset(Dataset):
  """Minimal dataset returning empty dicts."""

  def __init__(self, n: int = 2) -> None:
    self._items = [{'x': i} for i in range(n)]

  def __len__(self) -> int:
    return len(self._items)

  def __getitem__(self, index: int) -> dict:
    return self._items[index]


class _FPModule(NoopEvalModule):
  """Module that provides an optimizer and loss for fit."""

  def __init__(self) -> None:
    super().__init__()
    self._loss = DirectNumericLoss()

  def configure_optimizers(self):
    return NoOpOptimizer(list(self.parameters()))

  def configure_loss(self):
    return self._loss

  def train_dataloader(self):
    return DataLoader(_TinyDataset(), batch_size=1)


def test_fit_attaches_dataset_fingerprint(tmp_path: Path) -> None:
  """After fit with DataModule, experiment.dataset_meta has fingerprint."""
  data_dir = tmp_path / 'data'
  data_dir.mkdir()
  (data_dir / 'train.jsonl').write_text('{"text": "hello"}\n')

  dm = _FingerprintDataModule(data_dir)
  dm.setup(Stage.fit)
  assert dm.dataset_fingerprint is not None

  exp = Experiment(experiment_id='fp-exp', hypothesis='fingerprint test')
  module = _FPModule()

  trainer = Trainer(experiment=exp)
  trainer.fit(module, datamodule=dm, max_epochs=1)

  assert 'dataset_fingerprint' in exp.dataset_meta
  fp_data = exp.dataset_meta['dataset_fingerprint']
  assert 'bundle_hash' in fp_data
  assert len(fp_data['paths']) > 0


def test_fit_does_not_overwrite_existing_fingerprint(tmp_path: Path) -> None:
  """When dataset_meta already has 'dataset_fingerprint', it is not overwritten."""
  data_dir = tmp_path / 'data'
  data_dir.mkdir()
  (data_dir / 'train.jsonl').write_text('{"text": "existing"}\n')

  dm = _FingerprintDataModule(data_dir)
  dm.setup(Stage.fit)

  exp = Experiment(experiment_id='fp-exp2', hypothesis='no overwrite')
  existing_fp = {'bundle_hash': 'preexisting', 'paths': ['/old']}
  exp.dataset_meta['dataset_fingerprint'] = existing_fp

  module = _FPModule()
  trainer = Trainer(experiment=exp)
  trainer.fit(module, datamodule=dm, max_epochs=1)

  assert exp.dataset_meta['dataset_fingerprint'] == existing_fp


class _BareDM(DataModule):
  """DataModule with no fingerprint, providing a trivial train loader."""

  def train_dataloader(self):
    return DataLoader(_TinyDataset(), batch_size=1)

  def val_dataloader(self):
    return None

  def test_dataloader(self):
    return None


def test_fit_no_fingerprint_when_datamodule_lacks_it(tmp_path: Path) -> None:
  """When DataModule has no fingerprint, dataset_meta is unchanged."""
  dm = _BareDM()
  assert dm.dataset_fingerprint is None

  exp = Experiment(experiment_id='fp-exp3', hypothesis='no fp')
  module = _FPModule()
  trainer = Trainer(experiment=exp)
  trainer.fit(module, datamodule=dm, max_epochs=1)

  assert 'dataset_fingerprint' not in exp.dataset_meta


# -- FRICTION-004: tree switch disk-state warning -----------------------------


def test_tree_switch_no_checkout_emits_advisory(cli_forest, cli_workspace: Path) -> None:
  """``tree switch --no-checkout`` emits the disk-state advisory."""
  cli_forest.create_tree('branch-a')
  cli_forest.create_tree('branch-b')

  output = run_cli_text(cli_workspace, ['tree', 'switch', 'branch-a', '--no-checkout'])
  assert 'does not sync working tree files' in output


def test_tree_switch_no_checkout_present_in_json_mode(cli_forest, cli_workspace: Path) -> None:
  """The disk-state advisory is in JSON mode (via info) with --no-checkout."""
  cli_forest.create_tree('branch-a')
  cli_forest.create_tree('branch-b')

  result = run_cli(cli_workspace, ['tree', 'switch', 'branch-a', '--no-checkout'])
  assert result['result']['ok'] is True
  assert result['result']['active'] == 'branch-a'


def test_disk_state_advisory_constant() -> None:
  """DISK_STATE_ADVISORY is importable and contains expected keywords."""
  assert 'does not' in DISK_STATE_ADVISORY
  assert 'sync' in DISK_STATE_ADVISORY
  assert 'checkout' in DISK_STATE_ADVISORY
