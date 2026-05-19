"""Tests for core/status.py experiment status gathering via Forest."""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.artifacts.experiment import SummaryArtifact
from autopilot.core.config import AutoPilotConfig
from autopilot.core.enums import Status
from autopilot.core.errors import TrackingError
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from autopilot.core.status import get_experiment_status
from autopilot.tracking.io import atomic_write_json
from pathlib import Path
import pytest

_summary = SummaryArtifact()


def _build_forest(tmp_path: Path, slug: str, epoch: int = 0) -> FileForest:
  """Build a FileForest with one tree containing a single experiment node."""
  config = AutoPilotConfig(workspace=tmp_path)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')
  exp = Experiment(experiment_id=slug)
  exp.epoch = epoch
  tree.add(Node(experiment=exp))
  forest.save()
  return forest


class TestGetExperimentStatus:
  def test_basic_status(self, tmp_path: Path) -> None:
    forest = _build_forest(tmp_path, 'test-exp', epoch=0)
    exp_dir = forest.store.config.experiment_path(slug='test-exp')
    exp_dir.mkdir(parents=True, exist_ok=True)
    result = get_experiment_status(forest, 'test-exp')
    assert result['id'] == 'test-exp'
    assert result['epoch'] == 0

  def test_includes_stop_reason(self, tmp_path: Path) -> None:
    forest = _build_forest(tmp_path, 'test-exp', epoch=3)
    exp_dir = forest.store.config.experiment_path(slug='test-exp')
    exp_dir.mkdir(parents=True, exist_ok=True)
    _summary.write({'stop_reason': 'plateau', 'last_good_epoch': 2}, exp_dir)
    result = get_experiment_status(forest, 'test-exp')
    assert result['stop_reason'] == 'plateau'
    assert result['last_good_epoch'] == 2

  def test_crash_detection(self, tmp_path: Path) -> None:
    forest = _build_forest(tmp_path, 'test-exp', epoch=5)
    exp_dir = forest.store.config.experiment_path(slug='test-exp')
    exp_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
      exp_dir / 'run_state.json',
      {'epoch': 5, 'status': 'running'},
    )
    result = get_experiment_status(forest, 'test-exp')
    assert result['stop_reason'] == 'crash'

  def test_missing_node_raises(self, tmp_path: Path) -> None:
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    with pytest.raises(TrackingError):
      get_experiment_status(forest, 'nonexistent')

  def test_trained_epochs_count(self, tmp_path: Path) -> None:
    forest = _build_forest(tmp_path, 'test-exp', epoch=1)
    exp_dir = forest.store.config.experiment_path(slug='test-exp')
    exp_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(1, 4):
      (exp_dir / f'epoch_{ep}').mkdir()
    result = get_experiment_status(forest, 'test-exp')
    assert result['trained_epochs'] == 3

  def test_no_decision_fields_in_result(self, tmp_path: Path) -> None:
    forest = _build_forest(tmp_path, 'test-exp', epoch=2)
    exp_dir = forest.store.config.experiment_path(slug='test-exp')
    exp_dir.mkdir(parents=True, exist_ok=True)
    result = get_experiment_status(forest, 'test-exp')
    assert 'decision' not in result
    assert 'decision_reason' not in result


@pytest.mark.parametrize('status', list(Status))
def test_run_state_status_strings(tmp_path: Path, status: Status) -> None:
  forest = _build_forest(tmp_path, 'e', epoch=0)
  exp_dir = forest.store.config.experiment_path(slug='e')
  exp_dir.mkdir(parents=True, exist_ok=True)
  atomic_write_json(
    exp_dir / 'run_state.json',
    {
      'epoch': 0,
      'status': status.value,
      'stop_reason': 'kernel',
      'last_good_epoch': 7,
    },
  )
  result = get_experiment_status(forest, 'e')
  if status == Status.running:
    assert result['stop_reason'] == 'crash'
  else:
    assert result['stop_reason'] == 'kernel'
    assert result['last_good_epoch'] == 7


def test_run_state_supplements_when_no_summary(tmp_path: Path) -> None:
  forest = _build_forest(tmp_path, 'e', epoch=0)
  exp_dir = forest.store.config.experiment_path(slug='e')
  exp_dir.mkdir(parents=True, exist_ok=True)
  atomic_write_json(
    exp_dir / 'run_state.json',
    {
      'status': Status.completed.value,
      'stop_reason': 'done',
      'last_good_epoch': 4,
    },
  )
  result = get_experiment_status(forest, 'e')
  assert result['stop_reason'] == 'done'
  assert result['last_good_epoch'] == 4


def test_summary_wins_when_run_state_also_sets_stop_reason(tmp_path: Path) -> None:
  forest = _build_forest(tmp_path, 'e', epoch=0)
  exp_dir = forest.store.config.experiment_path(slug='e')
  exp_dir.mkdir(parents=True, exist_ok=True)
  _summary.write({'stop_reason': 'plateau', 'last_good_epoch': 2}, exp_dir)
  atomic_write_json(
    exp_dir / 'run_state.json',
    {
      'status': Status.completed.value,
      'stop_reason': 'ignored',
      'last_good_epoch': 99,
    },
  )
  result = get_experiment_status(forest, 'e')
  assert result['stop_reason'] == 'plateau'
  assert result['last_good_epoch'] == 2


def test_trained_epochs_ignores_bad_epoch_dirs(tmp_path: Path) -> None:
  forest = _build_forest(tmp_path, 'e', epoch=0)
  exp_dir = forest.store.config.experiment_path(slug='e')
  exp_dir.mkdir(parents=True, exist_ok=True)
  (exp_dir / 'epoch_1').mkdir()
  (exp_dir / 'epoch_foo').mkdir()
  (exp_dir / 'epoch_').mkdir()
  result = get_experiment_status(forest, 'e')
  assert result['trained_epochs'] == 1


def test_duplicate_experiment_id_deduplicates(tmp_path: Path) -> None:
  """BUG-044: duplicate experiment IDs across trees are deduplicated (first wins)."""
  config = AutoPilotConfig(workspace=tmp_path)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  tree1 = forest.create_tree('tree1')
  exp1 = Experiment(experiment_id='dup-exp')
  exp1.epoch = 5
  tree1.add(Node(experiment=exp1))

  tree2 = forest.create_tree('tree2')
  exp2 = Experiment(experiment_id='dup-exp')
  exp2.epoch = 10
  tree2.add(Node(experiment=exp2))

  forest.save()

  nodes = forest.query().all()
  ids = [n.experiment.id for n in nodes]
  assert ids.count('dup-exp') == 1
  matched = [n for n in nodes if n.experiment.id == 'dup-exp']
  assert matched[0].experiment.epoch == 5
