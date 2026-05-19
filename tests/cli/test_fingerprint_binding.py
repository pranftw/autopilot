"""Tests for dataset fingerprint auto-binding (Plan 13).

Covers:
  - ``experiment add --dataset-path`` fingerprint computation and persistence.
  - ``experiment compare`` drift detection with fingerprint differences.
  - ``query --json`` surfacing ``dataset_fingerprint`` in output rows.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_no_context, run_cli_text
import pytest


@pytest.fixture
def fp_workspace(tmp_path: Path) -> Path:
  """Workspace root with store dir created."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  return ws


@pytest.fixture
def fp_forest(fp_workspace: Path) -> FileForest:
  """Forest with an active tree named 'main'."""
  config = AutoPilotConfig(workspace=fp_workspace)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  forest.create_tree('main')
  forest.switch('main')
  forest.save()
  return forest


def _complete_experiment(workspace: Path, eid: str) -> None:
  """Complete an experiment so it becomes terminal for parenting."""
  run_cli(workspace, ['experiment', 'complete', eid])


# -- 4.1 CLI add and persistence --


def test_experiment_add_dataset_fingerprint_file(fp_workspace, fp_forest, tmp_path):
  """Add with --dataset-path pointing at a JSONL file; verify bundle_hash is 64-char hex."""
  data_file = tmp_path / 'data.jsonl'
  data_file.write_text('{"x": 1}\n{"x": 2}\n')

  output = run_cli(
    fp_workspace,
    ['experiment', 'add', '--hypothesis', 'test fp', '--dataset-path', str(data_file)],
  )
  result = output['result']
  assert result['ok'] is True
  assert 'dataset_fingerprint' in result
  fp = result['dataset_fingerprint']
  assert len(fp['bundle_hash']) == 64
  assert all(c in '0123456789abcdef' for c in fp['bundle_hash'])

  forest = FileForest(FileStore(AutoPilotConfig(workspace=fp_workspace)))
  tree = forest.active
  assert tree is not None
  node = tree.get(result['experiment_id'])
  assert node is not None
  stored_fp = node.experiment.dataset_meta['dataset_fingerprint']
  assert stored_fp['bundle_hash'] == fp['bundle_hash']


def test_experiment_add_dataset_same_file_same_hash(fp_workspace, fp_forest, tmp_path):
  """Two adds on same file yield equal bundle_hash."""
  data_file = tmp_path / 'data.jsonl'
  data_file.write_text('{"a": 1}\n')

  r1 = run_cli(
    fp_workspace,
    ['experiment', 'add', '--hypothesis', 'h1', '--id', 'exp1', '--dataset-path', str(data_file)],
  )
  _complete_experiment(fp_workspace, 'exp1')

  r2 = run_cli(
    fp_workspace,
    ['experiment', 'add', '--hypothesis', 'h2', '--id', 'exp2', '--dataset-path', str(data_file)],
  )
  h1 = r1['result']['dataset_fingerprint']['bundle_hash']
  h2 = r2['result']['dataset_fingerprint']['bundle_hash']
  assert h1 == h2


def test_experiment_add_dataset_modified_file_different_hash(fp_workspace, fp_forest, tmp_path):
  """Mutate file between adds; bundle_hash values differ."""
  data_file = tmp_path / 'data.jsonl'
  data_file.write_text('{"a": 1}\n')
  r1 = run_cli(
    fp_workspace,
    ['experiment', 'add', '--hypothesis', 'h1', '--id', 'exp1', '--dataset-path', str(data_file)],
  )
  _complete_experiment(fp_workspace, 'exp1')

  data_file.write_text('{"a": 2}\n')
  r2 = run_cli(
    fp_workspace,
    ['experiment', 'add', '--hypothesis', 'h2', '--id', 'exp2', '--dataset-path', str(data_file)],
  )
  h1 = r1['result']['dataset_fingerprint']['bundle_hash']
  h2 = r2['result']['dataset_fingerprint']['bundle_hash']
  assert h1 != h2


def test_experiment_add_dataset_nonexistent_path_errors(fp_workspace, fp_forest, tmp_path):
  """Nonexistent --dataset-path causes non-zero exit with descriptive message."""
  bad_path = tmp_path / 'missing.jsonl'
  with pytest.raises(SystemExit):
    run_cli(
      fp_workspace,
      ['experiment', 'add', '--hypothesis', 'h', '--dataset-path', str(bad_path)],
    )


def test_experiment_add_dataset_directory(fp_workspace, fp_forest, tmp_path):
  """Directory with known files; stable bundle_hash across identical content."""
  data_dir = tmp_path / 'dataset'
  data_dir.mkdir()
  (data_dir / 'a.txt').write_text('hello')
  (data_dir / 'b.txt').write_text('world')

  r1 = run_cli(
    fp_workspace,
    ['experiment', 'add', '--hypothesis', 'h1', '--id', 'exp1', '--dataset-path', str(data_dir)],
  )
  _complete_experiment(fp_workspace, 'exp1')

  r2 = run_cli(
    fp_workspace,
    ['experiment', 'add', '--hypothesis', 'h2', '--id', 'exp2', '--dataset-path', str(data_dir)],
  )
  h1 = r1['result']['dataset_fingerprint']['bundle_hash']
  h2 = r2['result']['dataset_fingerprint']['bundle_hash']
  assert h1 == h2
  assert len(h1) == 64


# -- 4.2 Compare and query --


def _seed_experiments_with_fingerprints(
  fp_forest: FileForest,
  fp_a: dict | None,
  fp_b: dict | None,
) -> tuple[str, str]:
  """Seed two experiments with optional fingerprint data.

  Args:
    fp_forest: Active forest.
    fp_a: Fingerprint dict for experiment a, or None.
    fp_b: Fingerprint dict for experiment b, or None.

  Returns:
    Tuple of (exp_a_id, exp_b_id).
  """
  tree = fp_forest.active
  assert tree is not None

  exp_a = Experiment(experiment_id='cmp-a', hypothesis='a')
  exp_a.start()
  exp_a.complete(metrics={'accuracy': 0.9})
  if fp_a is not None:
    exp_a.dataset_meta['dataset_fingerprint'] = fp_a

  exp_b = Experiment(experiment_id='cmp-b', hypothesis='b')
  exp_b.start()
  exp_b.complete(metrics={'accuracy': 0.85})
  if fp_b is not None:
    exp_b.dataset_meta['dataset_fingerprint'] = fp_b

  tree.add(Node(experiment=exp_a))
  tree.add(Node(experiment=exp_b))
  fp_forest.save()

  return 'cmp-a', 'cmp-b'


def test_compare_warns_when_fingerprints_differ(fp_workspace, fp_forest):
  """Drift info line emitted; JSON dataset_fingerprint_drift is True."""
  fp_a = {'paths': ['/data.jsonl'], 'hashes': ['aaa'], 'bundle_hash': 'a' * 64, 'timestamp': 't1'}
  fp_b = {'paths': ['/data.jsonl'], 'hashes': ['bbb'], 'bundle_hash': 'b' * 64, 'timestamp': 't2'}

  _seed_experiments_with_fingerprints(fp_forest, fp_a, fp_b)

  output = run_cli_no_context(
    fp_workspace,
    ['experiment', 'compare', 'cmp-a', 'cmp-b'],
  )
  result = output['result']
  assert result['dataset_fingerprint_drift'] is True

  text = run_cli_text(
    fp_workspace,
    ['experiment', 'compare', 'cmp-a', 'cmp-b'],
  )
  assert 'dataset fingerprint' in text.lower()


def test_compare_no_drift_when_fingerprints_match(fp_workspace, fp_forest):
  """Same dataset bound twice; drift flag False, no stray warn output."""
  fp_data = {
    'paths': ['/data.jsonl'],
    'hashes': ['abc123'],
    'bundle_hash': 'c' * 64,
    'timestamp': 't1',
  }

  _seed_experiments_with_fingerprints(fp_forest, fp_data, fp_data)

  output = run_cli_no_context(
    fp_workspace,
    ['experiment', 'compare', 'cmp-a', 'cmp-b'],
  )
  result = output['result']
  assert result['dataset_fingerprint_drift'] is False

  text = run_cli_text(
    fp_workspace,
    ['experiment', 'compare', 'cmp-a', 'cmp-b'],
  )
  assert 'dataset fingerprint' not in text.lower()


def test_compare_no_drift_when_one_side_missing_fingerprint(fp_workspace, fp_forest):
  """One experiment lacks fingerprint; drift flag None (unknown lineage)."""
  fp_data = {
    'paths': ['/data.jsonl'],
    'hashes': ['abc123'],
    'bundle_hash': 'd' * 64,
    'timestamp': 't1',
  }

  _seed_experiments_with_fingerprints(fp_forest, fp_data, None)

  output = run_cli_no_context(
    fp_workspace,
    ['experiment', 'compare', 'cmp-a', 'cmp-b'],
  )
  result = output['result']
  assert result['dataset_fingerprint_drift'] is None


def test_compare_no_drift_when_both_missing_fingerprint(fp_workspace, fp_forest):
  """Both experiments lack fingerprint; drift flag None (unknown lineage)."""
  _seed_experiments_with_fingerprints(fp_forest, None, None)

  output = run_cli_no_context(
    fp_workspace,
    ['experiment', 'compare', 'cmp-a', 'cmp-b'],
  )
  result = output['result']
  assert result['dataset_fingerprint_drift'] is None


# -- 4.3 Query --


def test_query_json_includes_dataset_fingerprint(fp_workspace, fp_forest, tmp_path):
  """Query row includes dataset_fingerprint mirroring dataset_meta."""
  data_file = tmp_path / 'data.jsonl'
  data_file.write_text('{"row": 1}\n')

  run_cli(
    fp_workspace,
    ['experiment', 'add', '--hypothesis', 'q', '--id', 'qexp', '--dataset-path', str(data_file)],
  )

  output = run_cli_no_context(fp_workspace, ['query'])
  experiments = output['result']['experiments']
  match = [e for e in experiments if e['id'] == 'qexp']
  assert len(match) == 1
  assert match[0]['dataset_fingerprint'] is not None
  assert 'bundle_hash' in match[0]['dataset_fingerprint']


def test_query_json_dataset_fingerprint_none_when_absent(fp_workspace, fp_forest):
  """Query row has dataset_fingerprint=None when no dataset was bound."""
  run_cli(
    fp_workspace,
    ['experiment', 'add', '--hypothesis', 'no dataset', '--id', 'nodata'],
  )

  output = run_cli_no_context(fp_workspace, ['query'])
  experiments = output['result']['experiments']
  match = [e for e in experiments if e['id'] == 'nodata']
  assert len(match) == 1
  assert match[0]['dataset_fingerprint'] is None
