"""Tests for experiment CLI commands: add, status, compare."""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from pathlib import Path
from tests.cli.conftest import run_cli, seed_tree_with_experiments
from unittest.mock import patch
import pytest


@pytest.fixture(autouse=True)
def _patch_store_checkout():
  """Patch FileStore.checkout for tests that don't create snapshots."""
  with patch('autopilot.ai.store.file_store.FileStore.checkout'):
    yield


@pytest.fixture
def ws(tmp_path: Path) -> Path:
  ws = tmp_path / 'ws'
  ws.mkdir()
  return ws


@pytest.fixture
def ws_with_tree(ws: Path) -> Path:
  """Workspace with an active empty tree."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  forest.create_tree('main')
  forest.switch('main')
  forest.save()
  return ws


@pytest.fixture
def ws_with_experiments(ws: Path) -> Path:
  """Workspace with a tree and experiments."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  seed_tree_with_experiments(
    forest,
    'main',
    [
      {
        'id': 'baseline',
        'hypothesis': 'default prompts',
        'status': 'completed',
        'metrics': {'accuracy': 0.72, 'latency': 120.0},
      },
      {
        'id': 'cot',
        'hypothesis': 'chain of thought',
        'status': 'completed',
        'metrics': {'accuracy': 0.78, 'latency': 115.0},
        'parent': 'baseline',
        'baseline': 'baseline',
      },
    ],
  )
  return ws


class TestExperimentAdd:
  def test_creates_root_experiment(self, ws_with_tree: Path) -> None:
    result = run_cli(
      ws_with_tree,
      [
        'experiment',
        'add',
        '--hypothesis',
        'test hypothesis',
        '--id',
        'exp-001',
      ],
    )
    assert result['result']['ok'] is True
    assert result['result']['experiment_id'] == 'exp-001'
    assert result['result']['hypothesis'] == 'test hypothesis'

  def test_auto_generates_id(self, ws_with_tree: Path) -> None:
    result = run_cli(
      ws_with_tree,
      [
        'experiment',
        'add',
        '--hypothesis',
        'auto id test',
      ],
    )
    assert result['result']['ok'] is True
    assert len(result['result']['experiment_id']) > 0

  def test_with_parent(self, ws_with_experiments: Path) -> None:
    result = run_cli(
      ws_with_experiments,
      [
        'experiment',
        'add',
        '--hypothesis',
        'child experiment',
        '--parent',
        'baseline',
        '--id',
        'child-001',
      ],
    )
    assert result['result']['ok'] is True
    assert result['result']['parent'] == 'baseline'

  def test_with_baseline(self, ws_with_experiments: Path) -> None:
    result = run_cli(
      ws_with_experiments,
      [
        'experiment',
        'add',
        '--hypothesis',
        'compare to baseline',
        '--parent',
        'cot',
        '--baseline',
        'baseline',
        '--id',
        'child-002',
      ],
    )
    assert result['result']['ok'] is True
    assert result['result']['baseline'] == 'baseline'

  def test_without_parent_on_empty_tree(self, ws_with_tree: Path) -> None:
    result = run_cli(
      ws_with_tree,
      [
        'experiment',
        'add',
        '--hypothesis',
        'root experiment',
        '--id',
        'root',
      ],
    )
    assert result['result']['ok'] is True
    assert result['result']['parent'] is None

  def test_json_output(self, ws_with_tree: Path) -> None:
    result = run_cli(
      ws_with_tree,
      [
        'experiment',
        'add',
        '--hypothesis',
        'json test',
        '--id',
        'json-exp',
      ],
    )
    assert 'ok' in result
    assert result['result']['ok'] is True

  def test_parent_defaults_to_head(self, ws_with_experiments: Path) -> None:
    """When HEAD is set and --parent omitted, parent defaults to HEAD."""
    run_cli(ws_with_experiments, ['checkout', 'cot'])
    result = run_cli(
      ws_with_experiments,
      [
        'experiment',
        'add',
        '--hypothesis',
        'auto parent from HEAD',
        '--id',
        'head-child',
      ],
    )
    assert result['result']['ok'] is True
    assert result['result']['parent'] == 'cot'

  def test_baseline_defaults_to_parent(self, ws_with_experiments: Path) -> None:
    """When --parent is given but --baseline omitted, baseline defaults to parent."""
    result = run_cli(
      ws_with_experiments,
      [
        'experiment',
        'add',
        '--hypothesis',
        'auto baseline',
        '--parent',
        'baseline',
        '--id',
        'auto-bl',
      ],
    )
    assert result['result']['ok'] is True
    assert result['result']['baseline'] == 'baseline'

  def test_no_active_tree_error(self, ws: Path) -> None:
    with pytest.raises(SystemExit):
      run_cli(ws, ['experiment', 'add', '--hypothesis', 'orphan'])

  def test_parent_not_found_error(self, ws_with_tree: Path) -> None:
    with pytest.raises(SystemExit):
      run_cli(
        ws_with_tree,
        [
          'experiment',
          'add',
          '--hypothesis',
          'bad parent',
          '--parent',
          'nonexistent',
          '--id',
          'bad',
        ],
      )


class TestExperimentStatus:
  def test_shows_by_id(self, ws_with_experiments: Path) -> None:
    result = run_cli(ws_with_experiments, ['experiment', 'status', 'baseline'])
    r = result['result']
    assert r['id'] == 'baseline'
    assert r['status'] == 'completed'
    assert r['hypothesis'] == 'default prompts'
    assert r['metrics']['accuracy'] == 0.72

  def test_shows_head(self, ws_with_experiments: Path) -> None:
    run_cli(ws_with_experiments, ['checkout', 'cot'])
    result = run_cli(ws_with_experiments, ['experiment', 'status'])
    assert result['result']['id'] == 'cot'

  def test_json_valid(self, ws_with_experiments: Path) -> None:
    result = run_cli(ws_with_experiments, ['experiment', 'status', 'baseline'])
    assert 'ok' in result
    assert result['result']['id'] == 'baseline'

  def test_nonexistent_error(self, ws_with_experiments: Path) -> None:
    with pytest.raises(SystemExit):
      run_cli(ws_with_experiments, ['experiment', 'status', 'ghost'])


class TestExperimentCompare:
  def test_shows_deltas(self, ws_with_experiments: Path) -> None:
    result = run_cli(ws_with_experiments, ['experiment', 'compare', 'baseline', 'cot'])
    deltas = result['result']['deltas']
    deltas_by_metric = {d['metric']: d for d in deltas}
    assert 'accuracy' in deltas_by_metric
    assert deltas_by_metric['accuracy']['baseline'] == 0.72
    assert deltas_by_metric['accuracy']['candidate'] == 0.78
    assert abs(deltas_by_metric['accuracy']['delta'] - 0.06) < 1e-9

  def test_json_valid(self, ws_with_experiments: Path) -> None:
    result = run_cli(ws_with_experiments, ['experiment', 'compare', 'baseline', 'cot'])
    assert result['result']['a'] == 'baseline'
    assert result['result']['b'] == 'cot'
    assert 'deltas' in result['result']

  def test_nonexistent_id_error(self, ws_with_experiments: Path) -> None:
    with pytest.raises(SystemExit):
      run_cli(ws_with_experiments, ['experiment', 'compare', 'baseline', 'ghost'])

  def test_text_output(self, ws_with_experiments: Path) -> None:
    from tests.cli.conftest import run_cli_text

    text = run_cli_text(ws_with_experiments, ['experiment', 'compare', 'baseline', 'cot'])
    assert 'accuracy' in text
    assert 'latency' in text
