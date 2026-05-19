"""Tests for query CLI command."""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from pathlib import Path
from tests.cli.conftest import run_cli_no_context, run_cli_text, seed_tree_with_experiments
import pytest


@pytest.fixture
def ws(tmp_path: Path) -> Path:
  ws = tmp_path / 'ws'
  ws.mkdir()
  return ws


@pytest.fixture
def populated_ws(ws: Path) -> Path:
  """Workspace with diverse experiment statuses and metrics."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  seed_tree_with_experiments(
    forest,
    'main',
    [
      {
        'id': 'exp-a',
        'hypothesis': 'baseline',
        'status': 'completed',
        'metrics': {'accuracy': 0.72, 'latency': 120.0},
      },
      {
        'id': 'exp-b',
        'hypothesis': 'improved',
        'status': 'completed',
        'metrics': {'accuracy': 0.85, 'latency': 100.0},
        'parent': 'exp-a',
        'baseline': 'exp-a',
      },
      {
        'id': 'exp-c',
        'hypothesis': 'failed attempt',
        'status': 'failed',
        'error': 'OOM',
        'parent': 'exp-a',
        'baseline': 'exp-a',
      },
      {
        'id': 'exp-d',
        'hypothesis': 'in progress',
        'status': 'running',
        'parent': 'exp-b',
        'baseline': 'exp-b',
      },
      {'id': 'exp-e', 'hypothesis': 'pending work', 'status': 'pending'},
      {
        'id': 'exp-f',
        'hypothesis': 'cancelled',
        'status': 'cancelled',
        'parent': 'exp-a',
        'baseline': 'exp-a',
      },
    ],
  )
  return ws


class TestQueryCompleted:
  def test_returns_only_completed(self, populated_ws: Path) -> None:
    result = run_cli_no_context(populated_ws, ['query', '--completed'])
    exps = result['result']['experiments']
    assert all(e['status'] == 'completed' for e in exps)
    assert len(exps) == 2

  def test_json_valid(self, populated_ws: Path) -> None:
    result = run_cli_no_context(populated_ws, ['query', '--completed'])
    assert 'ok' in result
    assert 'experiments' in result['result']
    assert 'count' in result['result']


class TestQueryFailed:
  def test_returns_only_failed(self, populated_ws: Path) -> None:
    result = run_cli_no_context(populated_ws, ['query', '--failed'])
    exps = result['result']['experiments']
    assert all(e['status'] == 'failed' for e in exps)
    assert len(exps) == 1
    assert exps[0]['id'] == 'exp-c'


class TestQueryRunning:
  def test_returns_only_running(self, populated_ws: Path) -> None:
    result = run_cli_no_context(populated_ws, ['query', '--running'])
    exps = result['result']['experiments']
    assert all(e['status'] == 'running' for e in exps)
    assert len(exps) == 1


class TestQueryPending:
  def test_returns_only_pending(self, populated_ws: Path) -> None:
    result = run_cli_no_context(populated_ws, ['query', '--pending'])
    exps = result['result']['experiments']
    assert all(e['status'] == 'pending' for e in exps)
    assert len(exps) == 1


class TestQueryTerminal:
  def test_returns_terminal(self, populated_ws: Path) -> None:
    result = run_cli_no_context(populated_ws, ['query', '--terminal'])
    exps = result['result']['experiments']
    statuses = {e['status'] for e in exps}
    assert statuses <= {'completed', 'failed', 'cancelled'}
    assert len(exps) == 4


class TestQueryFilter:
  def test_filter_status_completed(self, populated_ws: Path) -> None:
    result = run_cli_no_context(populated_ws, ['query', '--filter', 'status=completed'])
    exps = result['result']['experiments']
    assert all(e['status'] == 'completed' for e in exps)

  def test_filter_by_hypothesis(self, populated_ws: Path) -> None:
    result = run_cli_no_context(populated_ws, ['query', '--filter', 'hypothesis=baseline'])
    exps = result['result']['experiments']
    assert len(exps) == 1
    assert exps[0]['hypothesis'] == 'baseline'


class TestQueryMetricGt:
  def test_metric_gt_threshold(self, populated_ws: Path) -> None:
    result = run_cli_no_context(populated_ws, ['query', '--metric-gt', 'accuracy:0.8'])
    exps = result['result']['experiments']
    assert len(exps) == 1
    assert exps[0]['id'] == 'exp-b'
    assert exps[0]['metrics']['accuracy'] > 0.8


class TestQueryMetricLt:
  def test_metric_lt_threshold(self, populated_ws: Path) -> None:
    result = run_cli_no_context(populated_ws, ['query', '--metric-lt', 'latency:110'])
    exps = result['result']['experiments']
    assert len(exps) == 1
    assert exps[0]['id'] == 'exp-b'


class TestQueryBest:
  def test_best_accuracy(self, populated_ws: Path) -> None:
    result = run_cli_no_context(populated_ws, ['query', '--best', 'accuracy'])
    best = result['result']['best']
    assert best['id'] == 'exp-b'
    assert best['metrics']['accuracy'] == 0.85

  def test_best_lower_latency(self, populated_ws: Path) -> None:
    result = run_cli_no_context(populated_ws, ['query', '--best', 'latency', '--lower'])
    best = result['result']['best']
    assert best['id'] == 'exp-b'
    assert best['metrics']['latency'] == 100.0

  def test_best_higher_default(self, populated_ws: Path) -> None:
    result = run_cli_no_context(populated_ws, ['query', '--best', 'accuracy', '--higher'])
    best = result['result']['best']
    assert best['id'] == 'exp-b'


class TestQueryCombined:
  def test_completed_metric_gt_best(self, populated_ws: Path) -> None:
    result = run_cli_no_context(
      populated_ws,
      [
        'query',
        '--completed',
        '--metric-gt',
        'accuracy:0.5',
        '--best',
        'accuracy',
      ],
    )
    best = result['result']['best']
    assert best['id'] == 'exp-b'


class TestQueryEmpty:
  def test_empty_results_graceful(self, populated_ws: Path) -> None:
    result = run_cli_no_context(populated_ws, ['query', '--metric-gt', 'accuracy:0.99'])
    exps = result['result']['experiments']
    assert len(exps) == 0
    assert result['result']['count'] == 0

  def test_best_no_match(self, populated_ws: Path) -> None:
    result = run_cli_no_context(
      populated_ws, ['query', '--metric-gt', 'accuracy:0.99', '--best', 'accuracy']
    )
    assert result['result']['best'] is None

  def test_text_empty(self, populated_ws: Path) -> None:
    text = run_cli_text(populated_ws, ['query', '--metric-gt', 'accuracy:0.99'])
    assert 'no matching' in text.lower()


class TestQueryJson:
  def test_all_json_fields(self, populated_ws: Path) -> None:
    result = run_cli_no_context(populated_ws, ['query', '--completed'])
    exps = result['result']['experiments']
    assert len(exps) == 2
    for exp in exps:
      assert 'id' in exp
      assert 'status' in exp
      assert 'hypothesis' in exp
      assert 'metrics' in exp
      assert exp['status'] == 'completed'
    ids = {exp['id'] for exp in exps}
    assert ids == {'exp-a', 'exp-b'}
    exp_b = next(e for e in exps if e['id'] == 'exp-b')
    assert exp_b['hypothesis'] == 'improved'
    assert exp_b['metrics']['accuracy'] == 0.85

  def test_no_active_tree_error(self, ws: Path) -> None:
    with pytest.raises(SystemExit):
      run_cli_no_context(ws, ['query', '--completed'])
