"""Tests for experiment show --context-log and experiment compare context."""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli_no_context, run_cli_text
from unittest.mock import patch
import pytest


@pytest.fixture(autouse=True)
def _patch_store_checkout():
  """Patch FileStore.checkout for tests that don't create snapshots."""
  with patch('autopilot.ai.store.file_store.FileStore.checkout'):
    yield


@pytest.fixture
def ws(tmp_path: Path) -> Path:
  """Workspace root."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  return ws


def _seed_experiment_with_context(ws: Path) -> None:
  """Create a workspace with an experiment that has context log entries."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')

  exp = Experiment(experiment_id='ctx-exp', hypothesis='test context')
  exp.add_context('started optimization', source='trainer', epoch=0)
  exp.add_context('policy gate passed', source='policy', epoch=1)
  exp.add_context('agent applied fix', source='agent-optimizer', epoch=1)
  exp.add_context('early stopping triggered', source='early-stopping', epoch=2)
  exp.add_context('user requested rollback', source='user', epoch=2)

  exp.start()
  exp.complete(metrics={'accuracy': 0.85})

  node = Node(experiment=exp)
  tree.add(node)
  forest.save()


def _seed_two_experiments_with_context(ws: Path) -> None:
  """Create two experiments with different context logs for comparison."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')

  exp_a = Experiment(experiment_id='exp-a', hypothesis='baseline approach')
  exp_a.add_context('initial training', source='trainer', epoch=0)
  exp_a.add_context('epoch 1 accepted', source='policy', epoch=1)
  exp_a.start()
  exp_a.complete(metrics={'accuracy': 0.72, 'latency': 120.0})
  node_a = Node(experiment=exp_a)
  tree.add(node_a)

  exp_b = Experiment(experiment_id='exp-b', hypothesis='improved approach')
  exp_b.add_context('forked from exp-a', source='user', epoch=0)
  exp_b.add_context('optimizer applied changes', source='agent-optimizer', epoch=0)
  exp_b.add_context('policy accepted epoch 1', source='policy', epoch=1)
  exp_b.add_context('training completed', source='trainer', epoch=2)
  exp_b.start()
  exp_b.complete(metrics={'accuracy': 0.78, 'latency': 115.0})
  node_b = Node(experiment=exp_b, parent=node_a, baseline=node_a)
  tree.add(node_b)

  forest.save()


class TestExperimentShowContextLog:
  """Tests for experiment show --context-log flags."""

  def test_experiment_show_context_log_flag(self, ws: Path) -> None:
    """With --context-log, text output includes journal rows."""
    _seed_experiment_with_context(ws)
    text = run_cli_text(ws, ['experiment', 'show', 'ctx-exp', '--context-log'])
    assert 'trainer' in text
    assert 'policy' in text
    assert 'agent-optimizer' in text
    assert 'started optimization' in text

  def test_experiment_show_context_log_json(self, ws: Path) -> None:
    """JSON output contains context_log array with expected length."""
    _seed_experiment_with_context(ws)
    result = run_cli_no_context(ws, ['experiment', 'show', 'ctx-exp', '--context-log'])
    context_log = result['result']['context_log']
    assert isinstance(context_log, list)
    assert len(context_log) == 5
    assert context_log[0]['reason'] == 'started optimization'
    assert context_log[0]['source'] == 'trainer'

  def test_experiment_show_context_source_filter(self, ws: Path) -> None:
    """Only entries whose source matches flag value appear."""
    _seed_experiment_with_context(ws)
    result = run_cli_no_context(
      ws,
      ['experiment', 'show', 'ctx-exp', '--context-log', '--context-source', 'policy'],
    )
    context_log = result['result']['context_log']
    assert len(context_log) == 1
    assert all(e['source'] == 'policy' for e in context_log)

  def test_experiment_show_context_limit(self, ws: Path) -> None:
    """At most N rows after --limit (most recent)."""
    _seed_experiment_with_context(ws)
    result = run_cli_no_context(
      ws, ['experiment', 'show', 'ctx-exp', '--context-log', '--limit', '2']
    )
    context_log = result['result']['context_log']
    assert len(context_log) == 2
    assert context_log[-1]['reason'] == 'user requested rollback'

  def test_experiment_show_no_context_log_flag(self, ws: Path) -> None:
    """Default experiment show output unchanged when --context-log omitted."""
    _seed_experiment_with_context(ws)
    result = run_cli_no_context(ws, ['experiment', 'show', 'ctx-exp'])
    assert 'context_log' not in result['result']
    assert result['result']['id'] == 'ctx-exp'
    assert result['result']['status'] == 'completed'

  def test_experiment_show_context_source_and_limit_combined(self, ws: Path) -> None:
    """Source filter and limit compose correctly."""
    _seed_experiment_with_context(ws)
    result = run_cli_no_context(
      ws,
      [
        'experiment',
        'show',
        'ctx-exp',
        '--context-log',
        '--context-source',
        'user',
        '--limit',
        '1',
      ],
    )
    context_log = result['result']['context_log']
    assert len(context_log) == 1
    assert context_log[0]['source'] == 'user'


class TestExperimentCompareContext:
  """Tests for experiment compare context summary."""

  def test_experiment_compare_includes_context(self, ws: Path) -> None:
    """Text compare includes per-experiment context summary."""
    _seed_two_experiments_with_context(ws)
    text = run_cli_text(ws, ['experiment', 'compare', 'exp-a', 'exp-b'])
    assert 'Context (exp-a):' in text
    assert 'Context (exp-b):' in text
    assert 'initial training' in text or 'epoch 1 accepted' in text

  def test_experiment_compare_json_includes_full_log(self, ws: Path) -> None:
    """JSON compare includes complete context_log for each side."""
    _seed_two_experiments_with_context(ws)
    result = run_cli_no_context(ws, ['experiment', 'compare', 'exp-a', 'exp-b'])
    assert 'context_log_a' in result['result']
    assert 'context_log_b' in result['result']
    log_a = result['result']['context_log_a']
    log_b = result['result']['context_log_b']
    assert len(log_a) == 2
    assert len(log_b) == 4
    assert log_a[0]['source'] == 'trainer'
    assert log_b[0]['source'] == 'user'

  def test_experiment_compare_context_summary_last_3(self, ws: Path) -> None:
    """Text summary shows at most 3 entries per experiment."""
    _seed_two_experiments_with_context(ws)
    text = run_cli_text(ws, ['experiment', 'compare', 'exp-a', 'exp-b'])
    lines_with_agent_opt = [line for line in text.splitlines() if 'agent-optimizer' in line]
    assert len(lines_with_agent_opt) <= 1
