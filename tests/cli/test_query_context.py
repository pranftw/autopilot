"""Tests for query CLI context-log filters and JSON envelope."""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.context import ContextEntry
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli_no_context
import pytest


def _seed_with_context(ws: Path) -> None:
  """Seed a tree with experiments carrying distinct context logs."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')

  exp_a = Experiment(experiment_id='exp-a', hypothesis='baseline')
  exp_a.start()
  exp_a.add_context('initial setup', source='user', epoch=0)
  exp_a.add_context('policy gate accepted', source='policy', epoch=0)
  exp_a.complete(metrics={'accuracy': 0.72})

  exp_b = Experiment(experiment_id='exp-b', hypothesis='improved')
  exp_b.start()
  exp_b.add_context('rollback attempted', source='trainer', epoch=1)
  exp_b.add_context('agent applied fix', source='agent-optimizer', epoch=1)
  exp_b.complete(metrics={'accuracy': 0.85})

  exp_c = Experiment(experiment_id='exp-c', hypothesis='no context')
  exp_c.start()
  exp_c.complete(metrics={'accuracy': 0.60})

  node_a = Node(experiment=exp_a)
  tree.add(node_a)

  node_b = Node(experiment=exp_b, parent=node_a, baseline=node_a)
  tree.add(node_b)

  node_c = Node(experiment=exp_c, parent=node_a, baseline=node_a)
  tree.add(node_c)

  forest.save()


def _seed_with_timestamped_context(ws: Path) -> None:
  """Seed experiments with explicit timestamps for temporal filtering."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')

  exp_early = Experiment(experiment_id='exp-early', hypothesis='early')
  exp_early.start()
  entry_early = ContextEntry(
    timestamp='2024-01-01T00:00:00+00:00',
    reason='early event',
    source='user',
  )
  exp_early.context_log.record(entry_early)
  exp_early.complete(metrics={'score': 1.0})

  exp_late = Experiment(experiment_id='exp-late', hypothesis='late')
  exp_late.start()
  entry_late = ContextEntry(
    timestamp='2024-06-15T12:00:00+00:00',
    reason='late event',
    source='trainer',
  )
  exp_late.context_log.record(entry_late)
  exp_late.complete(metrics={'score': 2.0})

  exp_latest = Experiment(experiment_id='exp-latest', hypothesis='latest')
  exp_latest.start()
  entry_latest = ContextEntry(
    timestamp='2024-12-01T08:30:00+05:30',
    reason='offset event',
    source='policy',
  )
  exp_latest.context_log.record(entry_latest)
  exp_latest.complete(metrics={'score': 3.0})

  tree.add(Node(experiment=exp_early))
  node_late = Node(experiment=exp_late, parent=tree.get('exp-early'))
  tree.add(node_late)
  node_latest = Node(experiment=exp_latest, parent=tree.get('exp-early'))
  tree.add(node_latest)

  forest.save()


@pytest.fixture
def ctx_ws(tmp_path: Path) -> Path:
  """Workspace with context-bearing experiments."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  _seed_with_context(ws)
  return ws


@pytest.fixture
def ts_ws(tmp_path: Path) -> Path:
  """Workspace with timestamped context entries."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  _seed_with_timestamped_context(ws)
  return ws


class TestContextContains:
  def test_context_contains_filters_by_reason(self, ctx_ws: Path) -> None:
    result = run_cli_no_context(ctx_ws, ['query', '--context-contains', 'rollback'])
    exps = result['result']['experiments']
    assert len(exps) == 1
    assert exps[0]['id'] == 'exp-b'

  def test_context_contains_no_match_empty(self, ctx_ws: Path) -> None:
    result = run_cli_no_context(ctx_ws, ['query', '--context-contains', 'nonexistent-xyz'])
    exps = result['result']['experiments']
    assert len(exps) == 0
    assert result['result']['count'] == 0

  def test_context_contains_multiple_matches(self, ctx_ws: Path) -> None:
    result = run_cli_no_context(ctx_ws, ['query', '--context-contains', 'policy'])
    exps = result['result']['experiments']
    assert len(exps) == 1
    assert exps[0]['id'] == 'exp-a'


class TestContextSource:
  def test_context_source_filters_by_source(self, ctx_ws: Path) -> None:
    result = run_cli_no_context(ctx_ws, ['query', '--context-source', 'agent-optimizer'])
    exps = result['result']['experiments']
    assert len(exps) == 1
    assert exps[0]['id'] == 'exp-b'

  def test_context_source_user(self, ctx_ws: Path) -> None:
    result = run_cli_no_context(ctx_ws, ['query', '--context-source', 'user'])
    exps = result['result']['experiments']
    assert len(exps) == 1
    assert exps[0]['id'] == 'exp-a'

  def test_context_source_no_match(self, ctx_ws: Path) -> None:
    result = run_cli_no_context(ctx_ws, ['query', '--context-source', 'unknown-source'])
    exps = result['result']['experiments']
    assert len(exps) == 0


class TestContextAfter:
  def test_context_after_filters_by_timestamp(self, ts_ws: Path) -> None:
    result = run_cli_no_context(ts_ws, ['query', '--context-after', '2024-06-01T00:00:00+00:00'])
    exps = result['result']['experiments']
    ids = {e['id'] for e in exps}
    assert 'exp-late' in ids
    assert 'exp-latest' in ids
    assert 'exp-early' not in ids

  def test_context_after_inclusive(self, ts_ws: Path) -> None:
    result = run_cli_no_context(ts_ws, ['query', '--context-after', '2024-06-15T12:00:00+00:00'])
    exps = result['result']['experiments']
    ids = {e['id'] for e in exps}
    assert 'exp-late' in ids
    assert 'exp-latest' in ids

  def test_context_after_with_timezone(self, ts_ws: Path) -> None:
    result = run_cli_no_context(ts_ws, ['query', '--context-after', '2024-12-01T03:00:00+00:00'])
    exps = result['result']['experiments']
    ids = {e['id'] for e in exps}
    assert 'exp-latest' in ids
    assert 'exp-early' not in ids
    assert 'exp-late' not in ids


class TestCombinedFilters:
  def test_combined_filters(self, ctx_ws: Path) -> None:
    result = run_cli_no_context(
      ctx_ws, ['query', '--context-contains', 'agent', '--context-source', 'agent-optimizer']
    )
    exps = result['result']['experiments']
    assert len(exps) == 1
    assert exps[0]['id'] == 'exp-b'

  def test_combined_contradictory_empty(self, ctx_ws: Path) -> None:
    result = run_cli_no_context(
      ctx_ws, ['query', '--context-contains', 'rollback', '--context-source', 'user']
    )
    exps = result['result']['experiments']
    assert len(exps) == 0


class TestJsonOutput:
  def test_json_output_includes_context_log(self, ctx_ws: Path) -> None:
    result = run_cli_no_context(ctx_ws, ['query', '--completed'])
    exps = result['result']['experiments']
    for exp in exps:
      assert 'context_log' in exp
      assert isinstance(exp['context_log'], list)

  def test_json_context_log_entries_complete(self, ctx_ws: Path) -> None:
    result = run_cli_no_context(ctx_ws, ['query', '--context-source', 'user'])
    exps = result['result']['experiments']
    assert len(exps) == 1
    entries = exps[0]['context_log']
    assert len(entries) >= 1
    for entry in entries:
      assert 'timestamp' in entry
      assert 'reason' in entry
      assert 'source' in entry
      assert 'metadata' in entry

  def test_json_best_includes_context_log(self, ctx_ws: Path) -> None:
    result = run_cli_no_context(ctx_ws, ['query', '--best', 'accuracy'])
    best = result['result']['best']
    assert 'context_log' in best
    assert isinstance(best['context_log'], list)

  def test_json_empty_context_log(self, ctx_ws: Path) -> None:
    result = run_cli_no_context(ctx_ws, ['query', '--completed'])
    exps = result['result']['experiments']
    exp_c = next(e for e in exps if e['id'] == 'exp-c')
    assert exp_c['context_log'] == []


class TestQueryWithoutContextFlagsUnchanged:
  def test_query_without_context_flags_unchanged(self, ctx_ws: Path) -> None:
    result = run_cli_no_context(ctx_ws, ['query', '--completed'])
    exps = result['result']['experiments']
    assert len(exps) == 3
    ids = {e['id'] for e in exps}
    assert ids == {'exp-a', 'exp-b', 'exp-c'}
