"""Tests for workspace journal command.

Verifies aggregation across experiments, provenance annotation,
chronological sorting, source/limit/since filters, empty workspace
handling, JSON envelope shape, and context exemption.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.context import ContextEntry
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from autopilot.tracking.io import parse_timestamp
from pathlib import Path
from tests.cli.conftest import run_cli_no_context
import pytest


def _make_entry(
  reason: str,
  *,
  timestamp: str,
  source: str | None = None,
  epoch: int | None = None,
) -> ContextEntry:
  """Build a ContextEntry with an explicit timestamp (no clock dependency)."""
  return ContextEntry(
    timestamp=timestamp,
    reason=reason,
    source=source,
    epoch=epoch,
    metadata={},
  )


def _seed_workspace_with_journal(
  tmp_path: Path,
) -> tuple[Path, FileForest]:
  """Create a workspace with two trees and experiments bearing context entries.

  Tree 'alpha' has experiment 'exp-a' with 3 entries.
  Tree 'beta' has experiment 'exp-b' with 4 entries.

  Timestamps are deliberately out of disk order to verify sorting.

  Returns:
    Tuple of (workspace_path, forest).
  """
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  tree_a = forest.create_tree('alpha')

  exp_a = Experiment(experiment_id='exp-a', hypothesis='test a')
  exp_a.start()
  exp_a.context_log.append(
    _make_entry('step one', timestamp='2026-01-10T10:00:00+00:00', source='trainer', epoch=0)
  )
  exp_a.context_log.append(
    _make_entry('step three', timestamp='2026-01-12T10:00:00+00:00', source='user')
  )
  exp_a.context_log.append(
    _make_entry('step five', timestamp='2026-01-14T10:00:00+00:00', source='trainer', epoch=1)
  )
  tree_a.add(Node(experiment=exp_a))

  tree_b = forest.create_tree('beta')

  exp_b = Experiment(experiment_id='exp-b', hypothesis='test b')
  exp_b.start()
  exp_b.context_log.append(
    _make_entry('step two', timestamp='2026-01-11T10:00:00+00:00', source='policy')
  )
  exp_b.context_log.append(
    _make_entry('step four', timestamp='2026-01-13T10:00:00+00:00', source='trainer', epoch=0)
  )
  exp_b.context_log.append(
    _make_entry('step six', timestamp='2026-01-15T10:00:00+00:00', source='user')
  )
  exp_b.context_log.append(
    _make_entry('step seven', timestamp='2026-01-16T10:00:00+00:00', source='trainer', epoch=1)
  )
  tree_b.add(Node(experiment=exp_b))

  forest.switch('alpha')
  forest.save()
  return ws, forest


# ---------------------------------------------------------------------------
# 4.1: Aggregation and filters
# ---------------------------------------------------------------------------


class TestWorkspaceJournalAggregation:
  """workspace journal merges context log entries from all experiments."""

  def test_workspace_journal_aggregates_across_experiments(self, tmp_path: Path) -> None:
    """Merged entries length equals sum of per-experiment context entries."""
    ws, _ = _seed_workspace_with_journal(tmp_path)
    result = run_cli_no_context(ws, ['workspace', 'journal'])
    entries = result['result']['entries']
    assert len(entries) == 7

  def test_workspace_journal_annotates_experiment_id_and_tree(self, tmp_path: Path) -> None:
    """Every entry includes experiment_id and tree matching its origin."""
    ws, _ = _seed_workspace_with_journal(tmp_path)
    result = run_cli_no_context(ws, ['workspace', 'journal'])
    entries = result['result']['entries']

    for entry in entries:
      assert 'experiment_id' in entry
      assert 'tree' in entry

    alpha_entries = [e for e in entries if e['tree'] == 'alpha']
    beta_entries = [e for e in entries if e['tree'] == 'beta']
    assert all(e['experiment_id'] == 'exp-a' for e in alpha_entries)
    assert all(e['experiment_id'] == 'exp-b' for e in beta_entries)
    assert len(alpha_entries) == 3
    assert len(beta_entries) == 4

  def test_workspace_journal_sorted_by_timestamp(self, tmp_path: Path) -> None:
    """Output is sorted ascending by parsed timestamp."""
    ws, _ = _seed_workspace_with_journal(tmp_path)
    result = run_cli_no_context(ws, ['workspace', 'journal'])
    entries = result['result']['entries']

    timestamps = [parse_timestamp(e['timestamp']) for e in entries]
    assert timestamps == sorted(timestamps)

    reasons = [e['reason'] for e in entries]
    assert reasons == [
      'step one',
      'step two',
      'step three',
      'step four',
      'step five',
      'step six',
      'step seven',
    ]

  def test_workspace_journal_source_filter(self, tmp_path: Path) -> None:
    """--source trainer includes only trainer entries."""
    ws, _ = _seed_workspace_with_journal(tmp_path)
    result = run_cli_no_context(ws, ['workspace', 'journal', '--source', 'trainer'])
    entries = result['result']['entries']

    assert len(entries) == 4
    assert all(e['source'] == 'trainer' for e in entries)

  def test_workspace_journal_limit(self, tmp_path: Path) -> None:
    """--limit 3 retains only the 3 most recent entries, still sorted ascending."""
    ws, _ = _seed_workspace_with_journal(tmp_path)
    result = run_cli_no_context(ws, ['workspace', 'journal', '--limit', '3'])
    entries = result['result']['entries']

    assert len(entries) == 3
    reasons = [e['reason'] for e in entries]
    assert reasons == ['step five', 'step six', 'step seven']

    timestamps = [parse_timestamp(e['timestamp']) for e in entries]
    assert timestamps == sorted(timestamps)

  def test_workspace_journal_since_filter(self, tmp_path: Path) -> None:
    """--since excludes entries older than the threshold via datetime comparison."""
    ws, _ = _seed_workspace_with_journal(tmp_path)
    result = run_cli_no_context(
      ws, ['workspace', 'journal', '--since', '2026-01-14T00:00:00+00:00']
    )
    entries = result['result']['entries']

    assert len(entries) == 3
    for entry in entries:
      ts = parse_timestamp(entry['timestamp'])
      assert ts >= parse_timestamp('2026-01-14T00:00:00+00:00')

  def test_workspace_journal_empty_workspace(self, tmp_path: Path) -> None:
    """Forest with zero experiments yields entries=[], exit 0."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    forest.create_tree('empty')
    forest.switch('empty')
    forest.save()

    result = run_cli_no_context(ws, ['workspace', 'journal'])
    assert result['ok'] is True
    assert result['result']['entries'] == []

  def test_workspace_journal_combined_source_and_limit(self, tmp_path: Path) -> None:
    """--source + --limit composes: filter by source, then take N most recent."""
    ws, _ = _seed_workspace_with_journal(tmp_path)
    result = run_cli_no_context(ws, ['workspace', 'journal', '--source', 'trainer', '--limit', '2'])
    entries = result['result']['entries']

    assert len(entries) == 2
    assert all(e['source'] == 'trainer' for e in entries)
    reasons = [e['reason'] for e in entries]
    assert reasons == ['step five', 'step seven']

  def test_workspace_journal_since_and_limit(self, tmp_path: Path) -> None:
    """--since + --limit composes: filter by time, then take N most recent."""
    ws, _ = _seed_workspace_with_journal(tmp_path)
    result = run_cli_no_context(
      ws,
      ['workspace', 'journal', '--since', '2026-01-12T00:00:00+00:00', '--limit', '2'],
    )
    entries = result['result']['entries']

    assert len(entries) == 2
    reasons = [e['reason'] for e in entries]
    assert reasons == ['step six', 'step seven']

  def test_workspace_journal_invalid_since_fails(self, tmp_path: Path) -> None:
    """Malformed --since value triggers ctx.fail with actionable message."""
    ws, _ = _seed_workspace_with_journal(tmp_path)
    with pytest.raises(SystemExit):
      run_cli_no_context(ws, ['workspace', 'journal', '--since', 'not-a-date'])


# ---------------------------------------------------------------------------
# 4.2: CLI wiring
# ---------------------------------------------------------------------------


class TestWorkspaceJournalCLIWiring:
  """JSON envelope shape and context exemption."""

  def test_workspace_journal_json_shape(self, tmp_path: Path) -> None:
    """Envelope has ok, result.entries, and messages keys with correct types."""
    ws, _ = _seed_workspace_with_journal(tmp_path)
    result = run_cli_no_context(ws, ['workspace', 'journal'])

    assert result['ok'] is True
    assert 'result' in result
    assert 'entries' in result['result']
    assert isinstance(result['result']['entries'], list)
    assert 'messages' in result

    if result['result']['entries']:
      entry = result['result']['entries'][0]
      assert 'timestamp' in entry
      assert 'reason' in entry
      assert 'experiment_id' in entry
      assert 'tree' in entry

  def test_workspace_journal_context_exempt(self, tmp_path: Path) -> None:
    """workspace journal succeeds without --context (read-only command)."""
    ws, _ = _seed_workspace_with_journal(tmp_path)
    result = run_cli_no_context(ws, ['workspace', 'journal'])
    assert result['ok'] is True

  def test_workspace_journal_no_forest_returns_empty(self, tmp_path: Path) -> None:
    """When forest cannot be loaded, returns empty entries gracefully."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    result = run_cli_no_context(ws, ['workspace', 'journal'])
    assert result['result']['entries'] == []


class TestWorkspaceJournalTieBreaking:
  """Verify deterministic tie-breaking when timestamps are identical."""

  def test_same_timestamp_sorted_by_tree_then_experiment(self, tmp_path: Path) -> None:
    """Entries with identical timestamps use (tree, experiment_id) for ordering."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)

    same_ts = '2026-06-01T12:00:00+00:00'

    tree_a = forest.create_tree('aaa')
    exp_a = Experiment(experiment_id='exp-z', hypothesis='test')
    exp_a.start()
    exp_a.context_log.append(_make_entry('from aaa/exp-z', timestamp=same_ts, source='user'))
    tree_a.add(Node(experiment=exp_a))

    tree_b = forest.create_tree('bbb')
    exp_b = Experiment(experiment_id='exp-a', hypothesis='test')
    exp_b.start()
    exp_b.context_log.append(_make_entry('from bbb/exp-a', timestamp=same_ts, source='user'))
    tree_b.add(Node(experiment=exp_b))

    forest.switch('aaa')
    forest.save()

    result = run_cli_no_context(ws, ['workspace', 'journal'])
    entries = result['result']['entries']
    assert len(entries) == 2
    assert entries[0]['tree'] == 'aaa'
    assert entries[1]['tree'] == 'bbb'
