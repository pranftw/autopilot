"""Tests for context log filtering on experiment show (FR-016, Plan 18).

Verifies epoch range filters (``--epoch-from`` / ``--epoch-to``), reason
substring match (``--reason-contains``), context summary mode
(``--context-summary``), combined filter composition (AND semantics), and
text/JSON output parity for all new flags.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.commands.experiment.command import (
  _build_context_summary,
  _epoch_in_range,
  _filtered_context_entries,
)
from autopilot.core.config import AutoPilotConfig
from autopilot.core.context import ContextEntry
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_text
from unittest.mock import patch
import contextlib
import io
import json
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


def _seed_filtering_workspace(ws: Path) -> None:
  """Create a workspace with one experiment carrying varied context entries.

  Entries:
    epoch=0, source='trainer', reason='started training'
    epoch=0, source='policy', reason='policy gate accepted epoch 0'
    epoch=1, source='trainer', reason='training epoch 1 complete'
    epoch=1, source='policy', reason='policy gate accepted epoch 1'
    epoch=2, source='trainer', reason='training epoch 2 complete'
    epoch=None, source='user', reason='manual note added'
  """
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')

  exp = Experiment(experiment_id='filter-exp', hypothesis='filtering test')
  exp.add_context('started training', source='trainer', epoch=0)
  exp.add_context('policy gate accepted epoch 0', source='policy', epoch=0)
  exp.add_context('training epoch 1 complete', source='trainer', epoch=1)
  exp.add_context('policy gate accepted epoch 1', source='policy', epoch=1)
  exp.add_context('training epoch 2 complete', source='trainer', epoch=2)
  exp.add_context('manual note added', source='user')

  exp.start()
  exp.complete(metrics={'accuracy': 0.9})

  node = Node(experiment=exp)
  tree.add(node)
  forest.save()


class TestEpochInRange:
  """Unit tests for the _epoch_in_range helper."""

  def test_none_epoch_excluded(self) -> None:
    """Entries with epoch=None are excluded when any bound is set."""
    assert _epoch_in_range(None, 0, 2) is False
    assert _epoch_in_range(None, 0, None) is False
    assert _epoch_in_range(None, None, 2) is False

  def test_within_range(self) -> None:
    """Epoch within bounds passes."""
    assert _epoch_in_range(1, 0, 2) is True
    assert _epoch_in_range(0, 0, 0) is True

  def test_below_lower_bound(self) -> None:
    """Epoch below lower bound is excluded."""
    assert _epoch_in_range(0, 1, 3) is False

  def test_above_upper_bound(self) -> None:
    """Epoch above upper bound is excluded."""
    assert _epoch_in_range(3, 0, 2) is False

  def test_only_lower_bound(self) -> None:
    """With only epoch_from, no upper limit."""
    assert _epoch_in_range(5, 1, None) is True
    assert _epoch_in_range(0, 1, None) is False

  def test_only_upper_bound(self) -> None:
    """With only epoch_to, no lower limit."""
    assert _epoch_in_range(0, None, 2) is True
    assert _epoch_in_range(3, None, 2) is False


class TestBuildContextSummary:
  """Unit tests for _build_context_summary."""

  def test_counts_by_source(self) -> None:
    """Entries are grouped by source with correct counts."""
    entries = [
      ContextEntry.create('a', source='trainer'),
      ContextEntry.create('b', source='trainer'),
      ContextEntry.create('c', source='policy'),
    ]
    summary = _build_context_summary(entries)
    assert summary == {'trainer': 2, 'policy': 1}

  def test_sorted_desc_count_then_key(self) -> None:
    """Summary is sorted by descending count, then ascending key."""
    entries = [
      ContextEntry.create('a', source='beta'),
      ContextEntry.create('b', source='alpha'),
      ContextEntry.create('c', source='beta'),
      ContextEntry.create('d', source='alpha'),
      ContextEntry.create('e', source='gamma'),
    ]
    summary = _build_context_summary(entries)
    keys = list(summary.keys())
    assert keys == ['alpha', 'beta', 'gamma']
    assert summary == {'alpha': 2, 'beta': 2, 'gamma': 1}

  def test_empty_entries(self) -> None:
    """Empty input produces empty summary."""
    assert _build_context_summary([]) == {}

  def test_none_source_uses_empty_string(self) -> None:
    """Entries with source=None are grouped under empty string key."""
    entries = [ContextEntry.create('a'), ContextEntry.create('b')]
    summary = _build_context_summary(entries)
    assert summary == {'': 2}


class TestFilteredContextEntries:
  """Unit tests for _filtered_context_entries with new filter params."""

  def _make_experiment(self) -> Experiment:
    """Build an experiment with diverse context entries for filter testing."""
    exp = Experiment(experiment_id='test', hypothesis='test')
    exp.add_context('started training', source='trainer', epoch=0)
    exp.add_context('policy gate', source='policy', epoch=1)
    exp.add_context('training complete', source='trainer', epoch=2)
    exp.add_context('no epoch entry', source='user')
    return exp

  def test_epoch_from_filter(self) -> None:
    """epoch_from excludes entries below the bound and None-epoch entries."""
    exp = self._make_experiment()
    entries = _filtered_context_entries(exp, None, None, epoch_from=1)
    assert len(entries) == 2
    assert all(e.epoch is not None and e.epoch >= 1 for e in entries)

  def test_epoch_to_filter(self) -> None:
    """epoch_to excludes entries above the bound and None-epoch entries."""
    exp = self._make_experiment()
    entries = _filtered_context_entries(exp, None, None, epoch_to=0)
    assert len(entries) == 1
    assert entries[0].epoch == 0

  def test_epoch_range_filter(self) -> None:
    """Combined epoch_from and epoch_to restricts to inclusive range."""
    exp = self._make_experiment()
    entries = _filtered_context_entries(exp, None, None, epoch_from=1, epoch_to=1)
    assert len(entries) == 1
    assert entries[0].epoch == 1
    assert entries[0].source == 'policy'

  def test_reason_substr_filter(self) -> None:
    """reason_substr performs case-sensitive substring match."""
    exp = self._make_experiment()
    entries = _filtered_context_entries(exp, None, None, reason_substr='training')
    assert len(entries) == 2
    assert all('training' in e.reason for e in entries)

  def test_reason_substr_case_sensitive(self) -> None:
    """reason_substr is case-sensitive: 'Training' does not match 'training'."""
    exp = self._make_experiment()
    entries = _filtered_context_entries(exp, None, None, reason_substr='Training')
    assert len(entries) == 0

  def test_all_filters_combined(self) -> None:
    """source + epoch + reason compose with AND."""
    exp = self._make_experiment()
    entries = _filtered_context_entries(
      exp, 'trainer', None, epoch_from=2, epoch_to=2, reason_substr='complete'
    )
    assert len(entries) == 1
    assert entries[0].reason == 'training complete'
    assert entries[0].epoch == 2

  def test_limit_applied_after_filters(self) -> None:
    """limit takes the N most recent after all other filters."""
    exp = self._make_experiment()
    entries = _filtered_context_entries(exp, 'trainer', 1, epoch_from=0)
    assert len(entries) == 1
    assert entries[0].epoch == 2


class TestShowContextEpochRangeFilters:
  """CLI tests for --epoch-from and --epoch-to on experiment show."""

  def test_show_context_epoch_range_filters(self, ws: Path) -> None:
    """--epoch-from 1 --epoch-to 1 returns only epoch 1 entries."""
    _seed_filtering_workspace(ws)
    result = run_cli(
      ws,
      [
        'experiment',
        'show',
        'filter-exp',
        '--context-log',
        '--epoch-from',
        '1',
        '--epoch-to',
        '1',
      ],
    )
    context_log = result['result']['context_log']
    assert len(context_log) == 2
    assert all(e['epoch'] == 1 for e in context_log)

  def test_epoch_none_excluded(self, ws: Path) -> None:
    """Entries with epoch=None are excluded when epoch bounds are set."""
    _seed_filtering_workspace(ws)
    result = run_cli(
      ws,
      ['experiment', 'show', 'filter-exp', '--context-log', '--epoch-from', '0'],
    )
    context_log = result['result']['context_log']
    assert all(e['epoch'] is not None for e in context_log)
    assert len(context_log) == 5


class TestShowContextReasonContains:
  """CLI tests for --reason-contains on experiment show."""

  def test_show_context_reason_contains(self, ws: Path) -> None:
    """--reason-contains selects entries whose reason contains the substring."""
    _seed_filtering_workspace(ws)
    result = run_cli(
      ws,
      [
        'experiment',
        'show',
        'filter-exp',
        '--context-log',
        '--reason-contains',
        'policy gate',
      ],
    )
    context_log = result['result']['context_log']
    assert len(context_log) == 2
    assert all('policy gate' in e['reason'] for e in context_log)

  def test_reason_contains_case_sensitive(self, ws: Path) -> None:
    """--reason-contains is case-sensitive: uppercase mismatch yields 0 results."""
    _seed_filtering_workspace(ws)
    result = run_cli(
      ws,
      [
        'experiment',
        'show',
        'filter-exp',
        '--context-log',
        '--reason-contains',
        'Policy Gate',
      ],
    )
    context_log = result['result']['context_log']
    assert len(context_log) == 0


class TestShowContextSummary:
  """CLI tests for --context-summary on experiment show."""

  def test_show_context_summary_requires_context_log(self, ws: Path) -> None:
    """--context-summary without --context-log triggers ctx.fail."""
    _seed_filtering_workspace(ws)
    from autopilot.cli.context import build_context
    from autopilot.cli.main import build_parser

    parser = build_parser()
    full_argv = [
      'experiment',
      'show',
      'filter-exp',
      '--context-summary',
      '--workspace',
      str(ws),
      '--json',
      '--context',
      'test',
    ]
    parsed = parser.parse_args(full_argv)
    ctx = build_context(parsed)

    buf = io.StringIO()
    with pytest.raises(SystemExit), contextlib.redirect_stdout(buf):
      parsed.handler(ctx, parsed)

    output = buf.getvalue().strip()
    payload = json.loads(output)
    assert payload['ok'] is False
    assert '--context-log' in payload['error']

  def test_show_context_summary_counts(self, ws: Path) -> None:
    """--context-summary returns correct {source: count} distribution."""
    _seed_filtering_workspace(ws)
    result = run_cli(
      ws,
      ['experiment', 'show', 'filter-exp', '--context-log', '--context-summary'],
    )
    payload = result['result']
    summary = payload['context_summary']
    assert summary == {'trainer': 3, 'policy': 2, 'user': 1}

  def test_show_context_summary_json_shape(self, ws: Path) -> None:
    """JSON summary mode includes context_summary but omits context_log."""
    _seed_filtering_workspace(ws)
    result = run_cli(
      ws,
      ['experiment', 'show', 'filter-exp', '--context-log', '--context-summary'],
    )
    payload = result['result']
    assert 'context_summary' in payload
    assert 'context_log' not in payload
    assert isinstance(payload['context_summary'], dict)
    assert all(isinstance(v, int) for v in payload['context_summary'].values())

  def test_show_context_summary_text_table(self, ws: Path) -> None:
    """Text summary mode renders a source/count table."""
    _seed_filtering_workspace(ws)
    text = run_cli_text(
      ws,
      ['experiment', 'show', 'filter-exp', '--context-log', '--context-summary'],
    )
    assert 'Context summary' in text
    assert 'trainer' in text
    assert 'policy' in text


class TestShowContextCombinedFilters:
  """CLI tests for combined filter composition."""

  def test_show_context_combined_filters(self, ws: Path) -> None:
    """source + reason + epoch together reduce to a single matching row."""
    _seed_filtering_workspace(ws)
    result = run_cli(
      ws,
      [
        'experiment',
        'show',
        'filter-exp',
        '--context-log',
        '--context-source',
        'policy',
        '--epoch-from',
        '1',
        '--epoch-to',
        '1',
        '--reason-contains',
        'epoch 1',
      ],
    )
    context_log = result['result']['context_log']
    assert len(context_log) == 1
    assert context_log[0]['source'] == 'policy'
    assert context_log[0]['epoch'] == 1
    assert 'epoch 1' in context_log[0]['reason']

  def test_show_context_empty_after_filters(self, ws: Path) -> None:
    """Impossible filter combination yields empty context_log."""
    _seed_filtering_workspace(ws)
    result = run_cli(
      ws,
      [
        'experiment',
        'show',
        'filter-exp',
        '--context-log',
        '--context-source',
        'user',
        '--epoch-from',
        '0',
        '--epoch-to',
        '0',
      ],
    )
    context_log = result['result']['context_log']
    assert context_log == []

  def test_show_context_empty_after_filters_text(self, ws: Path) -> None:
    """Text mode with zero matching entries prints no journal rows."""
    _seed_filtering_workspace(ws)
    text = run_cli_text(
      ws,
      [
        'experiment',
        'show',
        'filter-exp',
        '--context-log',
        '--context-source',
        'user',
        '--epoch-from',
        '0',
        '--epoch-to',
        '0',
      ],
    )
    assert 'manual note' not in text

  def test_show_context_source_still_works(self, ws: Path) -> None:
    """Regression: existing --context-source alone still works unchanged."""
    _seed_filtering_workspace(ws)
    result = run_cli(
      ws,
      [
        'experiment',
        'show',
        'filter-exp',
        '--context-log',
        '--context-source',
        'trainer',
      ],
    )
    context_log = result['result']['context_log']
    assert len(context_log) == 3
    assert all(e['source'] == 'trainer' for e in context_log)


class TestShowContextSummaryWithFilters:
  """Summary mode interacts correctly with other filters."""

  def test_summary_with_epoch_filter(self, ws: Path) -> None:
    """Summary counts only entries passing epoch filter."""
    _seed_filtering_workspace(ws)
    result = run_cli(
      ws,
      [
        'experiment',
        'show',
        'filter-exp',
        '--context-log',
        '--context-summary',
        '--epoch-from',
        '0',
        '--epoch-to',
        '0',
      ],
    )
    summary = result['result']['context_summary']
    assert summary == {'policy': 1, 'trainer': 1}

  def test_summary_with_reason_filter(self, ws: Path) -> None:
    """Summary counts only entries matching reason substring."""
    _seed_filtering_workspace(ws)
    result = run_cli(
      ws,
      [
        'experiment',
        'show',
        'filter-exp',
        '--context-log',
        '--context-summary',
        '--reason-contains',
        'policy gate',
      ],
    )
    summary = result['result']['context_summary']
    assert summary == {'policy': 2}
