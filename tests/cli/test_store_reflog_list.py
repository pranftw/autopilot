"""Tests for the ``store reflog list`` CLI command (sub-plan 04).

Covers:
  - Command existence and basic invocation.
  - Context exemption (read-only, no --context required).
  - JSON output parity with ``debug store reflog``.
  - CLI stash-pop threads ``--context`` into reflog.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_no_context
from typing import Any


def _make_workspace_with_snapshot(
  tmp_path: Path,
  experiment_id: str = 'exp-reflog',
) -> dict[str, Any]:
  """Create a workspace with a forest, store, and one snapshot for reflog tests.

  Returns dict with keys: workspace, config, store, forest, source_dir, param.
  """
  ws = tmp_path / 'ws'
  ws.mkdir()
  source_dir = ws / 'src'
  source_dir.mkdir()
  (source_dir / 'main.py').write_text('print("hello")', encoding='utf-8')

  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)

  param = PathParameter(source=str(source_dir), pattern='*.py')
  store = FileStore(config)
  store.register_parameters({'source': param})
  store.snapshot(experiment_id, 0, context='initial')

  forest = FileForest(store)
  tree = forest.create_tree('main')
  exp = Experiment(experiment_id=experiment_id, hypothesis='test')
  exp.start()
  exp.complete(metrics={'score': 0.75})
  tree.add(Node(experiment=exp))
  forest.switch('main')
  forest.save()

  return {
    'workspace': ws,
    'config': config,
    'store': store,
    'forest': forest,
    'source_dir': source_dir,
    'param': param,
  }


class TestStoreReflogListExists:
  """Parser accepts ``store reflog list``; exits zero with minimal store."""

  def test_store_reflog_list_command_exists(self, tmp_path: Path) -> None:
    """store reflog list returns entries and count for a store with at least one entry."""
    ctx = _make_workspace_with_snapshot(tmp_path)
    ws = ctx['workspace']

    envelope = run_cli_no_context(ws, ['store', 'reflog', 'list'])
    result = envelope['result']
    assert 'entries' in result
    assert 'count' in result
    assert isinstance(result['entries'], list)
    assert result['count'] >= 1

  def test_store_reflog_list_empty_store(self, tmp_path: Path) -> None:
    """store reflog list on an empty store returns zero entries."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    tree = forest.create_tree('main')
    exp = Experiment(experiment_id='exp-empty', hypothesis='test')
    exp.start()
    exp.complete(metrics={'score': 0.5})
    tree.add(Node(experiment=exp))
    forest.switch('main')
    forest.save()

    envelope = run_cli_no_context(ws, ['store', 'reflog', 'list'])
    result = envelope['result']
    assert result['entries'] == []
    assert result['count'] == 0


class TestStoreReflogListContextExempt:
  """``store reflog list`` is context-exempt (read-only)."""

  def test_store_reflog_list_context_exempt(self, tmp_path: Path) -> None:
    """run_cli_no_context succeeds without --context."""
    ctx = _make_workspace_with_snapshot(tmp_path)
    ws = ctx['workspace']

    envelope = run_cli_no_context(ws, ['store', 'reflog', 'list'])
    assert 'entries' in envelope['result']


class TestStoreReflogListJsonMatchesDebug:
  """JSON payloads match between ``store reflog list`` and ``debug store reflog``."""

  def test_store_reflog_list_json_matches_debug(self, tmp_path: Path) -> None:
    """Both commands produce identical entries and count for the same store state."""
    ctx = _make_workspace_with_snapshot(tmp_path)
    ws = ctx['workspace']

    reflog_envelope = run_cli_no_context(ws, ['store', 'reflog', 'list'])
    debug_envelope = run_cli_no_context(ws, ['debug', 'store', 'reflog'])

    reflog_result = reflog_envelope['result']
    debug_result = debug_envelope['result']

    assert reflog_result['entries'] == debug_result['entries']
    assert reflog_result['count'] == debug_result['count']

  def test_store_reflog_list_json_matches_debug_with_limit(self, tmp_path: Path) -> None:
    """Both commands produce identical output when limited to N entries."""
    ctx = _make_workspace_with_snapshot(tmp_path)
    ws = ctx['workspace']
    store = ctx['store']
    source_dir = ctx['source_dir']

    (source_dir / 'main.py').write_text('v2', encoding='utf-8')
    store.snapshot('exp-reflog', 1, context='second')

    reflog_envelope = run_cli_no_context(ws, ['store', 'reflog', 'list', '-n', '1'])
    debug_envelope = run_cli_no_context(ws, ['debug', 'store', 'reflog', '-n', '1'])

    reflog_result = reflog_envelope['result']
    debug_result = debug_envelope['result']

    assert reflog_result['entries'] == debug_result['entries']
    assert reflog_result['count'] == debug_result['count']
    assert reflog_result['count'] == 1


class TestStashPopCliThreadsContext:
  """CLI stash-pop threads ``--context`` into the reflog entry."""

  def test_stash_pop_cli_threads_context(self, tmp_path: Path) -> None:
    """After CLI stash-pop with --context, reflog entry contains the context."""
    ctx = _make_workspace_with_snapshot(tmp_path)
    ws = ctx['workspace']

    run_cli(
      ws,
      [
        '--experiment',
        'exp-reflog',
        'store',
        'stash',
      ],
    )

    run_cli(
      ws,
      [
        '--experiment',
        'exp-reflog',
        'store',
        'stash-pop',
      ],
    )

    reflog_envelope = run_cli_no_context(ws, ['store', 'reflog', 'list'])
    entries = reflog_envelope['result']['entries']
    pop_entries = [e for e in entries if e.get('operation') == 'stash_pop']
    assert len(pop_entries) == 1
    assert pop_entries[0]['context'] == 'test'
