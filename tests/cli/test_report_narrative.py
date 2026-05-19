"""Tests for autopilot report narrative command (Plan 23)."""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.commands.report.narrative import (
  ReportNarrative,
  _build_narrative,
  _collect_recent_context,
  _collect_recent_executions,
  _collect_reflog_tail,
  _execution_record_subset,
  _render_narrative_text,
  _tree_summary,
)
from autopilot.cli.output import Output
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from autopilot.tracking.executions import ExecutionRecord
from autopilot.tracking.io import append_jsonl, utc_now_iso
from pathlib import Path
from tests.cli.conftest import run_cli_no_context
from unittest.mock import MagicMock, patch
import argparse


def _build_forest(
  tmp_path: Path,
  trees: dict[str, list[dict]],
) -> FileForest:
  """Build a FileForest with multiple trees and experiments.

  Args:
    tmp_path: Temporary directory for workspace.
    trees: Mapping of tree name to list of experiment dicts. Each dict
      has 'id', optional 'status' (default 'pending'), optional 'metrics',
      optional 'context_entries' (list of reason strings).
  """
  config = AutoPilotConfig(workspace=tmp_path)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  first_tree = None
  for tree_name, experiments in trees.items():
    tree = forest.create_tree(tree_name)
    if first_tree is None:
      first_tree = tree_name
    forest.switch(tree_name)
    for exp_data in experiments:
      exp = Experiment(experiment_id=exp_data['id'], hypothesis=exp_data.get('hypothesis'))
      status = exp_data.get('status', 'pending')
      metrics = exp_data.get('metrics', {})

      if status in {'running', 'completed', 'failed', 'cancelled'}:
        exp.start()
      if status == 'completed':
        exp.complete(metrics=metrics)
      elif status == 'failed':
        exp.fail()
      elif status == 'cancelled':
        exp.cancel()
      elif status == 'running' and metrics:
        exp.metrics = metrics

      for reason in exp_data.get('context_entries', []):
        exp.add_context(reason, source='test')

      node = Node(experiment=exp)
      tree.add(node)

      exp_dir = config.experiment_path(slug=exp_data['id'])
      exp_dir.mkdir(parents=True, exist_ok=True)

  if first_tree is not None:
    forest.switch(first_tree)
  forest.save()
  return forest


def _make_args(**kwargs: object) -> argparse.Namespace:
  """Build a Namespace with narrative defaults, overridden by kwargs."""
  values = {
    'metric': None,
    'higher': False,
    'lower': False,
    'context_tail': 10,
    'executions_tail': 10,
    'reflog_tail': 5,
    **kwargs,
  }
  return argparse.Namespace(**values)


class TestReportNarrativeEmptyForest:
  """report narrative on an empty forest exits cleanly."""

  def test_empty_forest_exit_zero(self, tmp_path: Path) -> None:
    """Empty forest produces tree_count=0 and exits 0."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    forest.save()

    ctx = MagicMock()
    ctx.config = config
    ctx.output = MagicMock(spec=Output)
    ctx.output.use_json = True

    args = _make_args()
    payload = _build_narrative(forest, ctx, args)
    assert payload['tree_count'] == 0
    assert payload['trees'] == []
    assert payload['recent_context'] == []

  def test_empty_forest_text_mode(self, tmp_path: Path) -> None:
    """Empty forest text mode shows neutral message."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    forest.save()

    ctx = MagicMock()
    ctx.config = config
    ctx.output = MagicMock(spec=Output)
    ctx.output.use_json = False

    payload = _build_narrative(forest, ctx, _make_args())
    _render_narrative_text(ctx, payload)
    info_calls = [c[0][0] for c in ctx.output.info.call_args_list]
    assert any('No trees' in msg or 'no trees' in msg.lower() for msg in info_calls)


class TestReportNarrativeMultiTreeCounts:
  """Multi-tree narrative reports correct counts and histograms."""

  def test_multi_tree_counts(self, tmp_path: Path) -> None:
    """Histogram sums match experiment counts per tree."""
    forest = _build_forest(
      tmp_path,
      {
        'main': [
          {'id': 'exp-a', 'status': 'completed', 'metrics': {'accuracy': 0.9}},
          {'id': 'exp-b', 'status': 'running'},
          {'id': 'exp-c', 'status': 'failed'},
        ],
        'dev': [
          {'id': 'exp-d', 'status': 'completed', 'metrics': {'accuracy': 0.7}},
          {'id': 'exp-e', 'status': 'pending'},
        ],
      },
    )

    ctx = MagicMock()
    ctx.config = forest.store.config
    ctx.output = MagicMock(spec=Output)
    ctx.output.use_json = True

    payload = _build_narrative(forest, ctx, _make_args())
    assert payload['tree_count'] == 2

    for tree_info in payload['trees']:
      histogram_sum = sum(tree_info['status_counts'].values())
      assert histogram_sum == tree_info['experiment_count']

    main_tree = next(t for t in payload['trees'] if t['name'] == 'main')
    assert main_tree['experiment_count'] == 3
    assert main_tree['status_counts']['completed'] == 1
    assert main_tree['status_counts']['running'] == 1
    assert main_tree['status_counts']['failed'] == 1

    dev_tree = next(t for t in payload['trees'] if t['name'] == 'dev')
    assert dev_tree['experiment_count'] == 2


class TestReportNarrativeBestMetric:
  """--metric flag selects best experiment per tree."""

  def test_best_metric_matches_winner(self, tmp_path: Path) -> None:
    """With --metric, best id matches the seeded winner."""
    forest = _build_forest(
      tmp_path,
      {
        'main': [
          {'id': 'exp-lo', 'status': 'completed', 'metrics': {'accuracy': 0.5}},
          {'id': 'exp-hi', 'status': 'completed', 'metrics': {'accuracy': 0.95}},
        ],
      },
    )

    ctx = MagicMock()
    ctx.config = forest.store.config
    ctx.output = MagicMock(spec=Output)
    ctx.output.use_json = True

    args = _make_args(metric='accuracy')
    payload = _build_narrative(forest, ctx, args)

    main_tree = payload['trees'][0]
    assert main_tree['best'] is not None
    assert main_tree['best']['id'] == 'exp-hi'
    assert main_tree['best']['metric'] == 'accuracy'
    assert main_tree['best']['value'] == 0.95

  def test_no_metric_omits_best(self, tmp_path: Path) -> None:
    """Without --metric, best section is absent from tree payload."""
    forest = _build_forest(
      tmp_path,
      {
        'main': [
          {'id': 'exp-a', 'status': 'completed', 'metrics': {'accuracy': 0.9}},
        ],
      },
    )

    ctx = MagicMock()
    ctx.config = forest.store.config
    ctx.output = MagicMock(spec=Output)
    ctx.output.use_json = True

    args = _make_args(metric=None)
    payload = _build_narrative(forest, ctx, args)
    assert 'best' not in payload['trees'][0]

  def test_best_metric_lower_is_better(self, tmp_path: Path) -> None:
    """--lower flag inverts best metric direction."""
    forest = _build_forest(
      tmp_path,
      {
        'main': [
          {'id': 'exp-lo', 'status': 'completed', 'metrics': {'loss': 0.1}},
          {'id': 'exp-hi', 'status': 'completed', 'metrics': {'loss': 0.9}},
        ],
      },
    )

    ctx = MagicMock()
    ctx.config = forest.store.config
    ctx.output = MagicMock(spec=Output)
    ctx.output.use_json = True

    args = _make_args(metric='loss', lower=True)
    payload = _build_narrative(forest, ctx, args)
    assert payload['trees'][0]['best']['id'] == 'exp-lo'


class TestReportNarrativeContextTail:
  """--context-tail limits recent context entries."""

  def test_context_tail_limits(self, tmp_path: Path) -> None:
    """--context-tail 2 returns at most 2 entries."""
    forest = _build_forest(
      tmp_path,
      {
        'main': [
          {
            'id': 'exp-a',
            'status': 'completed',
            'metrics': {},
            'context_entries': ['first', 'second', 'third', 'fourth'],
          },
        ],
      },
    )

    tree = forest.list_trees()[0]
    tree_nodes = [(tree.name, tree.query().all())]
    result = _collect_recent_context(tree_nodes, 2)
    assert len(result) == 2

  def test_context_sorted_by_timestamp_desc(self, tmp_path: Path) -> None:
    """Recent context entries are sorted by timestamp descending."""
    forest = _build_forest(
      tmp_path,
      {
        'main': [
          {
            'id': 'exp-a',
            'status': 'pending',
            'context_entries': ['early', 'later'],
          },
        ],
      },
    )

    tree = forest.list_trees()[0]
    tree_nodes = [(tree.name, tree.query().all())]
    result = _collect_recent_context(tree_nodes, 10)
    assert len(result) == 2
    assert result[0]['reason'] == 'later'
    assert result[1]['reason'] == 'early'

  def test_context_includes_experiment_id(self, tmp_path: Path) -> None:
    """Each context entry includes experiment_id attribution."""
    forest = _build_forest(
      tmp_path,
      {
        'main': [
          {'id': 'exp-ctx', 'status': 'pending', 'context_entries': ['test entry']},
        ],
      },
    )

    tree = forest.list_trees()[0]
    tree_nodes = [(tree.name, tree.query().all())]
    result = _collect_recent_context(tree_nodes, 10)
    assert result[0]['experiment_id'] == 'exp-ctx'

  def test_context_tail_zero(self, tmp_path: Path) -> None:
    """--context-tail 0 returns empty list."""
    forest = _build_forest(
      tmp_path,
      {
        'main': [
          {'id': 'exp-a', 'status': 'pending', 'context_entries': ['entry']},
        ],
      },
    )

    tree = forest.list_trees()[0]
    tree_nodes = [(tree.name, tree.query().all())]
    result = _collect_recent_context(tree_nodes, 0)
    assert result == []

  def test_context_empty_when_no_entries(self, tmp_path: Path) -> None:
    """No context entries yields empty list."""
    forest = _build_forest(
      tmp_path,
      {
        'main': [
          {'id': 'exp-a', 'status': 'pending'},
        ],
      },
    )

    tree = forest.list_trees()[0]
    tree_nodes = [(tree.name, tree.query().all())]
    result = _collect_recent_context(tree_nodes, 10)
    assert result == []


class TestReportNarrativeExecutionsTail:
  """--executions-tail reads from executions.jsonl."""

  def test_executions_tail(self, tmp_path: Path) -> None:
    """Tail N from executions.jsonl returns last N records."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    for i in range(5):
      append_jsonl(
        config.executions_path,
        {
          'timestamp': utc_now_iso(),
          'command': f'cmd-{i}',
          'args': [],
          'duration_ms': 0.0,
          'exit_code': 0,
          'experiment': None,
          'project': None,
          'extra': {},
          'context': None,
        },
      )

    ctx = MagicMock()
    ctx.config = config

    result = _collect_recent_executions(ctx, 3)
    assert len(result) == 3
    assert result[-1]['command'] == 'cmd-4'
    assert result[0]['command'] == 'cmd-2'

  def test_executions_empty_when_missing(self, tmp_path: Path) -> None:
    """Missing executions.jsonl returns empty list."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)

    ctx = MagicMock()
    ctx.config = config

    result = _collect_recent_executions(ctx, 10)
    assert result == []

  def test_execution_record_subset_fields(self) -> None:
    """_execution_record_subset extracts the right fields."""
    rec = ExecutionRecord(
      timestamp='2026-01-01T00:00:00Z',
      command='optimize train',
      args=['--max-epochs', '5'],
      duration_ms=1234.5,
      exit_code=0,
      stdout='output',
      stderr=None,
      experiment='exp-123',
      project='proj',
      extra={'key': 'val'},
      context='test run',
    )
    subset = _execution_record_subset(rec)
    assert subset['timestamp'] == '2026-01-01T00:00:00Z'
    assert subset['command'] == 'optimize train'
    assert subset['experiment'] == 'exp-123'
    assert subset['context'] == 'test run'
    assert subset['exit_code'] == 0
    assert 'stdout' not in subset
    assert 'args' not in subset


class TestReportNarrativeReflogTail:
  """--reflog-tail reads from store reflog.jsonl."""

  def test_reflog_tail(self, tmp_path: Path) -> None:
    """Tail N from reflog.jsonl returns last N entries."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    reflog_path = config.store_path / 'reflog.jsonl'
    for i in range(4):
      append_jsonl(
        reflog_path,
        {
          'timestamp': utc_now_iso(),
          'operation': f'op-{i}',
          'experiment_id': f'exp-{i}',
          'old_epoch': None,
          'new_epoch': i,
        },
      )

    ctx = MagicMock()
    ctx.config = config

    result = _collect_reflog_tail(ctx, 2)
    assert len(result) == 2
    assert result[-1]['operation'] == 'op-3'
    assert result[0]['operation'] == 'op-2'

  def test_reflog_empty_when_missing(self, tmp_path: Path) -> None:
    """Missing reflog returns empty list."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)

    ctx = MagicMock()
    ctx.config = config

    result = _collect_reflog_tail(ctx, 5)
    assert result == []

  def test_reflog_never_fails(self, tmp_path: Path) -> None:
    """Corrupt reflog lines are skipped, never raising."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    reflog_path = config.store_path / 'reflog.jsonl'
    reflog_path.write_text('not-json\n{"operation":"ok"}\n')

    ctx = MagicMock()
    ctx.config = config

    result = _collect_reflog_tail(ctx, 10)
    assert len(result) == 1
    assert result[0]['operation'] == 'ok'


class TestReportNarrativeContextExempt:
  """report narrative is context-exempt (read-only)."""

  def test_context_exempt_via_run_cli_no_context(self, tmp_path: Path) -> None:
    """run_cli_no_context succeeds for report narrative."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    forest.save()

    envelope = run_cli_no_context(tmp_path, ['report', 'narrative'])
    assert envelope['ok'] is True
    assert 'tree_count' in envelope['result']


class TestReportNarrativeJsonStableKeys:
  """JSON output has stable top-level keys regardless of optional sections."""

  def test_json_keys_present(self, tmp_path: Path) -> None:
    """Required top-level keys are always present."""
    forest = _build_forest(
      tmp_path,
      {
        'main': [
          {'id': 'exp-a', 'status': 'completed', 'metrics': {'accuracy': 0.9}},
        ],
      },
    )

    ctx = MagicMock()
    ctx.config = forest.store.config
    ctx.output = MagicMock(spec=Output)
    ctx.output.use_json = True

    payload = _build_narrative(forest, ctx, _make_args())
    required_keys = {'tree_count', 'trees', 'recent_context', 'recent_executions', 'reflog_tail'}
    assert required_keys <= set(payload.keys())

  def test_json_keys_with_metric(self, tmp_path: Path) -> None:
    """Keys are stable with --metric (adds best to tree entries)."""
    forest = _build_forest(
      tmp_path,
      {
        'main': [
          {'id': 'exp-a', 'status': 'completed', 'metrics': {'accuracy': 0.9}},
        ],
      },
    )

    ctx = MagicMock()
    ctx.config = forest.store.config
    ctx.output = MagicMock(spec=Output)
    ctx.output.use_json = True

    payload = _build_narrative(forest, ctx, _make_args(metric='accuracy'))
    required_keys = {'tree_count', 'trees', 'recent_context', 'recent_executions', 'reflog_tail'}
    assert required_keys <= set(payload.keys())
    assert 'best' in payload['trees'][0]

  def test_json_keys_empty_forest(self, tmp_path: Path) -> None:
    """Keys are stable even with empty forest."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    forest.save()

    ctx = MagicMock()
    ctx.config = config
    ctx.output = MagicMock(spec=Output)
    ctx.output.use_json = True

    payload = _build_narrative(forest, ctx, _make_args())
    required_keys = {'tree_count', 'trees', 'recent_context', 'recent_executions', 'reflog_tail'}
    assert required_keys <= set(payload.keys())


class TestTreeSummary:
  """Unit tests for _tree_summary helper."""

  def test_tree_summary_no_metric(self, tmp_path: Path) -> None:
    """Without metric, tree summary has no 'best' key."""
    forest = _build_forest(
      tmp_path,
      {
        'main': [{'id': 'e1', 'status': 'completed', 'metrics': {'x': 1}}],
      },
    )
    tree = forest.list_trees()[0]
    result = _tree_summary(tree, None, higher_is_better=True)
    assert result['name'] == 'main'
    assert result['experiment_count'] == 1
    assert 'best' not in result

  def test_tree_summary_with_metric(self, tmp_path: Path) -> None:
    """With metric, tree summary includes 'best' key."""
    forest = _build_forest(
      tmp_path,
      {
        'main': [
          {'id': 'e1', 'status': 'completed', 'metrics': {'x': 1}},
          {'id': 'e2', 'status': 'completed', 'metrics': {'x': 5}},
        ],
      },
    )
    tree = forest.list_trees()[0]
    result = _tree_summary(tree, 'x', higher_is_better=True)
    assert result['best']['id'] == 'e2'
    assert result['best']['value'] == 5

  def test_tree_summary_no_matching_metric(self, tmp_path: Path) -> None:
    """When no experiment has the metric, best is None."""
    forest = _build_forest(
      tmp_path,
      {
        'main': [{'id': 'e1', 'status': 'completed', 'metrics': {}}],
      },
    )
    tree = forest.list_trees()[0]
    result = _tree_summary(tree, 'nonexistent', higher_is_better=True)
    assert result['best'] is None


class TestRenderNarrativeText:
  """Tests for text mode rendering."""

  def test_text_renders_tree_info(self, tmp_path: Path) -> None:
    """Text output includes tree name and experiment count."""
    forest = _build_forest(
      tmp_path,
      {
        'main': [{'id': 'e1', 'status': 'completed', 'metrics': {}}],
      },
    )

    ctx = MagicMock()
    ctx.config = forest.store.config
    ctx.output = MagicMock(spec=Output)
    ctx.output.use_json = False

    payload = _build_narrative(forest, ctx, _make_args())
    _render_narrative_text(ctx, payload)
    info_calls = [c[0][0] for c in ctx.output.info.call_args_list]
    assert any('main' in msg for msg in info_calls)
    assert any('1 experiment' in msg for msg in info_calls)


class TestNarrativeForwardIntegration:
  """Integration test for ReportNarrative.forward via CLI path."""

  def test_forward_json_mode(self, tmp_path: Path) -> None:
    """ReportNarrative.forward emits result with correct schema."""
    forest = _build_forest(
      tmp_path,
      {
        'alpha': [
          {'id': 'exp-1', 'status': 'completed', 'metrics': {'f1': 0.8}},
        ],
        'beta': [
          {'id': 'exp-2', 'status': 'running'},
        ],
      },
    )

    ctx = MagicMock()
    ctx.config = forest.store.config
    ctx.output = MagicMock(spec=Output)
    ctx.output.use_json = True

    cmd = ReportNarrative()
    args = _make_args()

    with patch('autopilot.cli.commands.report.narrative.load_forest', return_value=forest):
      cmd.forward(ctx, args)

    ctx.output.result.assert_called_once()
    payload = ctx.output.result.call_args[0][0]
    assert payload['tree_count'] == 2
    assert len(payload['trees']) == 2

  def test_forward_text_mode(self, tmp_path: Path) -> None:
    """ReportNarrative.forward renders text when not JSON."""
    forest = _build_forest(
      tmp_path,
      {
        'main': [
          {'id': 'exp-a', 'status': 'pending'},
        ],
      },
    )

    ctx = MagicMock()
    ctx.config = forest.store.config
    ctx.output = MagicMock(spec=Output)
    ctx.output.use_json = False

    cmd = ReportNarrative()
    args = _make_args()

    with patch('autopilot.cli.commands.report.narrative.load_forest', return_value=forest):
      cmd.forward(ctx, args)

    ctx.output.result.assert_called_once()
    assert ctx.output.info.call_count > 0


class TestCrossTreeContextCollection:
  """Context entries from multiple trees are merged correctly."""

  def test_cross_tree_context_merge(self, tmp_path: Path) -> None:
    """Context from both trees appears in recent_context."""
    forest = _build_forest(
      tmp_path,
      {
        'tree-a': [
          {'id': 'ea', 'status': 'pending', 'context_entries': ['from tree-a']},
        ],
        'tree-b': [
          {'id': 'eb', 'status': 'pending', 'context_entries': ['from tree-b']},
        ],
      },
    )

    tree_nodes = [(t.name, t.query().all()) for t in forest.list_trees()]

    result = _collect_recent_context(tree_nodes, 10)
    reasons = {r['reason'] for r in result}
    assert 'from tree-a' in reasons
    assert 'from tree-b' in reasons
    tree_names = {r['tree'] for r in result}
    assert 'tree-a' in tree_names
    assert 'tree-b' in tree_names
