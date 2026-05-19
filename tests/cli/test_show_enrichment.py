"""Tests for experiment show enrichment (plan 03).

Covers P0#3 (show missing lineage + fingerprint), P1#20 (fingerprint parity
on show vs query), P3#30 (fail JSON error null), and P3#31 (metrics_trusted
flag for invalidated experiments).
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.context import build_context
from autopilot.cli.main import build_parser
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_no_context
import contextlib
import io
import json
import pytest


@pytest.fixture
def ws(tmp_path: Path) -> Path:
  """Workspace with store directory ready."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  return ws


@pytest.fixture
def forest(ws: Path) -> FileForest:
  """FileForest with an active tree 'main'."""
  cfg = AutoPilotConfig(workspace=ws)
  cfg.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(cfg)
  forest = FileForest(store)
  forest.create_tree('main')
  forest.switch('main')
  forest.save()
  return forest


def _add_experiment(
  forest: FileForest,
  eid: str,
  *,
  parent_node: Node | None = None,
  baseline_node: Node | None = None,
  status: str = 'pending',
  metrics: dict | None = None,
  dataset_fingerprint: dict | None = None,
) -> Node:
  """Helper to add an experiment node to the active tree."""
  exp = Experiment(experiment_id=eid, hypothesis=f'{eid} hypothesis')
  if dataset_fingerprint is not None:
    exp.dataset_meta['dataset_fingerprint'] = dataset_fingerprint
  if status in {'running', 'completed', 'failed'}:
    exp.start()
  if status == 'completed':
    exp.complete(metrics=metrics)
  elif status == 'failed':
    exp.fail(error='test failure')
  elif status == 'running' and metrics:
    exp.metrics = metrics
  node = Node(experiment=exp, parent=parent_node, baseline=baseline_node)
  tree = forest.active
  assert tree is not None
  tree.add(node)
  forest.save()
  return node


class TestShowEnrichment:
  """Tests for experiment show JSON enrichment fields."""

  def test_show_includes_parent(self, ws: Path, forest: FileForest) -> None:
    """P0#3: show includes parent experiment id."""
    parent = _add_experiment(forest, 'parent-exp', status='completed', metrics={'acc': 0.8})
    _add_experiment(
      forest, 'child-exp', parent_node=parent, status='completed', metrics={'acc': 0.9}
    )

    result = run_cli_no_context(ws, ['experiment', 'show', 'child-exp'])
    assert result['result']['parent'] == 'parent-exp'

  def test_show_includes_baseline(self, ws: Path, forest: FileForest) -> None:
    """P0#3: show includes baseline experiment id."""
    baseline = _add_experiment(forest, 'base-exp', status='completed', metrics={'acc': 0.7})
    _add_experiment(
      forest, 'derived-exp', baseline_node=baseline, status='completed', metrics={'acc': 0.9}
    )

    result = run_cli_no_context(ws, ['experiment', 'show', 'derived-exp'])
    assert result['result']['baseline'] == 'base-exp'

  def test_show_includes_dataset_fingerprint(self, ws: Path, forest: FileForest) -> None:
    """P1#20: show includes dataset_fingerprint matching query shape."""
    fp_dict = {
      'paths': ['/data/train.csv'],
      'hashes': ['abc123'],
      'bundle_hash': 'bundle_abc',
      'timestamp': '2026-01-01T00:00:00Z',
    }
    _add_experiment(
      forest,
      'fp-exp',
      status='completed',
      metrics={'acc': 0.9},
      dataset_fingerprint=fp_dict,
    )

    result = run_cli_no_context(ws, ['experiment', 'show', 'fp-exp'])
    assert result['result']['dataset_fingerprint'] == fp_dict

  def test_show_root_experiment_parent_null(self, ws: Path, forest: FileForest) -> None:
    """Root experiment (no parent) has parent: null."""
    _add_experiment(forest, 'root-exp', status='completed', metrics={'acc': 0.5})

    result = run_cli_no_context(ws, ['experiment', 'show', 'root-exp'])
    assert result['result']['parent'] is None

  def test_show_baseline_null_when_none(self, ws: Path, forest: FileForest) -> None:
    """Experiment with no baseline link has baseline: null."""
    _add_experiment(forest, 'no-base', status='completed', metrics={'acc': 0.6})

    result = run_cli_no_context(ws, ['experiment', 'show', 'no-base'])
    assert result['result']['baseline'] is None

  def test_show_fingerprint_null_when_no_dataset(self, ws: Path, forest: FileForest) -> None:
    """Experiment with no dataset_fingerprint has null value."""
    _add_experiment(forest, 'no-fp', status='completed', metrics={'acc': 0.7})

    result = run_cli_no_context(ws, ['experiment', 'show', 'no-fp'])
    assert result['result']['dataset_fingerprint'] is None

  def test_show_completed_metrics_trusted_true(self, ws: Path, forest: FileForest) -> None:
    """P3#31: completed experiment has metrics_trusted=True in show and query."""
    _add_experiment(forest, 'trusted', status='completed', metrics={'acc': 0.95})

    show_result = run_cli_no_context(ws, ['experiment', 'show', 'trusted'])
    assert show_result['result']['metrics_trusted'] is True

    query_result = run_cli_no_context(ws, ['query'])
    experiments = query_result['result']['experiments']
    matched = [e for e in experiments if e['id'] == 'trusted']
    assert len(matched) == 1
    assert matched[0]['metrics_trusted'] is True

  def test_invalidated_metrics_trusted_false(self, ws: Path, forest: FileForest) -> None:
    """P3#31: invalidated experiment has metrics_trusted=False in show and query."""
    _add_experiment(forest, 'inv-exp', status='completed', metrics={'acc': 0.9})

    run_cli(ws, ['experiment', 'invalidate', 'inv-exp', '--reason', 'data leak'])

    show_result = run_cli_no_context(ws, ['experiment', 'show', 'inv-exp'])
    assert show_result['result']['metrics_trusted'] is False
    assert show_result['result']['status'] == 'invalidated'

    query_result = run_cli_no_context(ws, ['query', '--include-invalidated'])
    experiments = query_result['result']['experiments']
    matched = [e for e in experiments if e['id'] == 'inv-exp']
    assert len(matched) == 1
    assert matched[0]['metrics_trusted'] is False

  def test_show_json_schema_all_fields(self, ws: Path, forest: FileForest) -> None:
    """All enriched fields present simultaneously with expected types."""
    fp_dict = {
      'paths': ['/data/test.csv'],
      'hashes': ['def456'],
      'bundle_hash': 'bundle_def',
      'timestamp': '2026-01-01T00:00:00Z',
    }
    parent = _add_experiment(forest, 'schema-parent', status='completed', metrics={'acc': 0.8})
    baseline = _add_experiment(forest, 'schema-base', status='completed', metrics={'acc': 0.7})
    _add_experiment(
      forest,
      'schema-exp',
      parent_node=parent,
      baseline_node=baseline,
      status='completed',
      metrics={'acc': 0.9},
      dataset_fingerprint=fp_dict,
    )

    payload = run_cli_no_context(ws, ['experiment', 'show', 'schema-exp'])['result']

    assert payload['parent'] == 'schema-parent'
    assert payload['baseline'] == 'schema-base'
    assert payload['dataset_fingerprint'] == fp_dict
    assert payload['metrics_trusted'] is True
    assert isinstance(payload['parent'], str)
    assert isinstance(payload['baseline'], str)
    assert isinstance(payload['dataset_fingerprint'], dict)
    assert isinstance(payload['metrics_trusted'], bool)
    assert payload['tree'] == 'main'
    assert payload['deployed_as'] is None

  def test_show_exit_code_missing_experiment(self, ws: Path, forest: FileForest) -> None:
    """Non-existent experiment id results in non-zero exit and error envelope."""
    parser = build_parser()
    argv = ['experiment', 'show', 'does-not-exist', '--workspace', str(ws), '--json']
    parsed = parser.parse_args(argv)
    ctx = build_context(parsed)

    buf = io.StringIO()
    with pytest.raises(SystemExit), contextlib.redirect_stdout(buf):
      parsed.handler(ctx, parsed)

    output = buf.getvalue().strip()
    envelope = json.loads(output)
    assert envelope['ok'] is False
    assert 'not found in any tree' in envelope['error']


class TestFailErrorField:
  """Tests for experiment fail JSON error resolution (P3#30)."""

  def test_fail_error_field_from_context(self, ws: Path, forest: FileForest) -> None:
    """P3#30: fail without --error uses --context as error fallback."""
    _add_experiment(forest, 'fail-ctx', status='running')

    parser = build_parser()
    argv = [
      'experiment',
      'fail',
      'fail-ctx',
      '--workspace',
      str(ws),
      '--json',
      '--context',
      'data was corrupted',
    ]
    parsed = parser.parse_args(argv)
    ctx = build_context(parsed)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      parsed.handler(ctx, parsed)
    result = json.loads(buf.getvalue().strip())
    assert result['result']['error'] == 'data was corrupted'

  def test_fail_error_field_with_error_flag(self, ws: Path, forest: FileForest) -> None:
    """Fail with --error populates error field directly."""
    _add_experiment(forest, 'fail-reason', status='running')

    result = run_cli(
      ws,
      ['experiment', 'fail', 'fail-reason', '--error', 'OOM on GPU 0'],
    )
    assert result['result']['error'] == 'OOM on GPU 0'
