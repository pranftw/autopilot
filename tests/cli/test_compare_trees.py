"""Tests for ``report compare --all-trees`` cross-tree comparison (Plan 12).

Validates the all-trees mode that picks the best experiment per tree
under a specified metric and recommends an overall winner via
MetricsComparator.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.context import build_context
from autopilot.cli.main import build_parser
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli_no_context
from typing import Any
import contextlib
import io
import json
import pytest


def _run_cli_expect_fail(workspace: Path, argv: list[str]) -> dict[str, Any]:
  """Run CLI expecting a SystemExit and return the JSON error envelope."""
  parser = build_parser()
  full_argv = [*list(argv), '--workspace', str(workspace), '--json']
  parsed = parser.parse_args(full_argv)
  ctx = build_context(parsed)

  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
    parsed.handler(ctx, parsed)

  output = buf.getvalue().strip()
  if output:
    return json.loads(output)
  return {}


@pytest.fixture
def two_tree_forest(cli_store: FileStore) -> FileForest:
  """Forest with 'alpha' (accuracy=0.9) and 'beta' (accuracy=0.85)."""
  forest = FileForest(cli_store)

  tree_a = forest.create_tree('alpha')
  exp_a = Experiment(experiment_id='exp-a', hypothesis='alpha approach')
  exp_a.start()
  exp_a.complete(metrics={'val_accuracy': 0.9, 'val_loss': 0.3})
  tree_a.add(Node(experiment=exp_a))

  tree_b = forest.create_tree('beta')
  exp_b = Experiment(experiment_id='exp-b', hypothesis='beta approach')
  exp_b.start()
  exp_b.complete(metrics={'val_accuracy': 0.85, 'val_loss': 0.4})
  tree_b.add(Node(experiment=exp_b))

  forest.switch('alpha')
  forest.save()
  return forest


@pytest.fixture
def three_tree_forest(cli_store: FileStore) -> FileForest:
  """Forest with three trees, one empty, for edge-case testing."""
  forest = FileForest(cli_store)

  tree_a = forest.create_tree('alpha')
  exp_a = Experiment(experiment_id='exp-a', hypothesis='alpha')
  exp_a.start()
  exp_a.complete(metrics={'val_accuracy': 0.9})
  tree_a.add(Node(experiment=exp_a))

  tree_b = forest.create_tree('beta')
  exp_b = Experiment(experiment_id='exp-b', hypothesis='beta')
  exp_b.start()
  exp_b.complete(metrics={'val_accuracy': 0.7})
  tree_b.add(Node(experiment=exp_b))

  forest.create_tree('gamma')

  forest.switch('alpha')
  forest.save()
  return forest


def test_report_compare_all_trees_multi_tree(
  cli_workspace: Path,
  two_tree_forest: FileForest,
) -> None:
  """Multi-tree comparison picks the tree with the best metric value."""
  envelope = run_cli_no_context(
    cli_workspace,
    ['report', 'compare', '--all-trees', '--metric', 'accuracy'],
  )
  result = envelope['result']

  assert result['metric'] == 'val_accuracy'
  assert result['higher_is_better'] is True
  assert len(result['trees']) == 2

  alpha_entry = next(t for t in result['trees'] if t['tree'] == 'alpha')
  beta_entry = next(t for t in result['trees'] if t['tree'] == 'beta')
  assert alpha_entry['best'] == 'exp-a'
  assert alpha_entry['metric_value'] == 0.9
  assert beta_entry['best'] == 'exp-b'
  assert beta_entry['metric_value'] == 0.85

  assert result['winner']['tree'] == 'alpha'
  assert result['winner']['id'] == 'exp-a'
  assert result['winner']['metric_value'] == 0.9


def test_report_compare_all_trees_single_tree_degenerates(
  cli_store: FileStore,
  cli_workspace: Path,
) -> None:
  """Single-tree forest: winner is the sole tree's best."""
  forest = FileForest(cli_store)
  tree = forest.create_tree('only')
  exp = Experiment(experiment_id='exp-only', hypothesis='single')
  exp.start()
  exp.complete(metrics={'score': 0.75})
  tree.add(Node(experiment=exp))
  forest.switch('only')
  forest.save()

  envelope = run_cli_no_context(
    cli_workspace,
    ['report', 'compare', '--all-trees', '--metric', 'score'],
  )
  result = envelope['result']

  assert len(result['trees']) == 1
  assert result['winner']['tree'] == 'only'
  assert result['winner']['id'] == 'exp-only'
  assert result['winner']['metric_value'] == 0.75


def test_report_compare_all_trees_empty_tree(
  cli_workspace: Path,
  three_tree_forest: FileForest,
) -> None:
  """Empty tree does not crash; represented with null best."""
  envelope = run_cli_no_context(
    cli_workspace,
    ['report', 'compare', '--all-trees', '--metric', 'accuracy'],
  )
  result = envelope['result']

  gamma_entry = next(t for t in result['trees'] if t['tree'] == 'gamma')
  assert gamma_entry['best'] is None
  assert gamma_entry['metric_value'] is None

  assert result['winner'] is not None
  assert result['winner']['tree'] == 'alpha'


def test_report_compare_all_trees_metric_key_resolution(
  cli_workspace: Path,
  two_tree_forest: FileForest,
) -> None:
  """Bare metric name 'accuracy' resolves to 'val_accuracy' via val-first strategy."""
  envelope = run_cli_no_context(
    cli_workspace,
    ['report', 'compare', '--all-trees', '--metric', 'accuracy'],
  )
  result = envelope['result']

  assert result['metric'] == 'val_accuracy'


def test_report_compare_all_trees_requires_metric_flag(
  cli_workspace: Path,
  two_tree_forest: FileForest,
) -> None:
  """--all-trees without --metric fails with an actionable message containing 'metric'."""
  envelope = _run_cli_expect_fail(
    cli_workspace,
    ['report', 'compare', '--all-trees'],
  )
  assert envelope['ok'] is False
  assert 'metric' in envelope['error'].lower()


def test_report_compare_all_trees_rejects_extra_slugs(
  cli_workspace: Path,
  two_tree_forest: FileForest,
) -> None:
  """--all-trees with positional slugs fails with an actionable message."""
  envelope = _run_cli_expect_fail(
    cli_workspace,
    ['report', 'compare', '--all-trees', '--metric', 'accuracy', 'exp-a'],
  )
  assert envelope['ok'] is False
  assert 'slugs' in envelope['error'].lower()


def test_report_compare_classic_mode_unchanged(
  cli_workspace: Path,
  two_tree_forest: FileForest,
) -> None:
  """Classic mode with two slugs still produces summaries and metric_comparisons."""
  envelope = run_cli_no_context(
    cli_workspace,
    ['report', 'compare', 'exp-a', 'exp-b'],
  )
  result = envelope['result']

  assert 'summaries' in result
  assert 'metric_comparisons' in result
  assert len(result['summaries']) == 2
  assert len(result['metric_comparisons']) == 1


def test_report_compare_all_trees_lower_flag(
  cli_workspace: Path,
  two_tree_forest: FileForest,
) -> None:
  """--lower inverts direction: lower metric value wins."""
  envelope = run_cli_no_context(
    cli_workspace,
    ['report', 'compare', '--all-trees', '--metric', 'loss', '--lower'],
  )
  result = envelope['result']

  assert result['higher_is_better'] is False
  assert result['winner']['tree'] == 'alpha'
  assert result['winner']['metric_value'] == 0.3


def test_report_compare_all_trees_multiple_experiments_per_tree(
  cli_store: FileStore,
  cli_workspace: Path,
) -> None:
  """Per-tree best picks the top experiment when a tree has multiple."""
  forest = FileForest(cli_store)

  tree_a = forest.create_tree('alpha')
  exp_a1 = Experiment(experiment_id='exp-a1', hypothesis='a1')
  exp_a1.start()
  exp_a1.complete(metrics={'score': 0.6})
  tree_a.add(Node(experiment=exp_a1))
  exp_a2 = Experiment(experiment_id='exp-a2', hypothesis='a2')
  exp_a2.start()
  exp_a2.complete(metrics={'score': 0.95})
  tree_a.add(Node(experiment=exp_a2))

  tree_b = forest.create_tree('beta')
  exp_b = Experiment(experiment_id='exp-b', hypothesis='b')
  exp_b.start()
  exp_b.complete(metrics={'score': 0.8})
  tree_b.add(Node(experiment=exp_b))

  forest.switch('alpha')
  forest.save()

  envelope = run_cli_no_context(
    cli_workspace,
    ['report', 'compare', '--all-trees', '--metric', 'score'],
  )
  result = envelope['result']

  alpha_entry = next(t for t in result['trees'] if t['tree'] == 'alpha')
  assert alpha_entry['best'] == 'exp-a2'
  assert alpha_entry['metric_value'] == 0.95

  assert result['winner']['tree'] == 'alpha'
  assert result['winner']['id'] == 'exp-a2'
