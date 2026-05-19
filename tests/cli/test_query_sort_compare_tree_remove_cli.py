"""Tests for query & compare enhancements (plan 11).

Covers GAP-001 (--sort), GAP-002 (--best prefix resolution),
GAP-004 (experiment list alias), GAP-005 (tree remove),
GAP-008/BUG-019 (cross-tree compare), GAP-020 (--all-trees tree attribution),
GAP-018 (compare verdict), TS-prefix-mismatch (metric prefix normalization).
"""

from autopilot.cli.commands.experiment.verdict import (
  compute_verdict,
  normalize_metric_prefixes,
)
from autopilot.cli.commands.query import resolve_metric_name
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli, seed_tree_with_experiments
import pytest

# -- GAP-001: --sort by metric (descending) ----------------------------------


def test_query_sort_by_metric_descending(cli_forest, cli_workspace: Path) -> None:
  """--sort accuracy returns rows ordered by that metric descending (highest first)."""
  seed_tree_with_experiments(
    cli_forest,
    'main',
    [
      {'id': 'exp-low', 'hypothesis': 'low', 'status': 'completed', 'metrics': {'accuracy': 0.3}},
      {'id': 'exp-mid', 'hypothesis': 'mid', 'status': 'completed', 'metrics': {'accuracy': 0.6}},
      {
        'id': 'exp-high',
        'hypothesis': 'high',
        'status': 'completed',
        'metrics': {'accuracy': 0.9},
      },
    ],
  )

  result = run_cli(cli_workspace, ['query', '--sort', 'accuracy'])
  experiments = result['result']['experiments']
  assert len(experiments) == 3
  assert experiments[0]['id'] == 'exp-high'
  assert experiments[1]['id'] == 'exp-mid'
  assert experiments[2]['id'] == 'exp-low'


# -- GAP-002: --best prefix resolution (val-first) ---------------------------


def test_best_resolves_val_prefix_first(cli_forest, cli_workspace: Path) -> None:
  """--best accuracy selects by val_accuracy when that key exists."""
  seed_tree_with_experiments(
    cli_forest,
    'main',
    [
      {
        'id': 'exp-a',
        'hypothesis': 'a',
        'status': 'completed',
        'metrics': {'val_accuracy': 0.8, 'train_accuracy': 0.95},
      },
      {
        'id': 'exp-b',
        'hypothesis': 'b',
        'status': 'completed',
        'metrics': {'val_accuracy': 0.9, 'train_accuracy': 0.7},
      },
    ],
  )

  result = run_cli(cli_workspace, ['query', '--best', 'accuracy'])
  assert result['result']['best']['id'] == 'exp-b'


def test_best_falls_back_to_train_prefix(cli_forest, cli_workspace: Path) -> None:
  """--best accuracy falls back to train_accuracy when val_accuracy is absent."""
  seed_tree_with_experiments(
    cli_forest,
    'main',
    [
      {
        'id': 'exp-a',
        'hypothesis': 'a',
        'status': 'completed',
        'metrics': {'train_accuracy': 0.7},
      },
      {
        'id': 'exp-b',
        'hypothesis': 'b',
        'status': 'completed',
        'metrics': {'train_accuracy': 0.95},
      },
    ],
  )

  result = run_cli(cli_workspace, ['query', '--best', 'accuracy'])
  assert result['result']['best']['id'] == 'exp-b'


def test_best_uses_bare_name_when_no_prefix_match(cli_forest, cli_workspace: Path) -> None:
  """--best accuracy uses bare 'accuracy' key when no prefixed variant exists."""
  seed_tree_with_experiments(
    cli_forest,
    'main',
    [
      {'id': 'exp-a', 'hypothesis': 'a', 'status': 'completed', 'metrics': {'accuracy': 0.4}},
      {'id': 'exp-b', 'hypothesis': 'b', 'status': 'completed', 'metrics': {'accuracy': 0.8}},
    ],
  )

  result = run_cli(cli_workspace, ['query', '--best', 'accuracy'])
  assert result['result']['best']['id'] == 'exp-b'


def test_resolve_metric_name_unit() -> None:
  """Unit test for resolve_metric_name helper."""
  exp1 = Experiment(experiment_id='e1')
  exp1.start()
  exp1.complete(metrics={'val_accuracy': 0.9, 'train_loss': 0.1})
  node1 = Node(experiment=exp1)

  exp2 = Experiment(experiment_id='e2')
  exp2.start()
  exp2.complete(metrics={'accuracy': 0.5})
  node2 = Node(experiment=exp2)

  assert resolve_metric_name([node1], 'accuracy') == 'val_accuracy'
  assert resolve_metric_name([node1], 'loss') == 'train_loss'
  assert resolve_metric_name([node2], 'accuracy') == 'accuracy'
  assert resolve_metric_name([], 'whatever') == 'whatever'


# -- GAP-005: tree remove persistence ----------------------------------------


def test_tree_remove_persists_in_forest(cli_forest, cli_workspace: Path) -> None:
  """tree remove removes tree from forest and persists the change."""
  cli_forest.create_tree('ephemeral', description='temp tree')
  cli_forest.create_tree('keeper', description='keep this')
  cli_forest.switch('keeper')

  result = run_cli(cli_workspace, ['tree', 'remove', 'ephemeral'])
  assert result['result']['ok'] is True
  assert result['result']['removed'] == 'ephemeral'

  result = run_cli(cli_workspace, ['tree', 'list'])
  tree_names = [t['name'] for t in result['result']['trees']]
  assert 'ephemeral' not in tree_names
  assert 'keeper' in tree_names


def test_tree_remove_nonexistent_fails(cli_forest, cli_workspace: Path) -> None:
  """tree remove on nonexistent tree fails with error."""
  cli_forest.create_tree('only-tree')
  cli_forest.switch('only-tree')

  with pytest.raises(SystemExit):
    run_cli(cli_workspace, ['tree', 'remove', 'ghost'])


# -- GAP-008/BUG-019: cross-tree experiment compare ---------------------------


def test_compare_finds_experiments_across_trees(cli_forest, cli_workspace: Path) -> None:
  """Compare locates experiments attached to different trees."""
  seed_tree_with_experiments(
    cli_forest,
    'tree-a',
    [
      {
        'id': 'exp-alpha',
        'hypothesis': 'alpha',
        'status': 'completed',
        'metrics': {'accuracy': 0.7},
      },
    ],
  )
  seed_tree_with_experiments(
    cli_forest,
    'tree-b',
    [
      {
        'id': 'exp-beta',
        'hypothesis': 'beta',
        'status': 'completed',
        'metrics': {'accuracy': 0.9},
      },
    ],
  )
  cli_forest.switch('tree-a')
  cli_forest.save()

  result = run_cli(cli_workspace, ['experiment', 'compare', 'exp-alpha', 'exp-beta'])
  comparison = result['result']
  assert comparison['a'] == 'exp-alpha'
  assert comparison['b'] == 'exp-beta'
  deltas_by_metric = {d['metric']: d for d in comparison['deltas']}
  assert deltas_by_metric['accuracy']['delta'] == pytest.approx(0.2)


def test_compare_rejects_missing_experiment(cli_forest, cli_workspace: Path) -> None:
  """Non-existent id yields error/fail."""
  seed_tree_with_experiments(
    cli_forest,
    'main',
    [
      {'id': 'exp-real', 'hypothesis': 'real', 'status': 'completed', 'metrics': {'acc': 0.5}},
    ],
  )

  with pytest.raises(SystemExit):
    run_cli(cli_workspace, ['experiment', 'compare', 'exp-real', 'exp-ghost'])


# -- GAP-020: tree attribution in --all-trees output --------------------------


def test_all_trees_duplicate_id_tree_field_matches_forest_dedup(
  cli_forest, cli_workspace: Path
) -> None:
  """When the same experiment id exists in multiple trees, Forest.query keeps the first tree's node.

  ``tree`` attribution must match that first-wins dedup (not last-wins overwrite).
  """
  seed_tree_with_experiments(
    cli_forest,
    'alpha',
    [
      {
        'id': 'shared-id',
        'hypothesis': 'from alpha',
        'status': 'completed',
        'metrics': {'accuracy': 0.99},
      },
    ],
  )
  seed_tree_with_experiments(
    cli_forest,
    'beta',
    [
      {
        'id': 'shared-id',
        'hypothesis': 'from beta',
        'status': 'completed',
        'metrics': {'accuracy': 0.01},
      },
    ],
  )

  result = run_cli(cli_workspace, ['query', '--all-trees'])
  experiments = result['result']['experiments']
  shared_rows = [e for e in experiments if e['id'] == 'shared-id']
  assert len(shared_rows) == 1
  row = shared_rows[0]
  assert row['metrics']['accuracy'] == pytest.approx(0.99)
  assert row['tree'] == 'alpha'


def test_all_trees_includes_tree_name(cli_forest, cli_workspace: Path) -> None:
  """--all-trees rows include tree name field."""
  seed_tree_with_experiments(
    cli_forest,
    'first',
    [
      {'id': 'exp-1', 'hypothesis': 'h1', 'status': 'completed', 'metrics': {'acc': 0.5}},
    ],
  )
  seed_tree_with_experiments(
    cli_forest,
    'second',
    [
      {'id': 'exp-2', 'hypothesis': 'h2', 'status': 'completed', 'metrics': {'acc': 0.8}},
    ],
  )

  result = run_cli(cli_workspace, ['query', '--all-trees'])
  experiments = result['result']['experiments']
  assert len(experiments) == 2

  tree_names = {e['id']: e['tree'] for e in experiments}
  assert tree_names['exp-1'] == 'first'
  assert tree_names['exp-2'] == 'second'


# -- TS-prefix-mismatch: metric prefix normalization in compare ---------------


def test_compare_normalizes_metric_prefixes(cli_forest, cli_workspace: Path) -> None:
  """Compare with mismatched metric prefixes still emits non-null deltas."""
  seed_tree_with_experiments(
    cli_forest,
    'main',
    [
      {
        'id': 'exp-a',
        'hypothesis': 'a',
        'status': 'completed',
        'metrics': {'val_accuracy': 0.7, 'val_loss': 0.3},
      },
      {
        'id': 'exp-b',
        'hypothesis': 'b',
        'status': 'completed',
        'metrics': {'accuracy': 0.85, 'loss': 0.15},
      },
    ],
  )

  result = run_cli(cli_workspace, ['experiment', 'compare', 'exp-a', 'exp-b'])
  deltas = result['result']['deltas']
  deltas_by_metric = {d['metric']: d for d in deltas}
  assert 'accuracy' in deltas_by_metric
  assert deltas_by_metric['accuracy']['delta'] is not None
  assert deltas_by_metric['accuracy']['delta'] == pytest.approx(0.15)
  assert 'loss' in deltas_by_metric
  assert deltas_by_metric['loss']['delta'] is not None
  assert deltas_by_metric['loss']['delta'] == pytest.approx(-0.15)


def testnormalize_metric_prefixes_unit() -> None:
  """Unit test for normalize_metric_prefixes helper."""
  a = {'val_accuracy': 0.8, 'train_loss': 0.2, 'lr': 0.001}
  b = {'accuracy': 0.9, 'loss': 0.1, 'lr': 0.0005}

  norm_a, norm_b = normalize_metric_prefixes(a, b)

  assert 'lr' in norm_a
  assert 'lr' in norm_b
  assert norm_a['lr'] == 0.001
  assert norm_b['lr'] == 0.0005

  assert 'accuracy' in norm_a
  assert 'accuracy' in norm_b
  assert norm_a['accuracy'] == 0.8
  assert norm_b['accuracy'] == 0.9

  assert 'loss' in norm_a
  assert 'loss' in norm_b
  assert norm_a['loss'] == 0.2
  assert norm_b['loss'] == 0.1


# -- GAP-018: compare verdict -------------------------------------------------


def test_compare_includes_verdict_field(cli_forest, cli_workspace: Path) -> None:
  """Compare JSON includes verdict field with expected values."""
  seed_tree_with_experiments(
    cli_forest,
    'main',
    [
      {
        'id': 'base',
        'hypothesis': 'baseline',
        'status': 'completed',
        'metrics': {'accuracy': 0.5, 'f1': 0.4},
      },
      {
        'id': 'better',
        'hypothesis': 'improved',
        'status': 'completed',
        'metrics': {'accuracy': 0.8, 'f1': 0.7},
      },
    ],
  )

  result = run_cli(cli_workspace, ['experiment', 'compare', 'base', 'better'])
  assert result['result']['verdict'] == 'improved'


def test_compare_detects_metric_regression(cli_forest, cli_workspace: Path) -> None:
  """Compare two experiments where candidate regresses; verdict reflects it."""
  seed_tree_with_experiments(
    cli_forest,
    'main',
    [
      {
        'id': 'good',
        'hypothesis': 'good',
        'status': 'completed',
        'metrics': {'accuracy': 0.9, 'f1': 0.85},
      },
      {
        'id': 'bad',
        'hypothesis': 'bad',
        'status': 'completed',
        'metrics': {'accuracy': 0.6, 'f1': 0.5},
      },
    ],
  )

  result = run_cli(cli_workspace, ['experiment', 'compare', 'good', 'bad'])
  assert result['result']['verdict'] == 'regressed'
  deltas_by_metric = {d['metric']: d for d in result['result']['deltas']}
  assert deltas_by_metric['accuracy']['delta'] == pytest.approx(-0.3)
  assert deltas_by_metric['f1']['delta'] == pytest.approx(-0.35)


def test_compare_inconclusive_verdict(cli_forest, cli_workspace: Path) -> None:
  """Mixed changes (one metric up, one down) produce inconclusive verdict."""
  seed_tree_with_experiments(
    cli_forest,
    'main',
    [
      {
        'id': 'exp-x',
        'hypothesis': 'x',
        'status': 'completed',
        'metrics': {'accuracy': 0.7, 'speed': 200.0},
      },
      {
        'id': 'exp-y',
        'hypothesis': 'y',
        'status': 'completed',
        'metrics': {'accuracy': 0.9, 'speed': 100.0},
      },
    ],
  )

  result = run_cli(cli_workspace, ['experiment', 'compare', 'exp-x', 'exp-y'])
  assert result['result']['verdict'] == 'inconclusive'


def test_compute_verdict_unit() -> None:
  """Unit test for compute_verdict helper with list-shaped deltas."""
  assert (
    compute_verdict(
      [
        {'metric': 'a', 'delta': 0.1, 'type': 'numeric'},
        {'metric': 'b', 'delta': 0.2, 'type': 'numeric'},
      ]
    )
    == 'improved'
  )
  assert (
    compute_verdict(
      [
        {'metric': 'a', 'delta': -0.1, 'type': 'numeric'},
        {'metric': 'b', 'delta': -0.2, 'type': 'numeric'},
      ]
    )
    == 'regressed'
  )
  assert (
    compute_verdict(
      [
        {'metric': 'a', 'delta': 0.1, 'type': 'numeric'},
        {'metric': 'b', 'delta': -0.1, 'type': 'numeric'},
      ]
    )
    == 'inconclusive'
  )
  assert compute_verdict([{'metric': 'a', 'delta': None, 'type': 'missing'}]) == 'inconclusive'
  assert compute_verdict([]) == 'inconclusive'


# -- experiment list alias (GAP-004) ------------------------------------------


def test_experiment_list_returns_experiments(cli_forest, cli_workspace: Path) -> None:
  """experiment list returns same results as query."""
  seed_tree_with_experiments(
    cli_forest,
    'main',
    [
      {'id': 'exp-1', 'hypothesis': 'h1', 'status': 'completed', 'metrics': {'acc': 0.5}},
      {'id': 'exp-2', 'hypothesis': 'h2', 'status': 'running', 'metrics': {}},
    ],
  )

  result = run_cli(cli_workspace, ['experiment', 'list'])
  assert result['result']['count'] == 2
  ids = [e['id'] for e in result['result']['experiments']]
  assert 'exp-1' in ids
  assert 'exp-2' in ids
