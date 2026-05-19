"""Tests for query JSON enrichment: deployed_as, has_notes, --best tree, --deployed advisory."""

from autopilot.ai.forest import FileForest
from pathlib import Path
from tests.cli.conftest import run_cli_no_context

# ---------------------------------------------------------------------------
# 4.1 Row shape and notes
# ---------------------------------------------------------------------------


class TestRowShapeAndNotes:
  """Test deployed_as and has_notes fields on query JSON rows."""

  def test_query_rows_include_deployed_as(
    self, multi_tree_forest: FileForest, cli_workspace: Path
  ) -> None:
    """Deployed label on a node surfaces in query --json experiments list."""
    tree = multi_tree_forest.get_tree('alpha')
    assert tree is not None
    node = tree.get('exp-alpha')
    assert node is not None
    node.deployed_as = 'production'
    multi_tree_forest.save()

    result = run_cli_no_context(cli_workspace, ['query'])
    assert result['ok'] is True
    experiments = result['result']['experiments']
    row = next(r for r in experiments if r['id'] == 'exp-alpha')
    assert row['deployed_as'] == 'production'

  def test_query_deployed_as_null(self, multi_tree_forest: FileForest, cli_workspace: Path) -> None:
    """Undeployed experiment row has deployed_as None in JSON."""
    result = run_cli_no_context(cli_workspace, ['query'])
    assert result['ok'] is True
    experiments = result['result']['experiments']
    row = next(r for r in experiments if r['id'] == 'exp-alpha')
    assert row['deployed_as'] is None

  def test_query_rows_include_has_notes(
    self, multi_tree_forest: FileForest, cli_workspace: Path
  ) -> None:
    """Experiment with notes set yields has_notes True."""
    tree = multi_tree_forest.get_tree('alpha')
    assert tree is not None
    node = tree.get('exp-alpha')
    assert node is not None
    node.experiment.notes = 'some important notes'
    multi_tree_forest.save()

    result = run_cli_no_context(cli_workspace, ['query'])
    assert result['ok'] is True
    experiments = result['result']['experiments']
    row = next(r for r in experiments if r['id'] == 'exp-alpha')
    assert row['has_notes'] is True

  def test_query_has_notes_false(self, multi_tree_forest: FileForest, cli_workspace: Path) -> None:
    """Experiment with notes None yields has_notes False."""
    result = run_cli_no_context(cli_workspace, ['query'])
    assert result['ok'] is True
    experiments = result['result']['experiments']
    row = next(r for r in experiments if r['id'] == 'exp-alpha')
    assert row['has_notes'] is False


# ---------------------------------------------------------------------------
# 4.2 --deployed and cross-tree behavior
# ---------------------------------------------------------------------------


class TestDeployedCrossTree:
  """Test --deployed advisory and cross-tree search."""

  def test_query_deployed_empty_warns(
    self, multi_tree_forest: FileForest, cli_workspace: Path
  ) -> None:
    """Active tree with no deployments emits advisory in JSON messages."""
    result = run_cli_no_context(cli_workspace, ['query', '--deployed'])
    assert result['ok'] is True
    assert result['result']['count'] == 0
    messages = result['messages']
    assert any(
      m['level'] == 'info'
      and m['message'] == 'No deployments in active tree. Use --all-trees to search all trees.'
      for m in messages
    )

  def test_query_deployed_all_trees_finds_cross_tree(
    self, multi_tree_forest: FileForest, cli_workspace: Path
  ) -> None:
    """Cross-tree --deployed --all-trees finds deployment in non-active tree."""
    tree_b = multi_tree_forest.get_tree('beta')
    assert tree_b is not None
    node_b = tree_b.get('exp-beta')
    assert node_b is not None
    node_b.deployed_as = 'staging'
    multi_tree_forest.save()

    result_active = run_cli_no_context(cli_workspace, ['query', '--deployed'])
    assert result_active['ok'] is True
    assert result_active['result']['count'] == 0
    messages = result_active['messages']
    assert any(
      m['message'] == 'No deployments in active tree. Use --all-trees to search all trees.'
      for m in messages
    )

    result_all = run_cli_no_context(cli_workspace, ['query', '--deployed', '--all-trees'])
    assert result_all['ok'] is True
    assert result_all['result']['count'] == 1
    assert result_all['result']['experiments'][0]['id'] == 'exp-beta'
    advisory = 'No deployments in active tree. Use --all-trees to search all trees.'
    for m in result_all['messages']:
      assert m['message'] != advisory

  def test_query_best_deployed_empty_warns(
    self, multi_tree_forest: FileForest, cli_workspace: Path
  ) -> None:
    """--best with --deployed on empty active tree emits advisory."""
    result = run_cli_no_context(cli_workspace, ['query', '--best', 'accuracy', '--deployed'])
    assert result['ok'] is True
    assert result['result']['best'] is None
    messages = result['messages']
    assert any(
      m['level'] == 'info'
      and m['message'] == 'No deployments in active tree. Use --all-trees to search all trees.'
      for m in messages
    )


# ---------------------------------------------------------------------------
# 4.3 Best + schema invariants
# ---------------------------------------------------------------------------


class TestBestAndSchema:
  """Test --best tree attribution and schema invariants."""

  def test_best_all_trees_includes_tree_field(
    self, multi_tree_forest: FileForest, cli_workspace: Path
  ) -> None:
    """--best accuracy --all-trees --json includes tree on the best object."""
    result = run_cli_no_context(cli_workspace, ['query', '--best', 'accuracy', '--all-trees'])
    assert result['ok'] is True
    best = result['result']['best']
    assert best is not None
    assert 'tree' in best
    assert best['tree'] == 'alpha'
    assert best['id'] == 'exp-alpha'

  def test_completed_metrics_trusted_true(
    self, multi_tree_forest: FileForest, cli_workspace: Path
  ) -> None:
    """Completed experiment row has metrics_trusted True."""
    result = run_cli_no_context(cli_workspace, ['query'])
    assert result['ok'] is True
    experiments = result['result']['experiments']
    row = next(r for r in experiments if r['id'] == 'exp-alpha')
    assert row['metrics_trusted'] is True

  def test_query_json_row_schema(self, multi_tree_forest: FileForest, cli_workspace: Path) -> None:
    """Representative query --json row contains all expected keys."""
    result = run_cli_no_context(cli_workspace, ['query'])
    assert result['ok'] is True
    experiments = result['result']['experiments']
    assert len(experiments) > 0
    row = experiments[0]
    expected_keys = {
      'id',
      'status',
      'hypothesis',
      'metrics',
      'spec_version',
      'created_at',
      'started_at',
      'dataset_fingerprint',
      'deployed_as',
      'has_notes',
      'context_log',
      'metrics_trusted',
    }
    assert expected_keys.issubset(set(row.keys()))
