"""CLI tests for ``report trend --all-trees`` (plan 06)."""

from tests.cli.conftest import run_cli_no_context, seed_tree_with_experiments


class TestReportTrendAllTreesJsonShape:
  """``report trend --all-trees --json`` yields ``result['trees']`` dict."""

  def test_report_trend_all_trees_json_shape(self, cli_workspace, cli_forest):
    """Each tree name appears as a key with a TrendResult dict or None."""
    seed_tree_with_experiments(
      cli_forest,
      'alpha',
      [
        {'id': 'a1', 'status': 'completed', 'metrics': {'accuracy': 0.5}},
        {'id': 'a2', 'status': 'completed', 'metrics': {'accuracy': 0.6}},
        {'id': 'a3', 'status': 'completed', 'metrics': {'accuracy': 0.7}},
      ],
    )
    seed_tree_with_experiments(
      cli_forest,
      'beta',
      [
        {'id': 'b1', 'status': 'completed', 'metrics': {'accuracy': 0.8}},
        {'id': 'b2', 'status': 'completed', 'metrics': {'accuracy': 0.85}},
        {'id': 'b3', 'status': 'completed', 'metrics': {'accuracy': 0.9}},
      ],
    )

    result = run_cli_no_context(cli_workspace, ['report', 'trend', 'accuracy', '--all-trees'])
    assert result['ok'] is True
    trees = result['result']['trees']
    assert 'alpha' in trees
    assert 'beta' in trees
    assert trees['alpha']['direction'] == 'improving'
    assert trees['beta']['direction'] == 'improving'
    assert trees['alpha']['metric'] == 'accuracy'

    expected_trend_keys = {
      'metric',
      'values',
      'experiment_ids',
      'direction',
      'best_value',
      'best_experiment_id',
      'latest_value',
      'improvement_rate',
    }
    assert set(trees['alpha'].keys()) == expected_trend_keys


class TestReportTrendAllTreesEmptyTree:
  """Tree with zero analyzable experiments maps to None."""

  def test_report_trend_all_trees_empty_tree(self, cli_workspace, cli_forest):
    """An empty tree produces None (no trend data)."""
    seed_tree_with_experiments(
      cli_forest,
      'populated',
      [
        {'id': 'p1', 'status': 'completed', 'metrics': {'accuracy': 0.5}},
        {'id': 'p2', 'status': 'completed', 'metrics': {'accuracy': 0.6}},
        {'id': 'p3', 'status': 'completed', 'metrics': {'accuracy': 0.7}},
      ],
    )
    seed_tree_with_experiments(cli_forest, 'empty', [])

    result = run_cli_no_context(cli_workspace, ['report', 'trend', 'accuracy', '--all-trees'])
    assert result['ok'] is True
    trees = result['result']['trees']
    assert trees['empty'] is None
    assert trees['populated'] is not None
    assert trees['populated']['direction'] == 'improving'


class TestReportTrendSingleTreeUnchanged:
  """Omitting ``--all-trees`` preserves pre-plan behavior."""

  def test_report_trend_single_tree_unchanged(self, cli_workspace, cli_forest):
    """Without --all-trees, JSON contains direct TrendResult keys (no ``trees`` wrapper)."""
    seed_tree_with_experiments(
      cli_forest,
      'main',
      [
        {'id': 'e1', 'status': 'completed', 'metrics': {'score': 0.5}},
        {'id': 'e2', 'status': 'completed', 'metrics': {'score': 0.6}},
        {'id': 'e3', 'status': 'completed', 'metrics': {'score': 0.7}},
      ],
    )

    result = run_cli_no_context(cli_workspace, ['report', 'trend', 'score'])
    assert result['ok'] is True
    inner = result['result']
    assert 'direction' in inner
    assert inner['direction'] == 'improving'
    assert inner['metric'] == 'score'
    assert 'trees' not in inner


class TestReportTrendSingleDatapoint:
  """Single completed experiment yields insufficient_data direction."""

  def test_report_trend_single_datapoint(self, cli_workspace, cli_forest):
    """One data point -> insufficient_data direction."""
    seed_tree_with_experiments(
      cli_forest,
      'main',
      [
        {'id': 'e1', 'status': 'completed', 'metrics': {'accuracy': 0.9}},
      ],
    )

    result = run_cli_no_context(cli_workspace, ['report', 'trend', 'accuracy'])
    assert result['ok'] is True
    inner = result['result']
    assert inner['direction'] == 'insufficient_data'
    assert len(inner['values']) == 1

  def test_report_trend_all_trees_single_datapoint(self, cli_workspace, cli_forest):
    """Single datapoint per tree in --all-trees mode."""
    seed_tree_with_experiments(
      cli_forest,
      'solo',
      [
        {'id': 's1', 'status': 'completed', 'metrics': {'accuracy': 0.75}},
      ],
    )

    result = run_cli_no_context(cli_workspace, ['report', 'trend', 'accuracy', '--all-trees'])
    assert result['ok'] is True
    trees = result['result']['trees']
    assert trees['solo'] is not None
    assert trees['solo']['direction'] == 'insufficient_data'
    assert len(trees['solo']['values']) == 1
