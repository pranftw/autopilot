"""CLI tests for ``report trend`` subcommand."""

from tests.cli.conftest import run_cli_no_context, seed_tree_with_experiments
import pytest


class TestTrendCliJson:
  """report trend JSON output."""

  def test_trend_cli_json(self, cli_workspace, cli_forest):
    """JSON output parses and contains result.direction."""
    seed_tree_with_experiments(
      cli_forest,
      'main',
      [
        {'id': 'e1', 'status': 'completed', 'metrics': {'accuracy': 0.5}},
        {'id': 'e2', 'status': 'completed', 'metrics': {'accuracy': 0.6}},
        {'id': 'e3', 'status': 'completed', 'metrics': {'accuracy': 0.7}},
      ],
    )
    result = run_cli_no_context(cli_workspace, ['report', 'trend', 'accuracy'])
    assert result['ok'] is True
    assert 'direction' in result['result']
    assert result['result']['direction'] == 'improving'
    assert result['result']['metric'] == 'accuracy'


class TestTrendCliExitCode:
  """report trend exit codes."""

  def test_trend_cli_exit_code_success(self, cli_workspace, cli_forest):
    """Success returns ok=True."""
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

  def test_trend_cli_no_tree_fails(self, cli_workspace, cli_forest):
    """Missing active tree -> non-zero exit."""
    with pytest.raises(SystemExit):
      run_cli_no_context(cli_workspace, ['report', 'trend', 'score'])


class TestTrendCliJsonSchema:
  """report trend JSON envelope key stability."""

  def test_trend_cli_json_schema(self, cli_workspace, cli_forest):
    """Top-level JSON envelope contains ok, result, messages keys."""
    seed_tree_with_experiments(
      cli_forest,
      'main',
      [
        {'id': 'e1', 'status': 'completed', 'metrics': {'accuracy': 0.5}},
        {'id': 'e2', 'status': 'completed', 'metrics': {'accuracy': 0.6}},
        {'id': 'e3', 'status': 'completed', 'metrics': {'accuracy': 0.7}},
      ],
    )
    result = run_cli_no_context(cli_workspace, ['report', 'trend', 'accuracy'])
    assert 'ok' in result
    assert 'result' in result
    assert 'messages' in result

    inner = result['result']
    expected_keys = {
      'metric',
      'values',
      'experiment_ids',
      'direction',
      'best_value',
      'best_experiment_id',
      'latest_value',
      'improvement_rate',
    }
    assert set(inner.keys()) == expected_keys
