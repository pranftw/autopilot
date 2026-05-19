"""CLI integration tests for experiment lifecycle enhancements (Plan 06).

Tests the CLI commands: experiment fail --metrics, experiment invalidate,
experiment deploy, and query --include-invalidated / --deployed.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_no_context
import pytest


def _setup_workspace(tmp_path: Path) -> tuple[Path, FileForest]:
  """Create a workspace with a forest containing one completed experiment."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')

  exp = Experiment(experiment_id='exp-1', hypothesis='test')
  exp.start()
  exp.complete(metrics={'accuracy': 0.95})
  tree.add(Node(experiment=exp))

  exp2 = Experiment(experiment_id='exp-2', hypothesis='test2')
  exp2.start()
  exp2.complete(metrics={'accuracy': 0.8})
  tree.add(Node(experiment=exp2))

  exp3 = Experiment(experiment_id='exp-pending', hypothesis='pending')
  tree.add(Node(experiment=exp3))

  forest.save()
  return ws, forest


class TestCLIExperimentFailMetrics:
  """CLI experiment fail --metrics integration."""

  def test_fail_with_metrics_json(self, tmp_path: Path) -> None:
    """experiment fail with --metrics persists and returns metrics."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    tree = forest.create_tree('main')
    forest.switch('main')
    exp = Experiment(experiment_id='fail-me', hypothesis='will fail')
    exp.start()
    tree.add(Node(experiment=exp))
    forest.save()

    result = run_cli(
      ws, ['experiment', 'fail', 'fail-me', '--error', 'oom', '--metrics', '{"acc": 0.1}']
    )
    assert result['ok'] is True
    assert result['result']['status'] == 'failed'
    assert result['result']['metrics'] == {'acc': 0.1}
    assert result['result']['error'] == 'oom'

  def test_fail_with_invalid_metrics_json(self, tmp_path: Path) -> None:
    """experiment fail with invalid --metrics JSON fails gracefully."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    tree = forest.create_tree('main')
    forest.switch('main')
    exp = Experiment(experiment_id='fail-bad', hypothesis='bad json')
    exp.start()
    tree.add(Node(experiment=exp))
    forest.save()

    with pytest.raises(SystemExit):
      run_cli(ws, ['experiment', 'fail', 'fail-bad', '--metrics', 'not-json'])

  def test_fail_with_non_dict_metrics(self, tmp_path: Path) -> None:
    """experiment fail with non-object --metrics JSON (e.g. array) fails."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    tree = forest.create_tree('main')
    forest.switch('main')
    exp = Experiment(experiment_id='fail-arr', hypothesis='array json')
    exp.start()
    tree.add(Node(experiment=exp))
    forest.save()

    with pytest.raises(SystemExit):
      run_cli(ws, ['experiment', 'fail', 'fail-arr', '--metrics', '[1, 2, 3]'])


class TestCLIExperimentInvalidate:
  """CLI experiment invalidate integration."""

  def test_invalidate_completed_experiment(self, tmp_path: Path) -> None:
    """Invalidate a completed experiment successfully."""
    ws, _ = _setup_workspace(tmp_path)
    result = run_cli(ws, ['experiment', 'invalidate', 'exp-1', '--reason', 'contaminated data'])
    assert result['ok'] is True
    assert result['result']['status'] == 'invalidated'
    assert result['result']['invalidated_at'] is not None

  def test_invalidate_running_experiment_fails(self, tmp_path: Path) -> None:
    """Cannot invalidate a running experiment."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    tree = forest.create_tree('main')
    forest.switch('main')
    exp = Experiment(experiment_id='run-exp', hypothesis='running')
    exp.start()
    tree.add(Node(experiment=exp))
    forest.save()

    with pytest.raises(SystemExit):
      run_cli(ws, ['experiment', 'invalidate', 'run-exp', '--reason', 'attempt'])

  def test_invalidate_empty_reason_fails(self, tmp_path: Path) -> None:
    """Invalidate with empty --reason fails."""
    ws, _ = _setup_workspace(tmp_path)
    with pytest.raises(SystemExit):
      run_cli(ws, ['experiment', 'invalidate', 'exp-1', '--reason', ''])


class TestCLIExperimentDeploy:
  """CLI experiment deploy integration."""

  def test_deploy_sets_deployed_as(self, tmp_path: Path) -> None:
    """Deploy sets the deployment label."""
    ws, _ = _setup_workspace(tmp_path)
    result = run_cli(ws, ['experiment', 'deploy', 'exp-1', '--as', 'production'])
    assert result['ok'] is True
    assert result['result']['deployed_as'] == 'production'

  def test_deploy_duplicate_name_conflict(self, tmp_path: Path) -> None:
    """Deploying a different experiment with the same name fails."""
    ws, _ = _setup_workspace(tmp_path)
    run_cli(ws, ['experiment', 'deploy', 'exp-1', '--as', 'production'])
    with pytest.raises(SystemExit):
      run_cli(ws, ['experiment', 'deploy', 'exp-2', '--as', 'production'])

  def test_deploy_same_experiment_idempotent(self, tmp_path: Path) -> None:
    """Deploying the same experiment with the same name is a no-op."""
    ws, _ = _setup_workspace(tmp_path)
    run_cli(ws, ['experiment', 'deploy', 'exp-1', '--as', 'production'])
    result = run_cli(ws, ['experiment', 'deploy', 'exp-1', '--as', 'production'])
    assert result['ok'] is True
    assert result['result']['idempotent'] is True

  def test_deploy_different_name_fails(self, tmp_path: Path) -> None:
    """Cannot redeploy an experiment under a different name."""
    ws, _ = _setup_workspace(tmp_path)
    run_cli(ws, ['experiment', 'deploy', 'exp-1', '--as', 'production'])
    with pytest.raises(SystemExit):
      run_cli(ws, ['experiment', 'deploy', 'exp-1', '--as', 'staging'])


class TestCLIQueryFilters:
  """CLI query --include-invalidated and --deployed filters."""

  def test_query_excludes_invalidated_by_default(self, tmp_path: Path) -> None:
    """Invalidated experiments excluded from query results by default."""
    ws, _ = _setup_workspace(tmp_path)
    run_cli(ws, ['experiment', 'invalidate', 'exp-1', '--reason', 'bad data'])
    result = run_cli_no_context(ws, ['query'])
    ids = [e['id'] for e in result['result']['experiments']]
    assert 'exp-1' not in ids
    assert 'exp-2' in ids

  def test_query_include_invalidated(self, tmp_path: Path) -> None:
    """--include-invalidated flag restores visibility."""
    ws, _ = _setup_workspace(tmp_path)
    run_cli(ws, ['experiment', 'invalidate', 'exp-1', '--reason', 'bad data'])
    result = run_cli_no_context(ws, ['query', '--include-invalidated'])
    ids = [e['id'] for e in result['result']['experiments']]
    assert 'exp-1' in ids

  def test_query_deployed_filter(self, tmp_path: Path) -> None:
    """--deployed returns only deployed nodes."""
    ws, _ = _setup_workspace(tmp_path)
    run_cli(ws, ['experiment', 'deploy', 'exp-1', '--as', 'production'])
    result = run_cli_no_context(ws, ['query', '--deployed'])
    ids = [e['id'] for e in result['result']['experiments']]
    assert 'exp-1' in ids
    assert 'exp-2' not in ids

  def test_query_deployed_empty_when_none_deployed(self, tmp_path: Path) -> None:
    """--deployed returns empty when no experiments are deployed."""
    ws, _ = _setup_workspace(tmp_path)
    result = run_cli_no_context(ws, ['query', '--deployed'])
    assert result['result']['count'] == 0
