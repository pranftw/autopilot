"""Deployment persistence and cross-tree targeting tests (Plan 07).

Tests:
  - test_deploy_replace_cross_tree_persistence
  - test_deploy_replace_immediate_query
  - test_deploy_events_after_forest_save
  - test_deploy_replace_clears_old_label_other_tree
  - test_deploy_cross_tree_without_switch
  - test_undeploy_cross_tree
"""

from autopilot.ai.deployment import DeploymentEvent, DeploymentLog, deployment_log_for_workspace
from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_no_context
from unittest.mock import patch
import pytest


def _setup_two_tree_workspace(tmp_path: Path) -> tuple[Path, FileForest]:
  """Create a workspace with two trees, each with one experiment.

  Tree 'alpha' has exp-alpha, tree 'beta' has exp-beta.
  'alpha' is the active tree.
  """
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  tree_a = forest.create_tree('alpha')
  exp_a = Experiment(experiment_id='exp-alpha', hypothesis='alpha')
  exp_a.start()
  exp_a.complete(metrics={'accuracy': 0.8})
  tree_a.add(Node(experiment=exp_a))

  tree_b = forest.create_tree('beta')
  exp_b = Experiment(experiment_id='exp-beta', hypothesis='beta')
  exp_b.start()
  exp_b.complete(metrics={'accuracy': 0.9})
  tree_b.add(Node(experiment=exp_b))

  forest.switch('alpha')
  forest.save()
  return ws, forest


class TestDeployPersistence:
  """Persistence and ordering tests for deploy --replace across trees."""

  def test_deploy_replace_cross_tree_persistence(self, tmp_path: Path) -> None:
    """Deploy with --replace across trees; reload forest; exactly one holder."""
    ws, _ = _setup_two_tree_workspace(tmp_path)

    run_cli(ws, ['experiment', 'deploy', 'exp-alpha', '--as', 'prod'])

    run_cli(ws, ['experiment', 'deploy', 'exp-beta', '--as', 'prod', '--replace'])

    config = AutoPilotConfig(workspace=ws)
    store = FileStore(config)
    reloaded = FileForest(store)
    reloaded.load()

    holders: list[str] = []
    for tree in reloaded.list_trees():
      holders.extend(
        node.experiment.id for node in tree.query().all() if node.deployed_as == 'prod'
      )

    assert holders == ['exp-beta']

  def test_deploy_replace_immediate_query(self, tmp_path: Path) -> None:
    """After replace, query --deployed shows new mapping immediately."""
    ws, _ = _setup_two_tree_workspace(tmp_path)

    run_cli(ws, ['experiment', 'deploy', 'exp-alpha', '--as', 'prod'])
    run_cli(ws, ['experiment', 'deploy', 'exp-beta', '--as', 'prod', '--replace'])

    result = run_cli_no_context(ws, ['query', '--deployed', '--all-trees'])
    experiments = result['result']['experiments']
    deployed_ids = [e['id'] for e in experiments]
    assert 'exp-beta' in deployed_ids
    assert 'exp-alpha' not in deployed_ids

  def test_deploy_events_after_forest_save(self, tmp_path: Path) -> None:
    """Deployment events are appended only after forest.save() succeeds.

    Patches FileForest.save to track call ordering relative to event append.
    """
    ws, _ = _setup_two_tree_workspace(tmp_path)

    call_order: list[str] = []

    original_save = FileForest.save

    def tracked_save(self_forest: FileForest) -> None:
      call_order.append('save')
      original_save(self_forest)

    original_append = DeploymentLog.append

    def tracked_append(self_log: DeploymentLog, event: DeploymentEvent) -> None:
      call_order.append('append')
      original_append(self_log, event)

    with (
      patch.object(FileForest, 'save', tracked_save),
      patch.object(DeploymentLog, 'append', tracked_append),
    ):
      run_cli(ws, ['experiment', 'deploy', 'exp-alpha', '--as', 'prod'])

    assert 'save' in call_order
    assert 'append' in call_order
    save_idx = call_order.index('save')
    append_idx = call_order.index('append')
    assert save_idx < append_idx, f'save must precede append; got order: {call_order}'

  def test_deploy_replace_clears_old_label_other_tree(self, tmp_path: Path) -> None:
    """Old holder on a different tree clears deployed_as after replace."""
    ws, _ = _setup_two_tree_workspace(tmp_path)

    run_cli(ws, ['experiment', 'deploy', 'exp-alpha', '--as', 'prod'])

    result = run_cli(ws, ['experiment', 'deploy', 'exp-beta', '--as', 'prod', '--replace'])
    assert result['ok'] is True
    assert result['result']['replaced_experiment_id'] == 'exp-alpha'

    show_alpha = run_cli_no_context(ws, ['experiment', 'show', 'exp-alpha'])
    assert show_alpha['result']['deployed_as'] is None

    show_beta = run_cli_no_context(ws, ['experiment', 'show', 'exp-beta'])
    assert show_beta['result']['deployed_as'] == 'prod'

  def test_deploy_no_event_on_save_failure(self, tmp_path: Path) -> None:
    """When save() raises, no deployment event is written."""
    ws, _ = _setup_two_tree_workspace(tmp_path)

    def failing_save(self_forest: FileForest) -> None:
      msg = 'simulated save failure'
      raise OSError(msg)

    log_path = deployment_log_for_workspace(ws)

    with (
      patch.object(FileForest, 'save', failing_save),
      pytest.raises(OSError, match='simulated save failure'),
    ):
      run_cli(ws, ['experiment', 'deploy', 'exp-alpha', '--as', 'prod'])

    events = log_path.query(label='prod')
    assert events == []


class TestDeployCrossTree:
  """Cross-tree deploy/undeploy without tree switch."""

  def test_deploy_cross_tree_without_switch(self, tmp_path: Path) -> None:
    """Experiment on non-active tree can be deployed without tree switch."""
    ws, _ = _setup_two_tree_workspace(tmp_path)

    result = run_cli(ws, ['experiment', 'deploy', 'exp-beta', '--as', 'svc'])
    assert result['ok'] is True
    assert result['result']['experiment_id'] == 'exp-beta'
    assert result['result']['deployed_as'] == 'svc'

    config = AutoPilotConfig(workspace=ws)
    store = FileStore(config)
    reloaded = FileForest(store)
    reloaded.load()

    assert reloaded.active is not None
    assert reloaded.active.name == 'alpha'

    found = reloaded.find_experiment('exp-beta')
    assert found is not None
    node, _ = found
    assert node.deployed_as == 'svc'

  def test_undeploy_cross_tree(self, tmp_path: Path) -> None:
    """Undeploy label whose holder sits on non-active tree succeeds."""
    ws, _ = _setup_two_tree_workspace(tmp_path)

    run_cli(ws, ['experiment', 'deploy', 'exp-beta', '--as', 'prod'])

    result = run_cli(ws, ['experiment', 'undeploy', 'prod'])
    assert result['ok'] is True
    assert result['result']['experiment_id'] == 'exp-beta'
    assert result['result']['label'] == 'prod'

    config = AutoPilotConfig(workspace=ws)
    store = FileStore(config)
    reloaded = FileForest(store)
    reloaded.load()

    found = reloaded.find_experiment('exp-beta')
    assert found is not None
    node, _ = found
    assert node.deployed_as is None

  def test_deploy_missing_experiment_fails(self, tmp_path: Path) -> None:
    """Deploy with nonexistent experiment id fails with guidance."""
    ws, _ = _setup_two_tree_workspace(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
      run_cli(ws, ['experiment', 'deploy', 'nonexistent', '--as', 'prod'])
    assert exc_info.value.code != 0

  def test_undeploy_events_after_save(self, tmp_path: Path) -> None:
    """Undeploy writes events only after forest.save() completes."""
    ws, _ = _setup_two_tree_workspace(tmp_path)
    run_cli(ws, ['experiment', 'deploy', 'exp-alpha', '--as', 'prod'])

    call_order: list[str] = []

    original_save = FileForest.save

    def tracked_save(self_forest: FileForest) -> None:
      call_order.append('save')
      original_save(self_forest)

    original_append = DeploymentLog.append

    def tracked_append(self_log: DeploymentLog, event: DeploymentEvent) -> None:
      call_order.append('append')
      original_append(self_log, event)

    with (
      patch.object(FileForest, 'save', tracked_save),
      patch.object(DeploymentLog, 'append', tracked_append),
    ):
      run_cli(ws, ['experiment', 'undeploy', 'prod'])

    assert 'save' in call_order
    assert 'append' in call_order
    save_idx = call_order.index('save')
    append_idx = call_order.index('append')
    assert save_idx < append_idx
