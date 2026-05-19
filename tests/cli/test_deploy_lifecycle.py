"""Deployment lifecycle tests (Plan 02): Forest deploy/undeploy + CLI.

Tests: 4 Forest unit tests + 12 CLI tests = 16 total.
Covers: Forest.deploy, Forest.undeploy, experiment deploy --replace,
experiment undeploy, JSON envelopes, context journaling, cross-tree replace.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.command import CLI
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_no_context
import pytest


def _setup_workspace(tmp_path: Path) -> tuple[Path, FileForest]:
  """Create a workspace with two completed experiments in one tree."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')

  exp_a = Experiment(experiment_id='exp-a', hypothesis='first')
  exp_a.start()
  exp_a.complete(metrics={'accuracy': 0.9})
  tree.add(Node(experiment=exp_a))

  exp_b = Experiment(experiment_id='exp-b', hypothesis='second')
  exp_b.start()
  exp_b.complete(metrics={'accuracy': 0.95})
  tree.add(Node(experiment=exp_b))

  forest.save()
  return ws, forest


def _setup_cross_tree_workspace(tmp_path: Path) -> tuple[Path, FileForest]:
  """Create a workspace with two trees, one experiment each."""
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


def _make_forest(tmp_path: Path) -> FileForest:
  """Create a minimal FileForest backed by a tmp_path store."""
  ws = tmp_path / 'unit'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  return FileForest(store)


# --- Section 4.1: Forest unit tests ---


class TestForestDeployUnit:
  """Forest.deploy and Forest.undeploy contract tests."""

  def test_forest_deploy_raises_on_conflict(self, tmp_path: Path) -> None:
    """deploy raises ValueError when label is held and replace=False."""
    forest = _make_forest(tmp_path)
    tree = forest.create_tree('t')

    exp_a = Experiment(experiment_id='a')
    node_a = Node(experiment=exp_a)
    tree.add(node_a)

    exp_b = Experiment(experiment_id='b')
    node_b = Node(experiment=exp_b)
    tree.add(node_b)

    forest.deploy(node_a, 'prod')
    with pytest.raises(ValueError, match='--replace'):
      forest.deploy(node_b, 'prod')

  def test_forest_deploy_replace_clears_previous(self, tmp_path: Path) -> None:
    """deploy with replace=True clears previous holder and sets new one."""
    forest = _make_forest(tmp_path)
    tree = forest.create_tree('t')

    exp_a = Experiment(experiment_id='a')
    node_a = Node(experiment=exp_a)
    tree.add(node_a)

    exp_b = Experiment(experiment_id='b')
    node_b = Node(experiment=exp_b)
    tree.add(node_b)

    forest.deploy(node_a, 'prod')
    previous = forest.deploy(node_b, 'prod', replace=True)

    assert previous is node_a
    assert node_a.deployed_as is None
    assert node_b.deployed_as == 'prod'

  def test_forest_undeploy_returns_node(self, tmp_path: Path) -> None:
    """undeploy returns the node that held the label and clears it."""
    forest = _make_forest(tmp_path)
    tree = forest.create_tree('t')

    exp = Experiment(experiment_id='x')
    node = Node(experiment=exp)
    tree.add(node)
    node.deployed_as = 'staging'

    result = forest.undeploy('staging')
    assert result is node
    assert node.deployed_as is None

  def test_forest_undeploy_unknown_returns_none(self, tmp_path: Path) -> None:
    """undeploy returns None when no node holds the label."""
    forest = _make_forest(tmp_path)
    forest.create_tree('t')

    result = forest.undeploy('nonexistent')
    assert result is None


# --- Section 4.1: Undeploy and query CLI tests ---


class TestUndeployCLI:
  """CLI experiment undeploy command tests."""

  def test_undeploy_clears_label(self, tmp_path: Path) -> None:
    """Happy path: deploy then undeploy leaves deployed_as as None."""
    ws, _ = _setup_workspace(tmp_path)
    run_cli(ws, ['experiment', 'deploy', 'exp-a', '--as', 'prod'])
    run_cli(ws, ['experiment', 'undeploy', 'prod'])

    show = run_cli_no_context(ws, ['experiment', 'show', 'exp-a'])
    assert show['result']['deployed_as'] is None

  def test_undeploy_unknown_label_fails(self, tmp_path: Path) -> None:
    """Unknown label causes non-zero exit."""
    ws, _ = _setup_workspace(tmp_path)
    with pytest.raises(SystemExit):
      run_cli(ws, ['experiment', 'undeploy', 'ghost'])

  def test_undeploy_requires_context(self) -> None:
    """Mutating command: 'experiment undeploy' is not context-exempt.

    Context enforcement happens in ``CLI.dispatch``, not in the handler.
    Verified via ``requires_context`` which is the authoritative check
    (same pattern as ``tests/cli/test_context_exemptions.py``).
    """
    cli = CLI()
    assert cli.requires_context('experiment undeploy') is True

  def test_undeploy_json(self, tmp_path: Path) -> None:
    """--json success envelope confirms label and experiment id."""
    ws, _ = _setup_workspace(tmp_path)
    run_cli(ws, ['experiment', 'deploy', 'exp-a', '--as', 'prod'])
    result = run_cli(ws, ['experiment', 'undeploy', 'prod'])
    assert result['ok'] is True
    assert result['result']['label'] == 'prod'
    assert result['result']['experiment_id'] == 'exp-a'

  def test_undeploy_exit_code_on_unknown_label(self, tmp_path: Path) -> None:
    """Explicit non-zero exit code assertion for unknown label."""
    ws, _ = _setup_workspace(tmp_path)
    with pytest.raises(SystemExit) as exc_info:
      run_cli(ws, ['experiment', 'undeploy', 'missing'])
    assert exc_info.value.code != 0

  def test_query_deployed_after_undeploy(self, tmp_path: Path) -> None:
    """After undeploy, query --deployed no longer lists the experiment."""
    ws, _ = _setup_workspace(tmp_path)
    run_cli(ws, ['experiment', 'deploy', 'exp-a', '--as', 'prod'])

    deployed = run_cli_no_context(ws, ['query', '--deployed'])
    ids_before = [e['id'] for e in deployed['result']['experiments']]
    assert 'exp-a' in ids_before

    run_cli(ws, ['experiment', 'undeploy', 'prod'])

    deployed_after = run_cli_no_context(ws, ['query', '--deployed'])
    ids_after = [e['id'] for e in deployed_after['result']['experiments']]
    assert 'exp-a' not in ids_after


# --- Section 4.2: Deploy, replace, and JSON CLI tests ---


class TestDeployReplaceCLI:
  """CLI experiment deploy --replace tests."""

  def test_deploy_replace_swaps_labels(self, tmp_path: Path) -> None:
    """--replace transfers label from A to B."""
    ws, _ = _setup_workspace(tmp_path)
    run_cli(ws, ['experiment', 'deploy', 'exp-a', '--as', 'prod'])
    run_cli(ws, ['experiment', 'deploy', 'exp-b', '--as', 'prod', '--replace'])

    show_a = run_cli_no_context(ws, ['experiment', 'show', 'exp-a'])
    assert show_a['result']['deployed_as'] is None

    show_b = run_cli_no_context(ws, ['experiment', 'show', 'exp-b'])
    assert show_b['result']['deployed_as'] == 'prod'

  def test_deploy_replace_emits_context(self, tmp_path: Path) -> None:
    """After replace, both experiments have deployment context entries."""
    ws, _ = _setup_workspace(tmp_path)
    run_cli(ws, ['experiment', 'deploy', 'exp-a', '--as', 'prod'])
    run_cli(ws, ['experiment', 'deploy', 'exp-b', '--as', 'prod', '--replace'])

    show_a = run_cli_no_context(ws, ['experiment', 'show', 'exp-a', '--context-log'])
    entries_a = show_a['result']['context_log']
    deploy_entries_a = [e for e in entries_a if e.get('source') == 'deployment']
    assert len(deploy_entries_a) >= 1

    show_b = run_cli_no_context(ws, ['experiment', 'show', 'exp-b', '--context-log'])
    entries_b = show_b['result']['context_log']
    deploy_entries_b = [e for e in entries_b if e.get('source') == 'deployment']
    assert len(deploy_entries_b) >= 1

  def test_deploy_replace_without_flag_fails(self, tmp_path: Path) -> None:
    """Without --replace, conflict preserved and hints --replace."""
    ws, _ = _setup_workspace(tmp_path)
    run_cli(ws, ['experiment', 'deploy', 'exp-a', '--as', 'prod'])
    with pytest.raises(SystemExit):
      run_cli(ws, ['experiment', 'deploy', 'exp-b', '--as', 'prod'])

  def test_deploy_replace_json(self, tmp_path: Path) -> None:
    """JSON result includes replaced_experiment_id when replace occurred."""
    ws, _ = _setup_workspace(tmp_path)
    run_cli(ws, ['experiment', 'deploy', 'exp-a', '--as', 'prod'])
    result = run_cli(ws, ['experiment', 'deploy', 'exp-b', '--as', 'prod', '--replace'])
    assert result['ok'] is True
    assert result['result']['replaced_experiment_id'] == 'exp-a'
    assert result['result']['deployed_as'] == 'prod'

  def test_deploy_replace_cross_tree(self, tmp_path: Path) -> None:
    """Label holder in non-active tree is cleared by replace from active tree."""
    ws, _ = _setup_cross_tree_workspace(tmp_path)

    run_cli(ws, ['experiment', 'deploy', 'exp-alpha', '--as', 'prod'])

    run_cli(ws, ['tree', 'switch', 'beta', '--no-checkout'])

    result = run_cli(ws, ['experiment', 'deploy', 'exp-beta', '--as', 'prod', '--replace'])
    assert result['ok'] is True
    assert result['result']['replaced_experiment_id'] == 'exp-alpha'

    run_cli(ws, ['tree', 'switch', 'alpha', '--no-checkout'])
    show_alpha = run_cli_no_context(ws, ['experiment', 'show', 'exp-alpha'])
    assert show_alpha['result']['deployed_as'] is None

  def test_deploy_cross_tree_no_switch(self, tmp_path: Path) -> None:
    """Deploy experiment on non-active tree without switching trees."""
    ws, _ = _setup_cross_tree_workspace(tmp_path)

    result = run_cli(ws, ['experiment', 'deploy', 'exp-beta', '--as', 'canary'])
    assert result['ok'] is True
    assert result['result']['experiment_id'] == 'exp-beta'
    assert result['result']['deployed_as'] == 'canary'

  def test_deploy_replace_cross_tree_no_switch(self, tmp_path: Path) -> None:
    """Replace across trees without switching; old holder on other tree cleared."""
    ws, _ = _setup_cross_tree_workspace(tmp_path)

    run_cli(ws, ['experiment', 'deploy', 'exp-beta', '--as', 'prod'])

    result = run_cli(ws, ['experiment', 'deploy', 'exp-alpha', '--as', 'prod', '--replace'])
    assert result['ok'] is True
    assert result['result']['replaced_experiment_id'] == 'exp-beta'

    show_beta = run_cli_no_context(ws, ['experiment', 'show', 'exp-beta'])
    assert show_beta['result']['deployed_as'] is None

  def test_deploy_json_ok_field(self, tmp_path: Path) -> None:
    """Plain deploy --json success includes ok: True and result with label."""
    ws, _ = _setup_workspace(tmp_path)
    result = run_cli(ws, ['experiment', 'deploy', 'exp-a', '--as', 'staging'])
    assert result['ok'] is True
    assert result['result']['deployed_as'] == 'staging'
    assert result['result']['experiment_id'] == 'exp-a'
