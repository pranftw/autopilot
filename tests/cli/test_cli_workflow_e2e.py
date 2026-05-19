"""End-to-end CLI workflow test exercising the full experiment lifecycle.

Steps:
  1. workspace init
  2. tree create main
  3. experiment add --hypothesis H --id exp-1
  4. tree list -> shows main with exp-1
  5. experiment status exp-1
  6. tree create feature
  7. tree switch feature
  8. experiment add --hypothesis H2 --id exp-2
  9. tree switch main
  10. query -> shows exp-1
  11. checkout exp-1
  12. experiment status (no id, uses HEAD)
  13. experiment compare exp-1 exp-2
      (cross-tree: exp-2 is in feature, compare searches all trees)
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.commands.workspace import WorkspaceCommand
from autopilot.cli.context import CLIContext
from autopilot.cli.output import Output
from autopilot.core.config import AutoPilotConfig
from pathlib import Path
from tests.cli.conftest import run_cli
from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture(autouse=True)
def _patch_store_checkout():
  """Patch FileStore.checkout for workflow tests that don't create snapshots."""
  with patch('autopilot.ai.store.file_store.FileStore.checkout'):
    yield


@pytest.fixture
def ws(tmp_path: Path) -> Path:
  ws = tmp_path / 'ws'
  ws.mkdir()
  return ws


def _complete_experiment_on_active_tree(
  forest: FileForest,
  experiment_id: str,
  metrics: dict[str, float],
) -> None:
  """Load node from forest.active, start it, complete with metrics, persist forest."""
  tree = forest.active
  assert tree is not None
  node = tree.get(experiment_id)
  assert node is not None
  node.experiment.start()
  node.experiment.complete(metrics=metrics)
  forest.save()


def _init_workspace_and_add_exp1(ws: Path) -> None:
  """Steps 1-5: init workspace, create tree, add exp-1, verify status."""
  config = AutoPilotConfig(workspace=ws)
  ctx = CLIContext(workspace=ws, config=config, output=Output(use_json=True))
  WorkspaceCommand().init(ctx, MagicMock())
  assert config.autopilot_path.is_dir()
  assert config.experiments_path.is_dir()

  result = run_cli(ws, ['tree', 'create', 'main'])
  assert result['ok'] is True
  assert result['result']['tree'] == 'main'

  result = run_cli(
    ws,
    ['experiment', 'add', '--hypothesis', 'baseline hypothesis', '--id', 'exp-1'],
  )
  assert result['ok'] is True
  assert result['result']['experiment_id'] == 'exp-1'
  assert result['result']['hypothesis'] == 'baseline hypothesis'

  config2 = AutoPilotConfig(workspace=ws)
  config2.store_path.mkdir(parents=True, exist_ok=True)
  forest = FileForest(FileStore(config2))
  _complete_experiment_on_active_tree(forest, 'exp-1', {'accuracy': 0.75})

  result = run_cli(ws, ['tree', 'list'])
  assert result['ok'] is True
  trees = result['result']['trees']
  assert len(trees) == 1
  assert trees[0]['name'] == 'main'
  assert trees[0]['nodes'] == 1
  assert trees[0]['active'] is True

  result = run_cli(ws, ['experiment', 'status', 'exp-1'])
  assert result['ok'] is True
  assert result['result']['id'] == 'exp-1'
  assert result['result']['hypothesis'] == 'baseline hypothesis'
  assert result['result']['status'] == 'completed'
  assert result['result']['metrics']['accuracy'] == 0.75


def _add_feature_tree_exp2(ws: Path) -> None:
  """Steps 6-9: create feature tree, add exp-2, switch back to main."""
  result = run_cli(ws, ['tree', 'create', 'feature'])
  assert result['ok'] is True

  result = run_cli(ws, ['tree', 'switch', 'feature', '--no-checkout'])
  assert result['ok'] is True
  assert result['result']['active'] == 'feature'

  result = run_cli(
    ws,
    ['experiment', 'add', '--hypothesis', 'feature hypothesis', '--id', 'exp-2'],
  )
  assert result['ok'] is True
  assert result['result']['experiment_id'] == 'exp-2'

  forest2 = FileForest(FileStore(AutoPilotConfig(workspace=ws)))
  feature_tree = forest2.active
  assert feature_tree is not None
  assert feature_tree.name == 'feature'
  _complete_experiment_on_active_tree(forest2, 'exp-2', {'accuracy': 0.82})

  result = run_cli(ws, ['tree', 'switch', 'main', '--no-checkout'])
  assert result['ok'] is True
  assert result['result']['active'] == 'main'


def _query_checkout_compare(ws: Path) -> None:
  """Steps 10-13: query, checkout, status, and compare experiments."""
  result = run_cli(ws, ['query', '--completed'])
  assert result['ok'] is True
  exp_ids = [e['id'] for e in result['result']['experiments']]
  assert 'exp-1' in exp_ids

  result = run_cli(ws, ['checkout', 'exp-1'])
  assert result['ok'] is True
  assert result['result']['experiment_id'] == 'exp-1'

  result = run_cli(ws, ['experiment', 'status'])
  assert result['ok'] is True
  assert result['result']['id'] == 'exp-1'

  result = run_cli(ws, ['experiment', 'compare', 'exp-1', 'exp-2'])
  assert result['ok'] is True
  assert result['result']['a'] == 'exp-1'
  assert result['result']['b'] == 'exp-2'
  deltas_by_metric = {d['metric']: d for d in result['result']['deltas']}
  assert deltas_by_metric['accuracy']['delta'] == pytest.approx(0.07)

  result = run_cli(
    ws,
    ['experiment', 'add', '--hypothesis', 'main variation', '--id', 'exp-3', '--parent', 'exp-1'],
  )
  assert result['result']['ok'] is True

  forest3 = FileForest(FileStore(AutoPilotConfig(workspace=ws)))
  _complete_experiment_on_active_tree(forest3, 'exp-3', {'accuracy': 0.80})

  result = run_cli(ws, ['experiment', 'compare', 'exp-1', 'exp-3'])
  assert result['ok'] is True
  deltas_by_metric = {d['metric']: d for d in result['result']['deltas']}
  assert 'accuracy' in deltas_by_metric
  assert deltas_by_metric['accuracy']['baseline'] == 0.75
  assert deltas_by_metric['accuracy']['candidate'] == 0.80


class TestFullWorkflow:
  """Full 13-step CLI workflow as specified in sub-plan 21."""

  def test_complete_workflow(self, ws: Path) -> None:
    _init_workspace_and_add_exp1(ws)
    _add_feature_tree_exp2(ws)
    _query_checkout_compare(ws)


class TestWorkflowTreeIsolation:
  """Verify that trees are isolated from each other."""

  def test_experiments_stay_in_their_tree(self, ws: Path) -> None:
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)

    run_cli(ws, ['tree', 'create', 'alpha'])
    run_cli(
      ws,
      [
        'experiment',
        'add',
        '--hypothesis',
        'alpha exp',
        '--id',
        'a-exp',
      ],
    )

    # Complete a-exp so it appears in completed query
    store = FileStore(config)
    forest = FileForest(store)
    tree = forest.active
    assert tree is not None
    node = tree.get('a-exp')
    assert node is not None
    node.experiment.start()
    node.experiment.complete(metrics={'score': 1.0})
    forest.save()

    run_cli(ws, ['tree', 'create', 'beta'])
    run_cli(ws, ['tree', 'switch', 'beta', '--no-checkout'])
    run_cli(
      ws,
      [
        'experiment',
        'add',
        '--hypothesis',
        'beta exp',
        '--id',
        'b-exp',
      ],
    )

    # Query on beta tree should only show b-exp
    result = run_cli(ws, ['query'])
    exp_ids = [e['id'] for e in result['result']['experiments']]
    assert 'b-exp' in exp_ids
    assert 'a-exp' not in exp_ids

    # Switch back to alpha, query should show a-exp
    run_cli(ws, ['tree', 'switch', 'alpha', '--no-checkout'])
    result = run_cli(ws, ['query', '--completed'])
    exp_ids = [e['id'] for e in result['result']['experiments']]
    assert 'a-exp' in exp_ids
    assert 'b-exp' not in exp_ids


class TestWorkflowAutoParentToHead:
  """When HEAD is set and --parent omitted, parent defaults to HEAD."""

  def test_auto_parent_chain(self, ws: Path) -> None:
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    run_cli(ws, ['tree', 'create', 'main'])

    # Add root experiment
    result = run_cli(
      ws,
      [
        'experiment',
        'add',
        '--hypothesis',
        'root',
        '--id',
        'root-exp',
      ],
    )
    assert result['result']['parent'] is None

    # Complete root so it can be parent
    store = FileStore(config)
    forest = FileForest(store)
    tree = forest.active
    assert tree is not None
    node = tree.get('root-exp')
    assert node is not None
    node.experiment.start()
    node.experiment.complete(metrics={'m': 1.0})
    forest.save()

    # Checkout root-exp to set HEAD
    run_cli(ws, ['checkout', 'root-exp'])

    # Add child without explicit --parent; should auto-parent to HEAD
    result = run_cli(
      ws,
      [
        'experiment',
        'add',
        '--hypothesis',
        'child',
        '--id',
        'child-exp',
      ],
    )
    assert result['result']['parent'] == 'root-exp'
    assert result['result']['baseline'] == 'root-exp'


class TestWorkflowParentMustBeTerminal:
  """Tree.add enforces that parent must be terminal."""

  def test_non_terminal_parent_rejected(self, ws: Path) -> None:
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    run_cli(ws, ['tree', 'create', 'main'])

    run_cli(
      ws,
      [
        'experiment',
        'add',
        '--hypothesis',
        'root',
        '--id',
        'pending-exp',
      ],
    )

    # pending-exp is still pending (not terminal), so using it as parent fails
    with pytest.raises(Exception, match='not terminal'):
      run_cli(
        ws,
        [
          'experiment',
          'add',
          '--hypothesis',
          'child',
          '--id',
          'child-exp',
          '--parent',
          'pending-exp',
        ],
      )


class TestWorkflowMultipleTreesQuery:
  """Query on active tree only returns experiments from that tree."""

  def test_query_scoped_to_active_tree(self, ws: Path) -> None:
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)

    # Create tree A with experiment
    run_cli(ws, ['tree', 'create', 'tree-a'])
    run_cli(
      ws,
      [
        'experiment',
        'add',
        '--hypothesis',
        'A',
        '--id',
        'exp-in-a',
      ],
    )

    # Create tree B with experiment
    run_cli(ws, ['tree', 'create', 'tree-b'])
    run_cli(
      ws,
      [
        'experiment',
        'add',
        '--hypothesis',
        'B',
        '--id',
        'exp-in-b',
      ],
    )

    # Active is now tree-b, query should show exp-in-b
    result = run_cli(ws, ['query'])
    exp_ids = [e['id'] for e in result['result']['experiments']]
    assert 'exp-in-b' in exp_ids
    assert 'exp-in-a' not in exp_ids

    # Switch to tree-a, query should show exp-in-a
    run_cli(ws, ['tree', 'switch', 'tree-a', '--no-checkout'])
    result = run_cli(ws, ['query'])
    exp_ids = [e['id'] for e in result['result']['experiments']]
    assert 'exp-in-a' in exp_ids
    assert 'exp-in-b' not in exp_ids


class TestWorkflowCheckoutSetsHead:
  """Checkout sets HEAD and subsequent status uses it."""

  def test_checkout_then_status_uses_head(self, ws: Path) -> None:
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)

    run_cli(ws, ['tree', 'create', 'main'])

    run_cli(
      ws,
      [
        'experiment',
        'add',
        '--hypothesis',
        'first',
        '--id',
        'first-exp',
      ],
    )

    # Complete first-exp
    store = FileStore(config)
    forest = FileForest(store)
    tree = forest.active
    assert tree is not None
    node = tree.get('first-exp')
    assert node is not None
    node.experiment.start()
    node.experiment.complete(metrics={'val': 0.5})
    forest.save()

    run_cli(
      ws,
      [
        'experiment',
        'add',
        '--hypothesis',
        'second',
        '--id',
        'second-exp',
        '--parent',
        'first-exp',
      ],
    )

    # Checkout first-exp
    run_cli(ws, ['checkout', 'first-exp'])

    # Status without ID uses HEAD
    result = run_cli(ws, ['experiment', 'status'])
    assert result['result']['id'] == 'first-exp'

    # Checkout second-exp
    run_cli(ws, ['checkout', 'second-exp'])
    result = run_cli(ws, ['experiment', 'status'])
    assert result['result']['id'] == 'second-exp'
