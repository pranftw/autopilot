"""Tests for experiment lineage command.

Verifies ancestor chain traversal, cross-tree resolution, JSON shape,
metrics inclusion, and error handling for unknown experiment IDs.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli_no_context
import pytest


@pytest.fixture
def lineage_workspace(tmp_path: Path) -> Path:
  """Workspace root for lineage tests."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  return ws


@pytest.fixture
def lineage_store(lineage_workspace: Path) -> FileStore:
  """FileStore for lineage tests."""
  config = AutoPilotConfig(workspace=lineage_workspace)
  config.store_path.mkdir(parents=True, exist_ok=True)
  return FileStore(config)


@pytest.fixture
def lineage_forest(lineage_store: FileStore) -> FileForest:
  """Forest with a three-deep chain for lineage testing.

  Chain: root -> parent -> child (active tree 'main').
  """
  forest = FileForest(lineage_store)
  tree = forest.create_tree('main')

  exp_root = Experiment(experiment_id='exp-root', hypothesis='root experiment')
  exp_root.start()
  exp_root.complete(metrics={'accuracy': 0.7, 'loss': 0.3})
  node_root = Node(experiment=exp_root)
  tree.add(node_root)

  exp_parent = Experiment(experiment_id='exp-parent', hypothesis='parent experiment')
  exp_parent.start()
  exp_parent.complete(metrics={'accuracy': 0.8, 'loss': 0.2})
  node_parent = Node(experiment=exp_parent, parent=node_root)
  tree.add(node_parent)

  exp_child = Experiment(experiment_id='exp-child', hypothesis='child experiment')
  exp_child.start()
  exp_child.complete(metrics={'accuracy': 0.9, 'loss': 0.1})
  node_child = Node(experiment=exp_child, parent=node_parent)
  tree.add(node_child)

  forest.switch('main')
  forest.save()
  return forest


def test_lineage_full_chain(lineage_workspace: Path, lineage_forest: FileForest) -> None:
  """Three-deep chain yields depth == 2 and ordered ancestors."""
  result = run_cli_no_context(lineage_workspace, ['experiment', 'lineage', 'exp-child'])

  assert result['ok'] is True
  payload = result['result']
  assert payload['experiment_id'] == 'exp-child'
  assert payload['tree'] == 'main'
  assert payload['depth'] == 2
  assert len(payload['ancestors']) == 2
  assert payload['ancestors'][0]['id'] == 'exp-parent'
  assert payload['ancestors'][1]['id'] == 'exp-root'


def test_lineage_root_experiment(lineage_workspace: Path, lineage_forest: FileForest) -> None:
  """Root experiment (parent is None) yields depth == 0 and empty ancestors."""
  result = run_cli_no_context(lineage_workspace, ['experiment', 'lineage', 'exp-root'])

  assert result['ok'] is True
  payload = result['result']
  assert payload['experiment_id'] == 'exp-root'
  assert payload['depth'] == 0
  assert payload['ancestors'] == []


def test_lineage_single_parent(lineage_workspace: Path, lineage_forest: FileForest) -> None:
  """Experiment with exactly one ancestor yields depth == 1."""
  result = run_cli_no_context(lineage_workspace, ['experiment', 'lineage', 'exp-parent'])

  assert result['ok'] is True
  payload = result['result']
  assert payload['experiment_id'] == 'exp-parent'
  assert payload['depth'] == 1
  assert len(payload['ancestors']) == 1
  assert payload['ancestors'][0]['id'] == 'exp-root'


def test_lineage_cross_tree(tmp_path: Path) -> None:
  """Lineage resolves experiment on non-active tree without switching."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  tree_a = forest.create_tree('alpha')
  exp_a = Experiment(experiment_id='exp-alpha', hypothesis='alpha')
  exp_a.start()
  exp_a.complete(metrics={'score': 0.5})
  tree_a.add(Node(experiment=exp_a))

  tree_b = forest.create_tree('beta')
  exp_b_root = Experiment(experiment_id='exp-b-root', hypothesis='beta root')
  exp_b_root.start()
  exp_b_root.complete(metrics={'score': 0.6})
  node_b_root = Node(experiment=exp_b_root)
  tree_b.add(node_b_root)

  exp_b_child = Experiment(experiment_id='exp-b-child', hypothesis='beta child')
  exp_b_child.start()
  exp_b_child.complete(metrics={'score': 0.7})
  node_b_child = Node(experiment=exp_b_child, parent=node_b_root)
  tree_b.add(node_b_child)

  forest.switch('alpha')
  forest.save()

  result = run_cli_no_context(ws, ['experiment', 'lineage', 'exp-b-child'])

  assert result['ok'] is True
  payload = result['result']
  assert payload['tree'] == 'beta'
  assert payload['depth'] == 1
  assert payload['ancestors'][0]['id'] == 'exp-b-root'


def test_lineage_includes_metrics(lineage_workspace: Path, lineage_forest: FileForest) -> None:
  """Each ancestor dict includes metrics matching the seeded values."""
  result = run_cli_no_context(lineage_workspace, ['experiment', 'lineage', 'exp-child'])

  assert result['ok'] is True
  ancestors = result['result']['ancestors']

  parent_entry = ancestors[0]
  assert parent_entry['metrics'] == {'accuracy': 0.8, 'loss': 0.2}

  root_entry = ancestors[1]
  assert root_entry['metrics'] == {'accuracy': 0.7, 'loss': 0.3}


def test_lineage_not_found(lineage_workspace: Path, lineage_forest: FileForest) -> None:
  """Unknown experiment id exits non-zero via ctx.fail."""
  with pytest.raises(SystemExit) as exc_info:
    run_cli_no_context(lineage_workspace, ['experiment', 'lineage', 'nonexistent-id'])

  assert exc_info.value.code != 0


def test_lineage_json_shape(lineage_workspace: Path, lineage_forest: FileForest) -> None:
  """Full JSON shape has correct structure and types."""
  result = run_cli_no_context(lineage_workspace, ['experiment', 'lineage', 'exp-child'])

  assert 'ok' in result
  assert result['ok'] is True
  assert 'result' in result
  assert 'messages' in result

  payload = result['result']
  assert set(payload.keys()) == {'experiment_id', 'tree', 'depth', 'ancestors'}
  assert isinstance(payload['experiment_id'], str)
  assert isinstance(payload['tree'], str)
  assert isinstance(payload['depth'], int)
  assert isinstance(payload['ancestors'], list)

  for ancestor in payload['ancestors']:
    assert set(ancestor.keys()) == {'id', 'status', 'metrics'}
    assert isinstance(ancestor['id'], str)
    assert isinstance(ancestor['status'], str)
    assert isinstance(ancestor['metrics'], dict)


def test_lineage_context_exempt(lineage_workspace: Path, lineage_forest: FileForest) -> None:
  """Lineage command succeeds without --context (read-only, context-exempt)."""
  result = run_cli_no_context(lineage_workspace, ['experiment', 'lineage', 'exp-child'])
  assert result['ok'] is True
  assert result['result']['experiment_id'] == 'exp-child'
