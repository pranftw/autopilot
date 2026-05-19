"""Tests for ``query --best`` JSON enrichment (Sub-plan 03, section 2.2).

Verifies that the ``--best`` JSON result includes ``deployed_as`` and ``tree``
fields for agent-consumable output.
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
def query_workspace(tmp_path: Path) -> Path:
  """Workspace with a completed experiment for query --best tests."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.init_workspace()
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  tree = forest.create_tree('main')

  exp = Experiment(experiment_id='exp-best', hypothesis='best candidate')
  exp.start()
  exp.complete(metrics={'accuracy': 0.95})
  tree.add(Node(experiment=exp))

  forest.switch('main')
  forest.save()
  return ws


@pytest.fixture
def query_workspace_deployed(tmp_path: Path) -> Path:
  """Workspace with a deployed experiment for query --best tests."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.init_workspace()
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  tree = forest.create_tree('main')

  exp = Experiment(experiment_id='exp-deployed', hypothesis='deployed candidate')
  exp.start()
  exp.complete(metrics={'accuracy': 0.99})
  node = Node(experiment=exp, deployed_as='production')
  tree.add(node)

  forest.switch('main')
  forest.save()
  return ws


class TestQueryBestIncludesDeployedAs:
  """query --best JSON includes deployed_as field."""

  def test_query_best_includes_deployed_as(self, query_workspace_deployed: Path) -> None:
    """Deployed experiment shows deployed_as with the label string."""
    envelope = run_cli_no_context(
      query_workspace_deployed,
      ['query', '--best', 'accuracy', '--completed'],
    )
    result = envelope.get('result', envelope)
    best = result['best']
    assert best is not None
    assert best['deployed_as'] == 'production'

  def test_query_best_deployed_as_null(self, query_workspace: Path) -> None:
    """Non-deployed experiment shows deployed_as as null."""
    envelope = run_cli_no_context(
      query_workspace,
      ['query', '--best', 'accuracy', '--completed'],
    )
    result = envelope.get('result', envelope)
    best = result['best']
    assert best is not None
    assert best['deployed_as'] is None


class TestQueryBestIncludesTree:
  """query --best JSON includes tree field."""

  def test_query_best_includes_tree(self, query_workspace: Path) -> None:
    """tree field equals the active tree name in single-tree mode."""
    envelope = run_cli_no_context(
      query_workspace,
      ['query', '--best', 'accuracy', '--completed'],
    )
    result = envelope.get('result', envelope)
    best = result['best']
    assert best is not None
    assert best['tree'] == 'main'

  def test_query_best_includes_tree_all_trees(self, query_workspace: Path) -> None:
    """tree field present in --all-trees mode."""
    envelope = run_cli_no_context(
      query_workspace,
      ['query', '--best', 'accuracy', '--completed', '--all-trees'],
    )
    result = envelope.get('result', envelope)
    best = result['best']
    assert best is not None
    assert best['tree'] == 'main'
