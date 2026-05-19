"""Tests for --all-trees query without an active tree (Plan 26 S2.2).

Verifies that ``--all-trees --json`` succeeds even when ``forest.active``
is None (no tree has been switched to).
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
def ws_no_active(tmp_path: Path) -> Path:
  """Workspace with trees but no active tree (active=null in forest.json)."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  tree_a = forest.create_tree('alpha')
  exp_a = Experiment(experiment_id='exp-alpha', hypothesis='alpha test')
  exp_a.start()
  exp_a.complete(metrics={'accuracy': 0.9})
  tree_a.add(Node(experiment=exp_a))

  tree_b = forest.create_tree('beta')
  exp_b = Experiment(experiment_id='exp-beta', hypothesis='beta test')
  exp_b.start()
  exp_b.complete(metrics={'accuracy': 0.8})
  tree_b.add(Node(experiment=exp_b))

  forest._active_name = None
  forest.save()
  return ws


class TestQueryAllTreesWithoutActiveTree:
  def test_all_trees_json_succeeds(self, ws_no_active: Path) -> None:
    """--all-trees --json query succeeds when forest.active is None."""
    result = run_cli_no_context(ws_no_active, ['query', '--all-trees'])
    assert result['ok'] is True
    experiments = result['result']['experiments']
    assert len(experiments) == 2

  def test_all_trees_includes_tree_attribution(self, ws_no_active: Path) -> None:
    """Each experiment row includes a 'tree' field for attribution."""
    result = run_cli_no_context(ws_no_active, ['query', '--all-trees'])
    experiments = result['result']['experiments']
    for exp in experiments:
      assert 'tree' in exp
    tree_names = {exp['tree'] for exp in experiments}
    assert tree_names == {'alpha', 'beta'}

  def test_single_tree_query_without_active_fails(self, ws_no_active: Path) -> None:
    """Default single-tree query fails when no active tree is set."""
    with pytest.raises(SystemExit):
      run_cli_no_context(ws_no_active, ['query'])
