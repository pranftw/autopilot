"""Tests for ExperimentImpact tree-children enrichment (plan 10, P1#19).

Verifies that ``experiment impact`` now includes direct tree children
(via ``Node.parent`` pointers) alongside dependency-graph dependents.
"""

from autopilot.ai.forest import FileForest
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli_no_context, seed_tree_with_experiments
import pytest


class TestImpactTreeChildren:
  """Tree-children enrichment in experiment impact."""

  def test_impact_includes_tree_children(
    self,
    cli_workspace: Path,
    cli_forest: FileForest,
  ) -> None:
    """Seed A -> B, C via parent pointers; impact A lists B and C as children."""
    seed_tree_with_experiments(
      cli_forest,
      'main',
      [
        {'id': 'exp-a', 'hypothesis': 'root', 'status': 'completed'},
        {'id': 'exp-b', 'hypothesis': 'child b', 'status': 'pending', 'parent': 'exp-a'},
        {'id': 'exp-c', 'hypothesis': 'child c', 'status': 'pending', 'parent': 'exp-a'},
      ],
    )
    result = run_cli_no_context(cli_workspace, ['experiment', 'impact', 'exp-a'])
    children = result['result']['children']
    assert children == ['exp-b', 'exp-c']
    assert result['result']['dependents'] == []

  def test_impact_both_deps_and_children(
    self,
    cli_workspace: Path,
    cli_forest: FileForest,
  ) -> None:
    """Target X has both a dependency dependent and a tree child."""
    seed_tree_with_experiments(
      cli_forest,
      'main',
      [
        {'id': 'exp-x', 'hypothesis': 'target', 'status': 'completed'},
        {'id': 'exp-child', 'hypothesis': 'child', 'status': 'pending', 'parent': 'exp-x'},
      ],
    )

    tree = cli_forest.active
    assert tree is not None
    dep_exp = Experiment(experiment_id='exp-dep', hypothesis='depends on x')
    dep_exp.start()
    dep_exp.dependencies = ['exp-x']
    dep_node = Node(experiment=dep_exp)
    tree.add(dep_node)
    cli_forest.save()

    result = run_cli_no_context(cli_workspace, ['experiment', 'impact', 'exp-x'])
    assert result['result']['children'] == ['exp-child']
    assert 'exp-dep' in result['result']['dependents']
    assert 'exp-dep' in result['result']['direct_dependents']

  def test_impact_json_includes_children(
    self,
    cli_workspace: Path,
    cli_forest: FileForest,
  ) -> None:
    """JSON envelope contains children key as a list of strings."""
    seed_tree_with_experiments(
      cli_forest,
      'main',
      [
        {'id': 'exp-root', 'hypothesis': 'root', 'status': 'completed'},
        {'id': 'exp-leaf', 'hypothesis': 'leaf', 'status': 'pending', 'parent': 'exp-root'},
      ],
    )
    result = run_cli_no_context(cli_workspace, ['experiment', 'impact', 'exp-root'])
    payload = result['result']
    assert 'children' in payload
    assert isinstance(payload['children'], list)
    assert all(isinstance(c, str) for c in payload['children'])
    assert payload['children'] == ['exp-leaf']
    assert 'experiment_id' in payload
    assert 'dependents' in payload
    assert 'direct_dependents' in payload

  def test_impact_exit_code_missing_experiment(
    self,
    cli_workspace: Path,
    cli_forest: FileForest,
  ) -> None:
    """Impact on a nonexistent experiment id exits non-zero."""
    seed_tree_with_experiments(
      cli_forest,
      'main',
      [
        {'id': 'exp-a', 'hypothesis': 'a', 'status': 'completed'},
      ],
    )
    with pytest.raises(SystemExit):
      run_cli_no_context(cli_workspace, ['experiment', 'impact', 'ghost-id'])

  def test_impact_no_children_empty_list(
    self,
    cli_workspace: Path,
    cli_forest: FileForest,
  ) -> None:
    """Experiment with no children returns empty children list."""
    seed_tree_with_experiments(
      cli_forest,
      'main',
      [
        {'id': 'exp-a', 'hypothesis': 'root', 'status': 'completed'},
        {'id': 'exp-b', 'hypothesis': 'child', 'status': 'pending', 'parent': 'exp-a'},
      ],
    )
    result = run_cli_no_context(cli_workspace, ['experiment', 'impact', 'exp-b'])
    assert result['result']['children'] == []

  def test_impact_children_sorted(
    self,
    cli_workspace: Path,
    cli_forest: FileForest,
  ) -> None:
    """Children are returned in sorted order."""
    seed_tree_with_experiments(
      cli_forest,
      'main',
      [
        {'id': 'exp-parent', 'hypothesis': 'root', 'status': 'completed'},
        {'id': 'exp-z', 'hypothesis': 'z', 'status': 'pending', 'parent': 'exp-parent'},
        {'id': 'exp-a', 'hypothesis': 'a', 'status': 'pending', 'parent': 'exp-parent'},
        {'id': 'exp-m', 'hypothesis': 'm', 'status': 'pending', 'parent': 'exp-parent'},
      ],
    )
    result = run_cli_no_context(cli_workspace, ['experiment', 'impact', 'exp-parent'])
    assert result['result']['children'] == ['exp-a', 'exp-m', 'exp-z']

  def test_impact_children_cross_tree(
    self,
    cli_workspace: Path,
    cli_forest: FileForest,
  ) -> None:
    """Children from different trees are all included (forest-wide scan)."""
    seed_tree_with_experiments(
      cli_forest,
      'tree-alpha',
      [
        {'id': 'exp-root', 'hypothesis': 'root', 'status': 'completed'},
        {'id': 'exp-child-a', 'hypothesis': 'a', 'status': 'pending', 'parent': 'exp-root'},
      ],
    )
    seed_tree_with_experiments(
      cli_forest,
      'tree-beta',
      [
        {'id': 'exp-root', 'hypothesis': 'root', 'status': 'completed'},
        {'id': 'exp-child-b', 'hypothesis': 'b', 'status': 'pending', 'parent': 'exp-root'},
      ],
    )
    result = run_cli_no_context(cli_workspace, ['experiment', 'impact', 'exp-root'])
    children = result['result']['children']
    assert 'exp-child-a' in children
    assert 'exp-child-b' in children
