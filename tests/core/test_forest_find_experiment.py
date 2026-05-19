"""Tests for Forest.find_experiment cross-tree lookup.

Covers:
  - Active tree is checked first.
  - Non-active tree is found on fallback.
  - Unknown id returns None.
  - Duplicate id across trees: first match (active-first) wins.
  - Active tree is not scanned twice during fallback.
"""

from autopilot.core.experiment import Experiment
from autopilot.core.forest import Forest
from autopilot.core.node import Node
from autopilot.core.store.base import Store
from unittest.mock import MagicMock, patch


def _make_forest() -> Forest:
  """Build an in-memory forest with two trees for testing."""
  store = MagicMock(spec=Store)
  forest = Forest(store)
  tree_a = forest.create_tree('alpha')
  tree_b = forest.create_tree('beta')

  exp_a = Experiment(experiment_id='exp-a', hypothesis='alpha exp')
  exp_a.start()
  exp_a.complete(metrics={'accuracy': 0.9})
  tree_a.add(Node(experiment=exp_a))

  exp_b = Experiment(experiment_id='exp-b', hypothesis='beta exp')
  exp_b.start()
  exp_b.complete(metrics={'accuracy': 0.8})
  tree_b.add(Node(experiment=exp_b))

  forest.switch('alpha')
  return forest


class TestForestFindExperimentActiveTree:
  """Experiment found on the active tree."""

  def test_forest_find_experiment_active_tree(self) -> None:
    """When the experiment lives on the active tree, returns (node, active_tree)."""
    forest = _make_forest()
    result = forest.find_experiment('exp-a')
    assert result is not None
    node, tree = result
    assert node.experiment.id == 'exp-a'
    assert tree is forest.active
    assert tree.name == 'alpha'


class TestForestFindExperimentOtherTree:
  """Experiment found on a non-active tree."""

  def test_forest_find_experiment_other_tree(self) -> None:
    """When the experiment is only on a non-active tree, returns (node, that_tree)."""
    forest = _make_forest()
    result = forest.find_experiment('exp-b')
    assert result is not None
    node, tree = result
    assert node.experiment.id == 'exp-b'
    assert tree.name == 'beta'


class TestForestFindExperimentNotFound:
  """Unknown experiment id returns None."""

  def test_forest_find_experiment_not_found(self) -> None:
    """Returns None for an id that does not exist in any tree."""
    forest = _make_forest()
    assert forest.find_experiment('no-such-exp') is None

  def test_forest_find_experiment_empty_forest(self) -> None:
    """Returns None when forest has no trees."""
    store = MagicMock(spec=Store)
    forest = Forest(store)
    assert forest.find_experiment('anything') is None

  def test_forest_find_experiment_no_active_tree(self) -> None:
    """Returns None when active is None but experiment exists on a tree."""
    store = MagicMock(spec=Store)
    forest = Forest(store)
    tree = forest.create_tree('only')
    exp = Experiment(experiment_id='exp-only', hypothesis='h')
    exp.start()
    exp.complete(metrics={})
    tree.add(Node(experiment=exp))

    result = forest.find_experiment('exp-only')
    assert result is not None
    node, found_tree = result
    assert node.experiment.id == 'exp-only'
    assert found_tree.name == 'only'


class TestForestFindExperimentDuplicateFirstWins:
  """Same id in multiple trees: active-first ordering wins."""

  def test_forest_find_experiment_duplicate_first_wins(self) -> None:
    """When the same experiment id is in both trees, active tree wins."""
    store = MagicMock(spec=Store)
    forest = Forest(store)
    tree_a = forest.create_tree('alpha')
    tree_b = forest.create_tree('beta')

    exp_a = Experiment(experiment_id='dup-exp', hypothesis='alpha copy')
    exp_a.start()
    exp_a.complete(metrics={'accuracy': 0.9})
    tree_a.add(Node(experiment=exp_a))

    exp_b = Experiment(experiment_id='dup-exp', hypothesis='beta copy')
    exp_b.start()
    exp_b.complete(metrics={'accuracy': 0.5})
    tree_b.add(Node(experiment=exp_b))

    forest.switch('alpha')
    result = forest.find_experiment('dup-exp')
    assert result is not None
    node, tree = result
    assert tree.name == 'alpha'
    assert node.experiment.hypothesis == 'alpha copy'

  def test_forest_find_experiment_duplicate_non_active_first_wins(self) -> None:
    """When no active tree, first in iteration order wins."""
    store = MagicMock(spec=Store)
    forest = Forest(store)
    tree_a = forest.create_tree('alpha')
    tree_b = forest.create_tree('beta')

    exp_a = Experiment(experiment_id='dup-exp', hypothesis='alpha copy')
    exp_a.start()
    exp_a.complete(metrics={})
    tree_a.add(Node(experiment=exp_a))

    exp_b = Experiment(experiment_id='dup-exp', hypothesis='beta copy')
    exp_b.start()
    exp_b.complete(metrics={})
    tree_b.add(Node(experiment=exp_b))

    result = forest.find_experiment('dup-exp')
    assert result is not None
    _node, tree = result
    assert tree.name == 'alpha'


class TestForestFindExperimentActiveNotScannedTwice:
  """Active tree nodes are visited exactly once."""

  def test_find_experiment_active_tree_not_scanned_twice(self) -> None:
    """Active tree's get() is called exactly once for a hit on active tree."""
    forest = _make_forest()
    active_tree = forest.active
    assert active_tree is not None

    with patch.object(active_tree, 'get', wraps=active_tree.get) as mock_get:
      forest.find_experiment('exp-a')
      assert mock_get.call_count == 1

  def test_find_experiment_miss_active_not_rescanned(self) -> None:
    """When experiment is on non-active tree, active is checked once only."""
    forest = _make_forest()
    active_tree = forest.active
    assert active_tree is not None

    with patch.object(active_tree, 'get', wraps=active_tree.get) as mock_get:
      result = forest.find_experiment('exp-b')
      assert result is not None
      assert result[1].name == 'beta'
      assert mock_get.call_count == 1
