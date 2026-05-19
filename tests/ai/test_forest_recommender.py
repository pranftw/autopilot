"""Tests for ForestRecommender -- forest-aware recommendation engine."""

from autopilot.ai.recommend import ForestRecommender
from autopilot.core.comparison import ComparatorMetric, MetricsComparator
from autopilot.core.experiment import Experiment
from autopilot.core.forest import Forest
from autopilot.core.node import Node
from autopilot.core.recommend import Recommender
from autopilot.core.store.base import Store
from tests.doubles import make_completed_experiment
import pytest


def _make_forest() -> Forest:
  """Create a Forest with a dummy store for testing."""
  store = Store.__new__(Store)
  store._trees = {}
  return Forest(store)


def _make_comparator(metric: str, higher_is_better: bool = True) -> MetricsComparator:
  """Build a comparator for a single metric."""
  return MetricsComparator([ComparatorMetric(metric, higher_is_better)])


class TestForestRecommenderDeploy:
  """Tests for the deploy recommendation path."""

  def test_forest_recommender_deploy(self) -> None:
    """Strong leader with significant delta produces deploy recommendation."""
    forest = _make_forest()
    tree = forest.create_tree('main')
    forest.switch('main')

    tree.add(Node(experiment=make_completed_experiment('exp-1', 'exp-1', {'score': 0.5})))
    tree.add(Node(experiment=make_completed_experiment('exp-2', 'exp-2', {'score': 0.95})))

    comparator = _make_comparator('score')
    recommender = ForestRecommender('score')
    rec = recommender.recommend(forest, comparator)

    assert rec.action == 'deploy'
    assert rec.experiment_id == 'exp-2'
    assert rec.confidence == 'high'


class TestForestRecommenderContinue:
  """Tests for the continue recommendation path."""

  def test_forest_recommender_continue(self) -> None:
    """Single completed experiment produces continue with low confidence."""
    forest = _make_forest()
    tree = forest.create_tree('main')
    forest.switch('main')

    tree.add(Node(experiment=make_completed_experiment('exp-1', 'exp-1', {'score': 0.7})))

    comparator = _make_comparator('score')
    recommender = ForestRecommender('score')
    rec = recommender.recommend(forest, comparator)

    assert rec.action == 'continue'
    assert rec.experiment_id == 'exp-1'
    assert rec.confidence == 'low'


class TestForestRecommenderRollback:
  """Tests for the rollback recommendation path."""

  def test_forest_recommender_rollback(self) -> None:
    """Regression in latest experiment triggers rollback recommendation."""
    forest = _make_forest()
    tree = forest.create_tree('main')
    forest.switch('main')

    tree.add(Node(experiment=make_completed_experiment('exp-1', 'exp-1', {'score': 0.85})))
    tree.add(Node(experiment=make_completed_experiment('exp-2', 'exp-2', {'score': 0.9})))
    tree.add(Node(experiment=make_completed_experiment('exp-3', 'exp-3', {'score': 0.4})))

    comparator = _make_comparator('score')
    recommender = ForestRecommender('score')
    rec = recommender.recommend(forest, comparator)

    assert rec.action == 'rollback'
    assert rec.confidence == 'medium'


class TestForestRecommenderInvestigate:
  """Tests for the investigate recommendation path."""

  def test_forest_recommender_investigate(self) -> None:
    """No completed experiments produces investigate recommendation."""
    forest = _make_forest()
    tree = forest.create_tree('main')
    forest.switch('main')

    exp = Experiment(experiment_id='exp-1')
    exp.start()
    tree.add(Node(experiment=exp))

    comparator = _make_comparator('score')
    recommender = ForestRecommender('score')
    rec = recommender.recommend(forest, comparator)

    assert rec.action == 'investigate'
    assert rec.confidence == 'low'
    assert rec.experiment_id is None


class TestForestRecommenderBranch:
  """Tests for the branch recommendation path."""

  def test_forest_recommender_branch(self) -> None:
    """Plateau across experiments produces branch recommendation."""
    forest = _make_forest()
    tree = forest.create_tree('main')
    forest.switch('main')

    tree.add(Node(experiment=make_completed_experiment('exp-1', 'exp-1', {'score': 0.800})))
    tree.add(Node(experiment=make_completed_experiment('exp-2', 'exp-2', {'score': 0.801})))
    tree.add(Node(experiment=make_completed_experiment('exp-3', 'exp-3', {'score': 0.800})))

    comparator = _make_comparator('score')
    recommender = ForestRecommender('score')
    rec = recommender.recommend(forest, comparator)

    assert rec.action == 'branch'
    assert rec.confidence == 'medium'


class TestForestRecommenderScope:
  """Tests for scoped recommendations."""

  def test_recommender_scoped(self) -> None:
    """Scoping to a tree restricts candidate experiments."""
    forest = _make_forest()
    tree_a = forest.create_tree('alpha')
    tree_b = forest.create_tree('beta')
    forest.switch('alpha')

    tree_a.add(Node(experiment=make_completed_experiment('exp-a', 'exp-a', {'score': 0.9})))
    tree_b.add(Node(experiment=make_completed_experiment('exp-b', 'exp-b', {'score': 0.5})))

    comparator = _make_comparator('score')
    recommender = ForestRecommender('score')
    rec = recommender.recommend(forest, comparator, scope='tree:alpha')

    assert rec.experiment_id == 'exp-a'

  def test_recommender_scoped_unknown_tree(self) -> None:
    """Unknown tree name in scope raises ValueError."""
    forest = _make_forest()
    forest.create_tree('main')
    forest.switch('main')

    comparator = _make_comparator('score')
    recommender = ForestRecommender('score')

    with pytest.raises(ValueError, match='unknown tree'):
      recommender.recommend(forest, comparator, scope='tree:nonexistent')


class TestForestRecommenderEvidence:
  """Tests for evidence in recommendations."""

  def test_recommender_evidence_includes_metrics(self) -> None:
    """Evidence dict includes metric data and candidate count."""
    forest = _make_forest()
    tree = forest.create_tree('main')
    forest.switch('main')

    tree.add(Node(experiment=make_completed_experiment('exp-1', 'exp-1', {'score': 0.7})))
    tree.add(Node(experiment=make_completed_experiment('exp-2', 'exp-2', {'score': 0.9})))

    comparator = _make_comparator('score')
    recommender = ForestRecommender('score')
    rec = recommender.recommend(forest, comparator)

    assert 'metrics' in rec.evidence
    assert 'best_value' in rec.evidence
    assert 'candidates_count' in rec.evidence
    assert rec.evidence['candidates_count'] == 2


class TestForestRecommenderEmpty:
  """Tests for empty/missing data scenarios."""

  def test_recommender_empty_forest(self) -> None:
    """Empty forest produces investigate with low confidence."""
    forest = _make_forest()

    comparator = _make_comparator('score')
    recommender = ForestRecommender('score')
    rec = recommender.recommend(forest, comparator)

    assert rec.action == 'investigate'
    assert rec.confidence == 'low'
    assert rec.experiment_id is None

  def test_recommender_missing_metric(self) -> None:
    """Metric not in any experiment produces investigate."""
    forest = _make_forest()
    tree = forest.create_tree('main')
    forest.switch('main')

    tree.add(Node(experiment=make_completed_experiment('exp-1', 'exp-1', {'accuracy': 0.9})))

    comparator = _make_comparator('score')
    recommender = ForestRecommender('score')
    rec = recommender.recommend(forest, comparator)

    assert rec.action == 'investigate'
    assert rec.confidence == 'low'
    assert any('primary metric not found' in r for r in rec.reasoning)


class TestRecommenderSubclass:
  """Tests for Recommender subclassing."""

  def test_forest_recommender_is_recommender_subclass(self) -> None:
    """ForestRecommender is a subclass of Recommender."""
    assert issubclass(ForestRecommender, Recommender)
    recommender = ForestRecommender('score')
    assert isinstance(recommender, Recommender)
