"""Unit tests for core recommendation types: Recommender and Recommendation."""

from autopilot.core.comparison import ComparatorMetric, MetricsComparator
from autopilot.core.forest import Forest
from autopilot.core.recommend import (
  VALID_CONFIDENCE_LEVELS,
  VALID_RECOMMENDATION_ACTIONS,
  Recommendation,
  Recommender,
)
from autopilot.core.store.base import Store
import pytest


class TestRecommendation:
  """Tests for the Recommendation dataclass."""

  def test_recommendation_roundtrip(self) -> None:
    """to_dict -> from_dict preserves all fields."""
    rec = Recommendation(
      action='deploy',
      experiment_id='exp-1',
      confidence='high',
      reasoning=['strong results', 'consistent improvement'],
      alternatives=['exp-2', 'exp-3'],
      evidence={'score': 0.95, 'delta': 0.1},
    )
    data = rec.to_dict()
    restored = Recommendation.from_dict(data)
    assert restored.action == rec.action
    assert restored.experiment_id == rec.experiment_id
    assert restored.confidence == rec.confidence
    assert restored.reasoning == rec.reasoning
    assert restored.alternatives == rec.alternatives
    assert restored.evidence == rec.evidence

  def test_recommendation_constructor_invalid_action(self) -> None:
    """Invalid action raises ValueError with guidance."""
    with pytest.raises(ValueError, match='Invalid recommendation action'):
      Recommendation(
        action='explode',
        experiment_id=None,
        confidence='high',
      )

  def test_recommendation_constructor_invalid_confidence(self) -> None:
    """Invalid confidence raises ValueError with guidance."""
    with pytest.raises(ValueError, match='Invalid confidence level'):
      Recommendation(
        action='deploy',
        experiment_id=None,
        confidence='extreme',
      )

  def test_recommendation_from_dict_missing_key(self) -> None:
    """from_dict with empty dict raises TypeError (missing required fields)."""
    with pytest.raises(TypeError):
      Recommendation.from_dict({})

  def test_recommendation_defaults(self) -> None:
    """Optional fields default to empty containers."""
    rec = Recommendation(
      action='investigate',
      experiment_id=None,
      confidence='low',
    )
    assert rec.reasoning == []
    assert rec.alternatives == []
    assert rec.evidence == {}


class TestRecommender:
  """Tests for the base Recommender class."""

  def test_recommender_base_not_implemented(self) -> None:
    """Bare Recommender.recommend() raises NotImplementedError."""
    recommender = Recommender()
    store = Store.__new__(Store)
    forest = Forest(store)
    comparator = MetricsComparator(
      [ComparatorMetric('score', higher_is_better=True)],
    )
    with pytest.raises(NotImplementedError):
      recommender.recommend(forest, comparator)

  def test_recommender_subclass(self) -> None:
    """Concrete subclass of Recommender returns a valid Recommendation."""

    class FixedRecommender(Recommender):
      """Always recommends continue."""

      def recommend(self, forest, comparator, *, scope=None):
        return Recommendation(
          action='continue',
          experiment_id='exp-fixed',
          confidence='medium',
          reasoning=['fixed recommendation'],
        )

    recommender = FixedRecommender()
    store = Store.__new__(Store)
    forest = Forest(store)
    comparator = MetricsComparator(
      [ComparatorMetric('score', higher_is_better=True)],
    )
    rec = recommender.recommend(forest, comparator)
    assert rec.action == 'continue'
    assert rec.experiment_id == 'exp-fixed'
    assert rec.confidence == 'medium'


class TestValidationConstants:
  """Tests for module-level validation frozensets."""

  def test_valid_actions_are_frozenset(self) -> None:
    """VALID_RECOMMENDATION_ACTIONS is a frozenset."""
    assert isinstance(VALID_RECOMMENDATION_ACTIONS, frozenset)
    assert 'deploy' in VALID_RECOMMENDATION_ACTIONS
    assert 'rollback' in VALID_RECOMMENDATION_ACTIONS
    assert 'continue' in VALID_RECOMMENDATION_ACTIONS
    assert 'branch' in VALID_RECOMMENDATION_ACTIONS
    assert 'investigate' in VALID_RECOMMENDATION_ACTIONS

  def test_valid_confidence_levels_are_frozenset(self) -> None:
    """VALID_CONFIDENCE_LEVELS is a frozenset."""
    assert isinstance(VALID_CONFIDENCE_LEVELS, frozenset)
    assert 'high' in VALID_CONFIDENCE_LEVELS
    assert 'medium' in VALID_CONFIDENCE_LEVELS
    assert 'low' in VALID_CONFIDENCE_LEVELS
