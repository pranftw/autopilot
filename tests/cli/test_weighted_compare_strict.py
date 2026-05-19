"""Tests for strict _compute_weighted_verdict after validation (QUALITY-002).

``_compute_weighted_verdict`` uses strict dict access (``[]``) instead of
``.get()`` with defaults. Callers must invoke ``_validate_weight_metrics``
first; skipping validation surfaces as ``KeyError``.
"""

from autopilot.cli.commands.experiment.compare import _compute_weighted_verdict
import pytest


class TestComputeWeightedVerdictStrict:
  """_compute_weighted_verdict uses strict entry access."""

  def test_compute_weighted_verdict_uses_strict_entry_access(self) -> None:
    """Validated deltas + normalized weights produce expected score/verdict."""
    deltas = [
      {
        'metric': 'accuracy',
        'type': 'numeric',
        'delta': 0.05,
        'higher_is_better': True,
        'baseline': 0.85,
        'candidate': 0.90,
      },
      {
        'metric': 'latency',
        'type': 'numeric',
        'delta': 10.0,
        'higher_is_better': False,
        'baseline': 100.0,
        'candidate': 110.0,
      },
    ]
    weights = {'accuracy': 0.7, 'latency': 0.3}

    score, verdict = _compute_weighted_verdict(deltas, weights)

    expected_score = 0.7 * 1.0 * 0.05 + 0.3 * (-1.0) * 10.0
    assert abs(score - expected_score) < 1e-12
    assert verdict == 'regressed'

  def test_compute_weighted_verdict_improved(self) -> None:
    """All higher-is-better metrics improved -> improved verdict."""
    deltas = [
      {
        'metric': 'f1',
        'type': 'numeric',
        'delta': 0.1,
        'higher_is_better': True,
        'baseline': 0.8,
        'candidate': 0.9,
      },
    ]
    weights = {'f1': 1.0}

    score, verdict = _compute_weighted_verdict(deltas, weights)

    assert score > 0
    assert verdict == 'improved'

  def test_compute_weighted_verdict_inconclusive(self) -> None:
    """Zero delta -> inconclusive verdict."""
    deltas = [
      {
        'metric': 'accuracy',
        'type': 'numeric',
        'delta': 0.0,
        'higher_is_better': True,
        'baseline': 0.9,
        'candidate': 0.9,
      },
    ]
    weights = {'accuracy': 1.0}

    score, verdict = _compute_weighted_verdict(deltas, weights)

    assert abs(score) < 1e-12
    assert verdict == 'inconclusive'

  def test_compute_weighted_verdict_missing_metric_raises_key_error(self) -> None:
    """Weights referencing absent metric -> KeyError (strict contract)."""
    deltas = [
      {
        'metric': 'accuracy',
        'type': 'numeric',
        'delta': 0.05,
        'higher_is_better': True,
        'baseline': 0.85,
        'candidate': 0.90,
      },
    ]
    weights = {'nonexistent': 1.0}

    with pytest.raises(KeyError, match='nonexistent'):
      _compute_weighted_verdict(deltas, weights)
