"""Tests for Experiment.spec_version tracking (Plan 21).

Covers:
  - state_dict / load_state_dict round-trip
  - Legacy dict without spec_version loads as None
  - Setting spec_version on Experiment instances
"""

from autopilot.core.experiment import Experiment


class TestSpecVersionRoundTrip:
  """Experiment.spec_version serialization round-trips."""

  def test_state_dict_preserves_string(self) -> None:
    exp = Experiment(experiment_id='e1')
    exp.spec_version = 'v2.1'
    state = exp.state_dict()
    assert state['spec_version'] == 'v2.1'

    restored = Experiment(experiment_id='tmp')
    restored.load_state_dict(state)
    assert restored.spec_version == 'v2.1'

  def test_state_dict_preserves_none(self) -> None:
    exp = Experiment(experiment_id='e2')
    assert exp.spec_version is None
    state = exp.state_dict()
    assert state['spec_version'] is None

    restored = Experiment(experiment_id='tmp')
    restored.load_state_dict(state)
    assert restored.spec_version is None

  def test_load_legacy_without_spec_version(self) -> None:
    """Missing key in persisted state deserializes as None."""
    exp = Experiment(experiment_id='e3')
    state = exp.state_dict()
    del state['spec_version']

    restored = Experiment(experiment_id='tmp')
    restored.load_state_dict(state)
    assert restored.spec_version is None

  def test_full_round_trip_with_version(self) -> None:
    exp = Experiment(experiment_id='e4', hypothesis='test hypo')
    exp.spec_version = '2024-01-schema'
    exp.start()
    exp.complete(metrics={'accuracy': 0.95})

    state = exp.state_dict()
    restored = Experiment(experiment_id='tmp')
    restored.load_state_dict(state)

    assert restored.spec_version == '2024-01-schema'
    assert restored.metrics == {'accuracy': 0.95}
    assert restored.id == 'e4'

  def test_default_is_none(self) -> None:
    exp = Experiment(experiment_id='fresh')
    assert exp.spec_version is None
