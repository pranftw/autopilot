"""Tests for DecisionEntry factory class.

Covers shape validation, invalid argument paths, context log integration,
machine-discriminable types, filtering by type, and metadata round-trips.
"""

from autopilot.core.context import ContextEntry
from autopilot.core.decision import DecisionEntry
from autopilot.core.experiment import Experiment
import pytest


class TestDeploymentEntry:
  """Tests for DecisionEntry.deployment factory."""

  def test_deployment_entry_shape(self) -> None:
    """Keys include _type==DEPLOYMENT_TYPE, label, experiment_id, optional fields."""
    result = DecisionEntry.deployment(
      label='production',
      experiment_id='exp-1',
      previous_id='exp-0',
      evidence={'accuracy_delta': 0.05},
    )
    assert result['_type'] == DecisionEntry.DEPLOYMENT_TYPE
    assert result['label'] == 'production'
    assert result['experiment_id'] == 'exp-1'
    assert result['previous_id'] == 'exp-0'
    assert result['evidence'] == {'accuracy_delta': 0.05}

  def test_deployment_entry_optional_fields_absent(self) -> None:
    """Optional fields are absent when not provided."""
    result = DecisionEntry.deployment(label='staging', experiment_id='exp-2')
    assert '_type' in result
    assert 'label' in result
    assert 'experiment_id' in result
    assert 'previous_id' not in result
    assert 'evidence' not in result

  def test_decision_entry_deployment_invalid_args(self) -> None:
    """Empty label raises ValueError."""
    with pytest.raises(ValueError, match='label must not be empty or whitespace-only'):
      DecisionEntry.deployment(label='', experiment_id='exp-1')

  def test_deployment_empty_experiment_id(self) -> None:
    """Empty experiment_id raises ValueError."""
    with pytest.raises(ValueError, match='experiment_id must not be empty or whitespace-only'):
      DecisionEntry.deployment(label='prod', experiment_id='')

  def test_deployment_rejects_whitespace_label(self) -> None:
    """Whitespace-only label raises ValueError."""
    with pytest.raises(ValueError, match='label must not be empty'):
      DecisionEntry.deployment(label='   ', experiment_id='e1')

  def test_deployment_rejects_whitespace_experiment_id(self) -> None:
    """Whitespace-only experiment_id raises ValueError."""
    with pytest.raises(ValueError, match='experiment_id must not be empty'):
      DecisionEntry.deployment(label='prod', experiment_id='   ')


class TestRollbackEntry:
  """Tests for DecisionEntry.rollback factory."""

  def test_rollback_entry_shape(self) -> None:
    """Keys include _type==ROLLBACK_TYPE, target_epoch, reason."""
    result = DecisionEntry.rollback(
      target_epoch=3,
      reason='accuracy dropped below threshold',
      metrics_before={'accuracy': 0.85},
    )
    assert result['_type'] == DecisionEntry.ROLLBACK_TYPE
    assert result['target_epoch'] == 3
    assert result['reason'] == 'accuracy dropped below threshold'
    assert result['metrics_before'] == {'accuracy': 0.85}

  def test_rollback_optional_fields_absent(self) -> None:
    """metrics_before absent when not provided."""
    result = DecisionEntry.rollback(target_epoch=0, reason='regression')
    assert 'metrics_before' not in result

  def test_decision_entry_rollback_invalid_args(self) -> None:
    """Empty reason raises ValueError."""
    with pytest.raises(ValueError, match='reason must not be empty or whitespace-only'):
      DecisionEntry.rollback(target_epoch=1, reason='')

  def test_rollback_rejects_whitespace_reason(self) -> None:
    """Whitespace-only reason raises ValueError."""
    with pytest.raises(ValueError, match='reason must not be empty'):
      DecisionEntry.rollback(target_epoch=0, reason='   ')


class TestComparisonEntry:
  """Tests for DecisionEntry.comparison factory."""

  def test_comparison_entry_shape(self) -> None:
    """Keys include _type==COMPARISON_TYPE, baseline_id, candidate_id, verdict."""
    result = DecisionEntry.comparison(
      baseline_id='exp-a',
      candidate_id='exp-b',
      verdict='improved',
      deltas=[{'metric': 'accuracy', 'delta': 0.03}],
      confidence='high',
    )
    assert result['_type'] == DecisionEntry.COMPARISON_TYPE
    assert result['baseline_id'] == 'exp-a'
    assert result['candidate_id'] == 'exp-b'
    assert result['verdict'] == 'improved'
    assert result['deltas'] == [{'metric': 'accuracy', 'delta': 0.03}]
    assert result['confidence'] == 'high'

  def test_comparison_optional_fields_absent(self) -> None:
    """deltas and confidence absent when not provided."""
    result = DecisionEntry.comparison(
      baseline_id='exp-a', candidate_id='exp-b', verdict='inconclusive'
    )
    assert 'deltas' not in result
    assert 'confidence' not in result

  def test_decision_entry_comparison_invalid_args(self) -> None:
    """Empty baseline_id raises ValueError."""
    with pytest.raises(ValueError, match='baseline_id must not be empty or whitespace-only'):
      DecisionEntry.comparison(baseline_id='', candidate_id='exp-2', verdict='improved')

  def test_comparison_empty_candidate_id(self) -> None:
    """Empty candidate_id raises ValueError."""
    with pytest.raises(ValueError, match='candidate_id must not be empty or whitespace-only'):
      DecisionEntry.comparison(baseline_id='exp-1', candidate_id='', verdict='improved')

  def test_comparison_empty_verdict(self) -> None:
    """Empty verdict raises ValueError."""
    with pytest.raises(ValueError, match='verdict must not be empty or whitespace-only'):
      DecisionEntry.comparison(baseline_id='exp-1', candidate_id='exp-2', verdict='')

  def test_comparison_rejects_whitespace_baseline_id(self) -> None:
    """Whitespace-only baseline_id raises ValueError."""
    with pytest.raises(ValueError, match='baseline_id must not be empty'):
      DecisionEntry.comparison(baseline_id='   ', candidate_id='c', verdict='v')

  def test_comparison_rejects_whitespace_candidate_id(self) -> None:
    """Whitespace-only candidate_id raises ValueError."""
    with pytest.raises(ValueError, match='candidate_id must not be empty'):
      DecisionEntry.comparison(baseline_id='b', candidate_id='   ', verdict='v')

  def test_comparison_rejects_whitespace_verdict(self) -> None:
    """Whitespace-only verdict raises ValueError."""
    with pytest.raises(ValueError, match='verdict must not be empty'):
      DecisionEntry.comparison(baseline_id='b', candidate_id='c', verdict='   ')


class TestPolicyGateEntry:
  """Tests for DecisionEntry.policy_gate factory."""

  def test_policy_gate_entry_shape(self) -> None:
    """Keys include _type==POLICY_GATE_TYPE, gate_name, passed, value, threshold."""
    result = DecisionEntry.policy_gate(
      gate_name='accuracy_gate',
      passed=True,
      value=0.95,
      threshold='>= 0.90',
    )
    assert result['_type'] == DecisionEntry.POLICY_GATE_TYPE
    assert result['gate_name'] == 'accuracy_gate'
    assert result['passed'] is True
    assert result['value'] == 0.95
    assert result['threshold'] == '>= 0.90'

  def test_policy_gate_optional_fields_absent(self) -> None:
    """value and threshold absent when not provided."""
    result = DecisionEntry.policy_gate(gate_name='budget', passed=False)
    assert 'value' not in result
    assert 'threshold' not in result

  def test_decision_entry_policy_gate_invalid_args(self) -> None:
    """Empty gate_name raises ValueError."""
    with pytest.raises(ValueError, match='gate_name must not be empty or whitespace-only'):
      DecisionEntry.policy_gate(gate_name='', passed=True)

  def test_policy_gate_rejects_whitespace_gate_name(self) -> None:
    """Whitespace-only gate_name raises ValueError."""
    with pytest.raises(ValueError, match='gate_name must not be empty'):
      DecisionEntry.policy_gate(gate_name='   ', passed=True)


class TestPlateauStopEntry:
  """Tests for DecisionEntry.plateau_stop factory."""

  def test_plateau_stop_entry_shape(self) -> None:
    """Keys include _type==PLATEAU_STOP_TYPE and all plateau fields."""
    result = DecisionEntry.plateau_stop(
      'accuracy',
      2,
      plateau_window=3,
      plateau_threshold=0.05,
      values=[0.5, 0.5, 0.5],
    )
    assert result['_type'] == DecisionEntry.PLATEAU_STOP_TYPE
    assert result['monitor'] == 'accuracy'
    assert result['epoch'] == 2
    assert result['plateau_window'] == 3
    assert result['plateau_threshold'] == 0.05
    assert result['values'] == [0.5, 0.5, 0.5]

  def test_plateau_stop_invalid_monitor(self) -> None:
    """Empty monitor raises ValueError."""
    with pytest.raises(ValueError, match='monitor must not be empty'):
      DecisionEntry.plateau_stop(
        '',
        0,
        plateau_window=3,
        plateau_threshold=0.05,
        values=[0.5, 0.5, 0.5],
      )

  def test_plateau_stop_whitespace_monitor(self) -> None:
    """Whitespace-only monitor raises ValueError."""
    with pytest.raises(ValueError, match='monitor must not be empty'):
      DecisionEntry.plateau_stop(
        '   ',
        0,
        plateau_window=3,
        plateau_threshold=0.05,
        values=[0.5, 0.5, 0.5],
      )


class TestContextLogIntegration:
  """Tests for DecisionEntry integration with ContextEntry / Experiment context log."""

  def test_decision_entry_in_context_log(self) -> None:
    """add_context stores metadata dict accurately in experiment context log."""
    exp = Experiment('test-exp', hypothesis='test')
    metadata = DecisionEntry.deployment(
      label='staging', experiment_id='test-exp', previous_id='prev-exp'
    )
    exp.add_context('deployed to staging', source='deployment', metadata=metadata)

    entries = exp.context_log.entries
    assert len(entries) == 1
    assert entries[0].reason == 'deployed to staging'
    assert entries[0].source == 'deployment'
    assert entries[0].metadata['_type'] == DecisionEntry.DEPLOYMENT_TYPE
    assert entries[0].metadata['label'] == 'staging'
    assert entries[0].metadata['experiment_id'] == 'test-exp'
    assert entries[0].metadata['previous_id'] == 'prev-exp'

  def test_decision_entry_machine_discriminable(self) -> None:
    """Different factory methods yield different _type values."""
    types = {
      DecisionEntry.deployment(label='x', experiment_id='e')['_type'],
      DecisionEntry.rollback(target_epoch=0, reason='r')['_type'],
      DecisionEntry.comparison(baseline_id='a', candidate_id='b', verdict='v')['_type'],
      DecisionEntry.policy_gate(gate_name='g', passed=True)['_type'],
      DecisionEntry.plateau_stop('acc', 0, plateau_window=3, plateau_threshold=0.01, values=[0.5])[
        '_type'
      ],
    }
    assert len(types) == 5

  def test_decision_entry_filter_by_type(self) -> None:
    """Filter context entries by _type to find specific decision kinds."""
    exp = Experiment('filter-test', hypothesis='test')

    exp.add_context(
      'deployed',
      source='deployment',
      metadata=DecisionEntry.deployment(label='prod', experiment_id='e1'),
    )
    exp.add_context(
      'compared',
      source='comparison',
      metadata=DecisionEntry.comparison(baseline_id='e0', candidate_id='e1', verdict='improved'),
    )
    exp.add_context(
      'gate checked',
      source='policy',
      metadata=DecisionEntry.policy_gate(gate_name='budget', passed=True),
    )

    comparisons = [
      e for e in exp.context_log if e.metadata.get('_type') == DecisionEntry.COMPARISON_TYPE
    ]
    assert len(comparisons) == 1
    assert comparisons[0].metadata['verdict'] == 'improved'

  def test_decision_entry_metadata_roundtrip(self) -> None:
    """Dict equality survives context log serialization round-trip."""
    original = DecisionEntry.rollback(
      target_epoch=5,
      reason='latency spike',
      metrics_before={'latency_p99': 450.0},
    )
    entry = ContextEntry.create(
      'rolling back due to latency',
      source='trainer',
      metadata=original,
    )
    serialized = entry.to_dict()
    restored = ContextEntry.from_dict(serialized)
    assert restored.metadata == original
    assert restored.metadata['_type'] == DecisionEntry.ROLLBACK_TYPE
    assert restored.metadata['target_epoch'] == 5
    assert restored.metadata['metrics_before'] == {'latency_p99': 450.0}
