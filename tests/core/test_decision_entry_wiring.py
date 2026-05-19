"""Tests for DecisionEntry wiring in rollback and context log filtering.

Verifies that ``Experiment.rollback()`` emits metadata built via
``DecisionEntry.rollback()`` (typed with ``_type='rollback'``) and that
mixed context entries can be filtered by ``_type`` discriminator.
"""

from autopilot.core.decision import DecisionEntry
from autopilot.core.experiment import Experiment
from unittest.mock import MagicMock


def _make_experiment_with_store(epoch: int = 1) -> Experiment:
  """Build a running experiment with a mock store for rollback testing."""
  exp = Experiment(experiment_id='exp-rollback')
  exp.start()
  exp.epoch = epoch
  store = MagicMock()
  exp.store = store
  return exp


class TestRollbackContextMetadata:
  """Rollback context entry metadata shape tests."""

  def test_rollback_context_has_type(self) -> None:
    exp = _make_experiment_with_store(epoch=3)
    exp.rollback(1)
    entries = list(exp.context_log)
    assert len(entries) == 1
    assert entries[0].metadata['_type'] == DecisionEntry.ROLLBACK_TYPE

  def test_rollback_metadata_has_target_epoch(self) -> None:
    exp = _make_experiment_with_store(epoch=5)
    exp.rollback(2)
    entries = list(exp.context_log)
    assert len(entries) == 1
    metadata = entries[0].metadata
    assert metadata['target_epoch'] == 2

  def test_rollback_metadata_has_reason(self) -> None:
    exp = _make_experiment_with_store(epoch=4)
    exp.rollback(1)
    entries = list(exp.context_log)
    metadata = entries[0].metadata
    assert metadata['reason'] == 'rolled back to epoch 1'

  def test_rollback_reason_string_preserved(self) -> None:
    exp = _make_experiment_with_store(epoch=3)
    exp.rollback(0)
    entries = list(exp.context_log)
    assert entries[0].reason == 'rolled back to epoch 0'

  def test_rollback_no_metrics_before_key(self) -> None:
    exp = _make_experiment_with_store(epoch=3)
    exp.rollback(1)
    entries = list(exp.context_log)
    metadata = entries[0].metadata
    assert 'metrics_before' not in metadata


class TestDecisionEntryFilterable:
  """Mixed entry filtering by ``_type`` discriminator."""

  def test_decision_entry_filterable_by_type(self) -> None:
    exp = Experiment(experiment_id='exp-filter')
    exp.start()

    exp.add_context(
      'policy accepted',
      source='policy',
      metadata=DecisionEntry.policy_gate(
        gate_name='MinGate',
        passed=True,
        value=0.9,
        threshold='>= 0.8',
      ),
    )

    store = MagicMock()
    exp.store = store
    exp.epoch = 3
    exp.rollback(1)

    exp.add_context(
      'deployed to prod',
      source='deployment',
      metadata=DecisionEntry.deployment(
        label='production',
        experiment_id='exp-filter',
      ),
    )

    all_entries = list(exp.context_log)
    assert len(all_entries) == 3

    policy_entries = [
      e for e in all_entries if e.metadata.get('_type') == DecisionEntry.POLICY_GATE_TYPE
    ]
    assert len(policy_entries) == 1
    assert policy_entries[0].reason == 'policy accepted'

    rollback_entries = [
      e for e in all_entries if e.metadata.get('_type') == DecisionEntry.ROLLBACK_TYPE
    ]
    assert len(rollback_entries) == 1
    assert rollback_entries[0].metadata['target_epoch'] == 1

    deployment_entries = [
      e for e in all_entries if e.metadata.get('_type') == DecisionEntry.DEPLOYMENT_TYPE
    ]
    assert len(deployment_entries) == 1
    assert deployment_entries[0].reason == 'deployed to prod'

  def test_filter_excludes_untyped_entries(self) -> None:
    exp = Experiment(experiment_id='exp-mixed')
    exp.start()

    exp.add_context('untyped note', source='user')

    store = MagicMock()
    exp.store = store
    exp.epoch = 2
    exp.rollback(0)

    all_entries = list(exp.context_log)
    assert len(all_entries) == 2

    typed_entries = [e for e in all_entries if e.metadata.get('_type') is not None]
    assert len(typed_entries) == 1
    assert typed_entries[0].metadata['_type'] == DecisionEntry.ROLLBACK_TYPE
