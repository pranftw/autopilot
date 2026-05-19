"""Tests for Experiment Traceable integration (plan 02).

Verifies that Experiment inherits from Traceable, carries a ContextLog,
serializes context_log in state_dict/load_state_dict (clean break), and
that AutoPilotExperiment MRO includes Traceable.
"""

from autopilot.ai.experiment import AutoPilotExperiment
from autopilot.core.context import ContextEntry, ContextLog
from autopilot.core.enums import Status
from autopilot.core.experiment import Experiment
from autopilot.core.traceable import Traceable


class TestExperimentContextLog:
  """Tests for context_log on the base Experiment."""

  def test_experiment_has_context_log(self) -> None:
    """Fresh Experiment exposes context_log as ContextLog."""
    exp = Experiment('id')
    assert isinstance(exp.context_log, ContextLog)

  def test_experiment_context_log_empty_on_creation(self) -> None:
    """New experiment has len(context_log) == 0."""
    exp = Experiment('id')
    assert len(exp.context_log) == 0

  def test_experiment_add_context(self) -> None:
    """add_context with minimal args increases entry count by one."""
    exp = Experiment('id')
    exp.add_context('test reason')
    assert len(exp.context_log) == 1

  def test_experiment_add_context_with_all_fields(self) -> None:
    """source, command, epoch, metadata passed through to entry."""
    exp = Experiment('id')
    exp.add_context(
      'detailed reason',
      source='user',
      command='experiment create',
      epoch=3,
      metadata={'key': 'value'},
    )
    entry = exp.context_log.entries[0]
    assert entry.reason == 'detailed reason'
    assert entry.source == 'user'
    assert entry.command == 'experiment create'
    assert entry.epoch == 3
    assert entry.metadata == {'key': 'value'}
    assert entry.timestamp

  def test_experiment_state_dict_includes_context_log(self) -> None:
    """state_dict() includes context_log key."""
    exp = Experiment('id')
    state = exp.state_dict()
    assert 'context_log' in state

  def test_experiment_state_dict_context_log_is_list(self) -> None:
    """context_log value is a list of dicts."""
    exp = Experiment('id')
    exp.add_context('entry one')
    state = exp.state_dict()
    assert isinstance(state['context_log'], list)
    for item in state['context_log']:
      assert isinstance(item, dict)

  def test_experiment_load_state_dict_restores_context_log(self) -> None:
    """load_state_dict preserves entries and fields."""
    exp = Experiment('id')
    exp.add_context('reason alpha', source='user')
    state = exp.state_dict()

    exp2 = Experiment('placeholder')
    exp2.load_state_dict(state)
    assert len(exp2.context_log) == 1
    restored = exp2.context_log.entries[0]
    assert restored.reason == 'reason alpha'
    assert restored.source == 'user'

  def test_experiment_load_state_dict_missing_context_log_raises(self) -> None:
    """KeyError when context_log key is absent (clean break)."""
    exp = Experiment('id')
    state = exp.state_dict()
    del state['context_log']
    import pytest

    with pytest.raises(KeyError, match='context_log'):
      exp.load_state_dict(state)

  def test_experiment_state_dict_roundtrip_with_entries(self) -> None:
    """Three heterogeneous entries survive dump + load."""
    exp = Experiment('id')
    exp.add_context('first', source='user', command='create')
    exp.add_context('second', source='policy', epoch=0, metadata={'gate': 'pass'})
    exp.add_context('third', source='trainer', epoch=1)
    state = exp.state_dict()

    exp2 = Experiment('placeholder')
    exp2.load_state_dict(state)
    assert len(exp2.context_log) == 3

    entries = exp2.context_log.entries
    assert entries[0].reason == 'first'
    assert entries[0].source == 'user'
    assert entries[0].command == 'create'
    assert entries[1].reason == 'second'
    assert entries[1].source == 'policy'
    assert entries[1].epoch == 0
    assert entries[1].metadata == {'gate': 'pass'}
    assert entries[2].reason == 'third'
    assert entries[2].source == 'trainer'
    assert entries[2].epoch == 1

  def test_experiment_create_context_log_override(self) -> None:
    """Subclass returning custom ContextLog subtype from create_context_log()."""

    class TaggedLog(ContextLog):
      """Custom log with a tag field."""

      tag: str = 'custom'

    class TaggedExperiment(Experiment):
      """Experiment using a custom log type."""

      def create_context_log(self) -> TaggedLog:
        return TaggedLog()

    exp = TaggedExperiment('id')
    assert isinstance(exp.context_log, TaggedLog)
    assert exp.context_log.tag == 'custom'


class TestAutoPilotExperimentContext:
  """Tests for AutoPilotExperiment Traceable chain."""

  def test_autopilot_experiment_has_context_log(self) -> None:
    """AutoPilotExperiment instance has ContextLog."""
    exp = AutoPilotExperiment('id')
    assert isinstance(exp.context_log, ContextLog)

  def test_autopilot_experiment_mro(self) -> None:
    """MRO includes Traceable before object."""
    expected = (AutoPilotExperiment, Experiment, Traceable, object)
    assert AutoPilotExperiment.__mro__ == expected


class TestExperimentContextRegression:
  """Regression smoke tests for Experiment + Traceable integration."""

  def test_existing_experiment_tests_still_pass(self) -> None:
    """In-process smoke: constructors, state_dict, load_state_dict round-trip."""
    exp = Experiment('regression-test', hypothesis='h')
    exp.start()
    exp.advance_epoch(metrics={'accuracy': 0.8})
    exp.notes = 'some note'
    state = exp.state_dict()

    assert 'context_log' in state
    assert isinstance(state['context_log'], list)

    exp2 = Experiment('placeholder')
    exp2.load_state_dict(state)
    assert exp2.id == 'regression-test'
    assert exp2.hypothesis == 'h'
    assert exp2.status == Status.running
    assert exp2.notes == 'some note'
    assert isinstance(exp2.context_log, ContextLog)

    ape = AutoPilotExperiment('ape-test')
    ape_state = ape.state_dict()
    assert 'context_log' in ape_state

    ape2 = AutoPilotExperiment('placeholder')
    ape2.load_state_dict(ape_state)
    assert ape2.id == 'ape-test'
    assert isinstance(ape2.context_log, ContextLog)

  def test_context_log_empty_after_state_dict_roundtrip(self) -> None:
    """Empty context_log serializes as [] and deserializes cleanly."""
    exp = Experiment('empty-log')
    state = exp.state_dict()
    assert state['context_log'] == []

    exp2 = Experiment('placeholder')
    exp2.load_state_dict(state)
    assert len(exp2.context_log) == 0

  def test_add_context_returns_entry(self) -> None:
    """add_context returns a ContextEntry (inherited from Traceable)."""
    exp = Experiment('id')
    result = exp.add_context('test')
    assert isinstance(result, ContextEntry)
    assert result.reason == 'test'
