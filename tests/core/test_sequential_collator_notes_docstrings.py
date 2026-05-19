"""Tests for Plan 09: core module fixes.

Covers:
  - AOE-002: empty Sequential raises ValueError
  - AgentCollator.build_prompt guard for missing 'data' key
  - ExperimentNotesShow docstring documents positional id requirement
"""

from autopilot.ai.gradient import AgentCollator
from autopilot.cli.commands.experiment.metadata import ExperimentNotesShow
from autopilot.core.module.module import Module
from autopilot.core.ops import Sequential
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from unittest.mock import MagicMock
import pytest


class _PassthroughModule(Module):
  """Trivial module returning a fresh Datum."""

  def forward(self, *args, **kwargs) -> Datum:
    return Datum(items=[Datum()])


class TestEmptySequential:
  """AOE-002: empty Sequential must raise at construction."""

  def test_aoe_empty_sequential_raises(self):
    """Sequential() with no modules raises ValueError."""
    with pytest.raises(ValueError, match='at least one module'):
      Sequential()

  def test_sequential_single_module_ok(self):
    """Sequential with one module constructs without error."""
    seq = Sequential(_PassthroughModule())
    assert 'module_0' in seq._modules

  def test_sequential_forward_still_works(self):
    """Regression: existing Sequential with modules still chains correctly."""
    m1 = _PassthroughModule()
    m2 = _PassthroughModule()
    seq = Sequential(m1, m2)
    result = seq(Datum(items=[Datum()]))
    assert isinstance(result, Datum)


class TestAgentCollatorBuildPrompt:
  """AgentCollator.build_prompt tolerance for malformed feedback."""

  def _make_collator(self) -> AgentCollator:
    agent = MagicMock()
    return AgentCollator(agent=agent)

  def test_agent_collator_build_prompt_missing_data_key(self):
    """Entry with no 'data' key does not raise KeyError."""
    collator = self._make_collator()
    param = Parameter()
    prompt = collator.build_prompt([{}], [param])
    assert 'missing data field' in prompt
    assert 'Item 1' in prompt

  def test_agent_collator_build_prompt_targets_only(self):
    """Entry with only 'targets' key (no 'data') renders fallback heading."""
    collator = self._make_collator()
    param = Parameter()
    prompt = collator.build_prompt([{'targets': Datum(items=[Datum()])}], [param])
    assert 'missing data field' in prompt

  def test_agent_collator_build_prompt_preserves_existing_path(self):
    """Regression: entry with 'data' as a Datum still renders item id."""
    collator = self._make_collator()
    param = Parameter()
    data = Datum(items=[Datum()])
    prompt = collator.build_prompt([{'data': data}], [param])
    assert f'id: {data.id}' in prompt
    assert 'missing data field' not in prompt

  def test_agent_collator_build_prompt_none_data_value(self):
    """Entry with data=None falls through to missing data branch."""
    collator = self._make_collator()
    param = Parameter()
    prompt = collator.build_prompt([{'data': None}], [param])
    assert 'missing data field' in prompt


class TestExperimentNotesShowDocstring:
  """ExperimentNotesShow docstring documents positional id."""

  def test_experiment_notes_show_docstring_mentions_positional_id(self):
    """Docstring mentions notes show, positional, and HEAD behavior."""
    docstring = ExperimentNotesShow.__doc__
    assert docstring is not None
    lower = docstring.lower()
    assert 'notes show' in lower
    assert 'positional' in lower
    assert 'head' in lower
