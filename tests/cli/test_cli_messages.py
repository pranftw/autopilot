"""Tests for autopilot.cli.messages constants."""

from autopilot.cli import messages
import inspect


def _public_msg_constants() -> dict[str, str]:
  """Collect all public MSG_* string constants from the messages module."""
  return {
    name: value
    for name, value in inspect.getmembers(messages)
    if name.startswith('MSG_') and isinstance(value, str)
  }


class TestMessageConstantsBasic:
  """Smoke tests: every MSG_* constant is a non-empty string."""

  def test_all_msg_constants_non_empty(self) -> None:
    constants = _public_msg_constants()
    assert len(constants) >= 4
    for name, value in constants.items():
      assert value, f'{name} is empty'

  def test_all_msg_constants_unique(self) -> None:
    constants = _public_msg_constants()
    values = list(constants.values())
    assert len(values) == len(set(values)), 'duplicate MSG_* values detected'


class TestMessageConstantsMatchOriginals:
  """Regression lock: critical MSG_* values match the exact inline strings
  they replaced from command handlers."""

  def test_experiment_slug_required_contains_flag(self) -> None:
    assert '--experiment' in messages.MSG_EXPERIMENT_SLUG_REQUIRED

  def test_experiment_slug_required_contains_action(self) -> None:
    assert 'experiment slug required' in messages.MSG_EXPERIMENT_SLUG_REQUIRED

  def test_no_active_tree_exact_text(self) -> None:
    assert messages.MSG_NO_ACTIVE_TREE == ('no active tree; create or switch to a tree first')

  def test_no_module_configured_contains_core_phrase(self) -> None:
    assert 'no module configured' in messages.MSG_NO_MODULE_CONFIGURED

  def test_no_trainer_configured_contains_core_phrase(self) -> None:
    assert 'no trainer configured' in messages.MSG_NO_TRAINER_CONFIGURED

  def test_epoch_required_exact_text(self) -> None:
    assert messages.MSG_EPOCH_REQUIRED == '--epoch is required'

  def test_epoch_invalid_is_template(self) -> None:
    rendered = messages.MSG_EPOCH_INVALID.format(value='bad')
    assert 'bad' in rendered
    assert 'latest' in rendered

  def test_epoch_not_found_is_template(self) -> None:
    rendered = messages.MSG_EPOCH_NOT_FOUND.format(
      epoch=5,
      experiment_id='exp-1',
      latest=3,
    )
    assert '5' in rendered
    assert 'exp-1' in rendered
    assert '3' in rendered

  def test_epoch_empty_store_is_template(self) -> None:
    rendered = messages.MSG_EPOCH_EMPTY_STORE.format(experiment_id='exp-1')
    assert 'exp-1' in rendered
    assert 'latest' in rendered


class TestMsgPrefixConvention:
  """All public string constants use the MSG_ prefix."""

  def test_no_non_msg_string_constants(self) -> None:
    for name, value in inspect.getmembers(messages):
      if name.startswith('_') or not isinstance(value, str):
        continue
      assert name.startswith('MSG_'), f'{name} is a public string but does not have the MSG_ prefix'
