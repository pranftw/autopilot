"""Tests for harness.simulator (UserSimulator)."""

from harness.simulator import MAX_USER_TURNS, UserSimulator
from unittest.mock import patch


class TestTurnBased:
  def test_turn_based_sends_turns(self) -> None:
    """With user_turns=['a','b'], returns each in order."""
    sim = UserSimulator()
    scenario = {'user_turns': ['a', 'b']}
    assert sim.next_message(scenario, 'agent says something', 0) == 'a'
    assert sim.next_message(scenario, 'agent says more', 1) == 'b'

  def test_turn_based_returns_none_after_last(self) -> None:
    """After consuming all user_turns, returns None."""
    sim = UserSimulator()
    scenario = {'user_turns': ['a', 'b']}
    assert sim.next_message(scenario, 'agent', 2) is None
    assert sim.next_message(scenario, 'agent', 10) is None


class TestReactive:
  def test_reactive_provides_name(self) -> None:
    """known_info name returned when agent asks for name."""
    sim = UserSimulator()
    scenario = {
      'user_instructions': {
        'reason_for_call': 'need help',
        'known_info': {'name': 'Jane Doe'},
        'task_instructions': 'check my order',
      },
    }
    reply = sim.next_message(scenario, 'What is your name?', 0)
    assert reply == 'Jane Doe'

  def test_reactive_provides_name_from_parts(self) -> None:
    """first_name + last_name used when name key absent."""
    sim = UserSimulator()
    scenario = {
      'user_instructions': {
        'reason_for_call': 'need help',
        'known_info': {'first_name': 'Jane', 'last_name': 'Doe'},
        'task_instructions': 'check my order',
      },
    }
    reply = sim.next_message(scenario, 'May I have your name?', 0)
    assert reply == 'Jane Doe'

  def test_reactive_provides_zip(self) -> None:
    """known_info zip returned when agent asks for zip code."""
    sim = UserSimulator()
    scenario = {
      'user_instructions': {
        'reason_for_call': 'need help',
        'known_info': {'zip': '90210'},
        'task_instructions': 'update address',
      },
    }
    reply = sim.next_message(scenario, 'Can you provide your zip code?', 0)
    assert reply == '90210'

  def test_reactive_provides_email(self) -> None:
    """known_info email returned when agent asks."""
    sim = UserSimulator()
    scenario = {
      'user_instructions': {
        'reason_for_call': 'need help',
        'known_info': {'email': 'jane@example.com'},
        'task_instructions': 'verify account',
      },
    }
    reply = sim.next_message(scenario, 'What is your email address?', 0)
    assert reply == 'jane@example.com'

  def test_reactive_provides_order_id(self) -> None:
    """known_info order_id returned when agent asks for order."""
    sim = UserSimulator()
    scenario = {
      'user_instructions': {
        'reason_for_call': 'check order',
        'known_info': {'order_id': '#W1234'},
        'task_instructions': 'check my order',
      },
    }
    reply = sim.next_message(scenario, 'What is your order number?', 0)
    assert reply == '#W1234'

  def test_reactive_confirms(self) -> None:
    """Agent asks for confirmation; reply is 'yes'."""
    sim = UserSimulator()
    scenario = {
      'user_instructions': {
        'reason_for_call': 'help',
        'known_info': {},
        'task_instructions': 'do something',
      },
    }
    reply = sim.next_message(scenario, 'Can you confirm this is correct?', 0)
    assert reply == 'yes'

  def test_reactive_ends_on_info_provided(self) -> None:
    """Agent message indicates info provided; returns None."""
    sim = UserSimulator()
    scenario = {
      'user_instructions': {
        'reason_for_call': 'order status',
        'known_info': {},
        'task_instructions': 'find order status',
      },
    }
    reply = sim.next_message(scenario, 'Your order status is shipped. Have a great day!', 0)
    assert reply is None

  def test_reactive_fallback_to_task_instructions(self) -> None:
    """When no heuristic matches, returns task_instructions."""
    sim = UserSimulator()
    scenario = {
      'user_instructions': {
        'reason_for_call': 'help',
        'known_info': {},
        'task_instructions': 'I need to return an item',
      },
    }
    reply = sim.next_message(scenario, 'I see, let me look into that.', 0)
    assert reply == 'I need to return an item'


class TestMaxTurns:
  def test_reactive_max_turns(self) -> None:
    """At MAX_USER_TURNS, returns None without crash."""
    sim = UserSimulator()
    scenario = {
      'user_instructions': {
        'reason_for_call': 'help',
        'known_info': {},
        'task_instructions': 'do something',
      },
    }
    assert sim.next_message(scenario, 'agent reply', MAX_USER_TURNS) is None

  def test_max_turns_patched(self) -> None:
    """Patching MAX_USER_TURNS to small value triggers None."""
    sim = UserSimulator()
    scenario = {
      'user_instructions': {
        'reason_for_call': 'help',
        'known_info': {},
        'task_instructions': 'do something',
      },
    }
    with patch('harness.simulator.MAX_USER_TURNS', 3):
      assert sim.next_message(scenario, 'agent reply', 3) is None


class TestEmptyScenario:
  def test_empty_scenario(self) -> None:
    """Scenario lacking user_instructions and user_turns returns None."""
    sim = UserSimulator()
    assert sim.next_message({}, 'hello', 0) is None

  def test_empty_user_turns_falls_through(self) -> None:
    """Empty user_turns list falls through to reactive mode."""
    sim = UserSimulator()
    scenario = {'user_turns': []}
    assert sim.next_message(scenario, 'hello', 0) is None

  def test_empty_user_instructions(self) -> None:
    """Empty user_instructions dict returns None."""
    sim = UserSimulator()
    scenario = {'user_instructions': {}}
    assert sim.next_message(scenario, 'hello', 0) is None
