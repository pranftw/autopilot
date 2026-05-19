"""Scripted user simulator for deterministic multi-turn conversations.

Provides ``UserSimulator`` whose ``next_message`` method generates the next
user utterance from scenario metadata.  Two modes:

1. **Turn-based** -- when ``scenario['user_turns']`` is a non-empty list, returns
   each element in order, then ``None`` after exhaustion.  Takes precedence over
   reactive mode.
2. **Reactive** -- uses ``scenario['user_instructions']`` (``reason_for_call``,
   ``known_info``, ``task_instructions``) to heuristically answer agent questions.
   Heuristic evaluation order: max-turn guard -> name/identity -> specific details
   (order id, email, zip, address, phone) -> confirmation -> conversation-end
   detection.  Ambiguity favors continuing unless the agent clearly provided the
   requested information.

A ``MAX_USER_TURNS`` named constant prevents infinite loops.
"""

MAX_USER_TURNS = 50

NAME_TRIGGERS = frozenset(
  {
    'your name',
    'who am i',
    'who are you',
    'may i have your name',
    'identify yourself',
    'what is your name',
  }
)

CONFIRMATION_TRIGGERS = frozenset(
  {
    'confirm',
    'is that correct',
    'do you agree',
    'shall i proceed',
    'would you like me to',
    'are you sure',
    'can i go ahead',
    'do you want me to',
    'yes or no',
  }
)

DETAIL_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
  ('order', ('order_id', 'order_number', 'order')),
  ('email', ('email', 'email_address')),
  ('zip', ('zip', 'zip_code', 'postal_code', 'postal')),
  ('address', ('address', 'street', 'shipping_address')),
  ('phone', ('phone', 'phone_number', 'telephone')),
)

END_INDICATORS = frozenset(
  {
    'your order status is',
    'your order has been',
    'here is the information',
    'i have processed',
    'is there anything else',
    'have a great day',
    'have a nice day',
    'is there anything else i can help',
    'your request has been',
    'i have updated',
    'i have cancelled',
    'the refund has been',
    'here are the details',
    'your account has been',
  }
)


class UserSimulator:
  """Scripted user for deterministic multi-turn conversations.

  Follows scenario instructions to respond to agent queries.  Provides
  information from ``known_info`` when the agent asks for it.

  Resolution order for reactive mode:

  1. Max-turn guard (``MAX_USER_TURNS``)
  2. Name / identity detection (checks ``name``, then ``first_name`` +
     ``last_name`` in ``known_info``)
  3. Specific detail request (order id, email, zip, address, phone) mapped
     from ``known_info`` keys
  4. Cooperative confirmation (replies ``'yes'``)
  5. End detection (agent provided the requested information)
  6. Fallback: re-state ``task_instructions`` to keep the conversation going
  """

  def next_message(self, scenario: dict, agent_response: str, turn: int) -> str | None:
    """Generate next user message based on scenario and agent response.

    Args:
      scenario: Scenario metadata with optional ``user_turns`` or
        ``user_instructions``.
      agent_response: The agent's last message.
      turn: Current turn number (0-based).

    Returns:
      Next user message, or ``None`` if conversation should end.
    """
    user_turns = scenario.get('user_turns')
    if isinstance(user_turns, list) and len(user_turns) > 0:
      return self._turn_based(user_turns, turn)
    return self._reactive(scenario, agent_response, turn)

  def _turn_based(self, user_turns: list[str], turn: int) -> str | None:
    """Return the next scripted user turn, or None if exhausted.

    Args:
      user_turns: Ordered list of scripted user messages.
      turn: Current turn index.

    Returns:
      The next user turn string, or ``None`` past the last turn.
    """
    if turn < len(user_turns):
      return user_turns[turn]
    return None

  def _reactive(self, scenario: dict, agent_response: str, turn: int) -> str | None:
    """Generate a heuristic reply from user_instructions.

    Args:
      scenario: Full scenario dict.
      agent_response: The agent's last message.
      turn: Current turn index.

    Returns:
      Heuristic user reply, or ``None`` to end.
    """
    if turn >= MAX_USER_TURNS:
      return None

    instructions = scenario.get('user_instructions')
    if not isinstance(instructions, dict) or not instructions:
      return None

    known_info = instructions.get('known_info', {})
    task_instructions = instructions.get('task_instructions', '')

    lower_response = agent_response.lower()

    name_reply = self._check_name(lower_response, known_info)
    if name_reply is not None:
      return name_reply

    detail_reply = self._check_details(lower_response, known_info)
    if detail_reply is not None:
      return detail_reply

    if self._is_confirmation_request(lower_response):
      return 'yes'

    if self._is_end_indicator(lower_response):
      return None

    if task_instructions:
      return task_instructions

    return None

  def _check_name(self, lower_response: str, known_info: dict) -> str | None:
    """Check if agent is asking for name and return it from known_info.

    Resolution order: ``name`` key first, then ``first_name`` + ``last_name``.

    Args:
      lower_response: Lowercased agent response.
      known_info: User knowledge dictionary.

    Returns:
      Name string if applicable, else ``None``.
    """
    if not any(trigger in lower_response for trigger in NAME_TRIGGERS):
      return None
    if 'name' in known_info:
      return str(known_info['name'])
    first = known_info.get('first_name', '')
    last = known_info.get('last_name', '')
    if first or last:
      return f'{first} {last}'.strip()
    return None

  def _check_details(self, lower_response: str, known_info: dict) -> str | None:
    """Check if agent is asking for a specific detail from known_info.

    Args:
      lower_response: Lowercased agent response.
      known_info: User knowledge dictionary.

    Returns:
      The detail value as string if found, else ``None``.
    """
    for keyword, info_keys in DETAIL_KEYWORDS:
      if keyword in lower_response:
        for key in info_keys:
          if key in known_info:
            return str(known_info[key])
    return None

  def _is_confirmation_request(self, lower_response: str) -> bool:
    """Check if agent is asking for yes/no confirmation.

    Args:
      lower_response: Lowercased agent response.

    Returns:
      ``True`` when a confirmation trigger is found.
    """
    return any(trigger in lower_response for trigger in CONFIRMATION_TRIGGERS)

  def _is_end_indicator(self, lower_response: str) -> bool:
    """Check if agent has provided the requested information.

    Args:
      lower_response: Lowercased agent response.

    Returns:
      ``True`` when an end indicator is found.
    """
    return any(indicator in lower_response for indicator in END_INDICATORS)
