"""HarnessAgent wrapping Pydantic AI for multi-turn conversation execution.

Provides ``HarnessAgent`` for running retail customer service conversations
and ``ConversationResult`` for capturing the trajectory, tool calls, turn
count, and token usage.  ``UserSimulator`` (from ``harness.simulator``) is
passed via the ``simulator`` parameter to ``run_conversation`` for multi-turn
interaction.
"""

from dataclasses import dataclass, field
from harness import DEFAULT_MODEL
from harness.database import RetailDB
from harness.tool_loader import HarnessDeps, register_tools
from typing import Any, Callable
import pydantic_ai
import traceback

DEFAULT_MAX_CONVERSATION_TURNS = 15


@dataclass
class ConversationResult:
  """Outcome of a single multi-turn conversation.

  Attributes:
    trajectory: List of role/content/turn dicts capturing the dialogue.
    tool_calls: Structured dicts logged by tools via ``HarnessDeps.tool_log``.
    turns: Number of completed assistant generations.
    error: Error string if the conversation failed, or ``None``.
    input_tokens: Total input tokens consumed across all turns.
    output_tokens: Total output tokens generated across all turns.
    api_calls: Total API requests made across all turns.
  """

  trajectory: list[dict] = field(default_factory=list)
  tool_calls: list[dict] = field(default_factory=list)
  turns: int = 0
  error: str | None = None
  input_tokens: int = 0
  output_tokens: int = 0
  api_calls: int = 0


class HarnessAgent:
  """Wraps Pydantic AI agent for multi-turn conversation execution.

  Args:
    model: The model identifier string for Pydantic AI
      (default: ``DEFAULT_MODEL``).
  """

  def __init__(self, model: str = DEFAULT_MODEL) -> None:
    self.model = model

  def run_conversation(
    self,
    instructions: str,
    tools: dict[str, Callable],
    scenario: dict,
    db: RetailDB,
    simulator: Any = None,
    max_turns: int | None = None,
  ) -> ConversationResult:
    """Run a complete multi-turn conversation for one scenario.

    Builds a fresh Pydantic AI ``Agent`` with the given instructions and
    tools, then loops up to the effective turn limit.  Each iteration
    calls ``run_sync``, records the assistant output, and obtains the
    next user message from ``simulator`` or the
    ``_get_next_user_message`` placeholder.

    Args:
      instructions: System prompt / policy text for the agent.
      tools: Dict of tool name -> callable (from ``load_tools``).
      scenario: Scenario dict with at least ``initial_message``.
      db: An already-cloned RetailDB for isolation.
      simulator: Optional ``UserSimulator``.  When provided,
        delegates next-user-message logic to ``simulator.next_message()``.
        When ``None``, uses ``_get_next_user_message`` placeholder
        (single-turn).
      max_turns: Maximum number of assistant turns before aborting with
        ``error='max_turns'``.  When ``None``, uses
        ``DEFAULT_MAX_CONVERSATION_TURNS`` (15).

    Returns:
      A ``ConversationResult`` capturing trajectory, tool calls, turns,
      error state (including full traceback when an exception occurs),
      and token usage.
    """
    limit = DEFAULT_MAX_CONVERSATION_TURNS if max_turns is None else max_turns
    deps = HarnessDeps(db=db, tool_log=[])
    agent = self._build_agent(instructions, tools)
    history = None
    trajectory: list[dict] = []
    error: str | None = None
    total_input_tokens = 0
    total_output_tokens = 0
    total_api_calls = 0
    user_msg = str(scenario.get('initial_message', ''))
    last_turn_index = -1
    for turn in range(limit):
      try:
        result = agent.run_sync(user_msg, deps=deps, message_history=history)
      except Exception as exc:
        tb = traceback.format_exc()
        error = f'{type(exc).__name__}: {exc}\n{tb}'
        break
      usage = result.usage()
      total_input_tokens += usage.input_tokens
      total_output_tokens += usage.output_tokens
      total_api_calls += usage.requests
      history = result.all_messages()
      trajectory.append({'role': 'assistant', 'content': result.output, 'turn': turn})
      last_turn_index = turn
      if simulator is not None:
        next_msg = simulator.next_message(scenario, result.output, turn)
      else:
        next_msg = self._get_next_user_message(scenario, result.output, turn)
      if next_msg is None:
        break
      trajectory.append({'role': 'user', 'content': next_msg, 'turn': turn + 1})
      user_msg = next_msg
    else:
      error = 'max_turns'
    completed_turns = last_turn_index + 1 if last_turn_index >= 0 else 0
    return ConversationResult(
      trajectory=trajectory,
      tool_calls=deps.tool_log,
      turns=completed_turns,
      error=error,
      input_tokens=total_input_tokens,
      output_tokens=total_output_tokens,
      api_calls=total_api_calls,
    )

  def _build_agent(
    self,
    instructions: str,
    tools: dict[str, Callable],
  ) -> pydantic_ai.Agent:
    """Construct a Pydantic AI agent with the given instructions and tools.

    Args:
      instructions: System prompt text.
      tools: Dict of tool name -> callable.

    Returns:
      A configured ``pydantic_ai.Agent`` instance.
    """
    agent = pydantic_ai.Agent(
      self.model,
      deps_type=HarnessDeps,
      instructions=instructions,
    )
    register_tools(agent, tools)
    return agent

  def _get_next_user_message(
    self,
    scenario: dict,
    agent_response: str,
    turn: int,
  ) -> str | None:
    """Placeholder for next user message logic.

    Returns ``None`` to end the conversation after one assistant turn.
    When a ``UserSimulator`` is passed to ``run_conversation``, it
    replaces this method for multi-turn interaction.

    Args:
      scenario: The scenario dict.
      agent_response: The agent's last response.
      turn: The current turn index.

    Returns:
      ``None`` (single-turn placeholder).
    """
    return None
