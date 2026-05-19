"""Shared conftest for dogfood regression test suite.

Centralizes skip markers for optional external dependencies and shared
agent stubs for deterministic testing.
"""

from autopilot.ai.agents.agent import Agent
from importlib.util import find_spec
import pytest

HAS_ANTHROPIC = find_spec('anthropic') is not None

skip_without_anthropic = pytest.mark.skipif(
  not HAS_ANTHROPIC,
  reason='anthropic SDK not installed',
)


class ScriptedAgent(Agent):
  """Agent stub returning a canned response for deterministic tests.

  Used by MergeAgent E2E and any test needing a predictable ``run()`` reply
  without real LLM calls.
  """

  def __init__(self, response: str) -> None:
    super().__init__()
    self._response = response

  def run(self, *args, **kwargs):
    """Return the canned response."""
    return self._response
