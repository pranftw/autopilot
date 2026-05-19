"""AgentCollator bridge adapters for AutoPilot Agent compatibility.

Provides ``PydanticAgent``, a thin wrapper around ``pydantic_ai.Agent.run_sync``
that satisfies the ``Agent.run() -> AgentResult`` contract expected by
``AgentCollator`` in the JudgeLoss collation path.  This is distinct from the
multi-turn ``HarnessAgent`` in ``harness/agent.py`` which drives full
conversations with tool use and history.
"""

from autopilot.ai.agents.agent import Agent, AgentResult
from harness import DEFAULT_MODEL
import pydantic_ai


class PydanticAgent(Agent):
  """Bridges pydantic_ai single-turn execution to the AutoPilot Agent contract.

  Used as the backing agent for ``AgentCollator`` when ``JudgeLoss`` needs
  to collate evaluation feedback into attributed ``TextGradient`` instances.
  Each call to ``run`` creates a fresh ``pydantic_ai.Agent``, invokes
  ``run_sync``, and wraps the result into an ``AgentResult``.

  Args:
    model: Model identifier string for pydantic_ai (default: harness
      ``DEFAULT_MODEL``).
    instructions: Optional system instructions passed to the pydantic_ai
      Agent constructor.
  """

  def __init__(
    self,
    model: str = DEFAULT_MODEL,
    instructions: str | None = None,
  ) -> None:
    """Initialize PydanticAgent with model and optional instructions.

    Args:
      model: Model identifier for pydantic_ai.
      instructions: Optional system prompt for the underlying agent.
    """
    super().__init__()
    self._model = model
    self._instructions = instructions

  def run(self, prompt: str, context: dict | None = None) -> AgentResult:
    """Execute a single-turn prompt and return an AgentResult.

    Creates a pydantic_ai Agent, calls ``run_sync`` with the prompt,
    and wraps ``result.output`` as a string into ``AgentResult``.
    The ``context`` parameter is accepted for API compatibility with
    callers that may pass extra metadata but is unused in this
    implementation.

    Args:
      prompt: The text prompt to send to the model.
      context: Optional context dict (unused; accepted for API
        compatibility with ``AgentCollator``).

    Returns:
      An ``AgentResult`` with the model's text output as a string.
    """
    agent = pydantic_ai.Agent(self._model, instructions=self._instructions)
    result = agent.run_sync(prompt)
    return AgentResult(output=str(result.output))
