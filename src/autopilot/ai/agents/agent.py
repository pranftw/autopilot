"""Agent abstraction for all AI agent types."""

from autopilot.ai.evaluation.steps import Step, collect_steps
from autopilot.ai.runtime import RateLimiter
from pydantic import BaseModel, Field
from typing import Any


class Agent:
  """Base agent. All agent types subclass this.

  Primary execution interface:
    run()        -- sync execution (subclasses define signature)
    async_run()  -- async execution (subclasses define signature)
  """

  def __init__(
    self,
    limiter: RateLimiter | None = None,
    num_parallel: int = 1,
  ) -> None:
    self._limiter = limiter
    self._num_parallel = num_parallel

  def run(self, *args: Any, **kwargs: Any) -> Any:
    """Sync execution. Override in subclasses."""
    raise NotImplementedError

  async def async_run(self, *args: Any, **kwargs: Any) -> Any:
    """Async execution. Override in subclasses."""
    raise NotImplementedError

  def name(self) -> str:
    """Agent identity. Override for custom names."""
    return type(self).__name__

  def setup(self, **kwargs: Any) -> None:
    """Called before the agent is used. Override for initialization."""

  def teardown(self) -> None:
    """Called when the agent is done. Override for cleanup."""

  @property
  def limiter(self) -> RateLimiter | None:
    return self._limiter

  @limiter.setter
  def limiter(self, value: RateLimiter | None) -> None:
    self._limiter = value

  def state_dict(self) -> dict[str, Any]:
    """Return agent state for checkpointing."""
    return {}

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    """Restore agent state from checkpoint."""

  def __repr__(self) -> str:
    return f'{type(self).__name__}()'


class StepAgent(Agent):
  """Agent that runs step-based LLM/Python workflows.

  Adds step orchestration to the Agent base.
  """

  def define_steps(self, config: Any) -> list[Step]:
    """Return ordered workflow steps. Default: collect from @step decorators."""
    return collect_steps(self)


class AgentResult(BaseModel):
  """Result from a prompt-based agent invocation."""

  output: str
  session_id: str | None = None
  metadata: dict[str, Any] = Field(default_factory=dict)
