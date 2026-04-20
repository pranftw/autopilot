"""Agent abstraction for code-modifying agents."""

from pydantic import BaseModel, Field
from typing import Any


class AgentResult(BaseModel):
  """Result from an agent invocation."""

  output: str
  session_id: str | None = None
  metadata: dict[str, Any] = Field(default_factory=dict)


class Agent:
  """Base agent. Override forward() to implement."""

  def forward(self, prompt: str, context: dict[str, Any] | None = None) -> AgentResult:
    raise NotImplementedError
