"""Evaluation output protocol for decoupling ai/evaluation from cli types."""

from typing import Protocol


class EvaluationOutputProtocol(Protocol):
  """Sink for evaluation progress and final structured results."""

  def info(self, message: str) -> None:
    """Emit an informational message to the user or log."""
    ...

  def warn(self, message: str) -> None:
    """Emit a non-fatal warning."""
    ...

  def result(self, payload: dict, ok: bool = True) -> None:
    """Emit a final or intermediate result payload."""
    ...
