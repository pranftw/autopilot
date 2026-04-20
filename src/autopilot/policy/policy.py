"""Protocol and base classes for policies."""

from autopilot.core.models import GateResult, Result
from typing import Protocol


class PolicyProtocol(Protocol):
  """Structural typing contract for policies."""

  def name(self) -> str: ...
  def forward(self, result: Result) -> GateResult: ...
  def explain(self, result: Result) -> str: ...


class Policy:
  """Base class for policies. Subclass and override forward()/explain()."""

  def name(self) -> str:
    return type(self).__name__

  def forward(self, result: Result) -> GateResult:
    return GateResult.PASS

  def __call__(self, result: Result) -> GateResult:
    return self.forward(result)

  def explain(self, result: Result) -> str:
    return 'default pass'
