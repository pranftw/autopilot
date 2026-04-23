"""Protocol and base classes for policies.

Policy evaluates experiment results after metrics exist. Used by Trainer
during fit() for epoch-level gating, or offline on persisted Result objects.
"""

from autopilot.core.models import Result
from autopilot.core.types import GateResult
from typing import Protocol


class PolicyProtocol(Protocol):
  """Structural typing contract for policies."""

  def name(self) -> str: ...
  def forward(self, result: Result) -> GateResult: ...
  def explain(self, result: Result) -> str: ...


class Policy:
  """Base class for policies. Subclass and override forward()/explain().

  Protocol:
    name() -> str                  -- stable identifier
    forward(result: Result) -> GateResult  -- PASS, FAIL, WARN, or SKIP
    explain(result: Result) -> str -- human-readable explanation
    __call__(result) -> GateResult -- delegates to forward()

  Integration with Trainer:
    Pass Policy instance via Trainer(policy=...). During fit(), after metrics
    are computed each epoch, the loop builds Result(metrics=...) and calls
    policy(result). On GateResult.FAIL, training stops and experiment.rollback()
    is called when a Store is available.

  For offline evaluation, load a persisted Result and call policy.forward()
  or policy.explain() directly.

  Built-in subclass: QualityFirstPolicy (policy/quality_first.py).
  """

  def name(self) -> str:
    return type(self).__name__

  def forward(self, result: Result) -> GateResult:
    return GateResult.PASS

  def __call__(self, result: Result) -> GateResult:
    return self.forward(result)

  def explain(self, result: Result) -> str:
    return 'default pass'
