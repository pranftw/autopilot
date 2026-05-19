"""Protocol and base classes for policies.

Policy evaluates experiment results after metrics exist. Used by Trainer
during fit() for epoch-level gating, or offline on persisted Result objects.
"""

from autopilot.core.models import Result
from autopilot.core.types import GateResult
from typing import Protocol


class PolicyProtocol(Protocol):
  """Structural typing contract for policies."""

  def name(self) -> str:
    """Return stable policy name."""
    ...

  def forward(self, result: Result) -> GateResult:
    """Evaluate ``result`` and return aggregate gate status."""
    ...

  def explain(self, result: Result) -> str:
    """Return a human-readable explanation for the last evaluation."""
    ...


class Policy:
  """Base class for policies. Subclass and override forward()/explain().

  Policy is deterministic by design: gate evaluation produces the same result
  given the same Result input (no mutable internal state between forward calls).
  state_dict()/load_state_dict() are provided for checkpoint serialization of
  gate configuration.

  Protocol:
    name() -> str                  -- stable identifier
    forward(result: Result) -> GateResult  -- PASSED, FAIL, WARN, or SKIP
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
    """Return the concrete class name as a stable identifier."""
    return type(self).__name__

  def forward(self, result: Result) -> GateResult:
    """Default policy always passes regardless of metrics.

    Returns:
      ``GateResult.PASSED`` unconditionally.
    """
    return GateResult.PASSED

  def __call__(self, result: Result) -> GateResult:
    """Delegate to ``forward``.

    Returns:
      Same ``GateResult`` as ``forward``.
    """
    return self.forward(result)

  def explain(self, result: Result) -> str:
    """Return a static explanation for the permissive default policy.

    Returns:
      Fixed ``'default pass'`` string.
    """
    return 'default pass'

  def state_dict(self) -> dict:
    """Serialize policy configuration for checkpoint persistence.

    Subclasses should override to include gate configuration. The base
    implementation returns an empty dict (stateless default policy).

    Returns:
      Dict representation of policy state.
    """
    return {}

  def load_state_dict(self, state: dict) -> None:
    """Restore policy configuration from checkpoint data.

    Subclasses should override to restore gate configuration. The base
    implementation is a no-op (stateless default policy).

    Args:
      state: Dict previously produced by ``state_dict()``.
    """
