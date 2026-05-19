"""Tests for the Optimizer.owns_step_gradient_context protocol.

Verifies BUG-004 fix (non-agentic AgentOptimizer no longer suppresses Trainer
gradient capture) and QUALITY-001 fix (public property replaces private
duck-typed hasattr check).
"""

from autopilot.ai.optimizer import AgentOptimizer
from autopilot.core.gradient import NumericGradient
from autopilot.core.module.module import Module
from autopilot.core.optimizer import Optimizer
from autopilot.core.parameter import ScalarParameter
from autopilot.core.trainer.journal import capture_gradient_summaries
from tests.doubles import NoOpOptimizer
from unittest.mock import MagicMock


class StubOwnsContextOptimizer(Optimizer):
  """Local stub that claims ownership of gradient context."""

  @property
  def owns_step_gradient_context(self) -> bool:
    """Always True for testing."""
    return True

  def step(self) -> None:
    pass


class _PlainSubclassOptimizer(Optimizer):
  """Plain subclass with no override -- inherits default False."""

  def step(self) -> None:
    pass


class _SingleParamModule(Module):
  """Module with one scalar parameter for capture tests."""

  def __init__(self) -> None:
    super().__init__()
    self.p = ScalarParameter(value=1.0)

  def forward(self, *args, **kwargs):
    pass


# --- 4.1 Property contract tests ---


class TestPropertyContract:
  """Tests for owns_step_gradient_context on Optimizer and AgentOptimizer."""

  def test_noop_optimizer_owns_step_gradient_context_false(self):
    param = ScalarParameter(value=1.0)
    opt = NoOpOptimizer([param])
    assert opt.owns_step_gradient_context is False

  def test_agent_optimizer_owns_context_matches_agentic_flag(self):
    agent = MagicMock()
    opt_non_agentic = AgentOptimizer(agent=agent, params=[], agentic=False)
    assert opt_non_agentic.owns_step_gradient_context is False

    opt_agentic = AgentOptimizer(agent=agent, params=[], agentic=True)
    assert opt_agentic.owns_step_gradient_context is True

  def test_custom_optimizer_subclass_defaults_to_false(self):
    param = ScalarParameter(value=1.0)
    opt = _PlainSubclassOptimizer([param])
    assert opt.owns_step_gradient_context is False


# --- 4.2 capture_gradient_summaries integration tests ---


class TestCaptureGradientSummaries:
  """Tests for capture_gradient_summaries using owns_step_gradient_context."""

  def test_capture_skips_when_optimizer_owns_context(self):
    module = _SingleParamModule()
    module.p.grad = NumericGradient(value=1.0)

    trainer = MagicMock()
    trainer._module = module
    trainer._optimizer = StubOwnsContextOptimizer([module.p])
    trainer._cached_grad_summaries = []

    capture_gradient_summaries(trainer)

    assert trainer._cached_grad_summaries == []

  def test_capture_runs_when_optimizer_does_not_own_context(self):
    module = _SingleParamModule()
    module.p.grad = NumericGradient(value=2.5)

    trainer = MagicMock()
    trainer._module = module
    trainer._optimizer = NoOpOptimizer([module.p])
    trainer._cached_grad_summaries = []

    capture_gradient_summaries(trainer)

    assert len(trainer._cached_grad_summaries) == 1
    row = trainer._cached_grad_summaries[0]
    assert row['param_name'] == 'p'
    assert row['param_type'] == 'ScalarParameter'
    assert row['gradient_type'] == 'NumericGradient'

  def test_capture_runs_when_optimizer_is_none(self):
    module = _SingleParamModule()
    module.p.grad = NumericGradient(value=3.0)

    trainer = MagicMock()
    trainer._module = module
    trainer._optimizer = None
    trainer._cached_grad_summaries = []

    capture_gradient_summaries(trainer)

    assert len(trainer._cached_grad_summaries) == 1
    assert trainer._cached_grad_summaries[0]['param_name'] == 'p'

  def test_capture_skips_when_module_none(self):
    trainer = MagicMock()
    trainer._module = None
    sentinel = ['sentinel_value']
    trainer._cached_grad_summaries = sentinel

    capture_gradient_summaries(trainer)

    assert trainer._cached_grad_summaries is sentinel
    assert trainer._cached_grad_summaries == ['sentinel_value']
