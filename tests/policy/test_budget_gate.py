"""Tests for BudgetGate and cost_usd accumulation plumbing."""

from autopilot.core.callbacks.cost import CostEntry, CostTrackerCallback
from autopilot.core.loops.epoch import EpochLoop
from autopilot.core.models import Result
from autopilot.core.types import GateResult
from autopilot.policy.gates import BudgetGate, MinGate, MonotonicGate
from autopilot.policy.quality_first import QualityFirstPolicy
from unittest.mock import MagicMock


class TestBudgetGateUnderBudget:
  """Under-budget scenarios should pass."""

  def test_budget_gate_under_budget_passes(self) -> None:
    gate = BudgetGate(max_usd=10.0)
    result = Result(metrics={'cost_usd': 5.0})
    assert gate.forward(result) == GateResult.PASSED

  def test_budget_gate_exactly_at_budget_passes(self) -> None:
    gate = BudgetGate(max_usd=10.0)
    result = Result(metrics={'cost_usd': 10.0})
    assert gate.forward(result) == GateResult.PASSED

  def test_budget_gate_zero_cost_passes(self) -> None:
    gate = BudgetGate(max_usd=10.0)
    result = Result(metrics={'cost_usd': 0.0})
    assert gate.forward(result) == GateResult.PASSED


class TestBudgetGateOverBudget:
  """Over-budget scenarios should fail."""

  def test_budget_gate_over_budget_fails(self) -> None:
    gate = BudgetGate(max_usd=10.0)
    result = Result(metrics={'cost_usd': 11.0})
    assert gate.forward(result) == GateResult.FAIL

  def test_budget_gate_slightly_over_fails(self) -> None:
    gate = BudgetGate(max_usd=10.0)
    result = Result(metrics={'cost_usd': 10.01})
    assert gate.forward(result) == GateResult.FAIL


class TestBudgetGateMissingMetric:
  """Missing cost_usd key should fail (fail-closed)."""

  def test_budget_gate_missing_cost_fails(self) -> None:
    gate = BudgetGate(max_usd=10.0)
    result = Result(metrics={})
    assert gate.forward(result) == GateResult.FAIL

  def test_budget_gate_other_metrics_only_fails(self) -> None:
    gate = BudgetGate(max_usd=10.0)
    result = Result(metrics={'accuracy': 0.95})
    assert gate.forward(result) == GateResult.FAIL


class TestBudgetGateExplain:
  """Tests for BudgetGate.explain() output."""

  def test_explain_under_budget(self) -> None:
    gate = BudgetGate(max_usd=50.0)
    result = Result(metrics={'cost_usd': 25.0})
    text = gate.explain(result)
    assert '25.0 <= 50.0' in text
    assert 'PASSED' in text

  def test_explain_over_budget(self) -> None:
    gate = BudgetGate(max_usd=50.0)
    result = Result(metrics={'cost_usd': 75.0})
    text = gate.explain(result)
    assert '75.0 <= 50.0' in text
    assert 'FAIL' in text

  def test_explain_missing(self) -> None:
    gate = BudgetGate(max_usd=50.0)
    result = Result(metrics={})
    text = gate.explain(result)
    assert 'missing -> FAIL' in text


class TestBudgetGateCallable:
  """Verify __call__ delegates to forward."""

  def test_call_wraps_forward(self) -> None:
    gate = BudgetGate(max_usd=10.0)
    result = Result(metrics={'cost_usd': 5.0})
    assert gate(result) == gate.forward(result)


class TestBudgetGateRequired:
  """Required flag semantics."""

  def test_required_defaults_to_true(self) -> None:
    gate = BudgetGate(max_usd=10.0)
    assert gate.required is True

  def test_required_false(self) -> None:
    gate = BudgetGate(max_usd=10.0, required=False)
    assert gate.required is False


class TestBudgetGateRepr:
  """Verify repr includes useful info."""

  def test_repr(self) -> None:
    gate = BudgetGate(max_usd=42.0)
    rep = repr(gate)
    assert 'BudgetGate' in rep
    assert 'cost_usd' in rep


class TestGatePublicProperties:
  """Tests for public accessor properties on gate classes."""

  def test_budget_gate_max_usd_property(self) -> None:
    """BudgetGate.max_usd returns configured budget ceiling."""
    assert BudgetGate(max_usd=50.0).max_usd == 50.0

  def test_budget_gate_max_usd_matches_internal(self) -> None:
    """Property returns same value as internal attribute."""
    gate = BudgetGate(max_usd=42.0)
    assert gate.max_usd == gate._max_usd

  def test_monotonic_gate_direction_property(self) -> None:
    """MonotonicGate.direction returns configured direction."""
    gate = MonotonicGate('accuracy', direction='non_decreasing')
    assert gate.direction == 'non_decreasing'

  def test_monotonic_gate_direction_non_increasing(self) -> None:
    """MonotonicGate.direction works for non_increasing."""
    gate = MonotonicGate('loss', direction='non_increasing')
    assert gate.direction == 'non_increasing'

  def test_monotonic_gate_epsilon_property(self) -> None:
    """MonotonicGate.epsilon returns configured tolerance."""
    gate = MonotonicGate('accuracy', epsilon=0.1)
    assert gate.epsilon == 0.1

  def test_monotonic_gate_epsilon_default(self) -> None:
    """MonotonicGate.epsilon defaults to 0.0."""
    gate = MonotonicGate('accuracy')
    assert gate.epsilon == 0.0

  def test_monotonic_gate_properties_match_internal(self) -> None:
    """Properties return same values as internal attributes."""
    gate = MonotonicGate('accuracy', direction='non_decreasing', epsilon=0.05)
    assert gate.direction == gate._direction
    assert gate.epsilon == gate._epsilon


class TestBudgetGateInsideCompositePolicy:
  """BudgetGate composed with other gates in QualityFirstPolicy."""

  def test_budget_gate_inside_composite_policy(self) -> None:
    """Budget + quality gate: both must pass for overall PASSED."""
    min_gate = MinGate('val_accuracy', 0.8)
    budget_gate = BudgetGate(max_usd=100.0)
    policy = QualityFirstPolicy(gates=[min_gate, budget_gate])

    result = Result(metrics={'val_accuracy': 0.9, 'cost_usd': 50.0})
    assert policy(result) == GateResult.PASSED

  def test_budget_fails_overrides_quality_pass(self) -> None:
    """Quality passes but budget fails -> overall FAIL."""
    min_gate = MinGate('val_accuracy', 0.8)
    budget_gate = BudgetGate(max_usd=100.0)
    policy = QualityFirstPolicy(gates=[min_gate, budget_gate])

    result = Result(metrics={'val_accuracy': 0.95, 'cost_usd': 150.0})
    assert policy(result) == GateResult.FAIL

  def test_quality_fails_overrides_budget_pass(self) -> None:
    """Budget passes but quality fails -> overall FAIL."""
    min_gate = MinGate('val_accuracy', 0.8)
    budget_gate = BudgetGate(max_usd=100.0)
    policy = QualityFirstPolicy(gates=[min_gate, budget_gate])

    result = Result(metrics={'val_accuracy': 0.5, 'cost_usd': 10.0})
    assert policy(result) == GateResult.FAIL

  def test_optional_budget_gate_warns_on_overspend(self) -> None:
    """Optional BudgetGate failure yields WARN, not FAIL."""
    min_gate = MinGate('val_accuracy', 0.8)
    budget_gate = BudgetGate(max_usd=100.0, required=False)
    policy = QualityFirstPolicy(gates=[min_gate, budget_gate])

    result = Result(metrics={'val_accuracy': 0.9, 'cost_usd': 200.0})
    assert policy(result) == GateResult.WARN


class TestCostUsdAccumulatesAcrossEpochs:
  """Integration: cumulative_usd from CostTrackerCallback flows to gate metrics."""

  def test_cost_usd_accumulates_across_epochs(self) -> None:
    """Two epochs with costs {1.0, 2.25} yield cumulative 3.25 in gate metrics."""

    class FixedCostTracker(CostTrackerCallback):
      """Override measure to return fixed cost_usd per epoch."""

      def __init__(self, costs: list[float]) -> None:
        super().__init__(None)
        self._costs = costs
        self._call_idx = 0

      def measure(self, epoch: int, elapsed: float, result: object = None) -> CostEntry:
        cost = self._costs[self._call_idx]
        self._call_idx += 1
        return CostEntry(epoch=epoch, wall_clock_s=elapsed, cost_usd=cost)

    tracker = FixedCostTracker([1.0, 2.25])
    trainer = MagicMock()
    trainer.callbacks = [tracker]

    tracker.on_epoch_start(trainer, None, 0)
    tracker.on_epoch_end(trainer, None, 0)
    assert tracker.cumulative_usd == 1.0

    tracker.on_epoch_start(trainer, None, 1)
    tracker.on_epoch_end(trainer, None, 1)
    assert tracker.cumulative_usd == 3.25

    loop = EpochLoop()
    captured_metrics: dict[str, float] = {}

    def capture_policy(result: Result) -> GateResult:
      captured_metrics.update(result.metrics)
      return GateResult.PASSED

    trainer.policy = capture_policy
    trainer.emit_context = MagicMock()

    gate_metrics: dict[str, float] = {'val_accuracy': 0.9}
    loop._check_policy_gate(trainer, 1, gate_metrics, None)

    assert captured_metrics['cost_usd'] == 3.25

  def test_cost_usd_not_overwritten_if_already_present(self) -> None:
    """If cost_usd is already in gate metrics, the callback value is not injected."""
    tracker = CostTrackerCallback(None)
    tracker.cumulative_usd = 99.0
    trainer = MagicMock()
    trainer.callbacks = [tracker]
    captured_metrics: dict[str, float] = {}

    def capture_policy(result: Result) -> GateResult:
      captured_metrics.update(result.metrics)
      return GateResult.PASSED

    trainer.policy = capture_policy
    trainer.emit_context = MagicMock()

    loop = EpochLoop()
    gate_metrics: dict[str, float] = {'cost_usd': 5.0}
    loop._check_policy_gate(trainer, 0, gate_metrics, None)

    assert captured_metrics['cost_usd'] == 5.0

  def test_budget_gate_with_accumulated_cost_integration(self) -> None:
    """End-to-end: BudgetGate fails when accumulated cost exceeds budget."""

    class FixedCostTracker(CostTrackerCallback):
      def __init__(self, costs: list[float]) -> None:
        super().__init__(None)
        self._costs = costs
        self._call_idx = 0

      def measure(self, epoch: int, elapsed: float, result: object = None) -> CostEntry:
        cost = self._costs[self._call_idx]
        self._call_idx += 1
        return CostEntry(epoch=epoch, wall_clock_s=elapsed, cost_usd=cost)

    tracker = FixedCostTracker([5.0, 6.0])
    budget_gate = BudgetGate(max_usd=10.0)
    policy = QualityFirstPolicy(gates=[budget_gate])

    trainer = MagicMock()
    trainer.callbacks = [tracker]
    trainer.policy = policy
    trainer.emit_context = MagicMock()
    loop = EpochLoop()

    tracker.on_epoch_start(trainer, None, 0)
    tracker.on_epoch_end(trainer, None, 0)
    result_e0 = loop._check_policy_gate(trainer, 0, {}, None)
    assert result_e0 is None

    tracker.on_epoch_start(trainer, None, 1)
    tracker.on_epoch_end(trainer, None, 1)
    result_e1 = loop._check_policy_gate(trainer, 1, {}, None)
    assert result_e1 is not None
    assert result_e1['stopped'] is True
