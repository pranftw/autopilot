"""Tests for CostTrackerCallback emit_context opt-in (plan 16)."""

from autopilot.core.callbacks.cost import (
  COST_ATTRIBUTION_TYPE,
  CostEntry,
  CostTrackerCallback,
)
from autopilot.core.experiment import Experiment
from unittest.mock import MagicMock


def _make_trainer_with_experiment() -> MagicMock:
  """Build a mock trainer with a real experiment attached."""
  trainer = MagicMock()
  trainer.experiment = Experiment(experiment_id='cost-exp')
  trainer.store = None
  trainer.current_epoch = 0
  return trainer


class TestCostEmitContextDefault:
  def test_cost_emit_context_false_default(self) -> None:
    """Default callback (emit_context=False): no emit_context calls across epochs."""
    ct = CostTrackerCallback(None)
    trainer = _make_trainer_with_experiment()

    for epoch in range(3):
      ct.on_epoch_start(trainer, None, epoch)
      ct.on_epoch_end(trainer, None, epoch)

    trainer.emit_context.assert_not_called()

  def test_explicit_false_no_emission(self) -> None:
    """Explicitly setting emit_context=False produces no context emissions."""
    ct = CostTrackerCallback(None, emit_context=False)
    trainer = _make_trainer_with_experiment()

    ct.on_epoch_start(trainer, None, 0)
    ct.on_epoch_end(trainer, None, 0)

    trainer.emit_context.assert_not_called()


class TestCostEmitContextTrue:
  def test_cost_emit_context_true(self) -> None:
    """emit_context=True: at least one context entry per epoch with source='cost'."""
    ct = CostTrackerCallback(None, emit_context=True)
    trainer = _make_trainer_with_experiment()

    for epoch in range(2):
      ct.on_epoch_start(trainer, None, epoch)
      ct.on_epoch_end(trainer, None, epoch)

    assert trainer.emit_context.call_count == 2
    for c in trainer.emit_context.call_args_list:
      assert c.kwargs['source'] == 'cost'

  def test_cost_context_type_discriminator(self) -> None:
    """Metadata _type matches COST_ATTRIBUTION_TYPE constant."""
    ct = CostTrackerCallback(None, emit_context=True)
    trainer = _make_trainer_with_experiment()

    ct.on_epoch_start(trainer, None, 0)
    ct.on_epoch_end(trainer, None, 0)

    _, kwargs = trainer.emit_context.call_args
    metadata = kwargs['metadata']
    assert metadata['_type'] == COST_ATTRIBUTION_TYPE
    assert COST_ATTRIBUTION_TYPE == 'cost_attribution'

  def test_cost_context_cumulative(self) -> None:
    """Cumulative field strictly increases across epochs with non-zero costs."""

    class FixedCost(CostTrackerCallback):
      def __init__(self, costs: list[float]) -> None:
        super().__init__(None, emit_context=True)
        self._costs = costs
        self._idx = 0

      def measure(self, epoch, elapsed, result=None):
        cost = self._costs[self._idx]
        self._idx += 1
        return CostEntry(epoch=epoch, wall_clock_s=elapsed, cost_usd=cost)

    ct = FixedCost([1.0, 2.0, 3.0])
    trainer = _make_trainer_with_experiment()

    cumulative_values: list[float] = []
    for epoch in range(3):
      ct.on_epoch_start(trainer, None, epoch)
      ct.on_epoch_end(trainer, None, epoch)
      _, kwargs = trainer.emit_context.call_args
      cumulative_values.append(kwargs['metadata']['cumulative'])

    assert cumulative_values == [1.0, 3.0, 6.0]
    for i in range(1, len(cumulative_values)):
      assert cumulative_values[i] > cumulative_values[i - 1]

  def test_cost_context_epoch_field(self) -> None:
    """Metadata epoch matches the epoch argument passed to on_epoch_end."""
    ct = CostTrackerCallback(None, emit_context=True)
    trainer = _make_trainer_with_experiment()

    ct.on_epoch_start(trainer, None, 3)
    ct.on_epoch_end(trainer, None, 3)

    _, kwargs = trainer.emit_context.call_args
    assert kwargs['metadata']['epoch'] == 3

  def test_cost_context_cost_usd_field(self) -> None:
    """Metadata cost_usd matches the epoch's measured cost."""

    class SingleCost(CostTrackerCallback):
      def measure(self, epoch, elapsed, result=None):
        return CostEntry(epoch=epoch, wall_clock_s=elapsed, cost_usd=5.25)

    ct = SingleCost(None, emit_context=True)
    trainer = _make_trainer_with_experiment()

    ct.on_epoch_start(trainer, None, 0)
    ct.on_epoch_end(trainer, None, 0)

    _, kwargs = trainer.emit_context.call_args
    assert kwargs['metadata']['cost_usd'] == 5.25

  def test_cost_context_reason_format(self) -> None:
    """Reason string includes dollar-formatted cost and cumulative values."""

    class FixedCost(CostTrackerCallback):
      def measure(self, epoch, elapsed, result=None):
        return CostEntry(epoch=epoch, wall_clock_s=elapsed, cost_usd=1.5)

    ct = FixedCost(None, emit_context=True)
    trainer = _make_trainer_with_experiment()

    ct.on_epoch_start(trainer, None, 0)
    ct.on_epoch_end(trainer, None, 0)

    args, kwargs = trainer.emit_context.call_args
    reason = args[0] if args else kwargs.get('reason')
    assert '$1.5000' in reason
    assert 'cumulative: $1.5000' in reason

  def test_no_emission_without_experiment(self) -> None:
    """No context emitted when trainer.experiment is None."""
    ct = CostTrackerCallback(None, emit_context=True)
    trainer = MagicMock()
    trainer.experiment = None
    trainer.store = None

    ct.on_epoch_start(trainer, None, 0)
    ct.on_epoch_end(trainer, None, 0)

    trainer.emit_context.assert_not_called()
