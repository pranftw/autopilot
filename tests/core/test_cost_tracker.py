"""Tests for CostTrackerCallback."""

from autopilot.core.callbacks.cost import CostEntry, CostTrackerCallback
from autopilot.core.errors import ConfigError
from autopilot.core.models import Result
from unittest.mock import MagicMock
import pytest
import time


class TestCostTrackerCallback:
  def test_single_epoch(self, tmp_path):
    ct = CostTrackerCallback(tmp_path)
    trainer = MagicMock()
    ct.on_epoch_start(trainer, None, 1)
    time.sleep(0.01)
    ct.on_epoch_end(trainer, None, 1, result=Result(metrics={'a': 1.0}))
    assert len(ct.per_epoch()) == 1
    assert ct.per_epoch()[0].wall_clock_s > 0

  def test_multi_epoch(self, tmp_path):
    ct = CostTrackerCallback(tmp_path)
    trainer = MagicMock()
    for i in range(3):
      ct.on_epoch_start(trainer, None, i + 1)
      ct.on_epoch_end(trainer, None, i + 1)
    assert len(ct.per_epoch()) == 3

  def test_total_aggregation(self, tmp_path):
    ct = CostTrackerCallback(tmp_path)
    trainer = MagicMock()
    for i in range(3):
      ct.on_epoch_start(trainer, None, i + 1)
      ct.on_epoch_end(trainer, None, i + 1)
    total = ct.total()
    assert total.wall_clock_s >= 0

  def test_on_loop_end_writes_artifact(self, tmp_path):
    ct = CostTrackerCallback(tmp_path)
    trainer = MagicMock()
    trainer.experiment = None
    trainer.tree = None
    ct.on_epoch_start(trainer, None, 1)
    ct.on_epoch_end(trainer, None, 1)
    ct.on_loop_end(trainer, None, {})
    assert (tmp_path / 'cost_summary.json').exists()

  def test_no_experiment_dir_raises_config_error(self):
    ct = CostTrackerCallback(None)
    trainer = MagicMock()
    trainer.experiment = None
    trainer.store = None
    ct.on_epoch_start(trainer, None, 1)
    ct.on_epoch_end(trainer, None, 1)
    with pytest.raises(ConfigError, match='cannot persist'):
      ct.on_loop_end(trainer, None, {})

  def test_state_dict_round_trip(self, tmp_path):
    ct = CostTrackerCallback(tmp_path)
    trainer = MagicMock()
    ct.on_epoch_start(trainer, None, 1)
    ct.on_epoch_end(trainer, None, 1)
    state = ct.state_dict()
    ct2 = CostTrackerCallback(tmp_path)
    ct2.load_state_dict(state)
    assert len(ct2.per_epoch()) == 1

  def test_measure_default(self, tmp_path):
    ct = CostTrackerCallback(tmp_path)
    entry = ct.measure(1, 1.234)
    assert entry.epoch == 1
    assert entry.wall_clock_s == 1.234

  def test_measure_with_result_metrics(self, tmp_path):
    ct = CostTrackerCallback(tmp_path)
    result = Result(metrics={'accuracy': 0.9})
    entry = ct.measure(1, 1.0, result=result)
    assert entry.metadata == {'accuracy': 0.9}

  def test_measure_override(self, tmp_path):
    class CustomCost(CostTrackerCallback):
      def measure(self, epoch, elapsed, result=None):
        return CostEntry(epoch=epoch, wall_clock_s=elapsed, api_calls=42)

    ct = CustomCost(tmp_path)
    trainer = MagicMock()
    ct.on_epoch_start(trainer, None, 1)
    ct.on_epoch_end(trainer, None, 1)
    assert ct.per_epoch()[0].api_calls == 42

  def test_artifact_registration(self, tmp_path):
    ct = CostTrackerCallback(tmp_path)
    assert 'cost_artifact' in ct.artifacts

  def test_load_state_dict_missing_entries_raises(self, tmp_path):
    """Missing entries key raises KeyError."""
    ct = CostTrackerCallback(tmp_path)
    with pytest.raises(KeyError, match='entries'):
      ct.load_state_dict({})

  def test_cost_tracker_raises_when_no_path_and_entries(self):
    """on_loop_end raises ConfigError when entries exist but no persistence path is available."""
    ct = CostTrackerCallback(None)
    trainer = MagicMock()
    trainer.experiment = None
    trainer.store = None
    ct.on_epoch_start(trainer, None, 0)
    ct.on_epoch_end(trainer, None, 0)
    with pytest.raises(ConfigError, match='cannot persist'):
      ct.on_loop_end(trainer, None, {})

  def test_no_error_when_no_entries(self):
    """on_loop_end does not raise when no entries have been recorded."""
    ct = CostTrackerCallback(None)
    trainer = MagicMock()
    trainer.experiment = None
    trainer.store = None
    ct.on_loop_end(trainer, None, {})

  def test_cumulative_usd_tracks_across_epochs(self):
    """cumulative_usd accumulates cost_usd from per-epoch CostEntry values."""

    class FixedCostTracker(CostTrackerCallback):
      def __init__(self, costs: list[float]) -> None:
        super().__init__(None)
        self._costs = costs
        self._call_idx = 0

      def measure(self, epoch, elapsed, result=None):
        cost = self._costs[self._call_idx]
        self._call_idx += 1
        return CostEntry(epoch=epoch, wall_clock_s=elapsed, cost_usd=cost)

    ct = FixedCostTracker([1.0, 2.25])
    trainer = MagicMock()
    ct.on_epoch_start(trainer, None, 0)
    ct.on_epoch_end(trainer, None, 0)
    assert ct.cumulative_usd == 1.0

    ct.on_epoch_start(trainer, None, 1)
    ct.on_epoch_end(trainer, None, 1)
    assert ct.cumulative_usd == 3.25

  def test_cumulative_usd_restored_from_state_dict(self):
    """load_state_dict restores cumulative_usd from persisted entries."""
    ct = CostTrackerCallback(None)
    ct._entries = [
      CostEntry(epoch=0, cost_usd=1.5),
      CostEntry(epoch=1, cost_usd=2.5),
    ]
    ct.cumulative_usd = 4.0
    state = ct.state_dict()

    ct2 = CostTrackerCallback(None)
    ct2.load_state_dict(state)
    assert ct2.cumulative_usd == 4.0

  def test_total_includes_cost_usd(self):
    """total() aggregates cost_usd across entries."""
    ct = CostTrackerCallback(None)
    ct._entries = [
      CostEntry(epoch=0, cost_usd=3.0),
      CostEntry(epoch=1, cost_usd=7.0),
    ]
    assert ct.total().cost_usd == 10.0


class TestCostEntryRoundTrip:
  def test_round_trip(self):
    c = CostEntry(epoch=1, wall_clock_s=5.0, api_calls=3, tokens_used=1000, cost_usd=12.5)
    d = c.to_dict()
    c2 = CostEntry.from_dict(d)
    assert c2.wall_clock_s == 5.0
    assert c2.tokens_used == 1000
    assert c2.cost_usd == 12.5

  def test_round_trip_default_cost_usd(self):
    """cost_usd defaults to 0.0 when not set."""
    c = CostEntry(epoch=0)
    d = c.to_dict()
    c2 = CostEntry.from_dict(d)
    assert c2.cost_usd == 0.0
