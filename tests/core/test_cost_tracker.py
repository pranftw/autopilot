"""Tests for CostTrackerCallback."""

from autopilot.core.callbacks.cost import CostEntry, CostTrackerCallback
from autopilot.core.models import Result
from unittest.mock import MagicMock
import time


class TestCostTrackerCallback:
  def test_single_epoch(self, tmp_path):
    ct = CostTrackerCallback(tmp_path)
    trainer = MagicMock()
    ct.on_epoch_start(trainer, 1)
    time.sleep(0.01)
    ct.on_epoch_end(trainer, 1, result=Result(metrics={'a': 1.0}))
    assert len(ct.per_epoch()) == 1
    assert ct.per_epoch()[0].wall_clock_s > 0

  def test_multi_epoch(self, tmp_path):
    ct = CostTrackerCallback(tmp_path)
    trainer = MagicMock()
    for i in range(3):
      ct.on_epoch_start(trainer, i + 1)
      ct.on_epoch_end(trainer, i + 1)
    assert len(ct.per_epoch()) == 3

  def test_total_aggregation(self, tmp_path):
    ct = CostTrackerCallback(tmp_path)
    trainer = MagicMock()
    for i in range(3):
      ct.on_epoch_start(trainer, i + 1)
      ct.on_epoch_end(trainer, i + 1)
    total = ct.total()
    assert total.wall_clock_s >= 0

  def test_on_loop_end_writes_artifact(self, tmp_path):
    ct = CostTrackerCallback(tmp_path)
    trainer = MagicMock()
    ct.on_epoch_start(trainer, 1)
    ct.on_epoch_end(trainer, 1)
    ct.on_loop_end(trainer, {})
    assert (tmp_path / 'cost_summary.json').exists()

  def test_no_experiment_dir_no_write(self):
    ct = CostTrackerCallback(None)
    trainer = MagicMock()
    ct.on_epoch_start(trainer, 1)
    ct.on_epoch_end(trainer, 1)
    ct.on_loop_end(trainer, {})

  def test_state_dict_round_trip(self, tmp_path):
    ct = CostTrackerCallback(tmp_path)
    trainer = MagicMock()
    ct.on_epoch_start(trainer, 1)
    ct.on_epoch_end(trainer, 1)
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
    ct.on_epoch_start(trainer, 1)
    ct.on_epoch_end(trainer, 1)
    assert ct.per_epoch()[0].api_calls == 42

  def test_artifact_registration(self, tmp_path):
    ct = CostTrackerCallback(tmp_path)
    assert 'cost_artifact' in ct.artifacts


class TestCostEntryRoundTrip:
  def test_round_trip(self):
    c = CostEntry(epoch=1, wall_clock_s=5.0, api_calls=3, tokens_used=1000)
    d = c.to_dict()
    c2 = CostEntry.from_dict(d)
    assert c2.wall_clock_s == 5.0
    assert c2.tokens_used == 1000
