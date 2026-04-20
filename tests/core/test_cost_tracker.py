"""Tests for CostTracker callback."""

from autopilot.core.cost_tracker import CostTracker
from autopilot.core.models import Result
from unittest.mock import MagicMock
import time


class TestCostTracker:
  def test_single_epoch(self, tmp_path):
    ct = CostTracker(tmp_path)
    trainer = MagicMock()
    ct.on_epoch_start(trainer, 1)
    time.sleep(0.01)
    ct.on_epoch_end(trainer, 1, result=Result(metrics={'a': 1.0}))
    assert len(ct.per_epoch()) == 1
    assert ct.per_epoch()[0].wall_clock_s > 0

  def test_multi_epoch(self, tmp_path):
    ct = CostTracker(tmp_path)
    trainer = MagicMock()
    for i in range(3):
      ct.on_epoch_start(trainer, i + 1)
      ct.on_epoch_end(trainer, i + 1)
    assert len(ct.per_epoch()) == 3

  def test_total_aggregation(self, tmp_path):
    ct = CostTracker(tmp_path)
    trainer = MagicMock()
    for i in range(3):
      ct.on_epoch_start(trainer, i + 1)
      ct.on_epoch_end(trainer, i + 1)
    total = ct.total()
    assert total.wall_clock_s >= 0

  def test_on_loop_end_writes_artifact(self, tmp_path):
    ct = CostTracker(tmp_path)
    trainer = MagicMock()
    ct.on_epoch_start(trainer, 1)
    ct.on_epoch_end(trainer, 1)
    ct.on_loop_end(trainer, {})
    assert (tmp_path / 'cost_summary.json').exists()

  def test_no_experiment_dir_no_write(self):
    ct = CostTracker(None)
    trainer = MagicMock()
    ct.on_epoch_start(trainer, 1)
    ct.on_epoch_end(trainer, 1)
    ct.on_loop_end(trainer, {})

  def test_state_dict_round_trip(self, tmp_path):
    ct = CostTracker(tmp_path)
    trainer = MagicMock()
    ct.on_epoch_start(trainer, 1)
    ct.on_epoch_end(trainer, 1)
    state = ct.state_dict()
    ct2 = CostTracker(tmp_path)
    ct2.load_state_dict(state)
    assert len(ct2.per_epoch()) == 1
