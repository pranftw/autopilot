"""Tests for RunStateCallback."""

from autopilot.core.callbacks.run_state import RunStateCallback
from autopilot.core.models import Result
from autopilot.tracking.io import read_json
from unittest.mock import MagicMock


class TestRunStateCallback:
  def test_on_epoch_end_writes_running(self, tmp_path):
    cb = RunStateCallback(tmp_path)
    result = Result(metrics={'accuracy': 0.8}, passed=True)
    cb.on_epoch_end(trainer=MagicMock(), epoch=3, result=result)

    state = read_json(tmp_path / 'run_state.json')
    assert state['epoch'] == 3
    assert state['status'] == 'running'
    assert 'timestamp' in state

  def test_on_loop_end_writes_completed(self, tmp_path):
    cb = RunStateCallback(tmp_path)
    loop_result = {
      'total_epochs': 5,
      'stop_reason': 'plateau',
      'last_good_epoch': 4,
    }
    cb.on_loop_end(trainer=MagicMock(), result=loop_result)

    state = read_json(tmp_path / 'run_state.json')
    assert state['status'] == 'completed'
    assert state['stop_reason'] == 'plateau'
    assert state['last_good_epoch'] == 4
    assert state['epoch'] == 5

  def test_crash_detection_pattern(self, tmp_path):
    cb = RunStateCallback(tmp_path)
    cb.on_epoch_end(trainer=MagicMock(), epoch=7, result=None)

    state = read_json(tmp_path / 'run_state.json')
    assert state['status'] == 'running'

  def test_state_dict_empty(self, tmp_path):
    cb = RunStateCallback(tmp_path)
    assert cb.state_dict() == {}

  def test_overwrite_on_subsequent_epochs(self, tmp_path):
    cb = RunStateCallback(tmp_path)
    cb.on_epoch_end(trainer=MagicMock(), epoch=1, result=None)
    cb.on_epoch_end(trainer=MagicMock(), epoch=2, result=None)

    state = read_json(tmp_path / 'run_state.json')
    assert state['epoch'] == 2

  def test_artifact_registration(self, tmp_path):
    cb = RunStateCallback(tmp_path)
    assert 'run_state_artifact' in cb.artifacts

  def test_write_read_round_trip(self, tmp_path):
    cb = RunStateCallback(tmp_path)
    cb.on_epoch_end(trainer=MagicMock(), epoch=5, result=None)
    data = cb.run_state_artifact.read(tmp_path)
    assert data['epoch'] == 5
    assert data['status'] == 'running'
