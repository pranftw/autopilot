"""Tests for RunStateCallback."""

from autopilot.core.callbacks.run_state import RunStateCallback
from autopilot.core.errors import ConfigError
from autopilot.core.models import Result
from autopilot.tracking.io import read_json
from typing import Any, cast
from unittest.mock import MagicMock
import pytest


def _mock_trainer() -> MagicMock:
  trainer = MagicMock()
  trainer.experiment = None
  trainer.tree = None
  return trainer


class TestRunStateCallback:
  def test_on_epoch_end_writes_running(self, tmp_path):
    cb = RunStateCallback(tmp_path)
    result = Result(metrics={'accuracy': 0.8})
    cb.on_epoch_end(trainer=_mock_trainer(), module=None, epoch=3, result=result)

    state_raw = read_json(tmp_path / 'run_state.json')
    assert state_raw is not None
    state = cast(dict[str, Any], state_raw)
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
    cb.on_loop_end(trainer=_mock_trainer(), module=None, result=loop_result)

    state_raw = read_json(tmp_path / 'run_state.json')
    assert state_raw is not None
    state = cast(dict[str, Any], state_raw)
    assert state['status'] == 'completed'
    assert state['stop_reason'] == 'plateau'
    assert state['last_good_epoch'] == 4
    assert state['epoch'] == 5

  def test_crash_detection_pattern(self, tmp_path):
    cb = RunStateCallback(tmp_path)
    cb.on_epoch_end(trainer=_mock_trainer(), module=None, epoch=7, result=None)

    state_raw = read_json(tmp_path / 'run_state.json')
    assert state_raw is not None
    state = cast(dict[str, Any], state_raw)
    assert state['status'] == 'running'

  def test_state_dict_empty(self, tmp_path):
    cb = RunStateCallback(tmp_path)
    assert cb.state_dict() == {}

  def test_overwrite_on_subsequent_epochs(self, tmp_path):
    cb = RunStateCallback(tmp_path)
    cb.on_epoch_end(trainer=_mock_trainer(), module=None, epoch=1, result=None)
    cb.on_epoch_end(trainer=_mock_trainer(), module=None, epoch=2, result=None)

    state_raw = read_json(tmp_path / 'run_state.json')
    assert state_raw is not None
    state = cast(dict[str, Any], state_raw)
    assert state['epoch'] == 2

  def test_artifact_registration(self, tmp_path):
    cb = RunStateCallback(tmp_path)
    assert 'run_state_artifact' in cb.artifacts

  def test_write_read_round_trip(self, tmp_path):
    cb = RunStateCallback(tmp_path)
    cb.on_epoch_end(trainer=_mock_trainer(), module=None, epoch=5, result=None)
    data = cb.run_state_artifact.read(tmp_path)
    assert data is not None
    assert data['epoch'] == 5
    assert data['status'] == 'running'

  def test_on_loop_end_total_epochs_required(self, tmp_path):
    """total_epochs uses strict access; missing key raises KeyError."""
    cb = RunStateCallback(tmp_path)
    with pytest.raises(KeyError, match='total_epochs'):
      cb.on_loop_end(trainer=_mock_trainer(), module=None, result={})

  def test_on_loop_end_without_orchestrator_keys(self, tmp_path):
    """last_good_epoch and stop_reason are optional (orchestrator-only)."""
    cb = RunStateCallback(tmp_path)
    cb.on_loop_end(trainer=_mock_trainer(), module=None, result={'total_epochs': 3})
    state_raw = read_json(tmp_path / 'run_state.json')
    assert state_raw is not None
    state = cast(dict[str, Any], state_raw)
    assert state['epoch'] == 3
    assert state['last_good_epoch'] is None
    assert state['stop_reason'] is None

  def test_run_state_raises_when_base_dir_none(self):
    """ConfigError raised when _resolve_dir returns None."""
    cb = RunStateCallback(path=None)
    trainer = _mock_trainer()
    with pytest.raises(ConfigError, match='run state directory not available'):
      cb.on_epoch_end(trainer=trainer, module=None, epoch=0, result=None)

  def test_run_state_raises_on_loop_end_when_base_dir_none(self):
    """ConfigError raised on on_loop_end when _resolve_dir returns None."""
    cb = RunStateCallback(path=None)
    trainer = _mock_trainer()
    with pytest.raises(ConfigError, match='run state directory not available'):
      cb.on_loop_end(trainer=trainer, module=None, result={'total_epochs': 1})
