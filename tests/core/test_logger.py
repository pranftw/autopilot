"""Tests for Logger base class and JSONLogger."""

from autopilot.core.logger import JSONLogger, Logger
from pathlib import Path
import pytest


class TestLoggerBase:
  def test_log_metrics_raises(self) -> None:
    with pytest.raises(NotImplementedError):
      Logger().log_metrics({'x': 1.0})

  def test_log_hyperparams_raises(self) -> None:
    with pytest.raises(NotImplementedError):
      Logger().log_hyperparams({'lr': 0.01})

  def test_log_raises(self) -> None:
    with pytest.raises(NotImplementedError):
      Logger().log('event')

  def test_finalize_is_noop(self) -> None:
    Logger().finalize('success')

  def test_name_default_none(self) -> None:
    assert Logger().name is None

  def test_version_default_none(self) -> None:
    assert Logger().version is None


class TestJSONLogger:
  def test_log_creates_file(self, tmp_path: Path) -> None:
    logger = JSONLogger(tmp_path)
    logger.log('test_event', 'hello')
    assert (tmp_path / 'events.jsonl').exists()

  def test_log_appends(self, tmp_path: Path) -> None:
    logger = JSONLogger(tmp_path)
    logger.log('a', 'first')
    logger.log('b', 'second')
    events = logger.load_events()
    assert len(events) == 2
    assert events[0].event_type == 'a'
    assert events[1].event_type == 'b'

  def test_log_metrics(self, tmp_path: Path) -> None:
    logger = JSONLogger(tmp_path)
    logger.log_metrics({'accuracy': 0.95}, step=1)
    events = logger.load_events()
    assert len(events) == 1
    assert events[0].event_type == 'metrics'
    assert events[0].metadata['metrics']['accuracy'] == 0.95
    assert events[0].metadata['step'] == 1

  def test_log_hyperparams(self, tmp_path: Path) -> None:
    logger = JSONLogger(tmp_path)
    logger.log_hyperparams({'lr': 0.001, 'batch_size': 32})
    events = logger.load_events()
    assert events[0].event_type == 'hyperparams'
    assert events[0].metadata['lr'] == 0.001

  def test_load_events_empty(self, tmp_path: Path) -> None:
    logger = JSONLogger(tmp_path)
    assert logger.load_events() == []

  def test_finalize_logs_event(self, tmp_path: Path) -> None:
    logger = JSONLogger(tmp_path)
    logger.finalize('success')
    events = logger.load_events()
    assert len(events) == 1
    assert events[0].event_type == 'finalize'
    assert events[0].message == 'success'

  def test_name_is_json(self) -> None:
    logger = JSONLogger(Path('/tmp'))
    assert logger.name == 'json'

  def test_creates_dir_on_log(self, tmp_path: Path) -> None:
    nested = tmp_path / 'a' / 'b'
    logger = JSONLogger(nested)
    logger.log('test', 'hi')
    assert nested.exists()


class TestCustomLogger:
  def test_subclass_override(self) -> None:
    class InMemoryLogger(Logger):
      def __init__(self):
        self.entries: list[tuple] = []

      def log(self, event_type, message='', metadata=None):
        self.entries.append((event_type, message))

      def log_metrics(self, metrics, step=None):
        self.entries.append(('metrics', metrics))

    logger = InMemoryLogger()
    logger.log('start', 'beginning')
    logger.log_metrics({'x': 1.0})
    assert len(logger.entries) == 2
