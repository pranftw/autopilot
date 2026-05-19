"""Tests for Logger base class, JSONLogger, and module-level event helpers."""

from autopilot.core.logger import JSONLogger, Logger, append_event, create_event, load_events
from pathlib import Path
from unittest.mock import patch
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

  def test_finalize_failed(self, tmp_path: Path) -> None:
    """JSONLogger records 'failed' finalize event."""
    logger = JSONLogger(tmp_path / 'logs')
    logger.finalize('failed')
    events = logger.load_events()
    finalize_events = [e for e in events if e.event_type == 'finalize']
    assert len(finalize_events) == 1
    assert finalize_events[0].message == 'failed'

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


class TestEventHelpers:
  def test_create_event_sets_timestamp_from_utc_now_iso(self) -> None:
    with patch(
      'autopilot.core.logger.utc_now_iso',
      return_value='2099-01-01T00:00:00+00:00',
    ):
      event = create_event('t')
      assert event.timestamp == '2099-01-01T00:00:00+00:00'
      assert event.metadata == {}

  def test_create_event_with_metadata(self) -> None:
    event = create_event('deploy', message='deploying', metadata={'env': 'prod'})
    assert event.event_type == 'deploy'
    assert event.message == 'deploying'
    assert event.metadata == {'env': 'prod'}

  def test_append_load_events_roundtrip(self, tmp_path: Path) -> None:
    append_event(tmp_path, create_event('a', message='m'))
    events = load_events(tmp_path)
    assert len(events) == 1
    assert events[0].event_type == 'a'
    assert events[0].message == 'm'

  def test_load_events_empty_dir(self, tmp_path: Path) -> None:
    events = load_events(tmp_path)
    assert events == []

  def test_append_multiple_events(self, tmp_path: Path) -> None:
    append_event(tmp_path, create_event('first'))
    append_event(tmp_path, create_event('second'))
    events = load_events(tmp_path)
    assert len(events) == 2
    assert events[0].event_type == 'first'
    assert events[1].event_type == 'second'
