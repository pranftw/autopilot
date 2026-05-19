from autopilot.core.config import AutoPilotConfig
from autopilot.tracking.executions import (
  ExecutionRecord,
  TeeWriter,
  capture_output,
  create_execution_record,
  filter_executions,
  load_executions,
  log_execution,
  resolve_command,
)
from typing import Any, cast
import argparse
import io
import pytest
import sys


def _make_record(**overrides: Any) -> ExecutionRecord:
  defaults: dict[str, Any] = {
    'timestamp': '2026-01-01T00:00:00+00:00',
    'command': 'execute',
    'args': ['--epochs', '5'],
    'duration_ms': 123.4,
    'exit_code': 0,
    'stdout': 'hello\n',
    'stderr': None,
    'experiment': 'run-1',
    'project': None,
    'extra': {},
  }
  defaults.update(overrides)
  return ExecutionRecord(**defaults)


# --- ExecutionRecord serialization ---


def test_execution_record_to_dict():
  record = _make_record()
  d = record.to_dict()
  assert d['timestamp'] == '2026-01-01T00:00:00+00:00'
  assert d['command'] == 'execute'
  assert d['args'] == ['--epochs', '5']
  assert d['duration_ms'] == 123.4
  assert d['exit_code'] == 0
  assert d['stdout'] == 'hello\n'
  assert d['stderr'] is None
  assert d['experiment'] == 'run-1'
  assert d['project'] is None
  assert d['extra'] == {}


def test_execution_record_from_dict_round_trip():
  record = _make_record()
  d = record.to_dict()
  restored = ExecutionRecord.from_dict(d)
  assert restored == record


def test_execution_record_defaults():
  record = ExecutionRecord(timestamp='t', command='c')
  assert record.args == []
  assert record.duration_ms == 0.0
  assert record.exit_code == 0
  assert record.stdout is None
  assert record.stderr is None
  assert record.experiment is None
  assert record.project is None
  assert record.extra == {}


def test_execution_record_ignores_unknown_keys():
  data = {
    'timestamp': 't',
    'command': 'c',
    'unknown_field': 'should be dropped',
    'another_unknown': 42,
  }
  record = ExecutionRecord.from_dict(data)
  assert record.timestamp == 't'
  assert record.command == 'c'
  assert not hasattr(record, 'unknown_field')


def test_execution_record_from_dict_missing_required():
  with pytest.raises(TypeError):
    ExecutionRecord.from_dict({})
  with pytest.raises(TypeError):
    ExecutionRecord.from_dict({'timestamp': 'x'})


def test_execution_record_extra_round_trip():
  record = _make_record(extra={'git_sha': 'abc123', 'agent_id': 'a1'})
  d = record.to_dict()
  restored = ExecutionRecord.from_dict(d)
  assert restored.extra == {'git_sha': 'abc123', 'agent_id': 'a1'}


def test_execution_record_extra_default_empty():
  record = ExecutionRecord(timestamp='t', command='c')
  d = record.to_dict()
  assert d['extra'] == {}


# --- create_execution_record ---


def test_create_execution_record_timestamp():
  record = create_execution_record(
    command='execute',
    args=[],
    duration_ms=10.0,
    exit_code=0,
  )
  assert record.timestamp
  assert 'T' in record.timestamp
  assert '+' in record.timestamp or 'Z' in record.timestamp


def test_create_execution_record_fields():
  record = create_execution_record(
    command='optimize train',
    args=['--lr', '0.01'],
    duration_ms=500.0,
    exit_code=1,
    stdout='output',
    stderr='err',
    experiment='exp-1',
    project='proj-1',
    extra={'key': 'val'},
  )
  assert record.command == 'optimize train'
  assert record.args == ['--lr', '0.01']
  assert record.duration_ms == 500.0
  assert record.exit_code == 1
  assert record.stdout == 'output'
  assert record.stderr == 'err'
  assert record.experiment == 'exp-1'
  assert record.project == 'proj-1'
  assert record.extra == {'key': 'val'}


def test_create_execution_record_extra_none_normalizes():
  record = create_execution_record(
    command='execute',
    args=[],
    duration_ms=0,
    exit_code=0,
    extra=None,
  )
  assert record.extra == {}


# --- log_execution / load_executions ---


def test_log_and_load_round_trip(tmp_path):
  log_path = tmp_path / 'executions.jsonl'
  record = create_execution_record(
    command='execute',
    args=['--n', '3'],
    duration_ms=42.0,
    exit_code=0,
    stdout='out',
    stderr='err',
    experiment='exp-1',
    project='proj-1',
    extra={'sha': 'abc'},
  )
  log_execution(log_path, record)
  loaded = load_executions(log_path)
  assert len(loaded) == 1
  assert loaded[0] == record


def test_log_appends(tmp_path):
  log_path = tmp_path / 'executions.jsonl'
  for i in range(2):
    record = create_execution_record(
      command=f'cmd-{i}',
      args=[],
      duration_ms=float(i),
      exit_code=0,
    )
    log_execution(log_path, record)
  loaded = load_executions(log_path)
  assert len(loaded) == 2


def test_load_missing_file(tmp_path):
  log_path = tmp_path / 'nonexistent.jsonl'
  assert load_executions(log_path) == []


# --- filter_executions ---


def _make_filter_records():
  return [
    _make_record(command='execute', project='p1', experiment='e1', exit_code=0, duration_ms=10),
    _make_record(command='optimize', project='p1', experiment='e2', exit_code=1, duration_ms=200),
    _make_record(command='execute', project='p2', experiment='e1', exit_code=0, duration_ms=50),
    _make_record(command='debug', project='p2', experiment='e3', exit_code=2, duration_ms=300),
  ]


def test_filter_by_command():
  records = _make_filter_records()
  result = filter_executions(records, command='execute')
  assert len(result) == 2
  assert all(r.command == 'execute' for r in result)


def test_filter_by_project():
  records = _make_filter_records()
  result = filter_executions(records, project='p2')
  assert len(result) == 2
  assert all(r.project == 'p2' for r in result)


def test_filter_by_experiment():
  records = _make_filter_records()
  result = filter_executions(records, experiment='e1')
  assert len(result) == 2
  assert all(r.experiment == 'e1' for r in result)


def test_filter_by_exit_code():
  records = _make_filter_records()
  result = filter_executions(records, exit_code=1)
  assert len(result) == 1
  assert result[0].command == 'optimize'


def test_filter_combined():
  records = _make_filter_records()
  result = filter_executions(records, command='execute', project='p1')
  assert len(result) == 1
  assert result[0].experiment == 'e1'


def test_filter_no_match():
  records = _make_filter_records()
  result = filter_executions(records, command='nonexistent')
  assert result == []


def test_filter_with_predicate():
  records = _make_filter_records()
  result = filter_executions(records, predicate=lambda r: r.duration_ms > 100)
  assert len(result) == 2
  assert all(r.duration_ms > 100 for r in result)


def test_filter_named_and_predicate():
  records = _make_filter_records()
  result = filter_executions(
    records,
    command='execute',
    predicate=lambda r: r.duration_ms > 20,
  )
  assert len(result) == 1
  assert result[0].project == 'p2'


def test_filter_no_filters_returns_all():
  records = _make_filter_records()
  result = filter_executions(records)
  assert len(result) == 4
  assert result == records


# --- TeeWriter ---


def test_teewriter_captures_and_passes_through():
  original = io.StringIO()
  buf = io.StringIO()
  tee = TeeWriter(original, buf)
  tee.write('hello')
  assert original.getvalue() == 'hello'
  assert buf.getvalue() == 'hello'


def test_teewriter_flush():
  class FlushTracker:
    def __init__(self):
      self.flushed = False

    def flush(self):
      self.flushed = True

  orig_tracker = FlushTracker()
  buf_tracker = FlushTracker()
  tee = TeeWriter(orig_tracker, cast(Any, buf_tracker))
  tee.flush()
  assert orig_tracker.flushed
  assert buf_tracker.flushed


def test_teewriter_isatty_delegates():
  class FakeTTY:
    def isatty(self):
      return True

  buf = io.StringIO()
  tee = TeeWriter(FakeTTY(), buf)
  assert tee.isatty() is True


def test_teewriter_isatty_no_method():
  class NoIsatty:
    pass

  buf = io.StringIO()
  tee = TeeWriter(NoIsatty(), buf)
  assert tee.isatty() is False


def test_teewriter_empty_write():
  original = io.StringIO()
  buf = io.StringIO()
  tee = TeeWriter(original, buf)
  result = tee.write('')
  assert result == 0
  assert not buf.getvalue()


def test_teewriter_writelines():
  original = io.StringIO()
  buf = io.StringIO()
  tee = TeeWriter(original, buf)
  tee.writelines(['a', 'b', 'c'])
  assert original.getvalue() == 'abc'
  assert buf.getvalue() == 'abc'


def test_teewriter_encoding_delegates():
  class EncodedStream:
    encoding = 'utf-8'

  buf = io.StringIO()
  tee = TeeWriter(EncodedStream(), buf)
  assert tee.encoding == 'utf-8'


def test_teewriter_fileno_delegates():
  class FakeFileno:
    def fileno(self):
      return 1

  buf = io.StringIO()
  tee = TeeWriter(FakeFileno(), buf)
  assert tee.fileno() == 1


# --- capture_output ---


def test_capture_output_basic():
  with capture_output() as (stdout_buf, _stderr_buf):
    print('captured')
  assert 'captured' in stdout_buf.getvalue()


def test_capture_output_exception_restores_streams():
  original_out = sys.stdout
  original_err = sys.stderr
  msg = 'boom'
  with pytest.raises(ValueError, match='boom'), capture_output():
    raise ValueError(msg)
  assert sys.stdout is original_out
  assert sys.stderr is original_err


def test_capture_output_systemexit_restores_streams():
  original_out = sys.stdout
  original_err = sys.stderr
  with pytest.raises(SystemExit), capture_output():
    sys.exit(1)
  assert sys.stdout is original_out
  assert sys.stderr is original_err


def test_capture_output_nested():
  original_out = sys.stdout
  original_err = sys.stderr
  with capture_output() as (outer_stdout, _outer_stderr):
    outer_tee_out = sys.stdout
    with capture_output() as (inner_stdout, _inner_stderr):
      print('inner')
    assert sys.stdout is outer_tee_out
    assert 'inner' in inner_stdout.getvalue()
    print('outer')
  assert sys.stdout is original_out
  assert sys.stderr is original_err
  assert 'inner' in outer_stdout.getvalue()
  assert 'outer' in outer_stdout.getvalue()


# --- resolve_command ---


def test_resolve_command_simple():
  ns = argparse.Namespace(command='execute')
  assert resolve_command(ns) == 'execute'


def test_resolve_command_with_action():
  ns = argparse.Namespace(command='optimize', optimize_action='train')
  assert resolve_command(ns) == 'optimize train'


def test_resolve_command_empty():
  ns = argparse.Namespace()
  assert resolve_command(ns) == 'unknown'


# --- Config executions_path ---


def test_executions_path_workspace(tmp_path):
  config = AutoPilotConfig(workspace=tmp_path)
  expected = tmp_path / '.autopilot' / 'executions.jsonl'
  assert config.executions_path == expected


def test_executions_path_project(tmp_path):
  config = AutoPilotConfig(workspace=tmp_path, project='myproj')
  expected = tmp_path / '.autopilot' / 'projects' / 'myproj' / 'executions.jsonl'
  assert config.executions_path == expected
