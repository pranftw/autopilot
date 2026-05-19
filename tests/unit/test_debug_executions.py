"""Tests for debug executions inspection subcommands (list, show, tail).

Covers ExecutionsCommand.list_executions, show_execution, tail_executions
with filtering, index semantics, edge cases, and JSON output.
"""

from autopilot.cli.commands.debug import EXEC_LIST_DEFAULT_LIMIT, ExecutionsCommand
from autopilot.cli.context import CLIContext
from autopilot.cli.output import Output
from autopilot.core.config import AutoPilotConfig
from autopilot.tracking.executions import ExecutionRecord, log_execution
import argparse
import json
import pytest


@pytest.fixture
def workspace(tmp_path):
  ap = tmp_path / '.autopilot'
  ap.mkdir()
  return tmp_path


@pytest.fixture
def config(workspace):
  return AutoPilotConfig(workspace=workspace)


def _make_ctx(config, use_json=False):
  ctx = CLIContext(
    workspace=config.workspace,
    config=config,
    output=Output(use_json=use_json),
  )
  return ctx


def _make_record(
  command='execute',
  exit_code=0,
  stdout=None,
  stderr=None,
  duration_ms=10.0,
  timestamp='2025-01-15T12:00:00+00:00',
):
  return ExecutionRecord(
    timestamp=timestamp,
    command=command,
    args=[],
    duration_ms=duration_ms,
    exit_code=exit_code,
    stdout=stdout,
    stderr=stderr,
  )


def _seed_records(config, records):
  for r in records:
    log_execution(config.executions_path, r)


def _make_list_args(
  limit=EXEC_LIST_DEFAULT_LIMIT,
  filter_command=None,
  failures=False,
  context_contains=None,
  summary=False,
):
  return argparse.Namespace(
    limit=limit,
    filter_command=filter_command,
    failures=failures,
    context_contains=context_contains,
    summary=summary,
  )


def _make_show_args(index):
  return argparse.Namespace(index=index)


def _make_tail_args(limit=10):
  return argparse.Namespace(limit=limit)


class TestListExecutions:
  def test_list_empty(self, config, capsys):
    ctx = _make_ctx(config, use_json=True)
    cmd = ExecutionsCommand()
    cmd.list_executions(ctx, _make_list_args())
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['count'] == 0
    table_msg = [m for m in envelope['messages'] if m.get('type') == 'table']
    assert len(table_msg) == 1
    assert table_msg[0]['rows'] == []

  def test_list_with_records(self, config, capsys):
    records = [_make_record(timestamp=f'2025-01-15T12:0{i}:00+00:00') for i in range(5)]
    _seed_records(config, records)
    ctx = _make_ctx(config)
    cmd = ExecutionsCommand()
    cmd.list_executions(ctx, _make_list_args())
    captured = capsys.readouterr()
    lines = [ln for ln in captured.out.strip().split('\n') if ln.strip()]
    data_lines = [
      ln
      for ln in lines
      if not ln.startswith('-') and 'idx' not in ln and 'count' not in ln and 'OK' not in ln
    ]
    assert len(data_lines) == 5

  def test_list_filter_command(self, config, capsys):
    records = [
      _make_record(command='execute'),
      _make_record(command='optimize'),
      _make_record(command='execute'),
    ]
    _seed_records(config, records)
    ctx = _make_ctx(config, use_json=True)
    cmd = ExecutionsCommand()
    cmd.list_executions(ctx, _make_list_args(filter_command='execute'))
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['count'] == 2
    table_msg = [m for m in envelope['messages'] if m.get('type') == 'table']
    rows = table_msg[0]['rows']
    assert len(rows) == 2
    for row in rows:
      assert row['command'] == 'execute'

  def test_list_filter_failures(self, config, capsys):
    records = [
      _make_record(exit_code=0),
      _make_record(exit_code=1),
      _make_record(exit_code=2),
    ]
    _seed_records(config, records)
    ctx = _make_ctx(config)
    cmd = ExecutionsCommand()
    cmd.list_executions(ctx, _make_list_args(failures=True))
    captured = capsys.readouterr()
    assert 'count: 2' in captured.out

  def test_list_limit(self, config, capsys):
    records = [_make_record(timestamp=f'2025-01-15T12:{i:02d}:00+00:00') for i in range(10)]
    _seed_records(config, records)
    ctx = _make_ctx(config)
    cmd = ExecutionsCommand()
    cmd.list_executions(ctx, _make_list_args(limit=3))
    captured = capsys.readouterr()
    assert 'count: 3' in captured.out

  def test_list_json_output(self, config, capsys):
    records = [_make_record()]
    _seed_records(config, records)
    ctx = _make_ctx(config, use_json=True)
    cmd = ExecutionsCommand()
    cmd.list_executions(ctx, _make_list_args())
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert 'count' in envelope['result']
    table_msg = [m for m in envelope['messages'] if m.get('type') == 'table']
    assert len(table_msg) == 1
    assert len(table_msg[0]['rows']) == 1

  def test_list_combined_filters(self, config, capsys):
    records = [
      _make_record(command='execute', exit_code=0),
      _make_record(command='execute', exit_code=1),
      _make_record(command='optimize', exit_code=1),
      _make_record(command='execute', exit_code=2),
    ]
    _seed_records(config, records)
    ctx = _make_ctx(config)
    cmd = ExecutionsCommand()
    cmd.list_executions(ctx, _make_list_args(filter_command='execute', failures=True))
    captured = capsys.readouterr()
    assert 'count: 2' in captured.out

  def test_list_global_index_with_filter(self, config, capsys):
    records = []
    for i in range(10):
      cmd_name = 'execute' if i % 2 == 0 else 'optimize'
      records.append(
        _make_record(
          command=cmd_name,
          timestamp=f'2025-01-15T12:{i:02d}:00+00:00',
        )
      )
    _seed_records(config, records)
    ctx = _make_ctx(config, use_json=True)
    cmd = ExecutionsCommand()
    cmd.list_executions(ctx, _make_list_args(filter_command='execute'))
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    table_msg = [m for m in envelope['messages'] if m.get('type') == 'table']
    rows = table_msg[0]['rows']
    expected_indices = [0, 2, 4, 6, 8]
    actual_indices = [r['idx'] for r in rows]
    assert actual_indices == expected_indices

    ctx2 = _make_ctx(config, use_json=True)
    cmd.show_execution(ctx2, _make_show_args(4))
    captured2 = capsys.readouterr()
    show_envelope = json.loads(captured2.out)
    assert show_envelope['result']['execution']['command'] == 'execute'
    assert show_envelope['result']['execution']['timestamp'].startswith('2025-01-15T12:04')


class TestShowExecution:
  def test_show_valid_index(self, config, capsys):
    records = [_make_record(command='execute', stdout='hello world')]
    _seed_records(config, records)
    ctx = _make_ctx(config, use_json=True)
    cmd = ExecutionsCommand()
    cmd.show_execution(ctx, _make_show_args(0))
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    result = envelope['result']
    assert 'execution' in result
    record = result['execution']
    assert record['command'] == 'execute'
    assert record['stdout'] == 'hello world'
    assert 'timestamp' in record

  def test_show_invalid_index(self, config, capsys):
    records = [_make_record()]
    _seed_records(config, records)
    ctx = _make_ctx(config)
    cmd = ExecutionsCommand()
    with pytest.raises(SystemExit):
      cmd.show_execution(ctx, _make_show_args(999))
    captured = capsys.readouterr()
    assert 'out of range' in captured.err

  def test_show_negative_index(self, config, capsys):
    records = [_make_record()]
    _seed_records(config, records)
    ctx = _make_ctx(config)
    cmd = ExecutionsCommand()
    with pytest.raises(SystemExit):
      cmd.show_execution(ctx, _make_show_args(-1))
    captured = capsys.readouterr()
    assert 'out of range' in captured.err

  def test_show_on_empty_history(self, config, capsys):
    ctx = _make_ctx(config)
    cmd = ExecutionsCommand()
    with pytest.raises(SystemExit):
      cmd.show_execution(ctx, _make_show_args(0))
    captured = capsys.readouterr()
    assert 'out of range' in captured.err

  def test_show_index_equal_to_len_fails(self, config, capsys):
    records = [_make_record() for _ in range(5)]
    _seed_records(config, records)
    ctx = _make_ctx(config)
    cmd = ExecutionsCommand()
    with pytest.raises(SystemExit):
      cmd.show_execution(ctx, _make_show_args(5))
    captured = capsys.readouterr()
    assert 'out of range' in captured.err


class TestTailExecutions:
  def test_tail_default(self, config, capsys):
    records = [_make_record(timestamp=f'2025-01-15T12:{i:02d}:00+00:00') for i in range(15)]
    _seed_records(config, records)
    ctx = _make_ctx(config, use_json=True)
    cmd = ExecutionsCommand()
    cmd.tail_executions(ctx, _make_tail_args())
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['count'] == 10
    assert envelope['result']['total'] == 15
    table_msg = [m for m in envelope['messages'] if m.get('type') == 'table']
    rows = table_msg[0]['rows']
    expected_indices = list(range(5, 15))
    actual_indices = [r['idx'] for r in rows]
    assert actual_indices == expected_indices

  def test_tail_custom_limit(self, config, capsys):
    records = [_make_record(timestamp=f'2025-01-15T12:{i:02d}:00+00:00') for i in range(15)]
    _seed_records(config, records)
    ctx = _make_ctx(config, use_json=True)
    cmd = ExecutionsCommand()
    cmd.tail_executions(ctx, _make_tail_args(limit=5))
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['count'] == 5
    assert envelope['result']['total'] == 15
    table_msg = [m for m in envelope['messages'] if m.get('type') == 'table']
    rows = table_msg[0]['rows']
    expected_indices = list(range(10, 15))
    actual_indices = [r['idx'] for r in rows]
    assert actual_indices == expected_indices

  def test_tail_stdout_preview_truncated(self, config, capsys):
    long_stdout = 'A' * 200
    records = [_make_record(stdout=long_stdout)]
    _seed_records(config, records)
    ctx = _make_ctx(config, use_json=True)
    cmd = ExecutionsCommand()
    cmd.tail_executions(ctx, _make_tail_args())
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    table_msg = [m for m in envelope['messages'] if m.get('type') == 'table']
    preview = table_msg[0]['rows'][0]['stdout_preview']
    assert len(preview) <= 80
    assert preview == 'A' * 80

    records_nl = [_make_record(stdout='line1\nline2\nline3')]
    config_path = config.executions_path
    config_path.unlink()
    _seed_records(config, records_nl)
    ctx2 = _make_ctx(config, use_json=True)
    cmd.tail_executions(ctx2, _make_tail_args())
    captured2 = capsys.readouterr()
    envelope2 = json.loads(captured2.out)
    table_msg2 = [m for m in envelope2['messages'] if m.get('type') == 'table']
    preview2 = table_msg2[0]['rows'][0]['stdout_preview']
    assert '\n' not in preview2
    assert preview2 == 'line1 line2 line3'
