"""Tests for executions list JSON field parity with ExecutionRecord (BUG-EXEC-LIST).

Validates that ``debug executions list --json`` rows include all fields
from ``ExecutionRecord`` (``context``, ``experiment``, ``project``,
``args``, ``extra``, ``exit_code``), and that the ``exit`` key has been
renamed to ``exit_code`` for consistency with ``to_dict()`` and ``show``.
"""

from autopilot.tracking.executions import create_execution_record, log_execution
from pathlib import Path
from tests.cli.conftest import run_cli_no_context, run_cli_text


def _seed_records(workspace: Path) -> None:
  """Seed ``executions.jsonl`` with two records for list tests."""
  exec_path = workspace / '.autopilot' / 'executions.jsonl'
  exec_path.parent.mkdir(parents=True, exist_ok=True)

  r1 = create_execution_record(
    command='optimize',
    args=['--max-epochs', '5'],
    duration_ms=1234.5,
    exit_code=0,
    experiment='exp-alpha',
    project='myproj',
    context='initial optimization run',
    extra={'agent': 'claude'},
  )
  log_execution(exec_path, r1)

  r2 = create_execution_record(
    command='execute',
    args=['-c', 'print("hello")'],
    duration_ms=42.0,
    exit_code=1,
    experiment=None,
    project=None,
    context=None,
  )
  log_execution(exec_path, r2)


def test_debug_executions_list_json_includes_context_experiment_args(tmp_path: Path) -> None:
  """BUG-EXEC-LIST: JSON rows must include context, experiment, and args."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  _seed_records(ws)

  result = run_cli_no_context(ws, ['debug', 'executions', 'list'])
  rows = result['result']['executions']
  assert len(rows) == 2

  row0 = rows[0]
  assert row0['context'] == 'initial optimization run'
  assert row0['experiment'] == 'exp-alpha'
  assert row0['project'] == 'myproj'
  assert row0['args'] == ['--max-epochs', '5']
  assert row0['extra'] == {'agent': 'claude'}
  assert row0['exit_code'] == 0
  assert 'exit' not in row0

  row1 = rows[1]
  assert row1['context'] is None
  assert row1['experiment'] is None
  assert row1['project'] is None
  assert row1['args'] == ['-c', 'print("hello")']
  assert row1['extra'] == {}
  assert row1['exit_code'] == 1
  assert 'exit' not in row1


def test_debug_executions_list_json_row_parity_with_to_dict(tmp_path: Path) -> None:
  """BUG-EXEC-LIST: JSON list rows include all ExecutionRecord fields.

  Verifies ``exit_code`` (not ``exit``), ``context`` (null vs str),
  ``experiment`` (null vs str), ``project`` (null vs str), ``args``
  (JSON array), and ``extra`` (dict) are present in every row.
  """
  ws = tmp_path / 'ws'
  ws.mkdir()
  _seed_records(ws)

  result = run_cli_no_context(ws, ['debug', 'executions', 'list'])
  required_keys = {
    'idx',
    'timestamp',
    'command',
    'exit_code',
    'duration_ms',
    'context',
    'experiment',
    'project',
    'args',
    'extra',
  }

  for row in result['result']['executions']:
    assert required_keys.issubset(row.keys()), f'missing keys: {required_keys - row.keys()}'
    assert 'exit' not in row, 'legacy "exit" key must be renamed to "exit_code"'
    assert isinstance(row['args'], list)
    assert isinstance(row['extra'], dict)
    assert isinstance(row['exit_code'], int)


def test_debug_executions_list_context_contains_filter_unchanged(tmp_path: Path) -> None:
  """BUG-EXEC-LIST: --context-contains filter still works after field additions."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  _seed_records(ws)

  result = run_cli_no_context(
    ws,
    ['debug', 'executions', 'list', '--context-contains', 'optimization'],
  )
  rows = result['result']['executions']
  assert len(rows) == 1
  assert rows[0]['context'] == 'initial optimization run'
  assert rows[0]['command'] == 'optimize'


def test_debug_executions_list_fields_match_show_execution(tmp_path: Path) -> None:
  """BUG-EXEC-LIST: list row fields match show output for the same record.

  Seeds a record with known context, experiment, and args, then asserts
  that the list row matches the show record for those keys.
  """
  ws = tmp_path / 'ws'
  ws.mkdir()
  _seed_records(ws)

  list_result = run_cli_no_context(ws, ['debug', 'executions', 'list'])
  list_rows = list_result['result']['executions']
  assert len(list_rows) == 2

  for row in list_rows:
    idx = row['idx']
    show_result = run_cli_no_context(ws, ['debug', 'executions', 'show', str(idx)])
    show_record = show_result['result']['execution']

    assert row['context'] == show_record['context']
    assert row['experiment'] == show_record['experiment']
    assert row['project'] == show_record['project']
    assert row['args'] == show_record['args']
    assert row['extra'] == show_record['extra']
    assert row['exit_code'] == show_record['exit_code']


def test_debug_executions_list_empty(tmp_path: Path) -> None:
  """Edge case: list with no records returns empty executions list."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  exec_path = ws / '.autopilot' / 'executions.jsonl'
  exec_path.parent.mkdir(parents=True, exist_ok=True)
  exec_path.touch()

  result = run_cli_no_context(ws, ['debug', 'executions', 'list'])
  assert result['result']['executions'] == []
  assert result['result']['count'] == 0


def test_debug_executions_list_failures_filter_with_new_fields(tmp_path: Path) -> None:
  """--failures filter still works and new fields are present on filtered rows."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  _seed_records(ws)

  result = run_cli_no_context(ws, ['debug', 'executions', 'list', '--failures'])
  rows = result['result']['executions']
  assert len(rows) == 1
  assert rows[0]['exit_code'] == 1
  assert rows[0]['command'] == 'execute'
  assert 'context' in rows[0]
  assert 'experiment' in rows[0]
  assert 'args' in rows[0]


def test_debug_executions_list_command_filter_with_new_fields(tmp_path: Path) -> None:
  """--command filter still works and new fields are present on filtered rows."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  _seed_records(ws)

  result = run_cli_no_context(
    ws,
    ['debug', 'executions', 'list', '--command', 'optimize'],
  )
  rows = result['result']['executions']
  assert len(rows) == 1
  assert rows[0]['command'] == 'optimize'
  assert rows[0]['context'] == 'initial optimization run'
  assert rows[0]['experiment'] == 'exp-alpha'


def test_debug_executions_list_table_columns_and_args_preview(tmp_path: Path) -> None:
  """Table display includes new column headers and truncated args preview.

  Verifies the human-readable table renders ``exit_code``, ``context``,
  ``experiment``, and ``args`` columns, and that long args are truncated
  with an ellipsis in text mode.
  """
  ws = tmp_path / 'ws'
  ws.mkdir()
  exec_path = ws / '.autopilot' / 'executions.jsonl'
  exec_path.parent.mkdir(parents=True, exist_ok=True)

  long_args = ['--' + ('x' * 20)] * 5
  record = create_execution_record(
    command='optimize',
    args=long_args,
    duration_ms=100.0,
    exit_code=0,
    experiment='exp-table',
    context='table test',
  )
  log_execution(exec_path, record)

  output = run_cli_text(ws, ['debug', 'executions', 'list'])

  assert 'exit_code' in output
  assert 'context' in output
  assert 'experiment' in output
  assert 'args' in output
  assert 'exp-table' in output
  assert 'table test' in output
  assert '...' in output
