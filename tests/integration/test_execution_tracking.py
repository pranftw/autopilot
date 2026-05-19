"""Integration tests for the execution tracking system.

End-to-end tests exercising CLI dispatch tracking, execute modes,
debug inspection commands, and project-scoped storage via the full
AutoPilotCLI entry point with in-process invocation.
"""

from autopilot.cli.main import AutoPilotCLI
from autopilot.core.config import AutoPilotConfig
from autopilot.tracking.executions import load_executions
import json


def _run_cli(*argv_parts):
  """Run AutoPilotCLI with given argv, returning the integer exit code."""
  try:
    AutoPilotCLI()(argv=list(argv_parts))
  except SystemExit as e:
    return e.code if isinstance(e.code, int) else 1
  else:
    return 0


def _records(workspace, project=None):
  config = AutoPilotConfig(workspace=workspace, project=project)
  return load_executions(config.executions_path)


def test_execute_inline_tracked(tmp_path):
  exit_code = _run_cli(
    '--workspace', str(tmp_path), '--context', 'test', 'execute', '-c', 'print(1)'
  )
  assert exit_code == 0
  records = _records(tmp_path)
  exec_records = [r for r in records if r.command == 'execute']
  assert len(exec_records) == 1


def test_execute_file_tracked(tmp_path):
  script = tmp_path / 'test_script.py'
  script.write_text('print("from file")\n')
  exit_code = _run_cli('--workspace', str(tmp_path), '--context', 'test', 'execute', str(script))
  assert exit_code == 0
  records = _records(tmp_path)
  exec_records = [r for r in records if r.command == 'execute']
  assert len(exec_records) == 1


def test_execute_module_tracked(tmp_path):
  exit_code = _run_cli('--workspace', str(tmp_path), '--context', 'test', 'execute', '-m', 'site')
  assert exit_code == 0
  records = _records(tmp_path)
  exec_records = [r for r in records if r.command == 'execute']
  assert len(exec_records) == 1


def test_regular_command_tracked(tmp_path):
  exit_code = _run_cli('--workspace', str(tmp_path), '--context', 'test', 'workspace', 'init')
  assert exit_code == 0
  records = _records(tmp_path)
  ws_records = [r for r in records if r.command == 'workspace init']
  assert len(ws_records) == 1


def test_failed_command_tracked(tmp_path):
  exit_code = _run_cli(
    '--workspace',
    str(tmp_path),
    '--context',
    'test',
    'execute',
    '-c',
    'import sys; sys.exit(1)',
  )
  assert exit_code != 0
  records = _records(tmp_path)
  failed = [r for r in records if r.exit_code != 0]
  assert len(failed) >= 1


def test_debug_executions_list(tmp_path, capsys):
  _run_cli('--workspace', str(tmp_path), '--context', 'test', 'workspace', 'init')
  capsys.readouterr()
  _run_cli('--workspace', str(tmp_path), '--context', 'test', 'workspace', 'init')
  capsys.readouterr()
  _run_cli('--workspace', str(tmp_path), '--context', 'test', 'execute', '-c', 'print(1)')
  capsys.readouterr()
  _run_cli('--workspace', str(tmp_path), '--json', 'debug', 'executions', 'list')
  captured = capsys.readouterr()
  envelope = json.loads(captured.out)
  assert envelope['ok'] is True
  assert envelope['result']['count'] == 3


def test_debug_executions_show(tmp_path, capsys):
  _run_cli('--workspace', str(tmp_path), '--context', 'test', 'workspace', 'init')
  capsys.readouterr()
  _run_cli('--workspace', str(tmp_path), '--json', 'debug', 'executions', 'show', '0')
  captured = capsys.readouterr()
  envelope = json.loads(captured.out)
  assert envelope['ok'] is True
  result = envelope['result']
  assert 'execution' in result
  record = result['execution']
  assert 'timestamp' in record
  assert 'command' in record
  assert 'exit_code' in record
  assert 'args' in record


def test_debug_executions_tail(tmp_path, capsys):
  for _ in range(5):
    _run_cli('--workspace', str(tmp_path), '--context', 'test', 'workspace', 'init')
    capsys.readouterr()
  _run_cli('--workspace', str(tmp_path), '--json', 'debug', 'executions', 'tail', '-n', '2')
  captured = capsys.readouterr()
  envelope = json.loads(captured.out)
  assert envelope['ok'] is True
  assert envelope['result']['count'] <= 2
  assert envelope['result']['total'] == 5


def test_project_scoped_tracking(tmp_path):
  _run_cli('--workspace', str(tmp_path), '-p', 'testproj', '--context', 'test', 'workspace', 'init')
  expected = tmp_path / '.autopilot' / 'projects' / 'testproj' / 'executions.jsonl'
  records = load_executions(expected)
  assert len(records) >= 1


def test_execute_inline_captures_stdout(tmp_path):
  _run_cli('--workspace', str(tmp_path), '--context', 'test', 'execute', '-c', "print('hello')")
  records = _records(tmp_path)
  exec_records = [r for r in records if r.command == 'execute']
  assert len(exec_records) == 1
  assert exec_records[0].stdout is not None
  assert 'hello' in exec_records[0].stdout


def test_failed_command_captures_stderr(tmp_path):
  _run_cli(
    '--workspace',
    str(tmp_path),
    '--context',
    'test',
    'execute',
    '-c',
    'import sys; sys.stderr.write("error msg"); sys.exit(1)',
  )
  records = _records(tmp_path)
  failed = [r for r in records if r.exit_code != 0]
  assert len(failed) >= 1
  assert failed[0].stderr is not None
  assert 'error' in failed[0].stderr


def test_two_commands_append_order(tmp_path):
  _run_cli('--workspace', str(tmp_path), '--context', 'test', 'workspace', 'init')
  _run_cli('--workspace', str(tmp_path), '--context', 'test', 'execute', '-c', 'print(1)')
  records = _records(tmp_path)
  assert len(records) == 2
  assert records[0].command == 'workspace init'
  assert records[1].command == 'execute'
