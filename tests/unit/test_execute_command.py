from autopilot.cli.commands.execute import ExecuteCommand
from autopilot.cli.context import CLIContext
from autopilot.cli.output import Output
from autopilot.core.config import AutoPilotConfig
from unittest.mock import MagicMock, patch
import argparse
import ast
import inspect
import pytest
import subprocess


@pytest.fixture
def workspace(tmp_path):
  ap = tmp_path / '.autopilot'
  ap.mkdir()
  return tmp_path


@pytest.fixture
def ctx(workspace):
  config = AutoPilotConfig(workspace=workspace)
  return CLIContext(
    workspace=workspace,
    config=config,
    output=Output(use_json=False),
  )


def _make_args(**kwargs):
  defaults = {
    'code': None,
    'module': None,
    'extra_args': [],
    'handler': None,
  }
  defaults.update(kwargs)
  return argparse.Namespace(**defaults)


def _mock_proc(returncode=0, stdout='', stderr=''):
  return subprocess.CompletedProcess(
    args=[],
    returncode=returncode,
    stdout=stdout,
    stderr=stderr,
  )


class TestParseMode:
  def test_parse_mode_inline(self):
    cmd = ExecuteCommand()
    args = _make_args(code='x')
    assert cmd.parse_mode(args) == ('inline', 'x', [])

  def test_parse_mode_module(self):
    cmd = ExecuteCommand()
    args = _make_args(module='pytest')
    assert cmd.parse_mode(args) == ('module', 'pytest', [])

  def test_parse_mode_module_with_extra(self):
    cmd = ExecuteCommand()
    args = _make_args(module='pytest', extra_args=['tests/'])
    assert cmd.parse_mode(args) == ('module', 'pytest', ['tests/'])

  def test_parse_mode_file(self):
    cmd = ExecuteCommand()
    args = _make_args(extra_args=['s.py', '--x'])
    assert cmd.parse_mode(args) == ('file', 's.py', ['--x'])

  def test_parse_mode_stdin(self):
    cmd = ExecuteCommand()
    args = _make_args()
    assert cmd.parse_mode(args) == ('stdin', None, [])

  def test_parse_mode_inline_with_extra(self):
    cmd = ExecuteCommand()
    args = _make_args(code='c', extra_args=['--a'])
    assert cmd.parse_mode(args) == ('inline', 'c', ['--a'])

  def test_parse_mode_inline_precedence_over_module(self):
    cmd = ExecuteCommand()
    args = _make_args(code='x', module='y')
    assert cmd.parse_mode(args) == ('inline', 'x', [])

  def test_parse_mode_strips_leading_separator(self):
    cmd = ExecuteCommand()
    args = _make_args(module='json.tool', extra_args=['--', '--help'])
    assert cmd.parse_mode(args) == ('module', 'json.tool', ['--help'])

  def test_parse_mode_file_strips_leading_separator(self):
    cmd = ExecuteCommand()
    args = _make_args(extra_args=['--', 'script.py', '--verbose'])
    assert cmd.parse_mode(args) == ('file', 'script.py', ['--verbose'])


class TestBuildCmd:
  def test_build_cmd_inline(self):
    cmd = ExecuteCommand()
    assert cmd.build_cmd('inline', 'code', []) == [
      'uv',
      'run',
      'python',
      '-c',
      'code',
    ]

  def test_build_cmd_module(self):
    cmd = ExecuteCommand()
    assert cmd.build_cmd('module', 'pytest', ['tests/']) == [
      'uv',
      'run',
      'python',
      '-m',
      'pytest',
      'tests/',
    ]

  def test_build_cmd_file(self):
    cmd = ExecuteCommand()
    assert cmd.build_cmd('file', 's.py', ['--e', '5']) == [
      'uv',
      'run',
      'python',
      's.py',
      '--e',
      '5',
    ]

  def test_build_cmd_stdin(self):
    cmd = ExecuteCommand()
    assert cmd.build_cmd('stdin', None, []) == ['uv', 'run', 'python']


class TestForward:
  def test_forward_inline_success(self, ctx):
    args = _make_args(code='print("hi")')
    with (
      patch('autopilot.cli.commands.execute.subprocess.run') as mock_run,
      patch('autopilot.cli.commands.execute.sys.stdout') as mock_out,
      patch('autopilot.cli.commands.execute.sys.stderr'),
    ):
      mock_run.return_value = _mock_proc(stdout='hi\n')
      cmd = ExecuteCommand()
      cmd.forward(ctx, args)
      mock_out.write.assert_any_call('hi\n')

  def test_forward_nonzero_exit(self, ctx):
    args = _make_args(code='fail')
    with (
      patch('autopilot.cli.commands.execute.subprocess.run') as mock_run,
      patch('autopilot.cli.commands.execute.sys.stdout'),
      patch('autopilot.cli.commands.execute.sys.stderr'),
    ):
      mock_run.return_value = _mock_proc(returncode=1, stderr='error\n')
      cmd = ExecuteCommand()
      with pytest.raises(SystemExit) as exc_info:
        cmd.forward(ctx, args)
      assert exc_info.value.code == 1

  def test_forward_no_self_tracking(self):
    src = inspect.getsource(ExecuteCommand)
    tree = ast.parse(src)
    names_used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs_used = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    all_names = names_used | attrs_used
    assert 'log_execution' not in all_names
    assert 'create_execution_record' not in all_names

    import_froms = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imported_modules = [node.module for node in import_froms if node.module]
    assert 'autopilot.tracking.executions' not in imported_modules

  def test_forward_stdin_tty_fails(self, ctx, capsys):
    args = _make_args()
    with patch('autopilot.cli.commands.execute.sys.stdin') as mock_stdin:
      mock_stdin.isatty.return_value = True
      cmd = ExecuteCommand()
      with pytest.raises(SystemExit):
        cmd.forward(ctx, args)
    captured = capsys.readouterr()
    assert 'provide -c' in captured.err

  def test_forward_stdin_success(self, ctx):
    args = _make_args()
    with (
      patch('autopilot.cli.commands.execute.sys.stdin') as mock_stdin,
      patch('autopilot.cli.commands.execute.subprocess.run') as mock_run,
      patch('autopilot.cli.commands.execute.sys.stdout'),
      patch('autopilot.cli.commands.execute.sys.stderr'),
    ):
      mock_stdin.isatty.return_value = False
      mock_stdin.read.return_value = 'print(1)'
      mock_run.return_value = _mock_proc(stdout='1\n')
      cmd = ExecuteCommand()
      cmd.forward(ctx, args)
      mock_run.assert_called_once()
      call_kwargs = mock_run.call_args[1]
      assert call_kwargs['input'] == 'print(1)'

  def test_forward_empty_stdout(self, ctx):
    args = _make_args(code='pass')
    ctx.output = MagicMock(spec=Output)
    ctx.output.use_json = False
    with (
      patch('autopilot.cli.commands.execute.subprocess.run') as mock_run,
      patch('autopilot.cli.commands.execute.sys.stdout') as mock_out,
      patch('autopilot.cli.commands.execute.sys.stderr') as mock_err,
    ):
      mock_run.return_value = _mock_proc(stdout='', stderr='')
      cmd = ExecuteCommand()
      cmd.forward(ctx, args)
      mock_out.write.assert_not_called()
      mock_err.write.assert_not_called()

  def test_forward_result_payload(self, ctx):
    args = _make_args(code='print(1)')
    ctx.output = MagicMock(spec=Output)
    ctx.output.use_json = False
    with (
      patch('autopilot.cli.commands.execute.subprocess.run') as mock_run,
      patch('autopilot.cli.commands.execute.sys.stdout'),
      patch('autopilot.cli.commands.execute.sys.stderr'),
    ):
      mock_run.return_value = _mock_proc(stdout='1\n')
      cmd = ExecuteCommand()
      cmd.forward(ctx, args)
      ctx.output.result.assert_called_once_with(
        {'mode': 'inline', 'exit_code': 0, 'stdout': '1\n', 'stderr': ''},
        ok=True,
      )
