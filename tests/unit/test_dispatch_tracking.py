from autopilot.cli.command import CLI
from autopilot.cli.context import CLIContext
from autopilot.cli.output import Output
from autopilot.core.config import AutoPilotConfig
from autopilot.tracking.executions import TeeWriter, load_executions, resolve_command
from unittest.mock import patch
import argparse
import pytest
import sys
import time


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
    context='test',
  )


@pytest.fixture
def cli():
  return CLI()


def _make_args(handler, command='test', **extra):
  return argparse.Namespace(handler=handler, command=command, **extra)


def _load_records(ctx):
  return load_executions(ctx.config.executions_path)


def test_dispatch_tracks_successful_command(cli, ctx):
  def handler(c, a):
    print('ok')

  args = _make_args(handler)
  cli.dispatch(ctx, args)
  records = _load_records(ctx)
  assert len(records) == 1
  assert records[0].exit_code == 0


def test_dispatch_tracks_failed_command(cli, ctx):
  def handler(c, a):
    msg = 'bad'
    raise ValueError(msg)

  args = _make_args(handler)
  with pytest.raises(SystemExit) as exc_info:
    cli.dispatch(ctx, args)
  assert exc_info.value.code == 1
  records = _load_records(ctx)
  assert len(records) == 1
  assert records[0].exit_code == 1


def test_dispatch_tracks_sys_exit(cli, ctx):
  def handler(c, a):
    sys.exit(42)

  args = _make_args(handler)
  with pytest.raises(SystemExit) as exc_info:
    cli.dispatch(ctx, args)
  assert exc_info.value.code == 42
  records = _load_records(ctx)
  assert len(records) == 1
  assert records[0].exit_code == 42


def test_dispatch_captures_stdout(cli, ctx):
  def handler(c, a):
    print('hello')

  args = _make_args(handler)
  cli.dispatch(ctx, args)
  records = _load_records(ctx)
  assert records[0].stdout is not None
  assert 'hello' in records[0].stdout


def test_dispatch_captures_stderr(cli, ctx):
  def handler(c, a):
    sys.stderr.write('err_msg')

  args = _make_args(handler)
  cli.dispatch(ctx, args)
  records = _load_records(ctx)
  assert records[0].stderr is not None
  assert 'err_msg' in records[0].stderr


def test_dispatch_records_timing(cli, ctx):
  def handler(c, a):
    time.sleep(0.001)

  args = _make_args(handler)
  cli.dispatch(ctx, args)
  records = _load_records(ctx)
  assert records[0].duration_ms > 0


def test_dispatch_records_project(cli, ctx):
  ctx.project = 'foo'

  def handler(c, a):
    pass

  args = _make_args(handler)
  cli.dispatch(ctx, args)
  records = _load_records(ctx)
  assert records[0].project == 'foo'


def test_dispatch_records_experiment(cli, ctx):
  ctx.experiment = 'exp-1'

  def handler(c, a):
    pass

  args = _make_args(handler)
  cli.dispatch(ctx, args)
  records = _load_records(ctx)
  assert records[0].experiment == 'exp-1'


def test_dispatch_records_command_name(cli, ctx):
  def handler(c, a):
    pass

  args = argparse.Namespace(
    handler=handler,
    command='optimize',
    optimize_action='train',
  )
  cli.dispatch(ctx, args)
  records = _load_records(ctx)
  assert records[0].command == 'optimize train'


def test_dispatch_streams_restored(cli, ctx):
  def handler(c, a):
    print('test')

  args = _make_args(handler)
  cli.dispatch(ctx, args)
  assert not isinstance(sys.stdout, TeeWriter)
  assert not isinstance(sys.stderr, TeeWriter)


def test_dispatch_logging_failure_silent(cli, ctx):
  def handler(c, a):
    print('success')

  args = _make_args(handler)
  with patch('autopilot.cli.command.log_execution', side_effect=OSError('disk full')):
    cli.dispatch(ctx, args)


def test_dispatch_no_handler_no_tracking(cli, ctx):
  args = argparse.Namespace(handler=None, command='test')
  with pytest.raises(SystemExit):
    cli.dispatch(ctx, args)
  assert not ctx.config.executions_path.exists()


def test_extract_argv_various_types(cli):
  args = argparse.Namespace(
    handler=lambda c, a: None,
    command='test',
    flag=True,
    name='val',
    items=['a', 'b'],
    missing=None,
  )
  result = cli._extract_argv(args)
  assert '--flag' in result
  assert '--name=val' in result
  assert 'a' in result
  assert 'b' in result
  assert '--missing' not in result
  assert '--missing=None' not in result


def test_dispatch_sys_exit_none_code(cli, ctx):
  def handler(c, a):
    sys.exit(None)

  args = _make_args(handler)
  with pytest.raises(SystemExit) as exc_info:
    cli.dispatch(ctx, args)
  assert exc_info.value.code == 1
  records = _load_records(ctx)
  assert records[0].exit_code == 1


def test_extract_argv_false_bool_skipped(cli):
  args = argparse.Namespace(
    handler=lambda c, a: None,
    command='test',
    flag=False,
  )
  result = cli._extract_argv(args)
  assert '--flag' not in result


def test_extract_argv_handler_command_skipped(cli):
  args = argparse.Namespace(
    handler=lambda c, a: None,
    command='test',
    optimize_action='train',
    name='val',
  )
  result = cli._extract_argv(args)
  for item in result:
    assert 'handler' not in item
    assert 'command' not in item
    assert 'optimize_action' not in item
  assert '--name=val' in result


def test_dispatch_empty_stdout_is_none(cli, ctx):
  def handler(c, a):
    pass

  args = _make_args(handler)
  cli.dispatch(ctx, args)
  records = _load_records(ctx)
  assert records[0].stdout is None


def test_dispatch_records_raw_argv(cli, ctx):
  def handler(c, a):
    pass

  args = _make_args(handler)
  raw = ['execute', '-c', 'print(1)']
  cli.dispatch(ctx, args, argv=raw)
  records = _load_records(ctx)
  assert records[0].args == ['execute', '-c', 'print(1)']


def test_dispatch_keyboard_interrupt_no_record(cli, ctx):
  def handler(c, a):
    raise KeyboardInterrupt

  args = _make_args(handler)
  with pytest.raises(KeyboardInterrupt):
    cli.dispatch(ctx, args)
  assert not ctx.config.executions_path.exists()


def test_resolve_command_preserves_cli_order():
  """Namespace with multiple actions preserves argparse insertion order."""
  ns = argparse.Namespace(
    command='store',
    reflog_action='reflog',
    expire_action='expire',
  )
  assert resolve_command(ns) == 'store reflog expire'


def test_resolve_command_single_action():
  """Namespace with single action resolves correctly."""
  ns = argparse.Namespace(command='store', recover_action='recover')
  assert resolve_command(ns) == 'store recover'
