"""Tests for CLI infrastructure fixes: JSON error envelopes, global flags on group
parsers, exit codes, collect_arguments/collect_subcommands MRO, ctx.fail(), and
removal of configure_commands().

Covers bugs 21-25 and 66 from the master plan.
"""

from autopilot.cli.command import CLI, Command
from autopilot.cli.context import CLIContext
from autopilot.cli.output import Output
from autopilot.cli.primitives import Argument, collect_arguments, collect_subcommands, subcommand
from autopilot.cli.resolvers import add_global_flags
from io import StringIO
from unittest.mock import patch
import argparse
import json
import pytest

# -- Test helpers --


class _SuccessCommand(Command):
  name = 'ok'
  help = 'succeeds'

  def forward(self, ctx, args):
    ctx.output.info('all good')
    ctx.output.result({'status': 'done'})


class _FailCommand(Command):
  name = 'fail'
  help = 'raises'

  def forward(self, ctx, args):
    msg = 'boom'
    raise RuntimeError(msg)


class _ErrorReturnCommand(Command):
  name = 'errret'
  help = 'errors and returns'

  def forward(self, ctx, args):
    ctx.fail('something went wrong')


def _build_test_cli(*cmds):
  """Build a CLI with the given command instances registered."""

  class TestCLI(CLI):
    prog = 'testcli'

    def __init__(self):
      super().__init__()
      for cmd in cmds:
        setattr(self, cmd.name, cmd)

  return TestCLI()


def _run_cli(cli, argv):
  """Run CLI.run() with captured stdout, returning (exit_code, stdout)."""
  out = StringIO()
  exit_code = 0
  merged = list(argv)
  if '--context' not in merged:
    merged.extend(['--context', 'test'])
  with patch('sys.stdout', out):
    try:
      cli(argv=merged)
    except SystemExit as e:
      exit_code = e.code if e.code is not None else 0
  return exit_code, out.getvalue()


# -- Bug 21: Exception with --json -> JSON error envelope --


class TestJsonErrorEnvelope:
  def test_exception_produces_json_envelope(self):
    cli = _build_test_cli(_FailCommand())
    code, stdout = _run_cli(cli, ['--json', 'fail'])
    assert code == 1
    envelope = json.loads(stdout)
    assert envelope['ok'] is False
    assert envelope['error'] == 'boom'
    assert 'messages' in envelope

  def test_exception_without_json_no_envelope(self):
    cli = _build_test_cli(_FailCommand())
    out = StringIO()
    err = StringIO()
    exit_code = 0
    with patch('sys.stdout', out), patch('sys.stderr', err):
      try:
        cli(argv=['fail', '--context', 'test'])
      except SystemExit as e:
        exit_code = e.code
    assert exit_code == 1
    assert not out.getvalue()
    assert 'boom' in err.getvalue()

  def test_buffered_messages_included_in_error_envelope(self):
    class InfoThenFail(Command):
      name = 'infofail'

      def forward(self, ctx, args):
        ctx.output.info('step 1 done')
        ctx.output.warn('step 2 warning')
        msg = 'step 3 crashed'
        raise RuntimeError(msg)

    cli = _build_test_cli(InfoThenFail())
    code, stdout = _run_cli(cli, ['--json', 'infofail'])
    assert code == 1
    envelope = json.loads(stdout)
    assert envelope['ok'] is False
    assert envelope['error'] == 'step 3 crashed'
    messages = envelope['messages']
    assert any(
      m.get('level') == 'error' and m.get('message') is not None and 'step 3' in m['message']
      for m in messages
    )


# -- Bug 22: Argparse failures --
# (Argparse errors go to stderr and exit 2 by default; full fix for --json
# wrapping argparse errors requires overriding parser.error(), which is planned
# for CLI integration tests. Here we verify baseline behavior.)


class TestArgparseErrors:
  def test_unknown_subcommand_exits_2(self):
    cli = _build_test_cli(_SuccessCommand())
    code, _ = _run_cli(cli, ['nonexistent'])
    assert code == 2

  def test_unknown_subcommand_with_json_exits_2(self):
    cli = _build_test_cli(_SuccessCommand())
    code, _ = _run_cli(cli, ['--json', 'nonexistent'])
    assert code == 2


# -- Bug 23: Global flags on group parsers --


class TestGlobalFlagsOnGroupParsers:
  def test_json_flag_between_group_and_leaf(self):
    """autopilot group --json leaf should work."""

    class LeafCmd(Command):
      name = 'leaf'
      help = 'a leaf'

      def forward(self, ctx, args):
        ctx.output.result({'answer': 42})

    class GroupCmd(Command):
      name = 'group'
      help = 'a group'

      def __init__(self):
        super().__init__()
        self.leaf = LeafCmd()

    cli = _build_test_cli(GroupCmd())
    code, stdout = _run_cli(cli, ['group', '--json', 'leaf'])
    assert code == 0
    envelope = json.loads(stdout)
    assert envelope['ok'] is True
    assert envelope['result']['answer'] == 42

  def test_json_flag_after_leaf(self):
    """autopilot group leaf --json should work."""

    class LeafCmd(Command):
      name = 'leaf'
      help = 'a leaf'

      def forward(self, ctx, args):
        ctx.output.result({'answer': 42})

    class GroupCmd(Command):
      name = 'group'
      help = 'a group'

      def __init__(self):
        super().__init__()
        self.leaf = LeafCmd()

    cli = _build_test_cli(GroupCmd())
    code, stdout = _run_cli(cli, ['group', 'leaf', '--json'])
    assert code == 0
    envelope = json.loads(stdout)
    assert envelope['ok'] is True

  def test_json_flag_before_group(self):
    """autopilot --json group leaf should work."""

    class LeafCmd(Command):
      name = 'leaf'
      help = 'a leaf'

      def forward(self, ctx, args):
        ctx.output.result({'answer': 42})

    class GroupCmd(Command):
      name = 'group'
      help = 'a group'

      def __init__(self):
        super().__init__()
        self.leaf = LeafCmd()

    cli = _build_test_cli(GroupCmd())
    code, stdout = _run_cli(cli, ['--json', 'group', 'leaf'])
    assert code == 0
    envelope = json.loads(stdout)
    assert envelope['ok'] is True


# -- Bug 24: Exit code consistency --


class TestExitCodes:
  def test_success_exits_0(self):
    cli = _build_test_cli(_SuccessCommand())
    code, _ = _run_cli(cli, ['ok'])
    assert code == 0

  def test_exception_exits_1(self):
    cli = _build_test_cli(_FailCommand())
    code, _ = _run_cli(cli, ['fail'])
    assert code == 1

  def test_ctx_fail_exits_1(self):
    cli = _build_test_cli(_ErrorReturnCommand())
    code, _ = _run_cli(cli, ['errret'])
    assert code == 1

  def test_ctx_fail_custom_exit_code(self):
    class CustomExit(Command):
      name = 'custom'

      def forward(self, ctx, args):
        ctx.fail('custom error', exit_code=3)

    cli = _build_test_cli(CustomExit())
    code, _ = _run_cli(cli, ['custom'])
    assert code == 3

  def test_ctx_fail_json_produces_envelope(self):
    cli = _build_test_cli(_ErrorReturnCommand())
    code, stdout = _run_cli(cli, ['--json', 'errret'])
    assert code == 1
    envelope = json.loads(stdout)
    assert envelope['ok'] is False
    assert envelope['error'] == 'something went wrong'

  def test_no_subcommand_exits_2(self):
    cli = _build_test_cli(_SuccessCommand())
    code, _ = _run_cli(cli, [])
    assert code == 2


# -- Bug 25: configure_commands removed --


class TestConfigureCommandsRemoved:
  def test_cli_has_no_configure_commands(self):
    assert not hasattr(CLI, 'configure_commands')


# -- Bug 66: collect_arguments MRO --


class TestCollectArgumentsMRO:
  def test_parent_arguments_inherited(self):
    class ParentCmd(Command):
      name = 'parent'
      alpha = Argument(default='a')

    class ChildCmd(ParentCmd):
      name = 'child'
      beta = Argument(default='b')

    args = collect_arguments(ChildCmd)
    names = [a.attr_name for a in args]
    assert 'beta' in names
    assert 'alpha' in names

  def test_child_override_wins(self):
    class ParentCmd(Command):
      name = 'parent'
      shared = Argument(default='parent_val')

    class ChildCmd(ParentCmd):
      name = 'child'
      shared = Argument(default='child_val')

    args = collect_arguments(ChildCmd)
    shared_args = [a for a in args if a.attr_name == 'shared']
    assert len(shared_args) == 1
    assert shared_args[0].kwargs['default'] == 'child_val'

  def test_grandchild_inherits_all(self):
    class GrandparentCmd(Command):
      name = 'gp'
      a = Argument(default='1')

    class ParentCmd(GrandparentCmd):
      name = 'p'
      b = Argument(default='2')

    class ChildCmd(ParentCmd):
      name = 'c'
      c = Argument(default='3')

    args = collect_arguments(ChildCmd)
    names = [a.attr_name for a in args]
    assert 'a' in names
    assert 'b' in names
    assert 'c' in names

  def test_no_arguments_returns_empty(self):
    class EmptyCmd(Command):
      name = 'empty'

    assert collect_arguments(EmptyCmd) == []


# -- Bug 66: collect_subcommands MRO --


class TestCollectSubcommandsMRO:
  def test_parent_subcommands_inherited(self):
    class ParentCmd(Command):
      name = 'parent'

      @subcommand('parent_action', help_text='from parent')
      def parent_action(self, ctx, args):
        pass

    class ChildCmd(ParentCmd):
      name = 'child'

      @subcommand('child_action', help_text='from child')
      def child_action(self, ctx, args):
        pass

    child = ChildCmd()
    subs = collect_subcommands(child)
    sub_names = [meta.name for meta, _ in subs]
    assert 'parent_action' in sub_names
    assert 'child_action' in sub_names

  def test_child_override_subcommand_wins(self):
    class ParentCmd(Command):
      name = 'parent'

      @subcommand('action', help_text='parent version')
      def action(self, ctx, args):
        pass

    class ChildCmd(ParentCmd):
      name = 'child'

      @subcommand('action', help_text='child version')
      def action(self, ctx, args):
        pass

    child = ChildCmd()
    subs = collect_subcommands(child)
    action_subs = [(m, fn) for m, fn in subs if m.name == 'action']
    assert len(action_subs) == 1
    assert action_subs[0][0].help == 'child version'

  def test_inherited_subcommands_are_bound_to_child(self):
    class ParentCmd(Command):
      name = 'parent'

      @subcommand('act', help_text='action')
      def act(self, ctx, args):
        pass

    class ChildCmd(ParentCmd):
      name = 'child'

    child = ChildCmd()
    subs = collect_subcommands(child)
    assert len(subs) == 1
    bound = subs[0][1]
    assert bound.__self__ is child


# -- Output.flush_error --


class TestFlushError:
  def test_flush_error_json_mode(self):
    out = Output(use_json=True)
    out.info('pre-message')
    buf = StringIO()
    with patch('sys.stdout', buf):
      out.flush_error('something broke')
    envelope = json.loads(buf.getvalue())
    assert envelope['ok'] is False
    assert envelope['error'] == 'something broke'
    assert len(envelope['messages']) == 1
    assert envelope['messages'][0]['message'] == 'pre-message'
    assert out._json_buffer == []

  def test_flush_error_text_mode_noop(self):
    out = Output(use_json=False)
    buf = StringIO()
    with patch('sys.stdout', buf):
      out.flush_error('something broke')
    assert not buf.getvalue()

  def test_flush_error_clears_buffer(self):
    out = Output(use_json=True)
    out.info('msg1')
    out.warn('msg2')
    buf = StringIO()
    with patch('sys.stdout', buf):
      out.flush_error('error')
    assert out._json_buffer == []


# -- CLIContext.fail --


class TestCtxFail:
  def test_fail_exits_with_code(self):
    ctx = CLIContext(output=Output(use_json=False))
    with pytest.raises(SystemExit) as exc_info:
      ctx.fail('bad thing')
    assert exc_info.value.code == 1

  def test_fail_custom_code(self):
    ctx = CLIContext(output=Output(use_json=False))
    with pytest.raises(SystemExit) as exc_info:
      ctx.fail('bad thing', exit_code=42)
    assert exc_info.value.code == 42

  def test_fail_json_produces_envelope(self):
    ctx = CLIContext(output=Output(use_json=True))
    ctx.output.info('prior msg')
    buf = StringIO()
    with patch('sys.stdout', buf), pytest.raises(SystemExit):
      ctx.fail('crash')
    envelope = json.loads(buf.getvalue())
    assert envelope['ok'] is False
    assert envelope['error'] == 'crash'
    assert any(m.get('message') == 'prior msg' for m in envelope['messages'])

  def test_fail_text_prints_error(self):
    ctx = CLIContext(output=Output(use_json=False))
    err_buf = StringIO()
    with patch('sys.stderr', err_buf), pytest.raises(SystemExit):
      ctx.fail('text error')
    assert 'text error' in err_buf.getvalue()


# -- Group parser global flags integration --


class TestGroupParserGlobalFlags:
  def test_group_parser_has_global_flags(self):
    """Verify that group parsers get global flags via register()."""

    class LeafCmd(Command):
      name = 'sub'
      help = 'a sub'

      def forward(self, ctx, args):
        pass

    class GroupCmd(Command):
      name = 'grp'
      help = 'a group'

      def __init__(self):
        super().__init__()
        self.sub = LeafCmd()

    parser = argparse.ArgumentParser()
    add_global_flags(parser)
    subs = parser.add_subparsers(dest='command')
    grp = GroupCmd()
    grp.register(subs)

    args = parser.parse_args(['grp', '--json', 'sub'])
    assert args.use_json is True

  def test_verbose_flag_on_group(self):
    class LeafCmd(Command):
      name = 'sub'
      help = 'a sub'

      def forward(self, ctx, args):
        pass

    class GroupCmd(Command):
      name = 'grp'
      help = 'a group'

      def __init__(self):
        super().__init__()
        self.sub = LeafCmd()

    parser = argparse.ArgumentParser()
    add_global_flags(parser)
    subs = parser.add_subparsers(dest='command')
    grp = GroupCmd()
    grp.register(subs)

    args = parser.parse_args(['grp', '--verbose', 'sub'])
    assert args.verbose is True


# -- Inline subcommand with group + global flags --


class TestInlineSubcommandGroupFlags:
  def test_inline_subcommand_on_group_accepts_json(self):
    class GroupCmd(Command):
      name = 'grp'
      help = 'a group'

      @subcommand('action', help_text='do something')
      def action(self, ctx, args):
        ctx.output.result({'done': True})

    cli = _build_test_cli(GroupCmd())
    code, stdout = _run_cli(cli, ['--json', 'grp', 'action'])
    assert code == 0
    envelope = json.loads(stdout)
    assert envelope['ok'] is True


# -- Inherited arguments actually work with argparse --


class TestInheritedArgumentsIntegration:
  def test_inherited_arguments_registered_on_parser(self):
    class BaseCmd(Command):
      name = 'base'
      base_opt = Argument('--base-opt', default='x')

    class DerivedCmd(BaseCmd):
      name = 'derived'
      derived_opt = Argument('--derived-opt', default='y')

      def forward(self, ctx, args):
        pass

    parser = argparse.ArgumentParser()
    add_global_flags(parser)
    subs = parser.add_subparsers(dest='command')
    derived = DerivedCmd()
    derived.register(subs)

    args = parser.parse_args(['derived', '--base-opt', 'hello', '--derived-opt', 'world'])
    assert args.base_opt == 'hello'
    assert args.derived_opt == 'world'
