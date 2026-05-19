"""Tests for --context CLI flag: parsing, enforcement, and experiment journaling.

Covers:
  - Global flag registration and default values (2.1/2.2)
  - ``CLI.requires_context()`` semantics (2.3)
  - Dispatch enforcement for mutating commands (2.3)
  - Whitespace / empty-string rejection (2.3)
  - User context journaling on experiment resolution (2.4)
  - No double-journaling between dispatch and handler (DRY-04)
  - Subparser SUPPRESS default behavior (2.1)
  - Instance-level context_exempt_commands merging
"""

from autopilot.cli.command import (
  _BASE_CONTEXT_EXEMPT,
  CLI,
  Command,
)
from autopilot.cli.context import CLIContext, build_context
from autopilot.cli.helpers import journal_user_context
from autopilot.cli.resolvers import add_global_flags
from autopilot.core.experiment import Experiment
from unittest.mock import patch
import argparse
import pytest

# 4.1  Parsing and CLIContext


class TestParsingAndCLIContext:
  def test_context_flag_parsed_on_root_parser(self) -> None:
    """--context on root parser yields CLIContext.context == value."""
    parser = argparse.ArgumentParser()
    parser.set_defaults(handler=None)
    add_global_flags(parser)
    args = parser.parse_args(['--context', 'because reasons'])
    assert args.context == 'because reasons'

    ctx = build_context(args)
    assert ctx.context == 'because reasons'

  def test_context_flag_default_none(self) -> None:
    """Omitting --context leaves CLIContext.context as None."""
    parser = argparse.ArgumentParser()
    parser.set_defaults(handler=None)
    add_global_flags(parser)
    args = parser.parse_args([])
    assert args.context is None

    ctx = build_context(args)
    assert ctx.context is None


# 4.2  CLI.requires_context() semantics


class TestRequiresContext:
  def test_requires_context_mutating_command(self) -> None:
    """Non-exempt top-level command requires context."""
    cli = CLI()
    assert cli.requires_context('execute') is True

  def test_requires_context_exempt_command(self) -> None:
    """Top-level exempt command does not require context."""
    cli = CLI()
    assert cli.requires_context('query') is False

  def test_requires_context_exempt_subcommand(self) -> None:
    """Multi-token exempt subcommand does not require context."""
    cli = CLI()
    assert cli.requires_context('experiment show') is False

  def test_requires_context_mutating_subcommand(self) -> None:
    """Subcommand under a group with partial exemptions requires context."""
    cli = CLI()
    assert cli.requires_context('experiment add') is True

  def test_requires_context_empty_string(self) -> None:
    """Empty command string requires context (fail-safe: unknown commands are not exempt)."""
    cli = CLI()
    assert cli.requires_context('') is True

  def test_requires_context_with_extra_exempt(self) -> None:
    """Extra exemptions passed at init are merged and honored."""
    cli = CLI(context_exempt_commands=frozenset({'custom-read'}))
    assert cli.requires_context('custom-read') is False
    assert cli.requires_context('execute') is True

  def test_requires_context_none_extra_same_as_base(self) -> None:
    """None extra_exempt is equivalent to empty frozenset (base only)."""
    cli_none = CLI(context_exempt_commands=None)
    cli_empty = CLI(context_exempt_commands=frozenset())
    assert cli_none.requires_context('query') is False
    assert cli_none.requires_context('execute') is True
    assert cli_empty.requires_context('query') is False
    assert cli_empty.requires_context('execute') is True
    assert cli_none.context_exempt_commands == cli_empty.context_exempt_commands
    assert cli_none.context_exempt_commands == _BASE_CONTEXT_EXEMPT

  def test_context_exempt_commands_property(self) -> None:
    """Property returns the merged frozenset."""
    extras = frozenset({'my-read', 'my-list'})
    cli = CLI(context_exempt_commands=extras)
    assert cli.context_exempt_commands == _BASE_CONTEXT_EXEMPT | extras

  def test_instance_isolation(self) -> None:
    """Two CLI instances have independent exempt sets."""
    cli_a = CLI(context_exempt_commands=frozenset({'a-only'}))
    cli_b = CLI(context_exempt_commands=frozenset({'b-only'}))
    assert cli_a.requires_context('a-only') is False
    assert cli_a.requires_context('b-only') is True
    assert cli_b.requires_context('b-only') is False
    assert cli_b.requires_context('a-only') is True

  def test_bare_experiment_requires_context(self) -> None:
    """Bare 'experiment' (no subcommand) requires context."""
    cli = CLI()
    assert cli.requires_context('experiment') is True


# 4.3  Dispatch enforcement


class TestDispatchEnforcement:
  def _make_cli_with_handler(self):
    """Build a minimal CLI with a leaf command that records calls."""
    calls = []

    class Leaf(Command):
      name = 'mutate'
      help = 'mutating command'

      def forward(self, ctx, args):
        calls.append('invoked')

    class TestCLI(CLI):
      def __init__(self):
        super().__init__()
        self.mutate = Leaf()

    return TestCLI(), calls

  def test_enforcement_fails_without_context(self) -> None:
    """Mutating command with ctx.context=None triggers sys.exit."""
    cli, calls = self._make_cli_with_handler()
    parser = cli.build_parser()
    args = parser.parse_args(['mutate'])

    ctx = build_context(args)
    assert ctx.context is None

    with pytest.raises(SystemExit):
      cli.dispatch(ctx, args)
    assert calls == []

  def test_enforcement_fails_with_whitespace_only_context(self) -> None:
    """Whitespace-only --context is treated as None and rejected."""
    cli, calls = self._make_cli_with_handler()
    parser = cli.build_parser()
    args = parser.parse_args(['mutate', '--context', '   '])

    ctx = build_context(args)
    with pytest.raises(SystemExit):
      cli.dispatch(ctx, args)
    assert calls == []

  def test_enforcement_fails_with_empty_string_context(self) -> None:
    """Empty string --context is rejected."""
    cli, calls = self._make_cli_with_handler()
    parser = cli.build_parser()
    args = parser.parse_args(['mutate', '--context', ''])

    ctx = build_context(args)
    with pytest.raises(SystemExit):
      cli.dispatch(ctx, args)
    assert calls == []

  def test_enforcement_passes_with_context(self) -> None:
    """Mutating command with non-None context proceeds to handler."""
    cli, calls = self._make_cli_with_handler()
    parser = cli.build_parser()
    args = parser.parse_args(['mutate', '--context', 'testing enforcement'])

    ctx = build_context(args)
    cli.dispatch(ctx, args)
    assert calls == ['invoked']

  def test_all_base_exempt_pass_without_context(self) -> None:
    """Every entry in _BASE_CONTEXT_EXEMPT returns False from requires_context."""
    cli = CLI()
    for cmd in _BASE_CONTEXT_EXEMPT:
      assert cli.requires_context(cmd) is False, f'{cmd!r} should be exempt'

  def test_project_cli_extra_exemptions(self) -> None:
    """Project CLI subclass with context_exempt_commands extends exemption set."""

    class ProjectCLI(CLI):
      def __init__(self):
        super().__init__(context_exempt_commands=frozenset({'project-read'}))

    cli = ProjectCLI()
    assert cli.requires_context('project-read') is False
    assert cli.requires_context('project-write') is True
    assert cli.requires_context('query') is False

  def test_dispatch_with_instance_extras(self) -> None:
    """Dispatch uses instance-level requires_context for exempt commands."""
    calls = []

    class Leaf(Command):
      name = 'myread'
      help = 'a read command'

      def forward(self, ctx, args):
        calls.append('called')

    class TestCLI(CLI):
      def __init__(self):
        super().__init__(context_exempt_commands=frozenset({'myread'}))
        self.myread = Leaf()

    cli = TestCLI()
    parser = cli.build_parser()
    args = parser.parse_args(['myread'])
    ctx = build_context(args)
    cli.dispatch(ctx, args)
    assert calls == ['called']


# 4.4  Experiment wiring and DRY


class TestExperimentJournaling:
  def test_context_wired_to_experiment_add_context(self) -> None:
    """journal_user_context records the user string on the experiment context_log."""
    exp = Experiment(experiment_id='test-exp', hypothesis='test')
    ctx = CLIContext(context='updating rules')

    ns = argparse.Namespace(command='experiment', experiment_action='add')
    journal_user_context(ctx, exp, ns)

    assert len(exp.context_log) == 1
    entry = next(iter(exp.context_log))
    assert entry.reason == 'updating rules'
    assert entry.source == 'user'
    assert entry.command == 'experiment add'

  def test_journal_user_context_noop_when_none(self) -> None:
    """journal_user_context is a no-op when ctx.context is None."""
    exp = Experiment(experiment_id='test-exp', hypothesis='test')
    ctx = CLIContext(context=None)

    ns = argparse.Namespace(command='experiment', experiment_action='add')
    journal_user_context(ctx, exp, ns)

    assert len(exp.context_log) == 0

  def test_no_double_journaling(self) -> None:
    """Dispatch does not call experiment.add_context -- only the handler does.

    dispatch owns ExecutionRecord.context via create_execution_record;
    handlers own experiment.add_context (DRY-04).
    """
    exp = Experiment(experiment_id='jrnl-test', hypothesis='double check')
    calls: list[str] = []

    class LeafCmd(Command):
      name = 'leaf'
      help = 'test'

      def forward(self, ctx, args):
        calls.append('handler')
        journal_user_context(ctx, exp, args)

    class TestCLI(CLI):
      def __init__(self):
        super().__init__()
        self.leaf = LeafCmd()

    cli = TestCLI()
    parser = cli.build_parser()
    args = parser.parse_args(['leaf', '--context', 'reason here'])
    ctx = build_context(args)

    with patch.object(exp, 'add_context', wraps=exp.add_context) as spy:
      cli.dispatch(ctx, args)

    assert calls == ['handler']
    spy.assert_called_once()
    assert spy.call_args[0][0] == 'reason here'
    assert spy.call_args[1]['source'] == 'user'


# 4.5  Parser UX


class TestParserUX:
  def test_context_flag_on_subparser_suppressed(self) -> None:
    """Subparser --context uses SUPPRESS default so it doesn't clobber root value."""
    root = argparse.ArgumentParser()
    add_global_flags(root)

    subs = root.add_subparsers(dest='command')
    sub = subs.add_parser('test')
    add_global_flags(sub, is_subparser=True)

    args = root.parse_args(['--context', 'root-reason', 'test'])
    assert args.context == 'root-reason'

  def test_subparser_without_context_preserves_root(self) -> None:
    """Omitting --context on subparser preserves root parser's value."""
    root = argparse.ArgumentParser()
    add_global_flags(root)

    subs = root.add_subparsers(dest='command')
    sub = subs.add_parser('sub')
    add_global_flags(sub, is_subparser=True)

    args = root.parse_args(['--context', 'from-root', 'sub'])
    assert args.context == 'from-root'

  def test_context_on_subparser_overrides_root(self) -> None:
    """--context on subparser overrides the root value."""
    root = argparse.ArgumentParser()
    add_global_flags(root)

    subs = root.add_subparsers(dest='command')
    sub = subs.add_parser('sub')
    add_global_flags(sub, is_subparser=True)

    args = root.parse_args(['--context', 'root', 'sub', '--context', 'override'])
    assert args.context == 'override'


# additional edge cases


class TestEdgeCases:
  def test_cli_default_exempt_equals_base(self) -> None:
    """Default CLI instance exempt set equals _BASE_CONTEXT_EXEMPT."""
    cli = CLI()
    assert cli.context_exempt_commands == _BASE_CONTEXT_EXEMPT

  def test_context_stripped_before_enforcement(self) -> None:
    """Leading/trailing whitespace is stripped; non-empty result passes."""
    cli_calls = []

    class Leaf(Command):
      name = 'act'
      help = 'action'

      def forward(self, ctx, args):
        cli_calls.append(ctx.context)

    class TestCLI(CLI):
      def __init__(self):
        super().__init__()
        self.act = Leaf()

    cli = TestCLI()
    parser = cli.build_parser()
    args = parser.parse_args(['act', '--context', '  trimmed  '])
    ctx = build_context(args)
    cli.dispatch(ctx, args)

    assert cli_calls == ['trimmed']


# AutoPilotCLI forwarding tests


class TestAutoPilotCLIForwarding:
  def test_autopilot_cli_accepts_context_exempt(self) -> None:
    """AutoPilotCLI forwards context_exempt_commands to base CLI."""
    from autopilot.cli.main import AutoPilotCLI

    extras = frozenset({'my-custom-read'})
    cli = AutoPilotCLI(context_exempt_commands=extras)
    assert cli.requires_context('my-custom-read') is False
    assert cli.requires_context('execute') is True

  def test_autopilot_cli_default_has_base_exempt(self) -> None:
    """Default AutoPilotCLI uses _BASE_CONTEXT_EXEMPT."""
    from autopilot.cli.main import AutoPilotCLI

    cli = AutoPilotCLI()
    assert cli.context_exempt_commands == _BASE_CONTEXT_EXEMPT

  def test_autopilot_cli_extras_preserve_base(self) -> None:
    """AutoPilotCLI with extras still includes all base exempt commands."""
    from autopilot.cli.main import AutoPilotCLI

    extras = frozenset({'project-status'})
    cli = AutoPilotCLI(context_exempt_commands=extras)
    assert cli.requires_context('query') is False
    assert cli.requires_context('debug cost') is False
    assert cli.requires_context('project-status') is False
