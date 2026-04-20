"""Tests for CLI command base classes: Argument, Flag, Command, CLI."""

from autopilot.cli.command import (
  CLI,
  Argument,
  Command,
  Flag,
  SubcommandMeta,
  argument,
  collect_arguments,
  collect_subcommands,
  subcommand,
)
import argparse
import pytest


class TestArgument:
  def test_set_name_derives_flag(self) -> None:
    class Cmd(Command):
      name = 'test'
      my_option = Argument(default='x', help='test')

    assert Cmd.my_option.flags == ('--my-option',)
    assert Cmd.my_option.attr_name == 'my_option'

  def test_explicit_flags_preserved(self) -> None:
    class Cmd(Command):
      name = 'test'
      limit = Argument('--limit', type=int, default=0)

    assert Cmd.limit.flags == ('--limit',)

  def test_add_to_parser(self) -> None:
    arg = Argument('--count', type=int, default=5, help='item count')
    parser = argparse.ArgumentParser()
    arg.add_to_parser(parser)
    parsed = parser.parse_args(['--count', '10'])
    assert parsed.count == 10

  def test_repr(self) -> None:
    arg = Argument('--verbose', action='store_true')
    r = repr(arg)
    assert 'Argument' in r
    assert '--verbose' in r


class TestFlag:
  def test_defaults_to_store_true(self) -> None:
    flag = Flag('--verbose')
    assert flag.kwargs['action'] == 'store_true'
    assert flag.kwargs['default'] is False


class TestSubcommandDecorator:
  def test_attaches_meta(self) -> None:
    @subcommand('run', help='execute')
    def handler(self, ctx, args):
      pass

    assert hasattr(handler, '_subcommand_meta')
    assert isinstance(handler._subcommand_meta, SubcommandMeta)

  def test_meta_fields(self) -> None:
    @subcommand('run', help='execute')
    def handler(self, ctx, args):
      pass

    meta = handler._subcommand_meta
    assert meta.name == 'run'
    assert meta.help == 'execute'
    assert meta.arguments == []

  def test_argument_decorator_appends(self) -> None:
    @argument('--count', type=int, default=0)
    @subcommand('run', help='execute')
    def handler(self, ctx, args):
      pass

    meta = handler._subcommand_meta
    assert len(meta.arguments) == 1
    assert meta.arguments[0][0] == ('--count',)

  def test_argument_without_subcommand_raises(self) -> None:
    with pytest.raises(TypeError, match='@argument must be stacked on top of @subcommand'):

      @argument('--count', type=int)
      def handler(self, ctx, args):
        pass


class TestCollectArguments:
  def test_collects_in_definition_order(self) -> None:
    class Cmd(Command):
      name = 'test'
      alpha = Argument(default='a')
      beta = Argument(default='b')
      gamma = Argument(default='c')

    args = collect_arguments(Cmd)
    assert len(args) == 3
    assert [a.attr_name for a in args] == ['alpha', 'beta', 'gamma']

  def test_ignores_non_argument_attrs(self) -> None:
    class Cmd(Command):
      name = 'test'
      help = 'a command'
      alpha = Argument(default='a')
      regular = 'not an argument'

    args = collect_arguments(Cmd)
    assert len(args) == 1
    assert args[0].attr_name == 'alpha'


class TestCollectSubcommands:
  def test_collects_in_definition_order(self) -> None:
    class Cmd(Command):
      name = 'test'

      @subcommand('first', help='1st')
      def first(self, ctx, args):
        pass

      @subcommand('second', help='2nd')
      def second(self, ctx, args):
        pass

    cmd = Cmd()
    subs = collect_subcommands(cmd)
    assert len(subs) == 2
    assert subs[0][0].name == 'first'
    assert subs[1][0].name == 'second'

  def test_ignores_undecorated(self) -> None:
    class Cmd(Command):
      name = 'test'

      @subcommand('only', help='only one')
      def only(self, ctx, args):
        pass

      def regular(self):
        pass

    cmd = Cmd()
    subs = collect_subcommands(cmd)
    assert len(subs) == 1

  def test_returns_bound_methods(self) -> None:
    class Cmd(Command):
      name = 'test'

      @subcommand('act', help='action')
      def act(self, ctx, args):
        pass

    cmd = Cmd()
    subs = collect_subcommands(cmd)
    bound = subs[0][1]
    assert bound.__self__ is cmd


class TestCommand:
  def test_setattr_auto_registers_children(self) -> None:
    class Parent(Command):
      name = 'parent'

    class Child(Command):
      name = 'child'

    p = Parent()
    p.child = Child()
    assert 'child' in p._commands
    assert p._commands['child'].name == 'child'

  def test_non_command_attrs_not_registered(self) -> None:
    class Cmd(Command):
      name = 'test'

    c = Cmd()
    c.value = 42
    assert 'value' not in c._commands

  def test_register_leaf(self) -> None:
    class Leaf(Command):
      name = 'leaf'
      help = 'a leaf'
      count = Argument('--count', type=int, default=0, help='item count')

      def forward(self, ctx, args):
        pass

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    leaf = Leaf()
    leaf.register(sub)
    args = parser.parse_args(['leaf', '--count', '5', '--workspace', '.'])
    assert args.count == 5

  def test_register_group(self) -> None:
    class Child(Command):
      name = 'child'
      help = 'child cmd'

      def forward(self, ctx, args):
        pass

    class Group(Command):
      name = 'group'
      help = 'group cmd'

      def __init__(self):
        super().__init__()
        self.child = Child()

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    g = Group()
    g.register(sub)
    args = parser.parse_args(['group', 'child', '--workspace', '.'])
    assert args.cmd == 'group'

  def test_register_nested_3_levels(self) -> None:
    class C(Command):
      name = 'c'
      help = 'level 3'

      def forward(self, ctx, args):
        pass

    class B(Command):
      name = 'b'
      help = 'level 2'

      def __init__(self):
        super().__init__()
        self.c = C()

    class A(Command):
      name = 'a'
      help = 'level 1'

      def __init__(self):
        super().__init__()
        self.b = B()

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    a = A()
    a.register(sub)
    args = parser.parse_args(['a', 'b', 'c', '--workspace', '.'])
    assert args.cmd == 'a'

  def test_register_inline_subcommands(self) -> None:
    class Cmd(Command):
      name = 'cmd'
      help = 'inline test'

      @subcommand('action', help='do action')
      def action(self, ctx, args):
        pass

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    cmd = Cmd()
    cmd.register(sub)
    args = parser.parse_args(['cmd', 'action', '--workspace', '.'])
    assert args.handler is not None

  def test_register_mixed_class_and_inline(self) -> None:
    class Leaf(Command):
      name = 'leaf'
      help = 'class child'

      def forward(self, ctx, args):
        pass

    class Mixed(Command):
      name = 'mixed'
      help = 'mixed'

      def __init__(self):
        super().__init__()
        self.leaf = Leaf()

      @subcommand('inline', help='inline sub')
      def inline(self, ctx, args):
        pass

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    m = Mixed()
    m.register(sub)
    args1 = parser.parse_args(['mixed', 'leaf', '--workspace', '.'])
    assert args1.cmd == 'mixed'
    args2 = parser.parse_args(['mixed', 'inline', '--workspace', '.'])
    assert args2.handler is not None

  def test_container_dunders(self) -> None:
    class Parent(Command):
      name = 'parent'

    class A(Command):
      name = 'a'

    class B(Command):
      name = 'b'

    p = Parent()
    p.a = A()
    p.b = B()
    assert p['a'].name == 'a'
    assert 'b' in p
    assert 'c' not in p
    assert len(p) == 2
    assert set(dict(p).keys()) == {'a', 'b'}

  def test_init_subclass_name_derivation(self) -> None:
    class TrainCommand(Command):
      pass

    assert TrainCommand.name == 'train'

  def test_init_subclass_preserves_explicit_name(self) -> None:
    class MyCmd(Command):
      name = 'custom'

    assert MyCmd.name == 'custom'

  def test_repr(self) -> None:
    class Grp(Command):
      name = 'grp'

    class Sub(Command):
      name = 'sub'

    g = Grp()
    g.sub = Sub()
    r = repr(g)
    assert 'Grp' in r
    assert 'sub' in r


class TestCLI:
  def test_setattr_auto_registers_commands(self) -> None:
    class MyCLI(CLI):
      pass

    class TestCmd(Command):
      name = 'test'

    cli = MyCLI()
    cli.test = TestCmd()
    assert 'test' in cli._commands

  def test_build_parser(self) -> None:
    class TestLeaf(Command):
      name = 'hello'
      help = 'say hello'

      def forward(self, ctx, args):
        pass

    class MyCLI(CLI):
      def __init__(self):
        super().__init__()
        self.hello = TestLeaf()

    cli = MyCLI()
    parser = cli.build_parser()
    args = parser.parse_args(['hello', '--workspace', '.'])
    assert args.command == 'hello'

  def test_configure_commands_hook(self) -> None:
    class DynCmd(Command):
      name = 'dynamic'

    class MyCLI(CLI):
      def configure_commands(self):
        return [DynCmd()]

    cli = MyCLI()
    result = cli.configure_commands()
    assert len(result) == 1
    assert result[0].name == 'dynamic'

  def test_init_subclass_registers_project(self) -> None:
    key = '_test_register_project'

    class ProjCLI(CLI, project=key):
      pass

    assert key in CLI._project_registry
    assert CLI._project_registry[key] is ProjCLI
    del CLI._project_registry[key]

  def test_init_subclass_no_project_keyword(self) -> None:
    class PlainCLI(CLI):
      pass

    assert 'PlainCLI' not in CLI._project_registry

  def test_project_registry_lookup(self) -> None:
    key = '_test_lookup'

    class LookupCLI(CLI, project=key):
      pass

    assert CLI._project_registry[key] is LookupCLI
    del CLI._project_registry[key]

  def test_commands_property(self) -> None:
    class Leaf(Command):
      name = 'leaf'

    class MyCLI(CLI):
      def __init__(self):
        super().__init__()
        self.leaf = Leaf()

    cli = MyCLI()
    assert 'leaf' in cli.commands

  def test_repr(self) -> None:
    class Leaf(Command):
      name = 'leaf'

    class MyCLI(CLI):
      def __init__(self):
        super().__init__()
        self.leaf = Leaf()

    cli = MyCLI()
    r = repr(cli)
    assert 'MyCLI' in r
    assert 'leaf' in r
