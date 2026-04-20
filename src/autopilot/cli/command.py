"""Unified recursive command system for the AutoPilot CLI.

Single Command class -- leaf or group determined by whether it has children,
not by type. Like nn.Module doesn't distinguish container vs leaf.

CLI is the top-level orchestrator (like Trainer). __init__ configures,
__call__ runs. Entry point: AutoPilotCLI()().
"""

from autopilot.cli.context import build_context
from autopilot.cli.shared import add_global_flags, make_subparser
from autopilot.core.trainer import Trainer
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import argparse
import autopilot.core.paths as paths
import runpy
import sys
import traceback

# argument descriptors


class Argument:
  """Declarative argument descriptor for Command classes.

  Stores argparse flags and kwargs. When assigned as a class attribute,
  __set_name__ auto-derives --flag-name from the Python attribute name
  if no explicit flags are given.
  """

  def __init__(self, *flags: str, **kwargs: Any) -> None:
    self.flags: tuple[str, ...] = flags
    self.kwargs: dict[str, Any] = kwargs
    self.attr_name: str = ''

  def __set_name__(self, owner: type, name: str) -> None:
    self.attr_name = name
    if not self.flags:
      self.flags = (f'--{name.replace("_", "-")}',)

  def __get__(self, obj: Any, objtype: type | None = None) -> 'Argument':
    return self

  def add_to_parser(self, parser: argparse.ArgumentParser) -> None:
    parser.add_argument(*self.flags, **self.kwargs)

  def __repr__(self) -> str:
    return f'Argument({self.flags!r}, {self.kwargs!r})'


class Flag(Argument):
  """Convenience descriptor for boolean flags. Sets action='store_true'."""

  def __init__(self, *flags: str, **kwargs: Any) -> None:
    kwargs.setdefault('action', 'store_true')
    kwargs.setdefault('default', False)
    super().__init__(*flags, **kwargs)


# subcommand decorator metadata


@dataclass
class SubcommandMeta:
  """Metadata attached to methods decorated with @subcommand."""

  name: str
  help: str = ''
  arguments: list[tuple[tuple, dict]] = field(default_factory=list)
  include_project_config: bool = True


def subcommand(name: str, *, help: str = '', include_project_config: bool = True):
  """Mark a method as an inline subcommand (like @llm_step)."""

  def decorator(fn):
    fn._subcommand_meta = SubcommandMeta(
      name=name,
      help=help,
      include_project_config=include_project_config,
    )
    return fn

  return decorator


def argument(*flags: str, **kwargs: Any):
  """Stack on top of @subcommand to add arguments to the inline subcommand."""

  def decorator(fn):
    if not hasattr(fn, '_subcommand_meta'):
      raise TypeError(
        f'@argument must be stacked on top of @subcommand; {fn.__name__} has no _subcommand_meta'
      )
    fn._subcommand_meta.arguments.append((flags, kwargs))
    return fn

  return decorator


# collection helpers


def collect_arguments(cls: type) -> list[Argument]:
  """Introspect cls.__dict__ for Argument instances in definition order."""
  args: list[Argument] = []
  for name in cls.__dict__:
    value = cls.__dict__[name]
    if isinstance(value, Argument):
      args.append(value)
  return args


def collect_subcommands(instance: object) -> list[tuple[SubcommandMeta, Any]]:
  """Introspect for methods with _subcommand_meta. Returns (meta, bound_method) pairs."""
  results: list[tuple[SubcommandMeta, Any]] = []
  for attr_name in type(instance).__dict__:
    method = getattr(type(instance), attr_name, None)
    if method is None or not hasattr(method, '_subcommand_meta'):
      continue
    meta: SubcommandMeta = method._subcommand_meta
    bound = getattr(instance, attr_name)
    results.append((meta, bound))
  return results


# command


class Command:
  """Recursive command node. Leaf or group determined by children, not type.

  Like nn.Module: __setattr__ auto-registers child Commands.
  Override forward() for leaf command logic.
  """

  name: str = ''
  help: str = ''
  include_project_config: bool = True

  def __init_subclass__(cls, **kwargs: Any) -> None:
    super().__init_subclass__(**kwargs)
    if not cls.name:
      raw = cls.__name__
      if raw.endswith('Command'):
        raw = raw[: -len('Command')]
      cls.name = raw.lower()

  def __init__(self) -> None:
    object.__setattr__(self, '_commands', {})

  def __setattr__(self, name: str, value: object) -> None:
    if isinstance(value, Command):
      self._commands[name] = value
    super().__setattr__(name, value)

  def forward(self, ctx: Any, args: argparse.Namespace) -> None:
    raise NotImplementedError(f'{type(self).__name__} must implement forward()')

  def __call__(self, ctx: Any, args: argparse.Namespace) -> None:
    return self.forward(ctx, args)

  def register(self, subparsers: argparse._SubParsersAction) -> None:
    """Register this command (and children) onto an argparse subparsers action."""
    inline_subs = collect_subcommands(self)
    has_children = bool(self._commands) or bool(inline_subs)

    if has_children:
      group = subparsers.add_parser(self.name, help=self.help)
      child_sub = group.add_subparsers(dest=f'{self.name}_action', required=True)

      for _attr_name, child_cmd in self._commands.items():
        child_cmd.register(child_sub)

      for meta, bound_method in inline_subs:
        sub_parser = make_subparser(
          child_sub,
          meta.name,
          meta.help,
          include_project_config=meta.include_project_config,
        )
        for arg_flags, arg_kwargs in meta.arguments:
          sub_parser.add_argument(*arg_flags, **arg_kwargs)
        sub_parser.set_defaults(handler=bound_method)
    else:
      sub_parser = make_subparser(
        subparsers,
        self.name,
        self.help,
        include_project_config=self.include_project_config,
      )
      for arg_desc in collect_arguments(type(self)):
        arg_desc.add_to_parser(sub_parser)
      sub_parser.set_defaults(handler=self)

  # container dunders

  def __getitem__(self, key: str) -> 'Command':
    return self._commands[key]

  def __iter__(self):
    return iter(self._commands.items())

  def __contains__(self, key: str) -> bool:
    return key in self._commands

  def __len__(self) -> int:
    return len(self._commands)

  @property
  def commands(self) -> dict[str, 'Command']:
    return dict(self._commands)

  def __repr__(self) -> str:
    children = ', '.join(self._commands.keys())
    return f'{type(self).__name__}({self.name!r}, children=[{children}])'


# cli


class CLI:
  """Top-level CLI orchestrator. Like Trainer: __init__ configures, __call__ runs.

  Subclass for project CLIs. Use __init_subclass__(project='...') to
  auto-register project CLI classes.
  """

  prog: str = 'autopilot'
  description: str = 'AutoPilot optimization CLI'

  _project_registry: dict[str, type['CLI']] = {}

  def __init_subclass__(cls, *, project: str | None = None, **kwargs: Any) -> None:
    super().__init_subclass__(**kwargs)
    if project is not None:
      CLI._project_registry[project] = cls

  def __init__(self) -> None:
    object.__setattr__(self, '_commands', {})
    self.module = None
    self.generator = None
    self.judge = None

  def __setattr__(self, name: str, value: object) -> None:
    if isinstance(value, Command):
      self._commands[name] = value
    super().__setattr__(name, value)

  def __call__(self, *, argv: list[str] | None = None) -> None:
    return self.run(argv=argv)

  def build_parser(self) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=self.prog, description=self.description)
    add_global_flags(parser)
    subparsers = parser.add_subparsers(dest='command', help='available commands')
    for _name, cmd in self._commands.items():
      cmd.register(subparsers)
    return parser

  def build_context(self, args: argparse.Namespace) -> Any:
    """Build CLIContext from parsed args. Override for custom context."""
    return build_context(args)

  def dispatch(self, ctx: Any, args: argparse.Namespace) -> None:
    """Dispatch to handler with error handling."""
    handler = getattr(args, 'handler', None)
    if handler is None:
      self.build_parser().print_help()
      sys.exit(1)
    try:
      handler(ctx, args)
    except Exception as e:
      ctx.output.error(str(e))
      if ctx.verbose:
        traceback.print_exc()
      sys.exit(1)

  def _pre_parse(
    self,
    argv: list[str] | None,
  ) -> tuple[str, str, list[str]]:
    """Extract -p/--project and --workspace before full parse."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('-p', '--project', default='')
    pre.add_argument('--workspace', default='.')
    known, remaining = pre.parse_known_args(argv)
    return known.project, known.workspace, remaining

  def run(self, *, argv: list[str] | None = None) -> None:
    """Single entry point. Handles project dispatch internally."""
    project, workspace, remaining = self._pre_parse(argv)

    if project:
      ws = Path(workspace).resolve()
      if project not in CLI._project_registry:
        cli_script = paths.project_cli(ws, project)
        if cli_script.exists():
          project_dir = paths.root(ws, project)
          sys.path.insert(0, str(project_dir))
          try:
            runpy.run_path(str(cli_script), run_name='__autopilot_project__')
          finally:
            sys.path.pop(0)

      if project in CLI._project_registry:
        project_cls = CLI._project_registry[project]
        project_cli = project_cls()
        full_argv = remaining + ['--workspace', str(ws)]
        project_cli._run_direct(argv=full_argv)
        return

    self._run_direct(argv=argv)

  def _run_direct(self, *, argv: list[str] | None = None) -> None:
    """Parse and dispatch without project resolution."""
    parser = self.build_parser()
    args = parser.parse_args(argv)

    if not args.command:
      parser.print_help()
      sys.exit(0)

    ctx = self.build_context(args)
    ctx.generator = self.generator
    ctx.judge = self.judge
    ctx.module = self.module
    if self.module:
      ctx.trainer = Trainer(dry_run=ctx.dry_run)

    self.dispatch(ctx, args)

  def configure_commands(self) -> list[Command] | None:
    """Optional dynamic command setup hook."""
    return None

  @property
  def commands(self) -> dict[str, Command]:
    return dict(self._commands)

  def __repr__(self) -> str:
    cmds = ', '.join(self._commands.keys())
    return f'{type(self).__name__}(commands=[{cmds}])'
