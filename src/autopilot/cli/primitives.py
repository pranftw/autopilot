"""Low-level argparse scaffolding and declarative descriptors for the CLI.

Contains the argument/flag descriptors, subcommand decorator and metadata,
parser error routing, and collection helpers. No ``Command`` or ``CLI``
classes live here -- those stay in ``cli.command``.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NoReturn, TypeVar
import argparse

# argparse error routing


class ArgparseCLIError(Exception):
  """Argparse failure routed through CLI instead of sys.exit.

  Raised by ``AutopilotArgumentParser`` so ``CLI.run_direct()`` can emit a
  JSON error envelope (when ``--json`` is on argv) or stderr usage text,
  then exit with the correct code.

  Attributes:
    message: Human-readable error string from argparse.
    exit_code: Process exit code (typically 2 for usage errors, 0 for --help).
  """

  def __init__(self, message: str, *, exit_code: int = 2) -> None:
    """Create an argparse CLI error with a message and exit code.

    Args:
      message: Human-readable error string from argparse.
      exit_code: Process exit code (typically 2 for usage errors).
    """
    super().__init__(message)
    self.message = message
    self.exit_code = exit_code


class AutopilotArgumentParser(argparse.ArgumentParser):
  """Argparse subclass that raises ``ArgparseCLIError`` instead of exiting.

  Stock ``ArgumentParser`` calls ``sys.exit()`` on usage errors and ``--help``,
  which bypasses ``CLI.dispatch()`` and the JSON envelope. This subclass
  converts those exits into catchable exceptions.
  """

  def error(self, message: str) -> NoReturn:
    """Raise ``ArgparseCLIError`` instead of printing usage and exiting.

    Args:
      message: Error message from argparse (e.g. unrecognized arguments).

    Raises:
      ArgparseCLIError: Always, with exit_code 2.
    """
    raise ArgparseCLIError(message, exit_code=2)

  def exit(self, status: int = 0, message: str | None = None) -> NoReturn:
    """Raise ``ArgparseCLIError`` instead of calling ``sys.exit``.

    Args:
      status: Exit code (0 for --help, non-zero for errors).
      message: Optional message from argparse.

    Raises:
      ArgparseCLIError: Always, preserving the original status code.
    """
    raise ArgparseCLIError(message or '', exit_code=status)


# argument descriptors


class Argument:
  """Declarative argument descriptor for Command classes.

  Stores argparse flags and kwargs. When assigned as a class attribute,
  __set_name__ auto-derives --flag-name from the Python attribute name
  if no explicit flags are given.
  """

  def __init__(self, *flags: str, **kwargs: Any) -> None:
    """Create an argument descriptor with optional flag names and argparse kwargs.

    Args:
      *flags: Long or short option strings (e.g. ``'--foo'``). If empty, flags are
        derived from the attribute name in ``__set_name__``.
      **kwargs: Forwarded to ``argparse.ArgumentParser.add_argument``.
    """
    self.flags: tuple[str, ...] = flags
    self.kwargs: dict[str, Any] = kwargs
    self.attr_name: str | None = None

  def __set_name__(self, owner: type, name: str) -> None:
    """Bind this descriptor to its class attribute name and derive default flags."""
    self.attr_name = name
    if not self.flags:
      self.flags = (f'--{name.replace("_", "-")}',)

  def __get__(self, obj: Any, objtype: type | None = None) -> 'Argument':
    """Return this descriptor instance (descriptor is not bound per-instance)."""
    return self

  def add_to_parser(self, parser: argparse.ArgumentParser) -> None:
    """Register this argument on the given parser."""
    parser.add_argument(*self.flags, **self.kwargs)

  def __repr__(self) -> str:
    """Return a debug representation of flags and kwargs."""
    return f'Argument({self.flags!r}, {self.kwargs!r})'


class Flag(Argument):
  """Convenience descriptor for boolean flags. Sets action='store_true'."""

  def __init__(self, *flags: str, **kwargs: Any) -> None:
    """Create a boolean flag with ``store_true`` and default ``False`` unless overridden.

    Args:
      *flags: Option strings for the flag.
      **kwargs: Passed to ``Argument``; ``action`` defaults to ``store_true``.
    """
    kwargs.setdefault('action', 'store_true')
    kwargs.setdefault('default', False)
    super().__init__(*flags, **kwargs)


# subcommand decorator metadata


_Decorated = TypeVar('_Decorated', bound=Callable[..., Any])


@dataclass
class SubcommandMeta:
  """Metadata attached to methods decorated with @subcommand."""

  name: str
  help: str | None = None
  arguments: list[tuple[tuple, dict]] = field(default_factory=list)


def subcommand(name: str, *, help_text: str | None = None) -> Callable[[_Decorated], _Decorated]:
  """Mark a method as an inline subcommand (like @llm_step).

  Args:
    name: Subcommand name exposed on the CLI.
    help_text: Optional help text for the subparser.

  Returns:
    Decorator that attaches ``subcommand_meta`` to the wrapped function.
  """

  def decorator(fn: _Decorated) -> _Decorated:
    fn.subcommand_meta = SubcommandMeta(
      name=name,
      help=help_text,
    )
    return fn

  return decorator


def argument(*flags: str, **kwargs: Any) -> Callable[[_Decorated], _Decorated]:
  """Stack on top of @subcommand to add arguments to the inline subcommand.

  Args:
    *flags: Positional args for ``add_argument`` (option strings).
    **kwargs: Keyword args for ``add_argument``.

  Returns:
    Decorator that appends this argument to the subcommand metadata.
  """

  def decorator(fn: _Decorated) -> _Decorated:
    if not hasattr(fn, 'subcommand_meta'):
      msg = f'@argument must be stacked on top of @subcommand; {fn.__name__} has no subcommand_meta'
      raise TypeError(msg)
    fn.subcommand_meta.arguments.append((flags, kwargs))
    return fn

  return decorator


# collection helpers


def collect_arguments(cls: type) -> list[Argument]:
  """Introspect class hierarchy for Argument instances in definition order.

  Args:
    cls: Command class (MRO is walked from root to leaf).

  Returns:
    ``Argument`` descriptors in stable definition order without duplicates.
  """
  seen_names: set[str] = set()
  args: list[Argument] = []
  for klass in cls.__mro__:
    for name in klass.__dict__:
      if name in seen_names:
        continue
      value = klass.__dict__[name]
      if isinstance(value, Argument):
        seen_names.add(name)
        args.append(value)
  return args


def collect_subcommands(instance: object) -> list[tuple[SubcommandMeta, Any]]:
  """Introspect class hierarchy for methods with subcommand_meta.

  Args:
    instance: Command instance whose type MRO is searched.

  Returns:
    Pairs of subcommand metadata and bound handler methods.
  """
  seen_names: set[str] = set()
  results: list[tuple[SubcommandMeta, Any]] = []
  for klass in type(instance).__mro__:
    for attr_name in klass.__dict__:
      if attr_name in seen_names:
        continue
      seen_names.add(attr_name)
      method = getattr(klass, attr_name, None)
      if method is None or not hasattr(method, 'subcommand_meta'):
        continue
      meta: SubcommandMeta = method.subcommand_meta
      bound = getattr(instance, attr_name)
      results.append((meta, bound))
  return results
