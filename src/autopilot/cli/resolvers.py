"""CLI argument resolution and shared parser configuration.

Global flags (--experiment, --epoch, --workspace, --context, etc.) are declared
on both the root parser and every subparser so `autopilot --experiment foo status`
and `autopilot status --experiment foo` both work.

Subparsers use `default=SUPPRESS` so an unset flag is absent from the namespace
rather than clobbering the root parser's value with a blank default.
"""

import argparse


def _default(is_subparser: bool, value):
  """Return SUPPRESS for subparsers so unset flags don't clobber root values.

  Args:
    is_subparser: Whether this default applies to a nested subparser.
    value: Concrete default for the root parser.

  Returns:
    ``argparse.SUPPRESS`` or ``value`` depending on ``is_subparser``.
  """
  return argparse.SUPPRESS if is_subparser else value


def add_global_flags(
  parser: argparse.ArgumentParser,
  *,
  is_subparser: bool = False,
) -> None:
  """Add global flags to a parser.

  On the root parser (is_subparser=False) real defaults are set.
  On subparsers (is_subparser=True) defaults are SUPPRESS so that
  an unset flag is absent from the namespace rather than clobbering
  the root parser's already-parsed value.
  """
  suppress_default = is_subparser
  parser.add_argument(
    '-p',
    '--project',
    default=_default(suppress_default, None),
    metavar='NAME',
    help='project name (auto-detected when cwd is under autopilot/projects/<name>)',
  )
  parser.add_argument(
    '--workspace',
    default=_default(suppress_default, '.'),
    metavar='PATH',
    help='workspace root directory (default: current directory)',
  )
  parser.add_argument(
    '--experiment',
    default=_default(suppress_default, None),
    metavar='SLUG',
    help='experiment slug',
  )
  parser.add_argument(
    '--dataset', default=_default(suppress_default, None), metavar='NAME', help='dataset name'
  )
  parser.add_argument(
    '--split',
    default=_default(suppress_default, None),
    metavar='NAME',
    help='dataset split (train, val, test)',
  )
  parser.add_argument(
    '--epoch',
    type=int,
    default=_default(suppress_default, None),
    metavar='N',
    help='epoch number (default: None; handlers resolve to latest when omitted)',
  )
  parser.add_argument(
    '--hyperparams',
    default=_default(suppress_default, None),
    metavar='PATH',
    help='path to hyperparameters JSON file',
  )
  parser.add_argument(
    '--dry-run',
    action='store_true',
    default=_default(suppress_default, value=False),
    help='show what would happen without executing',
  )
  parser.add_argument(
    '--verbose',
    action='store_true',
    default=_default(suppress_default, value=False),
    help='enable verbose output',
  )
  parser.add_argument(
    '--no-color',
    action='store_true',
    default=_default(suppress_default, value=False),
    help='disable color output',
  )
  parser.add_argument(
    '--json',
    action='store_true',
    default=_default(suppress_default, value=False),
    dest='use_json',
    help='output in JSON format',
  )
  parser.add_argument(
    '--expose',
    action='store_true',
    default=_default(suppress_default, value=False),
    help='include executed commands in JSON output',
  )
  parser.add_argument(
    '--context',
    type=str,
    default=_default(suppress_default, None),
    help='reason for this action (required for mutating commands)',
  )
  parser.add_argument(
    '--wait',
    type=int,
    default=_default(suppress_default, None),
    metavar='TIMEOUT_MS',
    help=(
      'max milliseconds to wait for workspace lock on contention. '
      '0 = wait forever, N > 0 = wait up to N ms. absent = fail fast.'
    ),
  )
  parser.add_argument(
    '--retry',
    type=int,
    default=_default(suppress_default, 0),
    metavar='N',
    help=(
      'max retries with exponential backoff on lock contention. '
      '0 = fail fast (default). mutually exclusive with --wait.'
    ),
  )


def make_subparser(
  subparsers: argparse._SubParsersAction,
  name: str,
  help_text: str | None,
) -> argparse.ArgumentParser:
  """Create a subcommand parser with global flags.

  Args:
    subparsers: Parent subparsers action.
    name: Subcommand name.
    help_text: Help string for the subcommand.

  Returns:
    New subparser with global flags attached.
  """
  sub = subparsers.add_parser(name, help=help_text)
  add_global_flags(sub, is_subparser=True)
  return sub
