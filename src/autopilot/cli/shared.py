"""Shared argument parsers and helpers for CLI commands.

Global flags (--experiment, --epoch, --workspace, etc.) are declared on both the
root parser and every subparser so `autopilot --experiment foo status` and
`autopilot status --experiment foo` both work.

Subparsers use `default=SUPPRESS` so an unset flag is absent from the namespace
rather than clobbering the root parser's value with a blank default.
"""

import argparse


def _default(is_subparser: bool, value):
  """Return SUPPRESS for subparsers so unset flags don't clobber root values."""
  return argparse.SUPPRESS if is_subparser else value


def add_global_flags(
  parser: argparse.ArgumentParser,
  *,
  include_project_config: bool = True,
  is_subparser: bool = False,
) -> None:
  """Add global flags to a parser.

  On the root parser (is_subparser=False) real defaults are set.
  On subparsers (is_subparser=True) defaults are SUPPRESS so that
  an unset flag is absent from the namespace rather than clobbering
  the root parser's already-parsed value.
  """
  s = is_subparser
  if include_project_config:
    parser.add_argument(
      '--config',
      default=_default(s, ''),
      metavar='PATH',
      help='path to project config override',
    )
  parser.add_argument(
    '-p',
    '--project',
    default=_default(s, ''),
    metavar='NAME',
    help='project name (auto-detected when cwd is under autopilot/projects/<name>)',
  )
  parser.add_argument(
    '--env', default=_default(s, ''), metavar='ENV', help='environment (staging, production)'
  )
  parser.add_argument(
    '--workspace',
    default=_default(s, '.'),
    metavar='PATH',
    help='workspace root directory (default: current directory)',
  )
  parser.add_argument(
    '--experiment', default=_default(s, ''), metavar='SLUG', help='experiment slug'
  )
  parser.add_argument('--dataset', default=_default(s, ''), metavar='NAME', help='dataset name')
  parser.add_argument(
    '--split',
    default=_default(s, ''),
    metavar='NAME',
    help='dataset split (train, val, test)',
  )
  parser.add_argument('--epoch', type=int, default=_default(s, 0), metavar='N', help='epoch number')
  parser.add_argument(
    '--hyperparams',
    default=_default(s, ''),
    metavar='PATH',
    help='path to hyperparameters JSON file',
  )
  parser.add_argument(
    '--dry-run',
    action='store_true',
    default=_default(s, False),
    help='show what would happen without executing',
  )
  parser.add_argument(
    '--verbose',
    action='store_true',
    default=_default(s, False),
    help='enable verbose output',
  )
  parser.add_argument(
    '--no-color',
    action='store_true',
    default=_default(s, False),
    help='disable color output',
  )
  parser.add_argument(
    '--json',
    action='store_true',
    default=_default(s, False),
    dest='use_json',
    help='output in JSON format',
  )
  parser.add_argument(
    '--expose',
    action='store_true',
    default=_default(s, False),
    help='include executed commands in JSON output',
  )


def make_subparser(
  subparsers: argparse._SubParsersAction,
  name: str,
  help_text: str,
  *,
  include_project_config: bool = True,
) -> argparse.ArgumentParser:
  """Create a subcommand parser with global flags."""
  sub = subparsers.add_parser(name, help=help_text)
  add_global_flags(sub, include_project_config=include_project_config, is_subparser=True)
  return sub
