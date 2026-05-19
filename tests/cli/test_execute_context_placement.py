"""Tests for execute context placement docs and report compare help.

Plan 05: verifies that help text documents global flag ordering for
``execute`` and baseline-centric comparison semantics for
``report compare``.
"""

from autopilot.cli.main import build_parser
import argparse


def _get_subparser(root: argparse.ArgumentParser, name: str) -> argparse.ArgumentParser:
  """Walk the root parser and return the subparser for the given command name."""
  for action in root._actions:
    if isinstance(action, argparse._SubParsersAction):
      return action.choices[name]
  msg = f'subparser {name!r} not found'
  raise KeyError(msg)


def _get_nested_subparser(
  root: argparse.ArgumentParser,
  *names: str,
) -> argparse.ArgumentParser:
  """Walk nested subparsers to reach a deeply nested command."""
  parser = root
  for name in names:
    parser = _get_subparser(parser, name)
  return parser


def test_execute_help_mentions_flag_ordering() -> None:
  """Execute help text documents that globals must precede the script path."""
  parser = build_parser()
  execute_parser = _get_subparser(parser, 'execute')

  help_text = execute_parser.format_help()

  assert '--context' in help_text
  assert '--json' in help_text
  assert '--experiment' in help_text
  assert 'BEFORE' in help_text or 'before' in help_text
  assert 'forwarded' in help_text.lower()
  assert 'REMAINDER' in help_text or 'remainder' in help_text.lower()


def test_execute_context_after_script_swallowed() -> None:
  """Demonstrate that --context after a script path lands in remainder."""
  parser = build_parser()
  args = parser.parse_args(['execute', 'script.py', '--context', 'test-value'])

  assert args.extra_args is not None
  assert '--context' in args.extra_args
  assert 'test-value' in args.extra_args
  assert args.context is None


def test_report_compare_help_mentions_baseline() -> None:
  """Report compare help describes baseline-centric comparison and pairwise."""
  parser = build_parser()
  compare_parser = _get_nested_subparser(parser, 'report', 'compare')

  help_text = compare_parser.format_help()

  assert 'baseline' in help_text.lower()
  assert 'candidate' in help_text.lower()
  assert 'baseline-centric' in help_text.lower()
  assert 'pairwise' in help_text.lower()
