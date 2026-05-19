"""Track arbitrary shell commands in the execution audit trail.

``autopilot track --context 'reason' -- <argv...>``

Runs a shell command via ``subprocess.run(argv, shell=False)`` with the
``REMAINDER`` args after ``--`` forwarded verbatim (no ``sh -c`` wrapping,
no string positional). Relies on the dispatch-level ``ExecutionRecord``
logging to persist one JSONL row. The child process inherits the teed
``sys.stdout`` / ``sys.stderr`` from the surrounding ``CLI.dispatch``
``capture_output()`` context so live output and the dispatch record
buffers stay consistent.

After the subprocess finishes, raises ``SystemExit(returncode)`` so
``exit_code`` in the JSONL row matches the child process.  When
``--dry-run`` is active, reports what would run without executing.
"""

from autopilot.cli.command import Command
from autopilot.cli.context import CLIContext
from autopilot.cli.resolvers import make_subparser
import argparse
import subprocess


class TrackCommand(Command):
  """Run and audit an arbitrary shell command in ``executions.jsonl``.

  The command tokens after ``--`` are executed via ``subprocess.run`` with
  ``shell=False``. No second ``log_execution`` call is made -- the dispatch
  wrapper handles the single JSONL row.
  """

  name = 'track'
  help = 'Run a shell command with execution tracking'

  def register(self, subparsers: argparse._SubParsersAction) -> None:
    """Attach the track parser with REMAINDER after ``--``."""
    sub_parser = make_subparser(subparsers, self.name, self.help)
    sub_parser.add_argument(
      'user_argv',
      nargs=argparse.REMAINDER,
      help='command to run (after --)',
    )
    sub_parser.set_defaults(handler=self)

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Execute the user command and propagate its exit code.

    Strips a leading ``--`` from the remainder tokens, runs the command
    with ``shell=False``, and raises ``SystemExit`` with the child's
    return code so the dispatch wrapper records the correct exit code.

    When ``ctx.dry_run`` is set, reports what would run without executing.

    Raises:
      SystemExit: Always (unless dry-run), with the child process return code.
    """
    user_argv = list(args.user_argv) if args.user_argv else []
    if user_argv and user_argv[0] == '--':
      user_argv = user_argv[1:]

    if not user_argv:
      ctx.fail('no command provided; usage: autopilot track --context "reason" -- <command...>')

    if ctx.dry_run:
      if ctx.output.use_json:
        ctx.output.result({'argv': user_argv, 'dry_run': True})
      else:
        ctx.output.info(f'dry-run: would execute {user_argv}')
      return

    proc = subprocess.run(
      user_argv,
      shell=False,
      check=False,
    )

    if ctx.output.use_json:
      ctx.output.result(
        {'exit_code': proc.returncode, 'argv': user_argv},
        ok=proc.returncode == 0,
      )

    raise SystemExit(proc.returncode)
