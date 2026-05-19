"""Run Python via uv with four input modes.

Modes:

1. Inline: ``autopilot execute -c 'print(1)'`` -- code string after ``-c``; remaining
   tokens go to the child Python process (e.g. as ``sys.argv``).
2. Module: ``autopilot execute -m pytest tests/`` -- ``-m`` module name; remainder
   forwarded like ``python -m``.
3. File: ``autopilot execute script.py --arg`` -- first remainder token is the script
   path; further tokens forwarded.
4. Stdin: ``echo 'print(2)' | autopilot execute`` -- no ``-c``, ``-m``, or file
   token; stdin must not be a TTY. Script body is read from stdin and passed to
   the subprocess via ``input=`` (equivalent to piping into ``python``).

Two output channels (FRICTION-002):

- **Human mode** (default): subprocess stdout/stderr are re-emitted to the
  terminal in real time via ``sys.stdout``/``sys.stderr``. Agents should not
  parse this stream.
- **JSON mode** (``--json``): subprocess output is NOT echoed to stdout.
  Instead, a single JSON envelope is emitted containing the full captured
  stdout and stderr inside the ``result`` fields. Agents should rely
  exclusively on the JSON envelope for structured output.

JSON result keys (``--json``):
  - ``mode``: execution mode (inline, module, file, stdin)
  - ``exit_code``: subprocess return code
  - ``stdout``: full captured subprocess stdout (text)
  - ``stderr``: full captured subprocess stderr (text)
"""

from autopilot.cli.command import Command
from autopilot.cli.context import CLIContext
from autopilot.cli.primitives import collect_arguments
from autopilot.cli.resolvers import make_subparser
import argparse
import subprocess
import sys

EXECUTE_EPILOG = (
  'IMPORTANT -- global flag ordering:\n'
  '  AutoPilot global flags (--context, --json, --experiment, --workspace,\n'
  '  --dry-run, --verbose, --epoch, --wait) are consumed by the CLI\n'
  '  dispatcher and are NOT forwarded to the subprocess.\n'
  '\n'
  '  These flags MUST appear BEFORE the execute subcommand or its\n'
  '  arguments. Tokens after the script path (file mode) or after -c/-m\n'
  '  value are forwarded verbatim to the child process via\n'
  '  argparse.REMAINDER.\n'
  '\n'
  "  Correct:   autopilot --context 'reason' execute script.py --my-flag\n"
  "  Wrong:     autopilot execute script.py --context 'reason'\n"
  '             (--context is forwarded to the script, not to autopilot)\n'
)


class ExecuteCommand(Command):
  """Execute Python under ``uv run python`` in four modes.

  Inline (``-c``), module (``-m``), file (first REMAINDER path), or stdin (no
  primary target, body from stdin). In text mode, subprocess output is
  re-emitted on ``sys.stdout`` / ``sys.stderr``. In JSON mode, output is
  captured only into the JSON envelope (not echoed). No ``log_execution`` or
  ``ExecutionRecord`` creation here -- SP3 dispatch instrumentation owns tracking.
  """

  name = 'execute'
  help = 'Execute Python code/files/modules via uv run python with tracking'

  def register(self, subparsers: argparse._SubParsersAction) -> None:
    """Attach the execute parser with ``-c`` / ``-m`` / remainder and global flags."""
    sub_parser = make_subparser(subparsers, self.name, self.help)
    sub_parser.epilog = EXECUTE_EPILOG
    sub_parser.formatter_class = argparse.RawDescriptionHelpFormatter
    sub_parser.add_argument(
      '-c',
      dest='code',
      default=None,
      metavar='CODE',
      help='Python code string (inline mode)',
    )
    sub_parser.add_argument(
      '-m',
      dest='module',
      default=None,
      metavar='MODULE',
      help='run library module as a script (same as python -m)',
    )
    sub_parser.add_argument(
      'extra_args',
      nargs=argparse.REMAINDER,
      default=None,
      help='script path and args (file mode), or extra args after -c/-m',
    )
    for arg_desc in collect_arguments(type(self)):
      arg_desc.add_to_parser(sub_parser)
    sub_parser.set_defaults(handler=self)

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Parse mode, build subprocess command, run, and re-emit output.

    Always uses ``capture_output=True`` so dispatch-level ``TeeWriter`` sees
    the same bytes. Stdin mode reads from ``sys.stdin`` and passes as
    ``input=`` to ``subprocess.run``.
    """
    mode, source, extra = self.parse_mode(args)
    if mode == 'stdin' and sys.stdin.isatty():
      ctx.fail('provide -c, -m, a script path, or pipe/heredoc to stdin')
    cmd = self.build_cmd(mode, source, extra)
    stdin_payload: str | None = None
    if mode == 'stdin':
      stdin_payload = sys.stdin.read()
    proc = subprocess.run(
      cmd,
      input=stdin_payload,
      capture_output=True,
      text=True,
      check=False,
    )
    if not ctx.output.use_json:
      if proc.stdout:
        sys.stdout.write(proc.stdout)
      if proc.stderr:
        sys.stderr.write(proc.stderr)
    ctx.output.result(
      {
        'mode': mode,
        'exit_code': proc.returncode,
        'stdout': proc.stdout or '',
        'stderr': proc.stderr or '',
      },
      ok=proc.returncode == 0,
    )
    if proc.returncode != 0:
      sys.exit(proc.returncode)

  def parse_mode(
    self,
    args: argparse.Namespace,
  ) -> tuple[str, str | None, list[str]]:
    """Determine execution mode from parsed arguments.

    Priority: ``-c`` (inline) > ``-m`` (module) > first positional (file) >
    stdin. Returns ``(mode, source, extra_args)``. A leading ``--`` in
    extra_args is stripped so it acts as an argparse separator without
    being forwarded to the subprocess (e.g. ``execute -m mod -- --help``).

    Returns:
      ``(mode, source, extra_args)`` where ``mode`` is one of ``inline``,
      ``module``, ``file``, or ``stdin``.
    """
    extra = args.extra_args if args.extra_args is not None else []
    if extra and extra[0] == '--':
      extra = extra[1:]
    if args.code is not None:
      return 'inline', args.code, extra
    if args.module is not None:
      return 'module', args.module, extra
    if extra:
      return 'file', extra[0], extra[1:]
    return 'stdin', None, []

  def build_cmd(
    self,
    mode: str,
    source: str | None,
    extra: list[str],
  ) -> list[str]:
    """Build the ``uv run python ...`` command list for the given mode.

    Stdin mode returns just the base command; the script body is passed via
    ``input=`` to ``subprocess.run`` in ``forward()``.

    Returns:
      Argument list for ``subprocess.run`` (``uv run python`` plus mode-specific tokens).
    """
    base = ['uv', 'run', 'python']
    if mode == 'inline':
      assert source is not None, 'inline mode requires a code string'
      return [*base, '-c', source, *extra]
    if mode == 'module':
      assert source is not None, 'module mode requires a module name'
      return [*base, '-m', source, *extra]
    if mode == 'file':
      assert source is not None, 'file mode requires a script path'
      return [*base, source, *extra]
    return base
