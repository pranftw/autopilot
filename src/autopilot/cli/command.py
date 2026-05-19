"""Unified recursive command system for the AutoPilot CLI.

Single Command class; leaf vs group follows child presence (like nn.Module).
CLI orchestrates like Trainer: ``__init__`` configures, ``__call__`` runs;
entry: ``AutoPilotCLI()()``.

Mutating commands need ``--context``; read-only commands use
``_BASE_CONTEXT_EXEMPT``. Project CLIs extend exemptions via
``CLI(context_exempt_commands=...)``. Enforcement: ``CLI.requires_context``.

Argparse errors with ``--json``: ``ArgparseCLIError`` plus ``CLI.run_direct``
emits JSON envelope or stderr usage (exit 2).
"""

from autopilot.cli.context import build_context
from autopilot.cli.primitives import (
  ArgparseCLIError,
  AutopilotArgumentParser,
  collect_arguments,
  collect_subcommands,
)
from autopilot.cli.resolvers import add_global_flags, make_subparser
from autopilot.core.config import AutoPilotConfig
from autopilot.core.trainer.trainer import Trainer
from autopilot.tracking.executions import (
  capture_output,
  create_execution_record,
  log_execution,
  resolve_command,
)
from autopilot.tracking.file_lock import LOCK_RETRY_AFTER_MS, ConcurrentMutationError
from pathlib import Path
from typing import Any, ClassVar
import argparse
import contextlib
import io
import json
import runpy
import sys
import time
import traceback

# context enforcement -- comprehensive set of read-only commands that do not
# require --context. entries must correspond to registered command names (single
# token for top-level commands, space-separated for compound subcommands).
# debug commands listed individually; no blanket 'debug' exemption. new debug
# subcommands default to requiring --context until explicitly added here.

_BASE_CONTEXT_EXEMPT = frozenset(
  {
    'query',
    'status',
    'diagnose',
    'trace',
    'report',
    'policy',
    'recommend',
    'debug executions list',
    'debug executions show',
    'debug executions tail',
    'debug cost',
    'debug parameters',
    'debug module-gradients',
    'debug gradients',
    'debug params',
    'debug optimizer-decisions',
    'debug collect',
    'debug commands',
    'debug profiler',
    'debug trend',
    'debug store reflog',
    'experiment show',
    'experiment status',
    'experiment compare',
    'experiment list',
    'experiment notes show',
    'experiment impact',
    'experiment lineage',
    'experiment timeline',
    'experiment deploy-log',
    'experiment metadata get',
    'experiment metadata show',
    'tree describe',
    'tree list',
    'tree show',
    'workspace doctor',
    'workspace journal',
    'workspace status',
    'workspace tree',
    'project list',
    'project doctor',
    'propose list',
    'store log',
    'store status',
    'store diff',
    'store worktree list',
    'store merge-analysis',
    'store merge-preview',
    'store merge',
    'store doctor',
    'store tag list',
    'store tag verify',
    'store stash-list',
    'store reflog list',
    'dataset list',
    'dataset show',
    'dataset split',
    'ai judge summarize',
    'ai judge distribution',
    'undo-guide',
  }
)


# command


class Command:
  """Recursive command node. Leaf or group determined by children, not type.

  Like nn.Module: __setattr__ auto-registers child Commands into _commands.
  Override forward(ctx, args) for leaf command logic. Container commands
  nest children and/or use @subcommand / @argument on methods for
  inline handlers.

  Registration model:
    register(subparsers) walks children and attaches argparse parsers.
    Declarative flags via Argument / Flag class attributes.
    Inline subcommands via @subcommand + @argument decorators.

  Name derivation: __init_subclass__ auto-derives name from class name
  (strips 'Command' suffix, lowercases) if not explicitly set.

  Error handling contract: exceptions in handler are caught by dispatch(),
  which calls flush_error() in JSON mode and sys.exit(1). Commands that
  detect errors should call ctx.fail(message) which emits the error and
  exits. Commands should NOT call ctx.output.error() + return (that exits 0).
  """

  name: str | None = None
  help: str | None = None
  _commands: dict[str, 'Command']

  def __init_subclass__(cls, **kwargs: Any) -> None:
    """Derive ``name`` from the class name when not set explicitly."""
    super().__init_subclass__(**kwargs)
    if not cls.name:
      raw = cls.__name__
      raw = raw.removesuffix('Command')
      cls.name = raw.lower()

  def __init__(self) -> None:
    """Initialize an empty child command registry."""
    object.__setattr__(self, '_commands', {})

  def __setattr__(self, name: str, value: object) -> None:
    """Register child ``Command`` instances on ``_commands`` before setting the attribute."""
    if isinstance(value, Command):
      self._commands[name] = value
    super().__setattr__(name, value)

  def forward(self, ctx: Any, args: argparse.Namespace) -> None:
    """Run this command; subclasses must implement.

    Args:
      ctx: CLI context.
      args: Parsed namespace for this handler.

    Raises:
      NotImplementedError: Always on the base class.
    """
    msg = f'{type(self).__name__} must implement forward()'
    raise NotImplementedError(msg)

  def __call__(self, ctx: Any, args: argparse.Namespace) -> None:
    """Delegate invocation to ``forward``."""
    return self.forward(ctx, args)

  def register(self, subparsers: argparse._SubParsersAction) -> None:
    """Register this command (and children) onto an argparse subparsers action."""
    assert self.name is not None, (
      f'{type(self).__name__} must have a name before register(); '
      'set name as a class variable or via __init_subclass__'
    )
    inline_subs = collect_subcommands(self)
    has_children = bool(self._commands) or bool(inline_subs)

    if has_children:
      group = subparsers.add_parser(self.name, help=self.help)
      add_global_flags(group, is_subparser=True)
      child_sub = group.add_subparsers(dest=f'{self.name}_action', required=True)

      for child_cmd in self._commands.values():
        child_cmd.register(child_sub)

      for meta, bound_method in inline_subs:
        sub_parser = make_subparser(
          child_sub,
          meta.name,
          meta.help,
        )
        for arg_flags, arg_kwargs in meta.arguments:
          sub_parser.add_argument(*arg_flags, **arg_kwargs)
        sub_parser.set_defaults(handler=bound_method)
    else:
      sub_parser = make_subparser(
        subparsers,
        self.name,
        self.help,
      )
      for arg_desc in collect_arguments(type(self)):
        arg_desc.add_to_parser(sub_parser)
      sub_parser.set_defaults(handler=self)

  # container protocol

  def __getitem__(self, key: str) -> 'Command':
    """Return the child command registered under ``key``."""
    return self._commands[key]

  def __iter__(self):
    """Iterate ``(name, child_command)`` pairs.

    Returns:
      Iterator over registered child command items.
    """
    return iter(self._commands.items())

  def __contains__(self, key: str) -> bool:
    """Return whether ``key`` names a registered child command."""
    return key in self._commands

  def __len__(self) -> int:
    """Return the number of registered child commands."""
    return len(self._commands)

  @property
  def commands(self) -> dict[str, 'Command']:
    """Copy of registered child commands by attribute name."""
    return dict(self._commands)

  def __repr__(self) -> str:
    """Return a debug representation listing child command names."""
    children = ', '.join(self._commands)
    return f'{type(self).__name__}({self.name!r}, children=[{children}])'


# cli


class CLI:
  """Top-level CLI orchestrator. Like Trainer: __init__ configures, __call__ runs.

  Subclass for project CLIs. Use __init_subclass__(project='...') to
  auto-register project CLI classes in _project_registry.

  __init__ wires: self.module, self.generator, self.judge, self.datamodule, and Command
  instances as attributes. __call__ -> run() -> pre-parse --project /
  --workspace -> optional project cli.py via runpy -> dispatch.

  Project resolution order: --project / -p flag > CWD detection.
  Entry point: AutoPilotCLI()() or MyCLI()().
  """

  prog: str = 'autopilot'
  description: str = 'AutoPilot optimization CLI'
  _commands: dict[str, Command]

  _project_registry: ClassVar[dict[str, type['CLI']]] = {}

  def __init_subclass__(cls, *, project: str | None = None, **kwargs: Any) -> None:
    """Optionally register this subclass under ``project`` for ``-p`` dispatch."""
    super().__init_subclass__(**kwargs)
    if project is not None:
      CLI._project_registry[project] = cls

  @classmethod
  def lookup_project(cls, slug: str | None) -> type['CLI'] | None:
    """Look up a registered project CLI class by slug.

    Args:
      slug: Project identifier registered via ``__init_subclass__(project=...)``.
        Returns None immediately when slug is None.

    Returns:
      The CLI subclass for the project, or None if not registered.
    """
    if slug is None:
      return None
    return cls._project_registry.get(slug)

  def __init__(
    self,
    *,
    context_exempt_commands: frozenset[str] | None = None,
  ) -> None:
    """Initialize command registry and optional wiring slots.

    Args:
      context_exempt_commands: Additional commands exempt from ``--context``
        enforcement. Merged with the base exempt set (read-only commands
        like query, status, etc.). Pass project-specific read-only
        commands here. None means use only the base set.
    """
    object.__setattr__(self, '_commands', {})
    self._context_exempt_commands = (
      _BASE_CONTEXT_EXEMPT | context_exempt_commands
      if context_exempt_commands is not None
      else _BASE_CONTEXT_EXEMPT
    )
    self.module = None
    self.generator = None
    self.judge = None
    self.datamodule = None

  @property
  def context_exempt_commands(self) -> frozenset[str]:
    """The full set of commands exempt from --context enforcement."""
    return self._context_exempt_commands

  def requires_context(self, command: str) -> bool:
    """Check whether a command requires ``--context``.

    Exact command strings are checked first (e.g. ``'experiment show'``),
    then the top-level token (e.g. ``'query'``). Returns True when the
    command must supply ``--context``.

    Args:
      command: Resolved command string (e.g. ``'experiment add'``).

    Returns:
      True when the command requires --context.
    """
    if command in self._context_exempt_commands:
      return False
    top_level = command.split(maxsplit=1)[0] if command else ''
    return top_level not in self._context_exempt_commands

  def __setattr__(self, name: str, value: object) -> None:
    """Register ``Command`` children on ``_commands`` like ``Command.__setattr__``."""
    if isinstance(value, Command):
      self._commands[name] = value
    super().__setattr__(name, value)

  def __call__(self, *, argv: list[str] | None = None) -> None:
    """Run the CLI with optional argv (same as ``run``)."""
    return self.run(argv=argv)

  def build_parser(self) -> argparse.ArgumentParser:
    """Construct the root argument parser with all registered commands.

    Uses ``AutopilotArgumentParser`` so usage errors raise
    ``ArgparseCLIError`` instead of calling ``sys.exit()``.

    Returns:
      Configured root ``ArgumentParser`` for this CLI.
    """
    parser = AutopilotArgumentParser(
      prog=self.prog,
      description=self.description,
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=(
        'Quick start:\n'
        "  autopilot workspace init --context 'init workspace'\n"
        "  autopilot tree create main --context 'create tree'\n"
        "  autopilot experiment add --id baseline --context 'first experiment'\n"
        '  autopilot experiment complete baseline'
        " --metrics '{\"accuracy\": 0.9}' --context 'done'\n"
        '  autopilot query --json'
      ),
    )
    parser.set_defaults(handler=None)
    add_global_flags(parser)
    subparsers = parser.add_subparsers(dest='command', help='available commands')
    for cmd in self._commands.values():
      cmd.register(subparsers)
    return parser

  def build_context(self, args: argparse.Namespace) -> Any:
    """Build CLIContext from parsed args. Override for custom context.

    Args:
      args: Parsed root-level namespace.

    Returns:
      Context object produced by ``build_context`` (default) or a subclass override.
    """
    return build_context(args)

  def _run_handler_with_retry(self, handler: Any, ctx: Any, args: argparse.Namespace) -> None:
    """Run handler with optional retry on ``ConcurrentMutationError``.

    When ``ctx.retry_max > 0`` and not ``ctx.dry_run``, retries up to N
    times with exponential backoff from ``LOCK_RETRY_AFTER_MS``. Clears
    buffered output messages between retries. Sets ``ctx.output.retry_attempts``
    so JSON envelopes reflect retry cost.

    Args:
      handler: Resolved command handler callable.
      ctx: CLI context carrying retry_max and output.
      args: Parsed argparse namespace.

    Raises:
      ConcurrentMutationError: When retries are exhausted or retry is disabled.
    """
    retry_enabled = ctx.retry_max > 0 and not ctx.dry_run
    retry_count = 0
    while True:
      try:
        ctx.output.retry_attempts = retry_count
        handler(ctx, args)
      except ConcurrentMutationError:
        if not retry_enabled or retry_count >= ctx.retry_max:
          raise
        sleep_seconds = (LOCK_RETRY_AFTER_MS / 1000.0) * (2**retry_count)
        time.sleep(sleep_seconds)
        retry_count += 1
        ctx.output.clear_messages()
      else:
        return

  def dispatch(
    self,
    ctx: Any,
    args: argparse.Namespace,
    *,
    argv: list[str] | None = None,
  ) -> None:
    """Dispatch to handler with error handling, retry, and execution tracking.

    Every successfully parsed command with a handler is logged to
    ``ctx.config.executions_path`` as a JSONL ``ExecutionRecord``.
    ``capture_output()`` installs ``TeeWriter`` on ``sys.stdout`` and
    ``sys.stderr`` so output appears on the terminal in real time while
    being captured for the record. Logging failures are silently swallowed
    (D6) so commands never fail because of the audit trail.

    When ``ctx.retry_max > 0``, the handler is retried up to N times on
    ``ConcurrentMutationError`` with exponential backoff starting from
    ``LOCK_RETRY_AFTER_MS`` (100ms, 200ms, 400ms, ...). The retry loop
    is disabled during ``--dry-run``. ``--retry`` and ``--wait`` are
    mutually exclusive.

    When *argv* is provided (from ``run_direct``), it is stored directly
    in the record's ``args`` field for exact reproducibility. Otherwise
    ``_extract_argv`` serializes the parsed ``Namespace``.
    """
    handler = args.handler
    if handler is None:
      self.build_parser().print_help()
      sys.exit(1)

    command = resolve_command(args)

    stripped = ctx.context.strip() if ctx.context else None
    ctx.context = stripped or None
    if not ctx.dry_run and self.requires_context(command) and ctx.context is None:
      ctx.fail(f'--context is required for {command!r}: explain why you are running this action')

    if ctx.retry_max > 0 and ctx.wait_timeout_ms is not None:
      ctx.fail(
        '--retry and --wait are mutually exclusive: use --retry N for '
        'bounded reruns with backoff or --wait TIMEOUT_MS to block inside '
        'lock acquisition'
      )

    start = time.monotonic()
    exit_code = 0
    out_buf = io.StringIO()
    err_buf = io.StringIO()

    try:
      with capture_output() as (out_buf, err_buf):
        try:
          self._run_handler_with_retry(handler, ctx, args)
        except ConcurrentMutationError as cme:
          if ctx.output.use_json:
            envelope = {
              'ok': False,
              'error': str(cme),
              'error_code': 'concurrent_mutation',
              'retry_after_ms': cme.retry_after_ms,
            }
            sys.stdout.write(json.dumps(envelope) + '\n')
          else:
            ctx.output.error(str(cme))
          exit_code = 1
        except Exception as e:
          ctx.output.error(str(e))
          if ctx.output.use_json:
            ctx.output.flush_error(str(e), error_code='handler_error')
          if ctx.verbose:
            traceback.print_exc()
          exit_code = 1
    except SystemExit as e:
      exit_code = e.code if isinstance(e.code, int) else 1

    duration_ms = (time.monotonic() - start) * 1000
    stdout_val = out_buf.getvalue() or None
    stderr_val = err_buf.getvalue() or None

    record = create_execution_record(
      command=command,
      args=argv if argv is not None else self._extract_argv(args),
      duration_ms=round(duration_ms, 1),
      exit_code=exit_code,
      stdout=stdout_val,
      stderr=stderr_val,
      experiment=ctx.experiment,
      project=ctx.project,
      context=ctx.context,
    )
    with contextlib.suppress(Exception):
      log_execution(ctx.config.executions_path, record)

    if exit_code != 0:
      sys.exit(exit_code)

  def _extract_argv(self, args: argparse.Namespace) -> list[str]:
    """Serialize a parsed Namespace to a stable list of CLI-like strings.

    Used as a fallback when raw ``argv`` is not available (callers that
    invoke ``dispatch`` directly without passing ``argv``). Serialization
    rules: bools ``True`` produce ``--flag``, ``False`` is skipped; scalars
    produce ``--key=value``; lists emit one entry per item; ``None`` is
    skipped; ``handler``, ``command``, and ``*_action`` keys are excluded.

    Returns:
      Stable CLI argv fragments approximating the namespace contents.
    """
    items: list[str] = []
    skip = {'handler', 'command'}
    for k, v in sorted(vars(args).items()):
      if k in skip or k.endswith('_action') or v is None:
        continue
      if isinstance(v, bool):
        if v:
          items.append(f'--{k}')
      elif isinstance(v, list):
        items.extend(str(item) for item in v)
      else:
        items.append(f'--{k}={v}')
    return items

  def _pre_parse(
    self,
    argv: list[str] | None,
  ) -> tuple[str, str, list[str]]:
    """Extract -p/--project and --workspace before full parse.

    Args:
      argv: Raw argv list or ``None`` (uses ``sys.argv`` semantics via parser).

    Returns:
      Tuple of ``(project, workspace, remaining_argv)`` after pre-parse.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('-p', '--project', default=None)
    pre.add_argument('--workspace', default='.')
    known, remaining = pre.parse_known_args(argv)
    return known.project, known.workspace, remaining

  def run(self, *, argv: list[str] | None = None) -> None:
    """Single entry point. Handles project dispatch internally."""
    project, workspace, remaining = self._pre_parse(argv)

    if project:
      ws = Path(workspace).resolve()
      if project not in CLI._project_registry:
        config = AutoPilotConfig(workspace=ws, project=project)
        cli_script = config.cli_file
        if cli_script.exists():
          project_dir = config.root
          sys.path.insert(0, str(project_dir))
          try:
            runpy.run_path(str(cli_script), run_name='__autopilot_project__')
          finally:
            sys.path.pop(0)

      if project in CLI._project_registry:
        project_cls = CLI._project_registry[project]
        project_cli = project_cls()
        full_argv = [*remaining, '--workspace', str(ws)]
        project_cli.run_direct(argv=full_argv)
        return

    self.run_direct(argv=argv)

  def run_direct(self, *, argv: list[str] | None = None) -> None:
    """Parse and dispatch without project resolution.

    Catches ``ArgparseCLIError`` from ``AutopilotArgumentParser`` and emits
    either a JSON error envelope (when ``--json`` appears in argv) or stderr
    usage text, then exits with the error's exit code (typically 2).

    Raises:
      SystemExit: On argparse errors (code 2), missing command (code 2),
        or dispatch failures (code from handler).
    """
    parser = self.build_parser()
    try:
      args = parser.parse_args(argv)
    except ArgparseCLIError as exc:
      if exc.exit_code == 0:
        raise SystemExit(0) from exc
      argv_list = argv if argv is not None else sys.argv[1:]
      json_requested = '--json' in argv_list
      if json_requested:
        sys.stdout.write(
          json.dumps({'ok': False, 'error': exc.message, 'error_code': 'cli_usage'}) + '\n'
        )
        raise SystemExit(exc.exit_code) from exc
      parser.print_usage(sys.stderr)
      sys.stderr.write(f'{parser.prog}: error: {exc.message}\n')
      raise SystemExit(exc.exit_code) from exc

    if not args.command:
      argv_list = argv if argv is not None else sys.argv[1:]
      if '--json' in argv_list:
        sys.stdout.write(
          json.dumps(
            {
              'ok': False,
              'error': 'no command specified',
              'error_code': 'cli_usage',
            }
          )
          + '\n'
        )
      else:
        parser.print_help()
      sys.exit(2)

    ctx = self.build_context(args)
    ctx.generator = self.generator
    ctx.judge = self.judge
    ctx.module = self.module
    ctx.datamodule = self.datamodule
    if self.module:
      ctx.trainer = Trainer(dry_run=ctx.dry_run)

    self.dispatch(ctx, args, argv=argv)

  @property
  def commands(self) -> dict[str, Command]:
    """Copy of top-level commands registered on this CLI."""
    return dict(self._commands)

  def __repr__(self) -> str:
    """Return a debug representation listing top-level command names."""
    cmds = ', '.join(self._commands)
    return f'{type(self).__name__}(commands=[{cmds}])'
