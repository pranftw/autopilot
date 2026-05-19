"""Tests for _BASE_CONTEXT_EXEMPT completeness and correctness.

Verifies that all read-only commands are exempt from --context enforcement,
that phantom entries (like 'store list') have been removed, that mutating
commands still require --context, that all compound exempt entries
correspond to real registered commands, and that the blanket 'debug'
exemption has been replaced with granular per-command entries (BUG-A002).

Extended by dogfood V4 sub-plan 09: matrix-driven parametrized tests,
top-level subtree coverage, set equality checks, parser completeness guard,
and invocation-level enforcement tests.
"""

from autopilot.cli.command import _BASE_CONTEXT_EXEMPT, CLI, Command
from autopilot.cli.context import build_context
from autopilot.cli.main import AutoPilotCLI
from autopilot.cli.primitives import collect_subcommands
from tests.cli.conftest import collect_leaf_commands
import pytest

# ---------------------------------------------------------------------------
# Authoritative command classification from CLAUDE.md CLI command matrix.
# Read-only commands (Mutating: No) and mutating commands (Mutating: Yes).
# When the matrix changes, tests fail until the oracle is updated.
# ---------------------------------------------------------------------------

READ_ONLY_COMMANDS = frozenset(
  {
    'ai judge distribution',
    'ai judge summarize',
    'dataset list',
    'dataset show',
    'dataset split',
    'debug collect',
    'debug commands',
    'debug cost',
    'debug executions list',
    'debug executions show',
    'debug executions tail',
    'debug gradients',
    'debug module-gradients',
    'debug optimizer-decisions',
    'debug parameters',
    'debug params',
    'debug profiler',
    'debug store reflog',
    'debug trend',
    'diagnose heatmap',
    'diagnose run',
    'experiment compare',
    'experiment deploy-log',
    'experiment impact',
    'experiment lineage',
    'experiment list',
    'experiment timeline',
    'experiment metadata get',
    'experiment metadata show',
    'experiment notes show',
    'experiment show',
    'experiment status',
    'policy check',
    'policy explain',
    'project doctor',
    'project list',
    'propose list',
    'query',
    'recommend',
    'report compare',
    'report narrative',
    'report summary',
    'report trend',
    'status',
    'store diff',
    'store doctor',
    'store log',
    'store merge',
    'store merge-analysis',
    'store merge-preview',
    'store reflog list',
    'store stash-list',
    'store status',
    'store tag list',
    'store tag verify',
    'store worktree list',
    'trace collect',
    'trace inspect',
    'trace verify',
    'tree describe',
    'tree list',
    'tree show',
    'undo-guide',
    'workspace doctor',
    'workspace journal',
    'workspace status',
    'workspace tree',
  }
)

MUTATING_COMMANDS = frozenset(
  {
    'ai generate dry-run',
    'ai generate resume',
    'ai generate run',
    'ai judge resume',
    'ai judge run',
    'checkout',
    'dataset seed',
    'debug store prune-orphans',
    'execute',
    'experiment add',
    'experiment cancel',
    'experiment complete',
    'experiment deploy',
    'experiment fail',
    'experiment invalidate',
    'experiment metadata set',
    'experiment notes write',
    'experiment remove',
    'experiment undeploy',
    'optimize deploy',
    'optimize loop',
    'optimize preflight',
    'optimize resume',
    'optimize set-hparams',
    'optimize test',
    'optimize train',
    'optimize validate',
    'project init',
    'propose create',
    'propose revert',
    'propose verify',
    'stabilize',
    'store branch',
    'store checkout',
    'store copy-epoch',
    'store create',
    'store merge-apply',
    'store merge-resolve',
    'store promote',
    'store recover',
    'store reflog expire',
    'store snapshot',
    'store stash',
    'store stash-pop',
    'store tag create',
    'store worktree create',
    'track',
    'tree create',
    'tree remove',
    'tree switch',
    'workspace init',
  }
)

# top-level tokens that exempt entire subtrees
TOP_LEVEL_PREFIXES = frozenset(
  {
    'query',
    'status',
    'diagnose',
    'trace',
    'report',
    'policy',
    'recommend',
  }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_child_by_name(
  parent: AutoPilotCLI | Command,
  name: str,
) -> Command | None:
  """Find a direct child command by its registered name."""
  for child in parent._commands.values():
    if child.name == name:
      return child
  return None


def _has_subcommand(parent: Command, name: str) -> bool:
  """Check if a command has a child or inline subcommand with the given name."""
  if _find_child_by_name(parent, name) is not None:
    return True
  return any(meta.name == name for meta, _ in collect_subcommands(parent))


def _resolve_compound_command(cli: AutoPilotCLI, command: str) -> bool:
  """Check whether a compound command resolves to a registered handler.

  Supports two-level ('experiment show') and three-level ('store worktree list',
  'debug executions list') compound commands.
  """
  parts = command.split()
  if len(parts) < 2:
    return True

  group_cmd = _find_child_by_name(cli, parts[0])
  if group_cmd is None:
    return False

  if len(parts) == 2:
    return _has_subcommand(group_cmd, parts[1])

  child_cmd = _find_child_by_name(group_cmd, parts[1])
  if child_cmd is None:
    return False
  return _has_subcommand(child_cmd, parts[2])


# ---------------------------------------------------------------------------
# 4.1: Parametrized exemption tests (read-only commands match matrix)
# ---------------------------------------------------------------------------


class TestExemptCommandsMatchMatrix:
  """Every read-only command from the CLAUDE.md matrix is context-exempt."""

  @pytest.mark.parametrize('command', sorted(READ_ONLY_COMMANDS))
  def test_exempt_commands_match_matrix(self, command: str) -> None:
    """Read-only command does not require --context."""
    cli = CLI()
    assert cli.requires_context(command) is False, (
      f'{command!r} is read-only per CLAUDE.md matrix but requires_context returned True'
    )


# ---------------------------------------------------------------------------
# 4.2: Parametrized mutating tests
# ---------------------------------------------------------------------------


class TestMutatingCommandsRequireContext:
  """Every mutating command from the CLAUDE.md matrix requires --context."""

  @pytest.mark.parametrize('command', sorted(MUTATING_COMMANDS))
  def test_mutating_commands_require_context(self, command: str) -> None:
    """Mutating command requires --context."""
    cli = CLI()
    assert cli.requires_context(command) is True, (
      f'{command!r} is mutating per CLAUDE.md matrix but requires_context returned False'
    )


# ---------------------------------------------------------------------------
# 4.3: Exempt set matches expected membership (set equality)
# ---------------------------------------------------------------------------


class TestExemptCountMatches:
  """_BASE_CONTEXT_EXEMPT matches the expected set from CLAUDE.md matrix."""

  def test_exempt_count_matches(self) -> None:
    """Build expected exempt set from matrix and compare via set equality.

    Top-level prefixes (query, status, etc.) cover entire subtrees, so
    only the prefix itself needs to be in _BASE_CONTEXT_EXEMPT. Compound
    read-only commands that are NOT covered by a prefix must be listed
    individually.
    """
    expected_exempt: set[str] = set()
    for cmd in READ_ONLY_COMMANDS:
      top_level = cmd.split(maxsplit=1)[0]
      if top_level in TOP_LEVEL_PREFIXES:
        expected_exempt.add(top_level)
      else:
        expected_exempt.add(cmd)

    actual = set(_BASE_CONTEXT_EXEMPT)

    missing_from_actual = expected_exempt - actual
    extra_in_actual = actual - expected_exempt

    assert missing_from_actual == set(), (
      f'expected in _BASE_CONTEXT_EXEMPT but missing: {sorted(missing_from_actual)}'
    )
    assert extra_in_actual == set(), (
      f'unexpected entries in _BASE_CONTEXT_EXEMPT: {sorted(extra_in_actual)}'
    )


# ---------------------------------------------------------------------------
# 4.4: Parser completeness guard -- all commands in matrix
# ---------------------------------------------------------------------------


class TestAllCommandsInMatrix:
  """Every registered leaf command appears in the CLAUDE.md classification."""

  def test_all_commands_in_matrix(self) -> None:
    """Walk parser tree; compare to matrix oracle so new commands cannot ship undocumented."""
    cli = AutoPilotCLI()
    parser = cli.build_parser()
    leaf_commands = collect_leaf_commands(parser)

    all_classified = READ_ONLY_COMMANDS | MUTATING_COMMANDS
    unclassified = [cmd for cmd in leaf_commands if cmd not in all_classified]

    assert unclassified == [], (
      f'commands registered in CLI but not classified in matrix oracle: {unclassified}'
    )

  def test_no_matrix_phantoms(self) -> None:
    """Every command in the matrix oracle actually exists in the parser."""
    cli = AutoPilotCLI()
    parser = cli.build_parser()
    leaf_commands = set(collect_leaf_commands(parser))

    all_classified = READ_ONLY_COMMANDS | MUTATING_COMMANDS
    phantoms = [cmd for cmd in sorted(all_classified) if cmd not in leaf_commands]

    assert phantoms == [], f'commands in matrix oracle but not registered in CLI parser: {phantoms}'


# ---------------------------------------------------------------------------
# 4.5: Top-level prefix subtree coverage
# ---------------------------------------------------------------------------


class TestTopLevelExemptCoversSubtree:
  """Top-level prefix entries exempt entire subtrees."""

  def test_report_subtree_exempt(self) -> None:
    """report summary, compare, narrative, and trend are all exempt."""
    cli = CLI()
    assert cli.requires_context('report summary') is False
    assert cli.requires_context('report compare') is False
    assert cli.requires_context('report narrative') is False
    assert cli.requires_context('report trend') is False

  def test_policy_subtree_exempt(self) -> None:
    """policy check and policy explain are exempt via 'policy' prefix."""
    cli = CLI()
    assert cli.requires_context('policy check') is False
    assert cli.requires_context('policy explain') is False

  def test_diagnose_subtree_exempt(self) -> None:
    """diagnose run and diagnose heatmap are exempt via 'diagnose' prefix."""
    cli = CLI()
    assert cli.requires_context('diagnose run') is False
    assert cli.requires_context('diagnose heatmap') is False

  def test_trace_subtree_exempt(self) -> None:
    """trace collect, trace inspect, and trace verify are exempt via 'trace' prefix."""
    cli = CLI()
    assert cli.requires_context('trace collect') is False
    assert cli.requires_context('trace inspect') is False
    assert cli.requires_context('trace verify') is False

  def test_query_exempt(self) -> None:
    """query is exempt as a top-level prefix."""
    cli = CLI()
    assert cli.requires_context('query') is False

  def test_status_exempt(self) -> None:
    """status is exempt as a top-level prefix."""
    cli = CLI()
    assert cli.requires_context('status') is False

  def test_prefixes_present_in_exempt_set(self) -> None:
    """All documented top-level prefixes are present in _BASE_CONTEXT_EXEMPT."""
    for prefix in TOP_LEVEL_PREFIXES:
      assert prefix in _BASE_CONTEXT_EXEMPT, f'{prefix!r} prefix missing from _BASE_CONTEXT_EXEMPT'


# ---------------------------------------------------------------------------
# 4.6: Invocation -- mutating without --context fails
# ---------------------------------------------------------------------------


class TestMutatingWithoutContextFails:
  """Dispatch of a mutating command without --context raises SystemExit."""

  def test_dispatch_rejects_mutating_without_context(self) -> None:
    """Dispatch of a mutating command without --context raises SystemExit."""
    calls: list[str] = []

    class Leaf(Command):
      name = 'create'
      help = 'create something'

      def forward(self, ctx, args):
        calls.append('invoked')

    class TreeGroup(Command):
      name = 'tree'
      help = 'tree ops'

      def __init__(self):
        super().__init__()
        self.create = Leaf()

    class TestCLI(CLI):
      def __init__(self):
        super().__init__()
        self.tree = TreeGroup()

    cli = TestCLI()
    parser = cli.build_parser()
    args = parser.parse_args(['tree', 'create'])
    ctx = build_context(args)

    with pytest.raises(SystemExit):
      cli.dispatch(ctx, args)
    assert calls == []


# ---------------------------------------------------------------------------
# 4.7: Invocation -- read-only without --context succeeds
# ---------------------------------------------------------------------------


class TestReadOnlyWithoutContextSucceeds:
  """Read-only commands pass dispatch without --context."""

  def test_exempt_command_dispatches_without_context(self) -> None:
    """An exempt command dispatches successfully without --context."""
    calls: list[str] = []

    class Leaf(Command):
      name = 'show'
      help = 'show something'

      def forward(self, ctx, args):
        calls.append('invoked')

    class TreeGroup(Command):
      name = 'tree'
      help = 'tree ops'

      def __init__(self):
        super().__init__()
        self.show = Leaf()

    class TestCLI(CLI):
      def __init__(self):
        super().__init__()
        self.tree = TreeGroup()

    cli = TestCLI()
    parser = cli.build_parser()
    args = parser.parse_args(['tree', 'show'])
    ctx = build_context(args)

    cli.dispatch(ctx, args)
    assert calls == ['invoked']

  def test_top_level_exempt_dispatches_without_context(self) -> None:
    """A top-level exempt command dispatches without --context."""
    calls: list[str] = []

    class StatusCmd(Command):
      name = 'status'
      help = 'show status'

      def forward(self, ctx, args):
        calls.append('invoked')

    class TestCLI(CLI):
      def __init__(self):
        super().__init__()
        self.status = StatusCmd()

    cli = TestCLI()
    parser = cli.build_parser()
    args = parser.parse_args(['status'])
    ctx = build_context(args)

    cli.dispatch(ctx, args)
    assert calls == ['invoked']


# ---------------------------------------------------------------------------
# 4.8: Whitespace-only context rejected
# ---------------------------------------------------------------------------


class TestWhitespaceContextRejected:
  """Whitespace-only --context is treated as omission for mutating commands."""

  def test_whitespace_context_rejected(self) -> None:
    """Mutating command with --context '  ' fails like omission."""
    calls: list[str] = []

    class Leaf(Command):
      name = 'create'
      help = 'create something'

      def forward(self, ctx, args):
        calls.append('invoked')

    class TreeGroup(Command):
      name = 'tree'
      help = 'tree ops'

      def __init__(self):
        super().__init__()
        self.create = Leaf()

    class TestCLI(CLI):
      def __init__(self):
        super().__init__()
        self.tree = TreeGroup()

    cli = TestCLI()
    parser = cli.build_parser()
    args = parser.parse_args(['tree', 'create', '--context', '   '])
    ctx = build_context(args)

    with pytest.raises(SystemExit):
      cli.dispatch(ctx, args)
    assert calls == []


# ---------------------------------------------------------------------------
# 4.9: Empty context rejected
# ---------------------------------------------------------------------------


class TestEmptyContextRejected:
  """Empty --context '' is treated as omission for mutating commands."""

  def test_empty_context_rejected(self) -> None:
    """Mutating command with --context '' fails like omission."""
    calls: list[str] = []

    class Leaf(Command):
      name = 'create'
      help = 'create something'

      def forward(self, ctx, args):
        calls.append('invoked')

    class TreeGroup(Command):
      name = 'tree'
      help = 'tree ops'

      def __init__(self):
        super().__init__()
        self.create = Leaf()

    class TestCLI(CLI):
      def __init__(self):
        super().__init__()
        self.tree = TreeGroup()

    cli = TestCLI()
    parser = cli.build_parser()
    args = parser.parse_args(['tree', 'create', '--context', ''])
    ctx = build_context(args)

    with pytest.raises(SystemExit):
      cli.dispatch(ctx, args)
    assert calls == []


# ---------------------------------------------------------------------------
# 4.10: --dry-run bypasses context enforcement
# ---------------------------------------------------------------------------


class TestDryRunBypassesContextEnforcement:
  """--dry-run is globally exempt from --context enforcement.

  Per CLAUDE.md: dry-run previews perform no durable mutation and should
  not require a reason string. The dispatch gate is:
  ``if not ctx.dry_run and self.requires_context(command) and ctx.context is None``
  """

  def test_dry_run_bypasses_context_requirement(self) -> None:
    """Mutating command with --dry-run and no --context does NOT fail."""
    calls: list[str] = []

    class Leaf(Command):
      name = 'create'
      help = 'create something'

      def forward(self, ctx, args):
        calls.append('invoked')

    class TreeGroup(Command):
      name = 'tree'
      help = 'tree ops'

      def __init__(self):
        super().__init__()
        self.create = Leaf()

    class TestCLI(CLI):
      def __init__(self):
        super().__init__()
        self.tree = TreeGroup()

    cli = TestCLI()
    parser = cli.build_parser()
    args = parser.parse_args(['tree', 'create', '--dry-run'])
    ctx = build_context(args)

    cli.dispatch(ctx, args)
    assert calls == ['invoked']


# ---------------------------------------------------------------------------
# Structural checks (preserved from original test file)
# ---------------------------------------------------------------------------


class TestExemptSetStructure:
  """Structural properties of _BASE_CONTEXT_EXEMPT."""

  def test_exempt_set_is_frozenset(self) -> None:
    """_BASE_CONTEXT_EXEMPT is a frozenset (immutable)."""
    assert isinstance(_BASE_CONTEXT_EXEMPT, frozenset)

  def test_no_empty_entries(self) -> None:
    """No empty or whitespace-only entries in the exempt set."""
    for entry in _BASE_CONTEXT_EXEMPT:
      assert entry.strip(), 'empty or whitespace-only entry in _BASE_CONTEXT_EXEMPT'

  def test_no_duplicate_semantics(self) -> None:
    """No two entries differ only by whitespace."""
    normalized = {entry.strip() for entry in _BASE_CONTEXT_EXEMPT}
    assert len(normalized) == len(_BASE_CONTEXT_EXEMPT)


# ---------------------------------------------------------------------------
# Compound command existence (all exempt entries are real commands)
# ---------------------------------------------------------------------------


class TestAllExemptCommandsExist:
  """Every compound entry in _BASE_CONTEXT_EXEMPT must be a real CLI command."""

  def test_all_exempt_commands_exist(self) -> None:
    """Every compound entry in _BASE_CONTEXT_EXEMPT corresponds to a registered command."""
    cli = AutoPilotCLI()
    compound_commands = [cmd for cmd in _BASE_CONTEXT_EXEMPT if ' ' in cmd]
    assert len(compound_commands) > 0, 'expected at least one compound command in exempt set'

    missing = [cmd for cmd in compound_commands if not _resolve_compound_command(cli, cmd)]

    assert missing == [], f'exempt entries not found as registered commands: {missing}'

  def test_top_level_exempt_commands_exist(self) -> None:
    """Every single-token entry in _BASE_CONTEXT_EXEMPT is a registered top-level command."""
    cli = AutoPilotCLI()
    top_level_names = {cmd.name for cmd in cli._commands.values()}
    single_token = [cmd for cmd in _BASE_CONTEXT_EXEMPT if ' ' not in cmd]
    assert len(single_token) > 0, 'expected at least one single-token command in exempt set'

    missing = [cmd for cmd in single_token if cmd not in top_level_names]
    assert missing == [], f'exempt entries not found as top-level commands: {missing}'


# ---------------------------------------------------------------------------
# BUG-A002: granular debug exemptions (replaces blanket 'debug' entry)
# ---------------------------------------------------------------------------


class TestGranularDebugExemptions:
  """Verify blanket 'debug' removed and individual debug commands classified correctly."""

  def test_debug_blanket_removed(self) -> None:
    """'debug' must not be in _BASE_CONTEXT_EXEMPT as a blanket top-level token."""
    assert 'debug' not in _BASE_CONTEXT_EXEMPT

  def test_prune_orphans_requires_context(self) -> None:
    """'debug store prune-orphans' is mutating and must require --context."""
    cli = CLI()
    assert cli.requires_context('debug store prune-orphans') is True

  def test_debug_read_only_commands_exempt(self) -> None:
    """All read-only debug commands are exempt from --context."""
    cli = CLI()
    read_only_debug = [
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
      'debug store reflog',
    ]
    for cmd in read_only_debug:
      assert not cli.requires_context(cmd), f'{cmd!r} should be exempt'


# ---------------------------------------------------------------------------
# Project CLI extension exemptions
# ---------------------------------------------------------------------------


class TestUndoGuideContextExempt:
  """undo-guide is read-only and context-exempt."""

  def test_undo_guide_context_exempt(self) -> None:
    """undo-guide succeeds without --context."""
    cli = CLI()
    assert cli.requires_context('undo-guide') is False


class TestProjectCLIExtensionExemptions:
  """Project CLIs can extend the exempt set via context_exempt_commands."""

  def test_project_extension_merges(self) -> None:
    """Custom project exemptions merge with base set."""
    custom = frozenset({'my-project read'})
    cli = CLI(context_exempt_commands=custom)
    assert cli.requires_context('my-project read') is False
    assert cli.requires_context('tree create') is True

  def test_base_set_preserved_with_extension(self) -> None:
    """Base exemptions survive when project extends."""
    cli = CLI(context_exempt_commands=frozenset({'custom read'}))
    assert cli.requires_context('query') is False
    assert cli.requires_context('status') is False
    assert cli.requires_context('store log') is False
