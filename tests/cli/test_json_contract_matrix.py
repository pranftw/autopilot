"""JSON envelope and exit-code contract matrix tests.

Parametrized CLI tests that every --json command emits parseable envelopes,
success paths include ``result``, and error paths exit nonzero with ``error``
strings.

Dogfood V4 sub-plan 08 of 10, Phase C.

Commands grouped by workspace setup depth (tiers A-E). The frozen command
set ``EXPECTED_JSON_COMMANDS`` stays aligned with the CLAUDE.md CLI command
matrix. ``test_all_json_commands_in_matrix`` is the drift guard.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.main import AutoPilotCLI
from autopilot.core.config import AutoPilotConfig
from io import StringIO
from pathlib import Path
from tests.cli.conftest import (
  collect_leaf_commands,
  run_cli,
  run_cli_no_context,
  seed_tree_with_experiments,
)
from typing import Any
from unittest.mock import patch
import json
import pytest

# ---------------------------------------------------------------------------
# Authoritative --json command set from CLAUDE.md CLI command matrix.
# Update this set when the matrix changes. test_all_json_commands_in_matrix
# catches drift between this set and the live CLI parser.
# ---------------------------------------------------------------------------

EXPECTED_JSON_COMMANDS = frozenset(
  {
    'ai generate run',
    'ai generate resume',
    'ai generate dry-run',
    'ai judge run',
    'ai judge resume',
    'ai judge summarize',
    'ai judge distribution',
    'dataset list',
    'dataset show',
    'dataset split',
    'debug commands',
    'debug profiler',
    'debug trend',
    'debug store reflog',
    'diagnose run',
    'diagnose heatmap',
    'execute',
    'experiment cancel',
    'experiment compare',
    'experiment deploy',
    'experiment deploy-log',
    'experiment undeploy',
    'experiment complete',
    'experiment fail',
    'experiment impact',
    'experiment lineage',
    'experiment timeline',
    'experiment invalidate',
    'experiment list',
    'experiment metadata get',
    'experiment metadata set',
    'experiment metadata show',
    'experiment notes show',
    'experiment notes write',
    'experiment show',
    'experiment status',
    'optimize preflight',
    'policy check',
    'policy explain',
    'project doctor',
    'project list',
    'propose create',
    'propose list',
    'propose revert',
    'propose verify',
    'query',
    'recommend',
    'report compare',
    'report narrative',
    'report summary',
    'report trend',
    'status',
    'store branch',
    'store checkout',
    'store copy-epoch',
    'store doctor',
    'store diff',
    'store log',
    'store merge',
    'store merge-analysis',
    'store merge-apply',
    'store merge-preview',
    'store merge-resolve',
    'store recover',
    'store reflog expire',
    'store reflog list',
    'store snapshot',
    'store stash',
    'store stash-list',
    'store stash-pop',
    'store status',
    'store tag create',
    'store tag list',
    'store tag verify',
    'store worktree list',
    'trace collect',
    'trace inspect',
    'trace verify',
    'track',
    'tree create',
    'tree describe',
    'tree list',
    'tree remove',
    'tree show',
    'undo-guide',
    'workspace doctor',
    'workspace journal',
    'workspace status',
    'workspace tree',
  }
)

KNOWN_NON_JSON_COMMANDS = frozenset(
  {
    'checkout',
    'dataset seed',
    'debug collect',
    'debug cost',
    'debug executions list',
    'debug executions show',
    'debug executions tail',
    'debug gradients',
    'debug module-gradients',
    'debug optimizer-decisions',
    'debug parameters',
    'debug params',
    'debug store prune-orphans',
    'experiment add',
    'experiment remove',
    'optimize deploy',
    'optimize loop',
    'optimize resume',
    'optimize set-hparams',
    'optimize test',
    'optimize train',
    'optimize validate',
    'project init',
    'stabilize',
    'store create',
    'store promote',
    'store worktree create',
    'tree switch',
    'workspace init',
  }
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_full_cli_json(argv: list[str]) -> tuple[int, str]:
  """Run AutoPilotCLI with captured stdout, returning (exit_code, stdout).

  Uses run_direct so argparse errors are also captured in JSON envelopes.

  Args:
    argv: Full CLI arguments including --json.

  Returns:
    Tuple of (exit_code, captured_stdout).
  """
  cli = AutoPilotCLI()
  out = StringIO()
  exit_code = 0
  with patch('sys.stdout', out), patch('sys.stderr', StringIO()):
    try:
      cli(argv=argv)
    except SystemExit as e:
      exit_code = int(e.code) if e.code is not None else 0
  return exit_code, out.getvalue()


def _assert_json_success_envelope(result: dict[str, Any]) -> None:
  """Assert the standard success JSON envelope structure.

  Args:
    result: Parsed JSON payload from CLI stdout.
  """
  assert 'ok' in result, f'missing "ok" in {list(result.keys())}'
  assert result['ok'] is True, f'expected ok=True, got {result["ok"]}'
  assert 'result' in result, f'missing "result" in {list(result.keys())}'


def _assert_json_error_envelope(stdout: str, exit_code: int) -> None:
  """Assert the standard error JSON envelope structure and nonzero exit.

  Args:
    stdout: Raw stdout from CLI invocation.
    exit_code: Process exit code.
  """
  assert exit_code != 0, f'expected nonzero exit, got {exit_code}'
  lines = [line.strip() for line in stdout.strip().split('\n') if line.strip()]
  assert lines, 'no output captured'
  envelope = json.loads(lines[-1])
  assert 'ok' in envelope, f'missing "ok" in error envelope: {list(envelope.keys())}'
  assert envelope['ok'] is False, f'expected ok=False, got {envelope["ok"]}'
  assert 'error' in envelope, f'missing "error" in error envelope: {list(envelope.keys())}'
  assert isinstance(envelope['error'], str), (
    f'error field should be str, got {type(envelope["error"]).__name__}'
  )


# ---------------------------------------------------------------------------
# Fixtures by tier
# ---------------------------------------------------------------------------


@pytest.fixture
def tier_a_workspace(tmp_path: Path) -> Path:
  """Tier A: minimal workspace with .autopilot layout (no store pre-created).

  Uses ``init_workspace()`` so workspace doctor sees all required directories.
  Store directory is NOT pre-created; ``workspace status`` only runs store
  doctor when ``store_existed`` is True before ``load_forest`` creates it.
  """
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.init_workspace()
  return ws


@pytest.fixture
def tier_b_workspace(tier_a_workspace: Path) -> Path:
  """Tier B: workspace with a tree (no experiments)."""
  config = AutoPilotConfig(workspace=tier_a_workspace)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  forest.create_tree('main')
  forest.switch('main')
  forest.save()
  return tier_a_workspace


@pytest.fixture
def tier_c_workspace(tier_a_workspace: Path) -> Path:
  """Tier C: workspace with tree, experiments, and HEAD set."""
  config = AutoPilotConfig(workspace=tier_a_workspace)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  seed_tree_with_experiments(
    forest,
    'main',
    [
      {
        'id': 'exp-base',
        'hypothesis': 'baseline approach',
        'status': 'completed',
        'metrics': {'accuracy': 0.72, 'latency': 120.0},
      },
      {
        'id': 'exp-cand',
        'hypothesis': 'candidate approach',
        'status': 'completed',
        'metrics': {'accuracy': 0.85, 'latency': 100.0},
        'parent': 'exp-base',
        'baseline': 'exp-base',
      },
      {
        'id': 'exp-running',
        'hypothesis': 'still running',
        'status': 'running',
        'metrics': {},
      },
    ],
  )
  tree = forest.active
  if tree is not None:
    tree.head = 'exp-base'
    forest.save()
  return tier_a_workspace


@pytest.fixture
def tier_d_workspace(tier_c_workspace: Path) -> Path:
  """Tier D: workspace with tree, experiments, and store with snapshots."""
  ws = tier_c_workspace
  config = AutoPilotConfig(workspace=ws)
  src_dir = ws / 'src'
  src_dir.mkdir(exist_ok=True)
  (src_dir / 'main.py').write_text('v0', encoding='utf-8')
  param = PathParameter(source=str(src_dir), pattern='**/*')
  store = FileStore(config)
  store.register_parameters({'source': param})
  store.snapshot('exp-base', 0)
  (src_dir / 'main.py').write_text('v1', encoding='utf-8')
  store.snapshot('exp-base', 1)
  return ws


# ---------------------------------------------------------------------------
# 4.1: Success-path envelope tests
# ---------------------------------------------------------------------------

TIER_A_READ_COMMANDS: list[tuple[str, list[str]]] = [
  ('workspace doctor', ['workspace', 'doctor']),
  ('workspace status', ['workspace', 'status']),
  ('workspace tree', ['workspace', 'tree']),
  ('project list', ['project', 'list']),
]

TIER_B_READ_COMMANDS: list[tuple[str, list[str]]] = [
  ('tree describe', ['tree', 'describe']),
  ('tree list', ['tree', 'list']),
  ('tree show', ['tree', 'show']),
]

TIER_C_READ_COMMANDS: list[tuple[str, list[str]]] = [
  ('query', ['query']),
  ('experiment list', ['experiment', 'list']),
  ('experiment show:exp-base', ['experiment', 'show', 'exp-base']),
  ('experiment status:exp-base', ['experiment', 'status', 'exp-base']),
  (
    'experiment compare',
    ['experiment', 'compare', 'exp-base', 'exp-cand'],
  ),
  ('experiment impact:exp-base', ['experiment', 'impact', 'exp-base']),
  (
    'experiment notes show:exp-base',
    ['experiment', 'notes', 'show', 'exp-base'],
  ),
  ('report summary', ['report', 'summary']),
  (
    'report compare',
    ['report', 'compare', 'exp-base', 'exp-cand'],
  ),
  ('report narrative', ['report', 'narrative']),
  (
    'status',
    ['--experiment', 'exp-base', 'status'],
  ),
]

TIER_D_READ_COMMANDS_NO_SOURCE: list[tuple[str, list[str]]] = [
  ('store doctor', ['store', 'doctor']),
  ('store tag list', ['store', 'tag', 'list']),
  ('store stash-list', ['store', 'stash-list']),
  ('store worktree list', ['store', 'worktree', 'list']),
  ('debug store reflog', ['debug', 'store', 'reflog']),
]


class TestTierASuccessEnvelope:
  """Tier A: workspace-only read commands produce valid JSON envelopes."""

  @pytest.mark.parametrize(
    ('label', 'argv'),
    TIER_A_READ_COMMANDS,
    ids=[c[0] for c in TIER_A_READ_COMMANDS],
  )
  def test_json_success_envelope(
    self,
    tier_a_workspace: Path,
    label: str,
    argv: list[str],
  ) -> None:
    result = run_cli_no_context(tier_a_workspace, argv)
    assert result, f'{label}: no JSON output'
    _assert_json_success_envelope(result)
    parsed = json.loads(json.dumps(result))
    assert isinstance(parsed, dict)


class TestTierBSuccessEnvelope:
  """Tier B: tree-level read commands produce valid JSON envelopes."""

  @pytest.mark.parametrize(
    ('label', 'argv'),
    TIER_B_READ_COMMANDS,
    ids=[c[0] for c in TIER_B_READ_COMMANDS],
  )
  def test_json_envelope_valid(
    self,
    tier_b_workspace: Path,
    label: str,
    argv: list[str],
  ) -> None:
    result = run_cli_no_context(tier_b_workspace, argv)
    assert result, f'{label}: no JSON output'
    _assert_json_success_envelope(result)


class TestTierCSuccessEnvelope:
  """Tier C: experiment-level read commands produce valid JSON envelopes."""

  @pytest.mark.parametrize(
    ('label', 'argv'),
    TIER_C_READ_COMMANDS,
    ids=[c[0] for c in TIER_C_READ_COMMANDS],
  )
  def test_json_envelope_valid(
    self,
    tier_c_workspace: Path,
    label: str,
    argv: list[str],
  ) -> None:
    result = run_cli_no_context(tier_c_workspace, argv)
    assert result, f'{label}: no JSON output'
    _assert_json_success_envelope(result)


class TestTierDSuccessEnvelope:
  """Tier D: store-level read commands produce valid JSON envelopes."""

  @pytest.mark.parametrize(
    ('label', 'argv'),
    TIER_D_READ_COMMANDS_NO_SOURCE,
    ids=[c[0] for c in TIER_D_READ_COMMANDS_NO_SOURCE],
  )
  def test_json_envelope_valid(
    self,
    tier_d_workspace: Path,
    label: str,
    argv: list[str],
  ) -> None:
    result = run_cli_no_context(tier_d_workspace, argv)
    assert result, f'{label}: no JSON output'
    _assert_json_success_envelope(result)

  def test_store_log_envelope(self, tier_d_workspace: Path) -> None:
    src = tier_d_workspace / 'src'
    result = run_cli_no_context(
      tier_d_workspace,
      ['store', 'log', '--source', str(src), '--experiment', 'exp-base'],
    )
    _assert_json_success_envelope(result)

  def test_store_status_envelope(self, tier_d_workspace: Path) -> None:
    src = tier_d_workspace / 'src'
    result = run_cli_no_context(
      tier_d_workspace,
      ['store', 'status', '--source', str(src), '--experiment', 'exp-base'],
    )
    _assert_json_success_envelope(result)

  def test_store_merge_analysis_envelope(self, tier_d_workspace: Path) -> None:
    ws = tier_d_workspace
    config = AutoPilotConfig(workspace=ws)
    src_dir = ws / 'src'
    param = PathParameter(source=str(src_dir), pattern='**/*')
    store = FileStore(config)
    store.register_parameters({'source': param})
    store.branch('exp-cand')
    (src_dir / 'main.py').write_text('v2-cand', encoding='utf-8')
    store.snapshot('exp-cand', 0)
    result = run_cli_no_context(
      ws,
      ['store', 'merge-analysis', 'exp-base', 'exp-cand'],
    )
    _assert_json_success_envelope(result)

  def test_store_merge_preview_envelope(self, tier_d_workspace: Path) -> None:
    ws = tier_d_workspace
    config = AutoPilotConfig(workspace=ws)
    src_dir = ws / 'src'
    param = PathParameter(source=str(src_dir), pattern='**/*')
    store = FileStore(config)
    store.register_parameters({'source': param})
    store.branch('exp-cand')
    (src_dir / 'main.py').write_text('v2-cand', encoding='utf-8')
    store.snapshot('exp-cand', 0)
    result = run_cli_no_context(
      ws,
      ['store', 'merge-preview', 'exp-base', 'exp-cand'],
    )
    _assert_json_success_envelope(result)

  def test_store_diff_envelope(self, tier_d_workspace: Path) -> None:
    src = tier_d_workspace / 'src'
    result = run_cli_no_context(
      tier_d_workspace,
      [
        'store',
        'diff',
        '--source',
        str(src),
        '--experiment',
        'exp-base',
        '--epoch-a',
        '0',
        '--epoch-b',
        '1',
      ],
    )
    _assert_json_success_envelope(result)


# ---------------------------------------------------------------------------
# 4.1 continued: Success-path mutating commands
# ---------------------------------------------------------------------------


class TestProjectDoctorSuccessEnvelope:
  """project doctor needs a real project with directories to report healthy."""

  def test_project_doctor_envelope(self, tier_a_workspace: Path) -> None:
    config = AutoPilotConfig(workspace=tier_a_workspace, project='testproj')
    config.init_project()
    config.root.mkdir(parents=True, exist_ok=True)
    config.experiments_path.mkdir(parents=True, exist_ok=True)
    config.datasets_path.mkdir(parents=True, exist_ok=True)
    config.records_path.mkdir(parents=True, exist_ok=True)
    config.cli_file.parent.mkdir(parents=True, exist_ok=True)
    config.cli_file.write_text('', encoding='utf-8')
    result = run_cli_no_context(
      tier_a_workspace,
      ['project', 'doctor', 'testproj'],
    )
    _assert_json_success_envelope(result)


class TestMutatingSuccessEnvelope:
  """Mutating commands that produce JSON envelopes on success."""

  def test_experiment_complete_envelope(self, tier_c_workspace: Path) -> None:
    result = run_cli(
      tier_c_workspace,
      ['experiment', 'complete', 'exp-running'],
    )
    _assert_json_success_envelope(result)

  def test_experiment_cancel_envelope(self, tier_c_workspace: Path) -> None:
    result = run_cli(
      tier_c_workspace,
      ['experiment', 'cancel', 'exp-running'],
    )
    _assert_json_success_envelope(result)

  def test_experiment_fail_envelope(self, tier_c_workspace: Path) -> None:
    result = run_cli(
      tier_c_workspace,
      ['experiment', 'fail', 'exp-running', '--error', 'test failure'],
    )
    _assert_json_success_envelope(result)

  def test_experiment_deploy_envelope(self, tier_c_workspace: Path) -> None:
    result = run_cli(
      tier_c_workspace,
      ['experiment', 'deploy', 'exp-base', '--as', 'prod'],
    )
    _assert_json_success_envelope(result)

  def test_experiment_undeploy_envelope(self, tier_c_workspace: Path) -> None:
    run_cli(
      tier_c_workspace,
      ['experiment', 'deploy', 'exp-base', '--as', 'staging'],
    )
    result = run_cli(
      tier_c_workspace,
      ['experiment', 'undeploy', 'staging'],
    )
    _assert_json_success_envelope(result)

  def test_experiment_invalidate_envelope(self, tier_c_workspace: Path) -> None:
    result = run_cli(
      tier_c_workspace,
      ['experiment', 'invalidate', 'exp-base', '--reason', 'bad data'],
    )
    _assert_json_success_envelope(result)

  def test_store_tag_create_envelope(self, tier_d_workspace: Path) -> None:
    result = run_cli(
      tier_d_workspace,
      ['store', 'tag', 'create', 'v1.0', '--experiment', 'exp-base', '--epoch', '0'],
    )
    _assert_json_success_envelope(result)

  def test_dataset_list_envelope(self, tier_a_workspace: Path) -> None:
    result = run_cli_no_context(tier_a_workspace, ['dataset', 'list'])
    _assert_json_success_envelope(result)

  def test_propose_list_envelope(self, tier_c_workspace: Path) -> None:
    result = run_cli_no_context(
      tier_c_workspace,
      ['--experiment', 'exp-base', 'propose', 'list'],
    )
    _assert_json_success_envelope(result)

  def test_policy_check_envelope(self, tier_a_workspace: Path) -> None:
    result = run_cli_no_context(
      tier_a_workspace,
      [
        'policy',
        'check',
        '--metrics',
        '{"accuracy": 0.9}',
        '--min',
        'accuracy:0.5',
      ],
    )
    _assert_json_success_envelope(result)

  def test_policy_explain_error_envelope(self, tier_c_workspace: Path) -> None:
    """policy explain requires a module; verify error envelope without one."""
    full = [
      '--experiment',
      'exp-base',
      'policy',
      'explain',
      '--json',
      '--workspace',
      str(tier_c_workspace),
    ]
    code, stdout = _run_full_cli_json(full)
    _assert_json_error_envelope(stdout, code)

  def test_execute_envelope(self, tier_a_workspace: Path) -> None:
    result = run_cli(
      tier_a_workspace,
      ['execute', '-c', 'print("hello")'],
    )
    _assert_json_success_envelope(result)

  def test_track_envelope(self, tier_a_workspace: Path) -> None:
    code, stdout = _run_full_cli_json(
      [
        '--json',
        '--context',
        'test',
        '--workspace',
        str(tier_a_workspace),
        'track',
        '--',
        'echo',
        'hi',
      ]
    )
    assert code == 0
    envelope = json.loads(stdout.strip())
    _assert_json_success_envelope(envelope)


# ---------------------------------------------------------------------------
# 4.2: Error-path envelope tests
# ---------------------------------------------------------------------------

ERROR_NONEXISTENT_EXPERIMENT: list[tuple[str, list[str]]] = [
  (
    'experiment show',
    ['experiment', 'show', '__nonexistent__'],
  ),
  (
    'experiment status',
    ['experiment', 'status', '__nonexistent__'],
  ),
  (
    'experiment compare',
    ['experiment', 'compare', '__nonexistent__', 'exp-base'],
  ),
  (
    'experiment impact',
    ['experiment', 'impact', '__nonexistent__'],
  ),
  (
    'experiment complete',
    ['experiment', 'complete', '__nonexistent__', '--context', 'test'],
  ),
  (
    'experiment fail',
    ['experiment', 'fail', '__nonexistent__', '--context', 'test'],
  ),
  (
    'experiment cancel',
    ['experiment', 'cancel', '__nonexistent__', '--context', 'test'],
  ),
  (
    'experiment invalidate',
    ['experiment', 'invalidate', '__nonexistent__', '--reason', 'bad', '--context', 'test'],
  ),
  (
    'experiment deploy',
    ['experiment', 'deploy', '__nonexistent__', '--as', 'prod', '--context', 'test'],
  ),
]


class TestErrorEnvelopeNonexistentExperiment:
  """Error paths for commands given a nonexistent experiment id."""

  @pytest.mark.parametrize(
    ('label', 'argv'),
    ERROR_NONEXISTENT_EXPERIMENT,
    ids=[c[0] for c in ERROR_NONEXISTENT_EXPERIMENT],
  )
  def test_json_error_exits_nonzero(
    self,
    tier_c_workspace: Path,
    label: str,
    argv: list[str],
  ) -> None:
    full = [*argv, '--json', '--workspace', str(tier_c_workspace)]
    code, stdout = _run_full_cli_json(full)
    _assert_json_error_envelope(stdout, code)


ERROR_NO_ACTIVE_TREE: list[tuple[str, list[str]]] = [
  ('query', ['query']),
  ('tree show', ['tree', 'show']),
  ('experiment list', ['experiment', 'list']),
]


class TestErrorEnvelopeNoActiveTree:
  """Error paths for commands when no active tree exists."""

  @pytest.mark.parametrize(
    ('label', 'argv'),
    ERROR_NO_ACTIVE_TREE,
    ids=[c[0] for c in ERROR_NO_ACTIVE_TREE],
  )
  def test_json_error_exits_nonzero(
    self,
    tier_a_workspace: Path,
    label: str,
    argv: list[str],
  ) -> None:
    full = [*argv, '--json', '--workspace', str(tier_a_workspace)]
    code, stdout = _run_full_cli_json(full)
    _assert_json_error_envelope(stdout, code)


ERROR_NO_EXPERIMENT_SPECIFIED: list[tuple[str, list[str]]] = [
  (
    'status without --experiment',
    ['status'],
  ),
]


class TestErrorEnvelopeNoExperiment:
  """Error paths for commands requiring an experiment when none specified."""

  @pytest.mark.parametrize(
    ('label', 'argv'),
    ERROR_NO_EXPERIMENT_SPECIFIED,
    ids=[c[0] for c in ERROR_NO_EXPERIMENT_SPECIFIED],
  )
  def test_json_error_exits_nonzero(
    self,
    tier_a_workspace: Path,
    label: str,
    argv: list[str],
  ) -> None:
    full = [*argv, '--json', '--workspace', str(tier_a_workspace), '--context', 'test']
    code, stdout = _run_full_cli_json(full)
    _assert_json_error_envelope(stdout, code)


ERROR_ARGPARSE_MISSING_ARGS: list[tuple[str, list[str]]] = [
  ('experiment complete (no id)', ['experiment', 'complete']),
  ('experiment fail (no id)', ['experiment', 'fail']),
  ('tree switch (no name)', ['tree', 'switch']),
  ('checkout (no id)', ['checkout']),
  ('store copy-epoch (no args)', ['store', 'copy-epoch']),
]


class TestErrorEnvelopeArgparseErrors:
  """Argparse usage errors still produce JSON envelopes."""

  @pytest.mark.parametrize(
    ('label', 'argv'),
    ERROR_ARGPARSE_MISSING_ARGS,
    ids=[c[0] for c in ERROR_ARGPARSE_MISSING_ARGS],
  )
  def test_json_error_on_missing_args(
    self,
    label: str,
    argv: list[str],
  ) -> None:
    full = [*argv, '--json']
    code, stdout = _run_full_cli_json(full)
    _assert_json_error_envelope(stdout, code)


class TestErrorEnvelopeHasErrorField:
  """Error envelopes contain a string ``error`` field."""

  def test_nonexistent_experiment_has_error_string(
    self,
    tier_c_workspace: Path,
  ) -> None:
    full = [
      'experiment',
      'show',
      '__nonexistent__',
      '--json',
      '--workspace',
      str(tier_c_workspace),
    ]
    code, stdout = _run_full_cli_json(full)
    assert code != 0
    lines = [line.strip() for line in stdout.strip().split('\n') if line.strip()]
    envelope = json.loads(lines[-1])
    assert isinstance(envelope['error'], str)
    assert len(envelope['error']) > 0

  def test_no_active_tree_has_error_string(
    self,
    tier_a_workspace: Path,
  ) -> None:
    full = [
      'query',
      '--json',
      '--workspace',
      str(tier_a_workspace),
    ]
    code, stdout = _run_full_cli_json(full)
    assert code != 0
    lines = [line.strip() for line in stdout.strip().split('\n') if line.strip()]
    envelope = json.loads(lines[-1])
    assert isinstance(envelope['error'], str)
    assert 'no active tree' in envelope['error']

  def test_argparse_error_has_error_string(self) -> None:
    full = ['experiment', 'complete', '--json']
    code, stdout = _run_full_cli_json(full)
    assert code != 0
    lines = [line.strip() for line in stdout.strip().split('\n') if line.strip()]
    envelope = json.loads(lines[-1])
    assert isinstance(envelope['error'], str)
    assert len(envelope['error']) > 0


# ---------------------------------------------------------------------------
# 4.3: Completeness test
# ---------------------------------------------------------------------------


class TestJsonCommandCompleteness:
  """Verify EXPECTED_JSON_COMMANDS stays aligned with the live CLI parser."""

  def test_all_json_commands_in_matrix(self) -> None:
    """Bidirectional drift guard: EXPECTED vs live CLI parser leaves.

    1. No phantom entries: every entry in EXPECTED_JSON_COMMANDS must be a
       real leaf command in the parser.
    2. No undocumented commands: every leaf command in the parser is either
       in EXPECTED_JSON_COMMANDS (has --json support) or in
       KNOWN_NON_JSON_COMMANDS (intentionally no --json output).

    Together these catch both stale matrix entries and newly-added commands
    that forgot to declare --json support.
    """
    cli = AutoPilotCLI()
    parser = cli.build_parser()
    leaf_commands = set(collect_leaf_commands(parser))

    not_in_parser = EXPECTED_JSON_COMMANDS - leaf_commands
    assert not not_in_parser, (
      f'EXPECTED_JSON_COMMANDS contains commands not found in the CLI parser: '
      f'{sorted(not_in_parser)}. Remove phantom entries or fix command registration.'
    )

    unaccounted = leaf_commands - EXPECTED_JSON_COMMANDS - KNOWN_NON_JSON_COMMANDS
    assert not unaccounted, (
      f'CLI leaf commands not accounted for in either EXPECTED_JSON_COMMANDS or '
      f'KNOWN_NON_JSON_COMMANDS: {sorted(unaccounted)}. '
      f'Add to EXPECTED_JSON_COMMANDS if the command emits --json output, '
      f'or to KNOWN_NON_JSON_COMMANDS if it intentionally does not.'
    )

  def test_expected_set_is_frozen(self) -> None:
    """EXPECTED_JSON_COMMANDS is a frozenset (immutable, not accidentally mutated)."""
    assert isinstance(EXPECTED_JSON_COMMANDS, frozenset)

  def test_expected_set_has_no_empty_entries(self) -> None:
    """No blank entries in the frozen command set."""
    for entry in EXPECTED_JSON_COMMANDS:
      assert entry.strip(), 'empty entry in EXPECTED_JSON_COMMANDS'

  def test_expected_set_count(self) -> None:
    """Baseline count to detect accidental additions or removals."""
    assert len(EXPECTED_JSON_COMMANDS) == 89

  def test_no_overlap_between_json_and_non_json(self) -> None:
    """Expected JSON and known non-JSON sets must be disjoint."""
    overlap = EXPECTED_JSON_COMMANDS & KNOWN_NON_JSON_COMMANDS
    assert not overlap, f'commands in both sets: {sorted(overlap)}'


# ---------------------------------------------------------------------------
# 4.4: Optional hygiene
# ---------------------------------------------------------------------------


class TestJsonEnvelopeParseability:
  """Spot checks that stdout is parseable JSON (not mixed with stderr text)."""

  def test_workspace_status_clean_json(self, tier_a_workspace: Path) -> None:
    result = run_cli_no_context(tier_a_workspace, ['workspace', 'status'])
    raw = json.dumps(result)
    reparsed = json.loads(raw)
    assert reparsed == result

  def test_tree_list_clean_json(self, tier_b_workspace: Path) -> None:
    result = run_cli_no_context(tier_b_workspace, ['tree', 'list'])
    raw = json.dumps(result)
    reparsed = json.loads(raw)
    assert reparsed == result


# ---------------------------------------------------------------------------
# 4.5: Promoted command envelope tests (plan 11)
# ---------------------------------------------------------------------------


class TestPromotedCommandEnvelopes:
  """Verify promoted commands produce correct JSON envelopes."""

  def test_store_snapshot_json_envelope(self, tier_d_workspace: Path) -> None:
    """store snapshot --json returns epoch (int) and skipped (bool)."""
    src = tier_d_workspace / 'src'
    (src / 'main.py').write_text('v-snapshot-test', encoding='utf-8')
    result = run_cli(
      tier_d_workspace,
      ['--experiment', 'exp-base', 'store', 'snapshot', '--source', str(src)],
    )
    _assert_json_success_envelope(result)
    inner = result['result']
    assert isinstance(inner['epoch'], int)
    assert isinstance(inner['skipped'], bool)

  def test_store_checkout_json_envelope(self, tier_d_workspace: Path) -> None:
    """store checkout --json returns slug and epoch."""
    src = tier_d_workspace / 'src'
    result = run_cli(
      tier_d_workspace,
      [
        '--experiment',
        'exp-base',
        'store',
        'checkout',
        '--source',
        str(src),
        '--epoch',
        '0',
      ],
    )
    _assert_json_success_envelope(result)
    inner = result['result']
    assert 'slug' in inner
    assert 'epoch' in inner

  def test_tree_create_json_envelope(self, tier_b_workspace: Path) -> None:
    """tree create --json returns ok and tree name."""
    result = run_cli(tier_b_workspace, ['tree', 'create', 'tmp-desc'])
    _assert_json_success_envelope(result)
    inner = result['result']
    assert inner['ok'] is True
    assert inner['tree'] == 'tmp-desc'

  def test_tree_remove_json_envelope(self, tier_b_workspace: Path) -> None:
    """tree remove --json returns ok and removed name."""
    run_cli(tier_b_workspace, ['tree', 'create', 'removable'])
    result = run_cli(tier_b_workspace, ['tree', 'remove', 'removable'])
    _assert_json_success_envelope(result)
    inner = result['result']
    assert inner['ok'] is True
    assert inner['removed'] == 'removable'

  def test_experiment_notes_write_json(self, tier_c_workspace: Path) -> None:
    """experiment notes write --json returns bytes_written without notes key."""
    result = run_cli(
      tier_c_workspace,
      ['experiment', 'notes', 'write', 'exp-base', '--body', 'hello'],
    )
    _assert_json_success_envelope(result)
    inner = result['result']
    assert inner['bytes_written'] == len(b'hello')
    assert 'notes' not in inner

  def test_json_matrix_count_updated(self) -> None:
    """len(EXPECTED_JSON_COMMANDS) matches the new baseline."""
    assert len(EXPECTED_JSON_COMMANDS) == 89

  def test_store_branch_json_envelope(self, tier_d_workspace: Path) -> None:
    """store branch --json returns experiment_id."""
    src = tier_d_workspace / 'src'
    result = run_cli(
      tier_d_workspace,
      ['--experiment', 'exp-cand', 'store', 'branch', '--source', str(src)],
    )
    _assert_json_success_envelope(result)
    inner = result['result']
    assert 'experiment_id' in inner
