"""Parametrized tests for CLI error handling normalization (Plan 02).

Verifies that handler-level validation failures exit with code 1 (not 0),
that argparse-level failures exit with code 2, and that --json mode produces
correct error envelopes.

Exit code semantics:
  2 = argparse usage error (unknown flag, missing required arg)
  1 = handler/runtime validation failure (ctx.fail)
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.main import AutoPilotCLI
from autopilot.core.config import AutoPilotConfig
from io import StringIO
from pathlib import Path
from tests.cli.conftest import seed_tree_with_experiments
from unittest.mock import patch
import json
import pytest


def _run_full_cli(argv: list[str]) -> tuple[int, str, str]:
  """Run the full AutoPilotCLI with captured stdout/stderr."""
  cli = AutoPilotCLI()
  out = StringIO()
  err = StringIO()
  exit_code = 0
  with patch('sys.stdout', out), patch('sys.stderr', err):
    try:
      cli(argv=argv)
    except SystemExit as e:
      exit_code = int(e.code) if e.code is not None else 0
  return exit_code, out.getvalue(), err.getvalue()


@pytest.fixture
def ws(tmp_path: Path) -> Path:
  ws = tmp_path / 'ws'
  ws.mkdir()
  return ws


@pytest.fixture
def ws_with_tree(ws: Path) -> Path:
  """Workspace with an active tree but no experiments."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  forest.create_tree('main')
  forest.switch('main')
  forest.save()
  return ws


@pytest.fixture
def ws_with_experiments(ws: Path) -> Path:
  """Workspace with tree and experiments."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  seed_tree_with_experiments(
    forest,
    'main',
    [
      {
        'id': 'exp-a',
        'hypothesis': 'baseline',
        'status': 'completed',
        'metrics': {'accuracy': 0.72},
      },
    ],
  )
  return ws


class TestHandlerFailureExitCode:
  """Handler validation failures produce exit code 1."""

  @pytest.mark.parametrize(
    ('argv_suffix', 'expected_fragment'),
    [
      (['status'], 'no experiment specified'),
      (['policy', 'explain'], 'experiment slug required'),
      (['debug', 'collect'], 'experiment slug required'),
    ],
  )
  def test_missing_experiment_exits_1(
    self,
    ws_with_tree: Path,
    argv_suffix: list[str],
    expected_fragment: str,
  ) -> None:
    exit_code, stdout, stderr = _run_full_cli(
      [*argv_suffix, '--context', 'test', '--workspace', str(ws_with_tree)]
    )
    assert exit_code == 1
    combined = stdout + stderr
    assert expected_fragment in combined

  def test_trace_collect_epoch_none_unit(self) -> None:
    """trace collect with epoch=None exits 1 (unit test with mock ctx).

    The global --epoch flag defaults to 0, so this path is only reachable
    programmatically. Covered by tests/cli/test_trace.py::test_collect_no_epoch.
    """
    from autopilot.cli.commands.trace import TraceCommand
    from autopilot.cli.context import CLIContext
    from unittest.mock import MagicMock

    ctx = MagicMock()
    ctx.epoch = None
    ctx.output = MagicMock()
    ctx.fail = CLIContext.fail.__get__(ctx, type(ctx))
    cmd = TraceCommand()
    args = MagicMock(epoch=None, limit=0)
    with pytest.raises(SystemExit) as exc_info:
      cmd.collect(ctx, args)
    assert exc_info.value.code == 1

  def test_diagnose_run_epoch_none_unit(self) -> None:
    """diagnose run with epoch=None exits 1 (unit test with mock ctx)."""
    from autopilot.cli.commands.diagnose import DiagnoseCommand
    from autopilot.cli.context import CLIContext
    from unittest.mock import MagicMock

    ctx = MagicMock()
    ctx.epoch = None
    ctx.output = MagicMock()
    ctx.fail = CLIContext.fail.__get__(ctx, type(ctx))
    cmd = DiagnoseCommand()
    args = MagicMock(epoch=None, category='', node='')
    with pytest.raises(SystemExit) as exc_info:
      cmd.run_diagnose(ctx, args)
    assert exc_info.value.code == 1

  def test_optimize_train_no_module_exits_1(self, ws_with_experiments: Path) -> None:
    exit_code, stdout, stderr = _run_full_cli(
      [
        'optimize',
        'train',
        '--context',
        'test',
        '--workspace',
        str(ws_with_experiments),
        '--experiment',
        'exp-a',
      ]
    )
    assert exit_code == 1
    combined = stdout + stderr
    assert 'no module configured' in combined or 'no trainer configured' in combined

  def test_optimize_set_hparams_invalid_json_exits_1(self, ws_with_experiments: Path) -> None:
    exit_code, stdout, stderr = _run_full_cli(
      [
        'optimize',
        'set-hparams',
        '--values',
        '{bad json',
        '--context',
        'test',
        '--workspace',
        str(ws_with_experiments),
        '--experiment',
        'exp-a',
      ]
    )
    assert exit_code == 1
    combined = stdout + stderr
    assert 'invalid JSON' in combined

  def test_stabilize_not_found_exits_1(self, ws_with_tree: Path) -> None:
    exit_code, stdout, stderr = _run_full_cli(
      ['stabilize', 'nonexistent', '--context', 'test', '--workspace', str(ws_with_tree)]
    )
    assert exit_code == 1
    combined = stdout + stderr
    assert 'not found' in combined

  def test_checkout_not_found_exits_1(self, ws_with_experiments: Path) -> None:
    exit_code, stdout, stderr = _run_full_cli(
      ['checkout', 'ghost', '--context', 'test', '--workspace', str(ws_with_experiments)]
    )
    assert exit_code == 1
    combined = stdout + stderr
    assert 'not found' in combined


class TestArgparseExitCode:
  """Argparse-level failures produce exit code 2 (not 1).

  Regression guard: argparse usage errors (exit 2) vs handler failures (exit 1).
  """

  @pytest.mark.parametrize(
    'argv',
    [
      ['checkout'],
      ['stabilize'],
      ['tree', 'create'],
    ],
  )
  def test_missing_positional_exits_2(self, ws: Path, argv: list[str]) -> None:
    exit_code, _, _ = _run_full_cli([*argv, '--context', 'test', '--workspace', str(ws)])
    assert exit_code == 2

  def test_unknown_flag_exits_2(self, ws: Path) -> None:
    exit_code, _, _ = _run_full_cli(
      ['checkout', 'x', '--nonexistent-flag', '--context', 'test', '--workspace', str(ws)]
    )
    assert exit_code == 2


class TestJsonErrorEnvelope:
  """Tests that --json on handler failure produces correct error envelope.

  Envelope format: {'ok': False, 'error': str, 'messages': list[...]}.
  """

  def test_checkout_store_failure_json(self, ws_with_experiments: Path) -> None:
    with patch('autopilot.core.tree.Tree.checkout', side_effect=RuntimeError('disk full')):
      exit_code, stdout, _ = _run_full_cli(
        [
          'checkout',
          'exp-a',
          '--json',
          '--context',
          'test',
          '--workspace',
          str(ws_with_experiments),
        ]
      )
    assert exit_code == 1
    payload = json.loads(stdout)
    assert payload['ok'] is False
    assert 'messages' in payload
    assert isinstance(payload['messages'], list)
    assert len(payload['messages']) >= 1

  def test_stabilize_not_found_json(self, ws_with_tree: Path) -> None:
    exit_code, stdout, _ = _run_full_cli(
      [
        'stabilize',
        'nonexistent',
        '--json',
        '--context',
        'test',
        '--workspace',
        str(ws_with_tree),
      ]
    )
    assert exit_code == 1
    payload = json.loads(stdout)
    assert payload['ok'] is False
    assert 'messages' in payload
    assert isinstance(payload['messages'], list)
    assert len(payload['messages']) >= 1

  def test_status_no_experiment_json(self, ws_with_tree: Path) -> None:
    exit_code, stdout, _ = _run_full_cli(
      [
        'status',
        '--json',
        '--context',
        'test',
        '--workspace',
        str(ws_with_tree),
      ]
    )
    assert exit_code == 1
    payload = json.loads(stdout)
    assert payload['ok'] is False
    assert 'messages' in payload
    assert isinstance(payload['messages'], list)

  def test_debug_collect_no_experiment_json(self, ws_with_tree: Path) -> None:
    exit_code, stdout, _ = _run_full_cli(
      [
        'debug',
        'collect',
        '--json',
        '--context',
        'test',
        '--workspace',
        str(ws_with_tree),
      ]
    )
    assert exit_code == 1
    payload = json.loads(stdout)
    assert payload['ok'] is False
    assert 'messages' in payload
    assert isinstance(payload['messages'], list)
    assert len(payload['messages']) >= 1


class TestExitCodeOneNotTwo:
  """Explicit split: argparse (2) vs handler (1).

  Regression guard ensuring we never confuse the two exit code domains.
  """

  def test_argparse_vs_handler_exit_codes_differ(self, ws_with_tree: Path) -> None:
    """argparse exit code (2) != handler failure exit code (1)."""
    argparse_code, _, _ = _run_full_cli(
      [
        'checkout',
        '--context',
        'test',
        '--workspace',
        str(ws_with_tree),
      ]
    )
    handler_code, _, _ = _run_full_cli(
      [
        'checkout',
        'nonexistent',
        '--context',
        'test',
        '--workspace',
        str(ws_with_tree),
      ]
    )
    assert argparse_code == 2
    assert handler_code == 1
    assert argparse_code != handler_code


class TestProposeFailurePaths:
  """propose verify/revert failures exit 1 through full CLI dispatch (section 4.5)."""

  def test_propose_verify_no_proposal_id_exits_2(self, ws_with_experiments: Path) -> None:
    exit_code, stdout, stderr = _run_full_cli(
      [
        'propose',
        'verify',
        '--context',
        'test',
        '--workspace',
        str(ws_with_experiments),
        '--experiment',
        'exp-a',
      ]
    )
    assert exit_code == 2
    combined = stdout + stderr
    assert '--proposal-id' in combined

  def test_propose_revert_no_proposal_id_exits_1(self, ws_with_experiments: Path) -> None:
    exit_code, stdout, stderr = _run_full_cli(
      [
        'propose',
        'revert',
        '--context',
        'test',
        '--workspace',
        str(ws_with_experiments),
        '--experiment',
        'exp-a',
      ]
    )
    assert exit_code == 1
    combined = stdout + stderr
    assert '--proposal-id is required' in combined

  def test_propose_verify_proposal_not_found_exits_1(self, ws_with_experiments: Path) -> None:
    exit_code, stdout, stderr = _run_full_cli(
      [
        'propose',
        'verify',
        '--proposal-id',
        'ghost',
        '--epoch',
        '0',
        '--context',
        'test',
        '--workspace',
        str(ws_with_experiments),
        '--experiment',
        'exp-a',
      ]
    )
    assert exit_code == 1
    combined = stdout + stderr
    assert 'not found' in combined

  def test_propose_revert_proposal_not_found_exits_1(self, ws_with_experiments: Path) -> None:
    exit_code, stdout, stderr = _run_full_cli(
      [
        'propose',
        'revert',
        '--proposal-id',
        'ghost',
        '--context',
        'test',
        '--workspace',
        str(ws_with_experiments),
        '--experiment',
        'exp-a',
      ]
    )
    assert exit_code == 1
    combined = stdout + stderr
    assert 'not found' in combined


class TestNoCtxOutputErrorReturn:
  """Verify no ctx.output.error(...) followed by return in command files."""

  def test_no_error_return_pattern_in_commands(self) -> None:
    """The pattern 'ctx.output.error(...); return' should not exist."""
    import re

    commands_dir = Path(__file__).parent.parent.parent / 'src' / 'autopilot' / 'cli' / 'commands'
    pattern = re.compile(r'ctx\.output\.error\([^\n]+\)\s*\n\s*return')
    violations = []
    for py_file in commands_dir.glob('*.py'):
      content = py_file.read_text()
      if pattern.search(content):
        violations.append(py_file.name)
    assert violations == [], f'files still using ctx.output.error + return: {violations}'

  def test_no_tree_head_in_checkout(self) -> None:
    """checkout.py must not access tree._head."""
    checkout_file = (
      Path(__file__).parent.parent.parent / 'src' / 'autopilot' / 'cli' / 'commands' / 'checkout.py'
    )
    content = checkout_file.read_text()
    assert '_head' not in content
