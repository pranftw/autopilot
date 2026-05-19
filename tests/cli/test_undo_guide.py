"""Tests for the undo-guide CLI command.

Verifies that the read-only undo-guide command correctly inspects
executions.jsonl and produces structured reversal suggestions with
appropriate confidence levels for various mutating commands.
"""

from autopilot.cli.command import CLI
from autopilot.cli.commands.undo import (
  UndoSuggestion,
  _extract_switch_target,
  _lookup_recipe,
  _recipe_experiment_deploy,
  _recipe_store_checkout,
  _recipe_store_snapshot,
  _recipe_terminal_operation,
)
from autopilot.core.config import AutoPilotConfig
from autopilot.tracking.executions import ExecutionRecord, log_execution
from autopilot.tracking.io import utc_now_iso
from pathlib import Path
from tests.cli.conftest import run_cli_no_context
import pytest


def _make_record(
  command: str,
  args: list[str] | None = None,
  context: str | None = 'test',
  exit_code: int = 0,
) -> ExecutionRecord:
  """Build a test execution record."""
  return ExecutionRecord(
    timestamp=utc_now_iso(),
    command=command,
    args=args if args is not None else [],
    duration_ms=100.0,
    exit_code=exit_code,
    context=context,
  )


def _seed_workspace(tmp_path: Path) -> Path:
  """Create minimal workspace structure."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  return ws


def _seed_records(ws: Path, records: list[ExecutionRecord]) -> None:
  """Write execution records to the workspace."""
  config = AutoPilotConfig(workspace=ws)
  config.executions_path.parent.mkdir(parents=True, exist_ok=True)
  for rec in records:
    log_execution(config.executions_path, rec)


class TestUndoGuideDeploySuggestsUndeploy:
  """experiment deploy -> suggests undeploy with label."""

  def test_deploy_suggests_undeploy(self, tmp_path: Path) -> None:
    """Seed execution row experiment deploy --as prod; assert undeploy suggestion."""
    ws = _seed_workspace(tmp_path)
    records = [
      _make_record('experiment deploy', ['exp-123', '--as', 'prod']),
    ]
    _seed_records(ws, records)

    result = run_cli_no_context(ws, ['undo-guide'])
    assert result['ok'] is True
    assert 'undeploy' in result['result']['suggested_undo']
    assert 'prod' in result['result']['suggested_undo']
    assert result['result']['confidence'] == 'high'


class TestUndoGuideTreeSwitchSuggestsSwitchBack:
  """Two tree switch records -> suggests switch to prior tree."""

  def test_tree_switch_suggests_switch_back(self, tmp_path: Path) -> None:
    """Two tree switch rows (b then a newest-first); assert suggests tree switch a."""
    ws = _seed_workspace(tmp_path)
    records = [
      _make_record('tree switch', ['a']),
      _make_record('tree switch', ['b']),
    ]
    _seed_records(ws, records)

    result = run_cli_no_context(ws, ['undo-guide'])
    assert result['ok'] is True
    assert result['result']['suggested_undo'] == 'tree switch a'
    assert result['result']['confidence'] == 'medium'


class TestUndoGuideExperimentAddSuggestsCancel:
  """experiment add -> suggests experiment cancel."""

  def test_experiment_add_suggests_cancel(self, tmp_path: Path) -> None:
    """Row for experiment add with known id; cancel suggestion contains id."""
    ws = _seed_workspace(tmp_path)
    records = [
      _make_record('experiment add', ['exp-new-42']),
    ]
    _seed_records(ws, records)

    result = run_cli_no_context(ws, ['undo-guide'])
    assert result['ok'] is True
    assert 'cancel' in result['result']['suggested_undo']
    assert 'exp-new-42' in result['result']['suggested_undo']
    assert result['result']['confidence'] == 'high'


class TestUndoGuideCheckoutSuggestsRecover:
  """store checkout -> suggests store recover."""

  def test_checkout_suggests_recover(self, tmp_path: Path) -> None:
    """store checkout maps to recover suggestion."""
    ws = _seed_workspace(tmp_path)
    records = [
      _make_record('store checkout', ['--epoch', '2']),
    ]
    _seed_records(ws, records)

    result = run_cli_no_context(ws, ['undo-guide'])
    assert result['ok'] is True
    assert 'recover' in result['result']['suggested_undo']
    assert result['result']['confidence'] == 'medium'


class TestUndoGuideNoMutations:
  """Only read-only rows -> fails with guidance."""

  def test_no_mutations(self, tmp_path: Path) -> None:
    """Only read-only records in execution log -> non-zero exit."""
    ws = _seed_workspace(tmp_path)
    records = [
      _make_record('query', []),
      _make_record('status', []),
      _make_record('tree list', []),
    ]
    _seed_records(ws, records)

    with pytest.raises(SystemExit):
      run_cli_no_context(ws, ['undo-guide'])


class TestUndoGuideTerminalOperation:
  """Terminal status transitions -> low confidence, no suggestion."""

  def test_terminal_operation_complete(self, tmp_path: Path) -> None:
    """experiment complete yields confidence='low' and terminal notes."""
    ws = _seed_workspace(tmp_path)
    records = [
      _make_record('experiment complete', ['exp-done']),
    ]
    _seed_records(ws, records)

    result = run_cli_no_context(ws, ['undo-guide'])
    assert result['ok'] is True
    assert result['result']['confidence'] == 'low'
    assert result['result']['suggested_undo'] is None
    assert 'terminal' in result['result']['notes']

  def test_terminal_operation_fail(self, tmp_path: Path) -> None:
    """experiment fail yields confidence='low' and terminal notes."""
    ws = _seed_workspace(tmp_path)
    records = [
      _make_record('experiment fail', ['exp-broke']),
    ]
    _seed_records(ws, records)

    result = run_cli_no_context(ws, ['undo-guide'])
    assert result['ok'] is True
    assert result['result']['confidence'] == 'low'
    assert 'terminal' in result['result']['notes']

  def test_terminal_operation_invalidate(self, tmp_path: Path) -> None:
    """experiment invalidate yields confidence='low' and terminal notes."""
    ws = _seed_workspace(tmp_path)
    records = [
      _make_record('experiment invalidate', ['exp-bad']),
    ]
    _seed_records(ws, records)

    result = run_cli_no_context(ws, ['undo-guide'])
    assert result['ok'] is True
    assert result['result']['confidence'] == 'low'
    assert 'terminal' in result['result']['notes']


class TestUndoGuideJsonShape:
  """Full JSON envelope structure validation."""

  def test_json_shape(self, tmp_path: Path) -> None:
    """Validate full nested structure + envelope keys ok, result, messages."""
    ws = _seed_workspace(tmp_path)
    records = [
      _make_record('experiment deploy', ['exp-1', '--as', 'staging']),
    ]
    _seed_records(ws, records)

    result = run_cli_no_context(ws, ['undo-guide'])
    assert 'ok' in result
    assert result['ok'] is True
    assert 'result' in result
    assert 'messages' in result

    inner = result['result']
    assert 'last_mutation' in inner
    assert 'suggested_undo' in inner
    assert 'confidence' in inner
    assert 'notes' in inner

    mutation = inner['last_mutation']
    assert 'command' in mutation
    assert 'args' in mutation
    assert 'timestamp' in mutation
    assert 'context' in mutation
    assert mutation['command'] == 'experiment deploy'
    assert mutation['args'] == ['exp-1', '--as', 'staging']


class TestUndoGuideConfidenceLevels:
  """Table-driven confidence assertions."""

  @pytest.mark.parametrize(
    ('command', 'args', 'expected_confidence'),
    [
      ('experiment deploy', ['e', '--as', 'prod'], 'high'),
      ('experiment add', ['new-exp'], 'high'),
      ('store checkout', [], 'medium'),
      ('experiment complete', ['e'], 'low'),
      ('experiment fail', ['e'], 'low'),
      ('store snapshot', [], 'low'),
    ],
  )
  def test_confidence_levels(
    self,
    tmp_path: Path,
    command: str,
    args: list[str],
    expected_confidence: str,
  ) -> None:
    """Confidence matches expected level for each recipe."""
    ws = _seed_workspace(tmp_path)
    records = [_make_record(command, args)]
    _seed_records(ws, records)

    result = run_cli_no_context(ws, ['undo-guide'])
    assert result['ok'] is True
    assert result['result']['confidence'] == expected_confidence


class TestUndoGuideMissingExecutionsFile:
  """No executions.jsonl -> exits 0 with empty guidance."""

  def test_missing_executions_file(self, tmp_path: Path) -> None:
    """Command exits 0 with null last_mutation and guidance message."""
    ws = _seed_workspace(tmp_path)

    result = run_cli_no_context(ws, ['undo-guide'])
    assert result['ok'] is True
    assert result['result']['last_mutation'] is None
    assert result['result']['suggested_undo'] is None
    assert 'no execution history' in result['result']['notes']


class TestUndoGuideCorruptJsonlLineSkipped:
  """Corrupt JSONL line skipped; valid records still drive output."""

  def test_corrupt_jsonl_line_skipped(self, tmp_path: Path) -> None:
    """One corrupt line + valid mutation row -> suggestion produced."""
    ws = _seed_workspace(tmp_path)
    config = AutoPilotConfig(workspace=ws)
    config.executions_path.parent.mkdir(parents=True, exist_ok=True)

    config.executions_path.write_text('THIS IS NOT JSON\n', encoding='utf-8')

    records = [_make_record('experiment deploy', ['x', '--as', 'live'])]
    _seed_records(ws, records)

    result = run_cli_no_context(ws, ['undo-guide'])
    assert result['ok'] is True
    assert result['result']['suggested_undo'] == 'experiment undeploy live'


class TestUndoGuideUnknownCommandLowConfidence:
  """Unrecognized command prefix -> low confidence + manual review."""

  def test_unknown_command_low_confidence(self, tmp_path: Path) -> None:
    """Latest mutating record is unrecognized; notes include review guidance."""
    ws = _seed_workspace(tmp_path)
    records = [
      _make_record('store worktree create', ['--branch', 'feature']),
    ]
    _seed_records(ws, records)

    result = run_cli_no_context(ws, ['undo-guide'])
    assert result['ok'] is True
    assert result['result']['confidence'] == 'low'
    assert 'manual review' in result['result']['notes']
    assert result['result']['suggested_undo'] is None


class TestUndoGuideSingleSwitchNoPrior:
  """Single tree switch in log -> cannot infer prior tree."""

  def test_single_switch_no_prior(self, tmp_path: Path) -> None:
    """One tree switch record -> suggested_undo is None with explanation."""
    ws = _seed_workspace(tmp_path)
    records = [
      _make_record('tree switch', ['new-tree']),
    ]
    _seed_records(ws, records)

    result = run_cli_no_context(ws, ['undo-guide'])
    assert result['ok'] is True
    assert result['result']['suggested_undo'] is None
    assert 'prior tree switch' in result['result']['notes']
    assert result['result']['confidence'] == 'low'


class TestUndoGuideUnitRecipes:
  """Unit tests for individual recipe functions."""

  def test_extract_switch_target_positional(self) -> None:
    """Extracts first non-flag token as tree name."""
    assert _extract_switch_target(['my-tree']) == 'my-tree'
    assert _extract_switch_target(['--no-checkout', 'other']) == 'other'

  def test_extract_switch_target_none(self) -> None:
    """Returns None when no positional found."""
    assert _extract_switch_target(['--no-checkout']) is None
    assert _extract_switch_target([]) is None

  def test_recipe_deploy_missing_label(self) -> None:
    """Deploy without --as label -> low confidence."""
    rec = _make_record('experiment deploy', ['exp-1'])
    result = _recipe_experiment_deploy(rec)
    assert result.confidence == 'low'
    assert result.suggested_undo is None

  def test_recipe_store_snapshot(self) -> None:
    """Snapshot -> low confidence, no undo."""
    rec = _make_record('store snapshot', [])
    result = _recipe_store_snapshot(rec)
    assert result.confidence == 'low'
    assert result.suggested_undo is None

  def test_recipe_terminal(self) -> None:
    """Terminal recipe -> low confidence, terminal notes."""
    rec = _make_record('experiment complete', [])
    result = _recipe_terminal_operation(rec)
    assert result.confidence == 'low'
    assert result.notes is not None
    assert 'terminal' in result.notes

  def test_recipe_store_checkout(self) -> None:
    """Checkout -> medium confidence with recover command."""
    rec = _make_record('store checkout', [])
    result = _recipe_store_checkout(rec)
    assert result.confidence == 'medium'
    assert result.suggested_undo is not None
    assert 'recover' in result.suggested_undo

  def test_lookup_recipe_fallback(self) -> None:
    """Unknown command -> low confidence fallback."""
    rec = _make_record('store promote', ['exp-1'])
    cli = CLI()
    result = _lookup_recipe('store promote', rec, [rec], cli)
    assert result.confidence == 'low'
    assert result.notes is not None
    assert 'manual review' in result.notes

  def test_undo_suggestion_frozen(self) -> None:
    """UndoSuggestion is a frozen dataclass."""
    s = UndoSuggestion(suggested_undo='cmd', confidence='high', notes=None)
    with pytest.raises(AttributeError):
      s.confidence = 'low'  # type: ignore[ty:invalid-assignment]
