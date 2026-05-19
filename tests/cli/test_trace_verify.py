"""Tests for the ``trace verify`` CLI subcommand.

Covers JSON shape, context exemption, experiment slug requirement,
epoch inference from policy entries, and failure when empty log
has no --epochs.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.context import ContextEntry
from autopilot.core.decision import DecisionEntry
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli_no_context


def _policy_entry(epoch: int) -> ContextEntry:
  """Build a policy gate context entry for a given epoch."""
  return ContextEntry.create(
    f'epoch {epoch} accepted by policy gate',
    source='policy',
    epoch=epoch,
    metadata={'_type': DecisionEntry.POLICY_GATE_TYPE, 'gates': []},
  )


def _gradient_entry(epoch: int) -> ContextEntry:
  """Build a gradient journal context entry."""
  return ContextEntry.create(
    'gradient feedback recorded',
    source='trainer',
    epoch=epoch,
    metadata={
      'gradient_summaries': [
        {
          'param_name': 'p',
          'param_type': 'ScalarParameter',
          'gradient_type': 'TextGradient',
          'summary': 'improve X',
        },
      ],
    },
  )


def _setup_workspace(
  tmp_path: Path,
  experiment_id: str,
  entries: list[ContextEntry] | None = None,
) -> Path:
  """Create a workspace with a forest, tree, and experiment carrying context entries.

  Args:
    tmp_path: Pytest tmp_path root.
    experiment_id: Experiment id to create.
    entries: Context entries to seed on the experiment.

  Returns:
    Workspace path.
  """
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')

  exp = Experiment(experiment_id=experiment_id, hypothesis='test')
  exp.start()
  exp.complete(metrics={'accuracy': 0.9})

  if entries:
    for entry in entries:
      exp.context_log.record(entry)

  tree.add(Node(experiment=exp))
  forest.save()
  return ws


class TestTraceVerifyJsonShape:
  """trace verify returns proper JSON envelope."""

  def test_trace_verify_json_shape(self, tmp_path: Path) -> None:
    """Seed experiment with complete trace; verify JSON result shape and completeness."""
    entries = [
      _policy_entry(0),
      _policy_entry(1),
      _gradient_entry(1),
    ]
    ws = _setup_workspace(tmp_path, 'exp-001', entries)
    result = run_cli_no_context(
      ws,
      ['trace', 'verify', '--experiment', 'exp-001', '--epochs', '2'],
    )
    assert result['ok'] is True
    assert result['result']['epochs_run'] == 2
    assert result['result']['experiment_id'] == 'exp-001'
    report = result['result']['report']
    assert report['complete'] is True
    assert isinstance(report['dimensions'], list)
    assert report['gaps'] == []


class TestTraceVerifyContextExempt:
  """trace verify is context-exempt (read-only)."""

  def test_trace_verify_context_exempt(self, tmp_path: Path) -> None:
    """Invoke without --context; exit 0 via run_cli_no_context."""
    entries = [_policy_entry(0), _gradient_entry(0)]
    ws = _setup_workspace(tmp_path, 'exp-001', entries)
    result = run_cli_no_context(
      ws,
      ['trace', 'verify', '--experiment', 'exp-001', '--epochs', '1'],
    )
    assert result['ok'] is True


class TestTraceVerifyRequiresExperimentSlug:
  """trace verify fails without --experiment."""

  def test_trace_verify_requires_experiment_slug(self, tmp_path: Path) -> None:
    """No --experiment -> non-zero exit, error mentions experiment."""
    ws = _setup_workspace(tmp_path, 'exp-001')
    try:
      result = run_cli_no_context(ws, ['trace', 'verify', '--epochs', '1'])
      assert result.get('ok') is False
      error_msg = result.get('error')
      assert error_msg is not None
      assert 'experiment' in error_msg.lower()
    except SystemExit:
      pass


class TestTraceVerifyInfersEpochsFromPolicyEntries:
  """trace verify infers epochs from policy gate entries when --epochs omitted."""

  def test_trace_verify_infers_epochs_from_policy_entries(self, tmp_path: Path) -> None:
    """Log with policy entries for epochs 0..2; omit --epochs -> epochs_run == 3."""
    entries = [
      _policy_entry(0),
      _policy_entry(1),
      _policy_entry(2),
      _gradient_entry(2),
    ]
    ws = _setup_workspace(tmp_path, 'exp-001', entries)
    result = run_cli_no_context(
      ws,
      ['trace', 'verify', '--experiment', 'exp-001'],
    )
    assert result['ok'] is True
    assert result['result']['epochs_run'] == 3


class TestTraceVerifyFailsWithoutEpochsWhenEmptyLog:
  """trace verify fails when context log is empty and --epochs not given."""

  def test_trace_verify_fails_without_epochs_when_empty_log(self, tmp_path: Path) -> None:
    """Empty context log, no --epochs -> non-zero exit with guidance to pass --epochs."""
    ws = _setup_workspace(tmp_path, 'exp-001')
    try:
      result = run_cli_no_context(
        ws,
        ['trace', 'verify', '--experiment', 'exp-001'],
      )
      assert result.get('ok') is False
      error_msg = result.get('error')
      assert error_msg is not None
      assert '--epochs' in error_msg
    except SystemExit:
      pass
