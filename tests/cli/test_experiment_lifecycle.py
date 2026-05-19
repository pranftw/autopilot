"""Tests for experiment lifecycle CLI: complete, fail, cancel.

Covers Plan 01 (FR-001, FR-002): experiment complete/fail/cancel via CLI,
forest persistence, pending->completed transition, and context enforcement.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.helpers import journal_user_context
from autopilot.core.config import AutoPilotConfig
from autopilot.core.enums import Status
from autopilot.core.errors import ExperimentError
from autopilot.core.experiment import Experiment
from pathlib import Path
from tests.cli.conftest import run_cli, seed_tree_with_experiments
from unittest.mock import patch
import pytest


@pytest.fixture(autouse=True)
def _patch_store_checkout():
  """Patch FileStore.checkout for tests that don't create snapshots."""
  with patch('autopilot.ai.store.file_store.FileStore.checkout'):
    yield


@pytest.fixture
def ws(tmp_path: Path) -> Path:
  """Workspace root."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  return ws


@pytest.fixture
def ws_with_pending(ws: Path) -> Path:
  """Workspace with an active tree containing a pending experiment."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  seed_tree_with_experiments(
    forest,
    'main',
    [{'id': 'exp-pending', 'hypothesis': 'test pending', 'status': 'pending'}],
  )
  return ws


@pytest.fixture
def ws_with_running(ws: Path) -> Path:
  """Workspace with an active tree containing a running experiment."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  seed_tree_with_experiments(
    forest,
    'main',
    [{'id': 'exp-running', 'hypothesis': 'test running', 'status': 'running'}],
  )
  return ws


@pytest.fixture
def ws_with_completed(ws: Path) -> Path:
  """Workspace with an active tree containing a completed experiment."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  seed_tree_with_experiments(
    forest,
    'main',
    [
      {
        'id': 'exp-done',
        'hypothesis': 'already done',
        'status': 'completed',
        'metrics': {'accuracy': 0.5},
      },
    ],
  )
  return ws


# -- 4.1 core transitions --


class TestCoreLifecycleTransitions:
  """Verify core Experiment lifecycle changes from Plan 01 section 2.1."""

  def test_experiment_complete_from_pending_updates_metrics_and_status(self) -> None:
    """Pending -> completed with metrics applied (FR-001)."""
    exp = Experiment(experiment_id='test-1', hypothesis='h')
    assert exp.status == Status.pending
    exp.complete({'x': 1.0})
    assert exp.status == Status.completed
    assert exp.metrics['x'] == 1.0
    assert exp.completed_at is not None

  def test_experiment_complete_from_running_still_works(self) -> None:
    """Running -> completed is the existing Trainer workflow."""
    exp = Experiment(experiment_id='test-2', hypothesis='h')
    exp.start()
    assert exp.status == Status.running
    exp.complete({'y': 2.0})
    assert exp.status == Status.completed
    assert exp.metrics['y'] == 2.0

  def test_experiment_complete_rejects_terminal_via_experiment_error(self) -> None:
    """Terminal experiments cannot be completed -- ExperimentError raised."""
    for setup_status in ['completed', 'failed', 'cancelled']:
      exp = Experiment(experiment_id=f'test-{setup_status}', hypothesis='h')
      exp.start()
      if setup_status == 'completed':
        exp.complete()
      elif setup_status == 'failed':
        exp.fail('err')
      elif setup_status == 'cancelled':
        exp.cancel()

      with pytest.raises(ExperimentError, match='cannot complete'):
        exp.complete({'z': 3.0})

  def test_complete_from_pending_without_metrics(self) -> None:
    """Pending -> completed with no metrics leaves empty dict."""
    exp = Experiment(experiment_id='test-no-metrics', hypothesis='h')
    exp.complete()
    assert exp.status == Status.completed
    assert exp.metrics == {}

  def test_require_status_error_shows_both_allowed(self) -> None:
    """Error message includes both allowed statuses when multiple."""
    exp = Experiment(experiment_id='test-msg', hypothesis='h')
    exp.start()
    exp.complete()
    with pytest.raises(ExperimentError, match='pending or running'):
      exp.complete()


# -- 4.2 CLI integration --


class TestCLIExperimentComplete:
  """CLI experiment complete command tests."""

  def test_cli_experiment_complete_persists_forest_metrics(self, ws_with_running: Path) -> None:
    """Complete persists metrics to the forest (section 2.2)."""
    result = run_cli(
      ws_with_running,
      ['experiment', 'complete', 'exp-running', '--metrics', '{"acc": 0.9}'],
    )
    assert result['result']['ok'] is True
    assert result['result']['status'] == 'completed'
    assert result['result']['metrics']['acc'] == 0.9

    config = AutoPilotConfig(workspace=ws_with_running)
    store = FileStore(config)
    forest = FileForest(store)
    tree = forest.active
    assert tree is not None
    node = tree.get('exp-running')
    assert node is not None
    assert node.experiment.status == Status.completed
    assert node.experiment.metrics['acc'] == 0.9

  def test_cli_experiment_complete_json_metrics_round_trip(self, ws_with_running: Path) -> None:
    """--metrics JSON parses to float and persists (test 4.2.4)."""
    result = run_cli(
      ws_with_running,
      ['experiment', 'complete', 'exp-running', '--metrics', '{"score": 0.5}'],
    )
    assert result['result']['metrics']['score'] == 0.5

  def test_cli_experiment_complete_from_pending(self, ws_with_pending: Path) -> None:
    """CLI can complete a pending experiment directly (FR-001)."""
    result = run_cli(
      ws_with_pending,
      ['experiment', 'complete', 'exp-pending', '--metrics', '{"val": 1.0}'],
    )
    assert result['result']['ok'] is True
    assert result['result']['status'] == 'completed'

  def test_cli_experiment_complete_without_metrics(self, ws_with_running: Path) -> None:
    """Complete without --metrics preserves existing empty metrics."""
    result = run_cli(
      ws_with_running,
      ['experiment', 'complete', 'exp-running'],
    )
    assert result['result']['ok'] is True
    assert result['result']['status'] == 'completed'

  def test_cli_experiment_complete_invalid_json(self, ws_with_running: Path) -> None:
    """Invalid --metrics JSON causes a clean exit failure."""
    with pytest.raises(SystemExit):
      run_cli(
        ws_with_running,
        ['experiment', 'complete', 'exp-running', '--metrics', 'not-json'],
      )

  def test_cli_experiment_complete_terminal_fails(self, ws_with_completed: Path) -> None:
    """Cannot complete an already-completed experiment."""
    with pytest.raises(SystemExit):
      run_cli(
        ws_with_completed,
        ['experiment', 'complete', 'exp-done'],
      )


class TestCLIExperimentFail:
  """CLI experiment fail command tests."""

  def test_cli_experiment_fail_with_error(self, ws_with_running: Path) -> None:
    """Fail a running experiment with an error string."""
    result = run_cli(
      ws_with_running,
      ['experiment', 'fail', 'exp-running', '--error', 'out of budget'],
    )
    assert result['result']['ok'] is True
    assert result['result']['status'] == 'failed'
    assert result['result']['error'] == 'out of budget'

  def test_cli_experiment_fail_without_error(self, ws_with_running: Path) -> None:
    """Fail without --error uses --context as error fallback (P3#30)."""
    result = run_cli(
      ws_with_running,
      ['experiment', 'fail', 'exp-running'],
    )
    assert result['result']['status'] == 'failed'
    assert result['result']['error'] == 'test'

  def test_cli_experiment_fail_from_pending(self, ws_with_pending: Path) -> None:
    """BUG-DFV1-002: CLI fail on pending experiment succeeds."""
    result = run_cli(
      ws_with_pending,
      ['experiment', 'fail', 'exp-pending', '--error', 'bad data'],
    )
    assert result['result']['ok'] is True
    assert result['result']['status'] == 'failed'
    assert result['result']['error'] == 'bad data'

  def test_cli_experiment_fail_persists(self, ws_with_running: Path) -> None:
    """Fail persists to forest on disk."""
    run_cli(
      ws_with_running,
      ['experiment', 'fail', 'exp-running', '--error', 'bad data'],
    )
    config = AutoPilotConfig(workspace=ws_with_running)
    store = FileStore(config)
    forest = FileForest(store)
    assert forest.active is not None
    node = forest.active.get('exp-running')
    assert node is not None
    assert node.experiment.status == Status.failed


class TestCLIExperimentCancel:
  """CLI experiment cancel command tests."""

  def test_cli_experiment_cancel_persists_status(self, ws_with_pending: Path) -> None:
    """Cancel yields 'cancelled' on reload (test 4.2.5)."""
    result = run_cli(
      ws_with_pending,
      ['experiment', 'cancel', 'exp-pending'],
    )
    assert result['result']['ok'] is True
    assert result['result']['status'] == 'cancelled'

    config = AutoPilotConfig(workspace=ws_with_pending)
    store = FileStore(config)
    forest = FileForest(store)
    assert forest.active is not None
    node = forest.active.get('exp-pending')
    assert node is not None
    assert node.experiment.status == Status.cancelled

  def test_cli_experiment_cancel_from_running(self, ws_with_running: Path) -> None:
    """Running experiments can be cancelled too."""
    result = run_cli(
      ws_with_running,
      ['experiment', 'cancel', 'exp-running'],
    )
    assert result['result']['status'] == 'cancelled'

  def test_cli_experiment_cancel_terminal_fails(self, ws_with_completed: Path) -> None:
    """Cannot cancel an already-completed experiment."""
    with pytest.raises(SystemExit):
      run_cli(
        ws_with_completed,
        ['experiment', 'cancel', 'exp-done'],
      )


class TestCLIExperimentRemoveJournalContext:
  """BUG-DFV1-006: experiment remove calls journal_user_context."""

  def test_experiment_remove_journals_user_context(self, ws_with_pending: Path) -> None:
    """BUG-DFV1-006: experiment remove calls journal_user_context before removal."""
    with patch(
      'autopilot.cli.commands.experiment.lifecycle.journal_user_context',
      wraps=journal_user_context,
    ) as mock_journal:
      run_cli(
        ws_with_pending,
        ['experiment', 'remove', 'exp-pending'],
      )
      assert mock_journal.call_count == 1
      call_args = mock_journal.call_args
      assert call_args[0][1].id == 'exp-pending'


def test_cli_mutations_require_context() -> None:
  """All lifecycle commands (complete, fail, cancel) require --context (test 2.2).

  Context enforcement is handled by ``CLI.dispatch()`` via ``requires_context``.
  Verifies the new commands are NOT in the exempt set.
  """
  from autopilot.cli.command import CLI

  cli = CLI()
  assert cli.requires_context('experiment complete') is True
  assert cli.requires_context('experiment fail') is True
  assert cli.requires_context('experiment cancel') is True
