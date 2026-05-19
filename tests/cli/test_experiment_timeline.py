"""Tests for the experiment timeline CLI command."""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.command import CLI
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from autopilot.tracking.executions import ExecutionRecord, log_execution
from autopilot.tracking.io import utc_now_iso
from pathlib import Path
from tests.cli.conftest import run_cli_no_context


def _seed_workspace(tmp_path: Path) -> tuple[Path, str]:
  """Create workspace with a tree, experiment, and context log entry.

  Returns:
    Tuple of (workspace_path, experiment_id).
  """
  ws = tmp_path / 'ws'
  ws.mkdir()

  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  tree = forest.create_tree('main')
  exp = Experiment(experiment_id='exp-timeline-001', hypothesis='timeline test')
  exp.start()
  exp.add_context('context event alpha', source='trainer', epoch=0)
  exp.complete(metrics={'accuracy': 0.9})
  tree.add(Node(experiment=exp))
  forest.switch('main')
  forest.save()

  return ws, 'exp-timeline-001'


class TestTimelineCLIJsonShape:
  """experiment timeline --json returns correct envelope shape."""

  def test_timeline_cli_json_shape(self, tmp_path: Path) -> None:
    """JSON envelope has ok=True and result.entries as a list."""
    ws, eid = _seed_workspace(tmp_path)

    result = run_cli_no_context(ws, ['experiment', 'timeline', eid])

    assert result['ok'] is True
    entries = result['result']['entries']
    assert isinstance(entries, list)
    assert len(entries) >= 1

    first = entries[0]
    assert 'timestamp' in first
    assert 'stream' in first
    assert 'reason' in first


class TestTimelineContextExempt:
  """experiment timeline is read-only and context-exempt."""

  def test_timeline_context_exempt(self) -> None:
    """experiment timeline does not require --context."""
    cli = CLI()
    assert cli.requires_context('experiment timeline') is False


class TestTimelineFiltersExecutionByExperiment:
  """Timeline only includes execution records for the target experiment."""

  def test_timeline_filters_execution_by_experiment(self, tmp_path: Path) -> None:
    """Only execution records matching the experiment appear in timeline."""
    ws, eid = _seed_workspace(tmp_path)

    config = AutoPilotConfig(workspace=ws)
    exec_path = config.executions_path

    matching = ExecutionRecord(
      timestamp=utc_now_iso(),
      command='optimize train',
      args=['--max-epochs', '3'],
      duration_ms=500.0,
      exit_code=0,
      experiment=eid,
    )
    log_execution(exec_path, matching)

    other = ExecutionRecord(
      timestamp=utc_now_iso(),
      command='optimize train',
      args=['--max-epochs', '5'],
      duration_ms=800.0,
      exit_code=0,
      experiment='exp-other-999',
    )
    log_execution(exec_path, other)

    result = run_cli_no_context(ws, ['experiment', 'timeline', eid])

    assert result['ok'] is True
    entries = result['result']['entries']

    exec_entries = [e for e in entries if e['stream'] == 'execution']
    assert len(exec_entries) == 1
    assert exec_entries[0]['metadata']['command'] == 'optimize train'
    assert exec_entries[0]['metadata']['args'] == ['--max-epochs', '3']

    all_exec_experiments = [
      e['metadata'].get('command') for e in entries if e['stream'] == 'execution'
    ]
    assert 'exp-other-999' not in str(all_exec_experiments)
