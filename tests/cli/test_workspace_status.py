"""Tests for ``autopilot workspace status`` command (Plan 20 + Sub-plan 03 + Sub-plan 05).

Covers JSON payload structure, experiment/tree counts, store doctor
integration, executions summary, context exemption,
execution context field in the JSON payload (sub-plan 03, section 2.4),
and deployments inventory / trees.detail (sub-plan 05).
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.command import CLI
from autopilot.cli.context import build_context
from autopilot.cli.main import build_parser
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from autopilot.tracking.executions import create_execution_record, log_execution
from pathlib import Path
from tests.cli.conftest import run_cli_no_context
from typing import Any
import contextlib
import io
import pytest


def _status_payload(workspace: Path) -> dict[str, Any]:
  """Run workspace status and return the inner result payload."""
  envelope = run_cli_no_context(workspace, ['workspace', 'status'])
  return envelope.get('result', envelope)


def _status_text(workspace: Path) -> str:
  """Run workspace status in text mode and return captured stdout."""
  parser = build_parser()
  full_argv = ['workspace', 'status', '--workspace', str(workspace)]
  parsed = parser.parse_args(full_argv)
  ctx = build_context(parsed)
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
    parsed.handler(ctx, parsed)
  return buf.getvalue()


@pytest.fixture
def status_workspace(tmp_path: Path) -> Path:
  """Healthy workspace with full layout and store for workspace status tests."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.init_workspace()
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  FileForest(store).save()
  return ws


# 4.1 layout and exemptions


class TestWorkspaceStatusJsonEmptyWorkspace:
  """test_workspace_status_json_empty_workspace: all top-level sections present."""

  def test_all_sections_present(self, status_workspace: Path) -> None:
    """Fresh workspace has all five top-level keys in the status payload."""
    result = _status_payload(status_workspace)
    expected_keys = {'trees', 'experiments', 'store', 'executions', 'health'}
    assert expected_keys <= set(result.keys())

  def test_empty_workspace_trees_zero(self, status_workspace: Path) -> None:
    """Empty workspace reports zero trees."""
    result = _status_payload(status_workspace)
    assert result['trees']['count'] == 0
    assert result['trees']['active'] is None

  def test_empty_workspace_experiments_zero(self, status_workspace: Path) -> None:
    """Empty workspace reports zero experiments for all statuses."""
    result = _status_payload(status_workspace)
    for count in result['experiments'].values():
      assert count == 0

  def test_empty_workspace_executions_zero(self, status_workspace: Path) -> None:
    """Empty workspace reports zero total executions."""
    result = _status_payload(status_workspace)
    assert result['executions']['total'] == 0
    assert result['executions']['recent'] == []


class TestWorkspaceStatusPopulatedCounts:
  """test_workspace_status_populated_counts: non-zero experiment and tree stats."""

  def test_populated_counts(self, tmp_path: Path) -> None:
    """Workspace with trees and experiments reports accurate counts."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.init_workspace()
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)

    tree = forest.create_tree('main')
    exp = Experiment(experiment_id='exp-1', hypothesis='test')
    exp.start()
    exp.complete(metrics={'score': 0.9})
    tree.add(Node(experiment=exp))

    exp2 = Experiment(experiment_id='exp-2', hypothesis='test 2')
    tree.add(Node(experiment=exp2))

    forest.switch('main')
    forest.save()

    result = _status_payload(ws)
    assert result['trees']['count'] >= 1
    assert result['trees']['active'] == 'main'
    assert result['experiments']['completed'] >= 1
    assert result['experiments']['pending'] >= 1

  def test_multiple_trees_counted(self, tmp_path: Path) -> None:
    """Multiple trees are reflected in the count."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.init_workspace()
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)

    forest.create_tree('alpha')
    forest.create_tree('beta')
    forest.switch('alpha')
    forest.save()

    result = _status_payload(ws)
    assert result['trees']['count'] == 2


# 4.2 health and store doctor


class TestWorkspaceStatusStoreDoctorPresent:
  """test_workspace_status_includes_store_doctor_when_store_present."""

  def test_store_doctor_keys(self, status_workspace: Path) -> None:
    """health.store_doctor subtree contains FileStore.doctor() contract keys."""
    result = _status_payload(status_workspace)
    store_doctor = result['health']['store_doctor']
    expected_keys = {
      'healthy',
      'manifest_errors',
      'missing_blobs',
      'orphan_blobs',
      'orphan_count',
      'refs_issues',
    }
    assert expected_keys <= set(store_doctor.keys())

  def test_store_doctor_healthy_on_empty_store(self, status_workspace: Path) -> None:
    """Empty store is healthy (no snapshots, no corruption)."""
    result = _status_payload(status_workspace)
    assert result['health']['store_doctor']['healthy'] is True

  def test_no_store_doctor_when_store_absent(self, tmp_path: Path) -> None:
    """health.store_doctor is absent when store directory does not exist."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.init_workspace()
    result = _status_payload(ws)
    assert 'store_doctor' not in result['health']

  def test_workspace_doctor_checks_present(self, status_workspace: Path) -> None:
    """health.workspace_doctor contains healthy flag and checks dict."""
    result = _status_payload(status_workspace)
    ws_doctor = result['health']['workspace_doctor']
    assert 'healthy' in ws_doctor
    assert 'checks' in ws_doctor

  def test_store_doctor_error_produces_unhealthy(self, tmp_path: Path) -> None:
    """StoreError during doctor produces store_doctor with healthy=False and error.

    Raises:
      SystemExit: Expected exit 1 because unhealthy store makes overall status unhealthy.
    """
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.init_workspace()
    store_path = config.store_path
    store_path.mkdir(parents=True, exist_ok=True)
    refs_file = store_path / 'refs.json'
    refs_file.write_text('not valid json{{{')

    with pytest.raises(SystemExit) as exc_info:
      _status_payload(ws)
    assert exc_info.value.code == 1


# 4.2 executions summary


class TestWorkspaceStatusExecutionsSummary:
  """test_workspace_status_executions_summary: JSONL round-trip visible in status."""

  def test_executions_visible_after_logging(self, status_workspace: Path) -> None:
    """Logged execution records appear in the status payload."""
    config = AutoPilotConfig(workspace=status_workspace)
    record = create_execution_record(
      command='query',
      args=['--json'],
      duration_ms=42.0,
      exit_code=0,
    )
    log_execution(config.executions_path, record)

    result = _status_payload(status_workspace)
    assert result['executions']['total'] >= 1
    recent = result['executions']['recent']
    assert len(recent) >= 1
    assert recent[-1]['command'] == 'query'

  def test_recent_capped_at_limit(self, status_workspace: Path) -> None:
    """Recent executions are capped at WORKSPACE_STATUS_RECENT_EXECUTIONS."""
    config = AutoPilotConfig(workspace=status_workspace)
    for i in range(10):
      record = create_execution_record(
        command=f'cmd-{i}',
        args=[],
        duration_ms=1.0,
        exit_code=0,
      )
      log_execution(config.executions_path, record)

    result = _status_payload(status_workspace)
    assert result['executions']['total'] == 10
    assert len(result['executions']['recent']) == 5

  def test_recent_entries_have_required_fields(self, status_workspace: Path) -> None:
    """Each recent entry has timestamp, command, and exit_code."""
    config = AutoPilotConfig(workspace=status_workspace)
    record = create_execution_record(
      command='optimize',
      args=[],
      duration_ms=100.0,
      exit_code=1,
    )
    log_execution(config.executions_path, record)

    result = _status_payload(status_workspace)
    entry = result['executions']['recent'][-1]
    assert 'timestamp' in entry
    assert entry['command'] == 'optimize'
    assert entry['exit_code'] == 1


# 4.1 context exemption


class TestWorkspaceStatusContextExempt:
  """test_workspace_status_context_exempt: no --context required."""

  def test_context_exempt_via_run_cli_no_context(self, status_workspace: Path) -> None:
    """workspace status succeeds without --context (read-only, context-exempt)."""
    result = _status_payload(status_workspace)
    assert 'trees' in result

  def test_requires_context_returns_false(self) -> None:
    """CLI.requires_context('workspace status') is False."""
    cli = CLI()
    assert cli.requires_context('workspace status') is False


# store section


class TestWorkspaceStatusStoreSection:
  """Tests for the store section in the status payload."""

  def test_store_exists_when_present(self, status_workspace: Path) -> None:
    """store.exists is True when the store directory exists."""
    result = _status_payload(status_workspace)
    assert result['store']['exists'] is True
    assert 'path' in result['store']

  def test_store_not_exists_when_absent(self, tmp_path: Path) -> None:
    """store.exists is False when the store directory is absent."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.init_workspace()
    result = _status_payload(ws)
    assert result['store']['exists'] is False


# experiment status counting


class TestExperimentStatusCounting:
  """Verify all Status values are counted correctly."""

  def test_all_statuses_counted(self, tmp_path: Path) -> None:
    """Each status value is correctly counted across experiments."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.init_workspace()
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)

    tree = forest.create_tree('main')

    exp_pending = Experiment(experiment_id='exp-pending', hypothesis='p')
    tree.add(Node(experiment=exp_pending))

    exp_running = Experiment(experiment_id='exp-running', hypothesis='r')
    exp_running.start()
    tree.add(Node(experiment=exp_running))

    exp_completed = Experiment(experiment_id='exp-completed', hypothesis='c')
    exp_completed.start()
    exp_completed.complete(metrics={'score': 1.0})
    tree.add(Node(experiment=exp_completed))

    exp_failed = Experiment(experiment_id='exp-failed', hypothesis='f')
    exp_failed.start()
    exp_failed.fail(error='boom')
    tree.add(Node(experiment=exp_failed))

    exp_cancelled = Experiment(experiment_id='exp-cancelled', hypothesis='x')
    exp_cancelled.start()
    exp_cancelled.cancel()
    tree.add(Node(experiment=exp_cancelled))

    forest.switch('main')
    forest.save()

    result = _status_payload(ws)
    assert result['experiments']['pending'] == 1
    assert result['experiments']['running'] == 1
    assert result['experiments']['completed'] == 1
    assert result['experiments']['failed'] == 1
    assert result['experiments']['cancelled'] == 1


# 4.4 execution context in workspace status (sub-plan 03)


class TestWorkspaceStatusExecutionsIncludeContext:
  """test_workspace_status_executions_include_context: context field in recent entries."""

  def test_workspace_status_executions_include_context(self, status_workspace: Path) -> None:
    """Recent execution with context shows same string in JSON."""
    config = AutoPilotConfig(workspace=status_workspace)
    record = create_execution_record(
      command='optimize train',
      args=['--max-epochs', '5'],
      duration_ms=500.0,
      exit_code=0,
      context='run training loop',
    )
    log_execution(config.executions_path, record)

    result = _status_payload(status_workspace)
    recent = result['executions']['recent']
    assert len(recent) >= 1
    last_entry = recent[-1]
    assert 'context' in last_entry
    assert last_entry['context'] == 'run training loop'

  def test_workspace_status_executions_context_null(self, status_workspace: Path) -> None:
    """Execution lacking context shows null in the context field."""
    config = AutoPilotConfig(workspace=status_workspace)
    record = create_execution_record(
      command='query',
      args=['--json'],
      duration_ms=10.0,
      exit_code=0,
    )
    log_execution(config.executions_path, record)

    result = _status_payload(status_workspace)
    recent = result['executions']['recent']
    assert len(recent) >= 1
    last_entry = recent[-1]
    assert 'context' in last_entry
    assert last_entry['context'] is None


# sub-plan 05: deployments and trees.detail


def _make_workspace_with_forest(
  tmp_path: Path,
  name: str = 'ws',
) -> tuple[Path, FileForest]:
  """Create a workspace with initialized layout and an empty forest.

  Returns:
    Tuple of workspace path and FileForest instance.
  """
  ws = tmp_path / name
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.init_workspace()
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  return ws, forest


class TestWorkspaceStatusIncludesDeploymentsSection:
  """test_workspace_status_includes_deployments_section."""

  def test_deployments_key_is_list(self, status_workspace: Path) -> None:
    """workspace status --json result includes deployments as a list."""
    result = _status_payload(status_workspace)
    assert 'deployments' in result
    assert isinstance(result['deployments'], list)


class TestWorkspaceStatusDeploymentsEmptyWhenNone:
  """test_workspace_status_deployments_empty_when_none."""

  def test_empty_deployments(self, tmp_path: Path) -> None:
    """No deployed_as set anywhere; deployments is an empty list."""
    ws, forest = _make_workspace_with_forest(tmp_path)
    tree = forest.create_tree('main')
    exp = Experiment(experiment_id='exp-1', hypothesis='test')
    tree.add(Node(experiment=exp))
    forest.switch('main')
    forest.save()

    result = _status_payload(ws)
    assert result['deployments'] == []


class TestWorkspaceStatusDeploymentsCrossTree:
  """test_workspace_status_deployments_cross_tree."""

  def test_cross_tree_deployments(self, tmp_path: Path) -> None:
    """Deployments on different trees appear with distinct tree values."""
    ws, forest = _make_workspace_with_forest(tmp_path)
    tree_a = forest.create_tree('alpha')
    tree_b = forest.create_tree('beta')

    exp_a = Experiment(experiment_id='exp-a', hypothesis='a')
    exp_a.start()
    exp_a.complete(metrics={'score': 0.9})
    node_a = Node(experiment=exp_a, deployed_as='prod')
    tree_a.add(node_a)

    exp_b = Experiment(experiment_id='exp-b', hypothesis='b')
    exp_b.start()
    exp_b.complete(metrics={'score': 0.8})
    node_b = Node(experiment=exp_b, deployed_as='staging')
    tree_b.add(node_b)

    forest.switch('alpha')
    forest.save()

    result = _status_payload(ws)
    deployments = result['deployments']
    assert len(deployments) == 2
    labels = {d['label'] for d in deployments}
    assert labels == {'prod', 'staging'}
    trees_in_result = {d['tree'] for d in deployments}
    assert trees_in_result == {'alpha', 'beta'}

    prod = next(d for d in deployments if d['label'] == 'prod')
    assert prod['experiment_id'] == 'exp-a'
    assert prod['tree'] == 'alpha'

    staging = next(d for d in deployments if d['label'] == 'staging')
    assert staging['experiment_id'] == 'exp-b'
    assert staging['tree'] == 'beta'


class TestWorkspaceStatusTreesDetailExperimentCounts:
  """test_workspace_status_trees_detail_experiment_counts."""

  def test_experiment_counts(self, tmp_path: Path) -> None:
    """Each trees.detail row has correct experiment_count."""
    ws, forest = _make_workspace_with_forest(tmp_path)
    tree_a = forest.create_tree('alpha')
    tree_b = forest.create_tree('beta')

    for i in range(3):
      exp = Experiment(experiment_id=f'a-{i}', hypothesis='a')
      tree_a.add(Node(experiment=exp))

    exp_b = Experiment(experiment_id='b-0', hypothesis='b')
    tree_b.add(Node(experiment=exp_b))

    forest.switch('alpha')
    forest.save()

    result = _status_payload(ws)
    detail = result['trees']['detail']
    assert len(detail) == 2

    alpha_detail = next(d for d in detail if d['name'] == 'alpha')
    assert alpha_detail['experiment_count'] == 3

    beta_detail = next(d for d in detail if d['name'] == 'beta')
    assert beta_detail['experiment_count'] == 1


class TestWorkspaceStatusTreesDetailActiveFlag:
  """test_workspace_status_trees_detail_active_flag."""

  def test_active_flag(self, tmp_path: Path) -> None:
    """Exactly one tree has active=True matching forest.active.name."""
    ws, forest = _make_workspace_with_forest(tmp_path)
    forest.create_tree('alpha')
    forest.create_tree('beta')
    forest.switch('beta')
    forest.save()

    result = _status_payload(ws)
    detail = result['trees']['detail']
    active_rows = [d for d in detail if d['active'] is True]
    assert len(active_rows) == 1
    assert active_rows[0]['name'] == 'beta'

    inactive_rows = [d for d in detail if d['active'] is False]
    assert len(inactive_rows) == 1
    assert inactive_rows[0]['name'] == 'alpha'


class TestWorkspaceStatusTreesDetailDescription:
  """test_workspace_status_trees_detail_description."""

  def test_description_present(self, tmp_path: Path) -> None:
    """Tree with description appears correctly; None maps to null."""
    ws, forest = _make_workspace_with_forest(tmp_path)
    forest.create_tree('with-desc', description='my description')
    forest.create_tree('no-desc')
    forest.switch('with-desc')
    forest.save()

    result = _status_payload(ws)
    detail = result['trees']['detail']

    with_desc = next(d for d in detail if d['name'] == 'with-desc')
    assert with_desc['description'] == 'my description'

    no_desc = next(d for d in detail if d['name'] == 'no-desc')
    assert no_desc['description'] is None


class TestWorkspaceStatusNoActiveTree:
  """test_workspace_status_no_active_tree."""

  def test_no_active_tree(self, tmp_path: Path) -> None:
    """forest.active is None; trees.active is null, all detail rows active=False."""
    ws, forest = _make_workspace_with_forest(tmp_path)
    forest.create_tree('orphan')
    forest.save()

    result = _status_payload(ws)
    assert result['trees']['active'] is None
    for row in result['trees']['detail']:
      assert row['active'] is False


class TestWorkspaceStatusDeploymentsSortOrder:
  """test_workspace_status_deployments_sort_order."""

  def test_sort_order(self, tmp_path: Path) -> None:
    """Deployments are sorted by (tree.name, label)."""
    ws, forest = _make_workspace_with_forest(tmp_path)
    tree_b = forest.create_tree('beta')
    tree_a = forest.create_tree('alpha')

    exp1 = Experiment(experiment_id='e1', hypothesis='h')
    tree_b.add(Node(experiment=exp1, deployed_as='z-deploy'))

    exp2 = Experiment(experiment_id='e2', hypothesis='h')
    tree_b.add(Node(experiment=exp2, deployed_as='a-deploy'))

    exp3 = Experiment(experiment_id='e3', hypothesis='h')
    tree_a.add(Node(experiment=exp3, deployed_as='m-deploy'))

    forest.switch('alpha')
    forest.save()

    result = _status_payload(ws)
    deployments = result['deployments']
    assert len(deployments) == 3

    for i in range(len(deployments) - 1):
      curr = (deployments[i]['tree'], deployments[i]['label'])
      nxt = (deployments[i + 1]['tree'], deployments[i + 1]['label'])
      assert curr <= nxt, f'sort order violated: {curr} > {nxt}'

    assert deployments[0]['tree'] == 'alpha'
    assert deployments[0]['label'] == 'm-deploy'
    assert deployments[1]['tree'] == 'beta'
    assert deployments[1]['label'] == 'a-deploy'
    assert deployments[2]['tree'] == 'beta'
    assert deployments[2]['label'] == 'z-deploy'


class TestWorkspaceStatusNoTrees:
  """test_workspace_status_no_trees: fresh workspace with zero trees."""

  def test_no_trees_json(self, status_workspace: Path) -> None:
    """Fresh workspace: deployments empty, trees.detail empty, no crash."""
    result = _status_payload(status_workspace)
    assert result['deployments'] == []
    assert result['trees']['detail'] == []
    assert result['trees']['count'] == 0

  def test_no_trees_text_mode(self, status_workspace: Path) -> None:
    """Text mode does not crash on fresh workspace with zero trees."""
    text = _status_text(status_workspace)
    assert 'Trees:' in text
    assert 'Traceback' not in text


class TestWorkspaceStatusEmptyForest:
  """test_workspace_status_empty_forest: forest with zero trees."""

  def test_empty_forest_json(self, tmp_path: Path) -> None:
    """Forest with zero trees: detail empty, experiments all zero."""
    ws, forest = _make_workspace_with_forest(tmp_path)
    forest.save()

    result = _status_payload(ws)
    assert result['trees']['detail'] == []
    assert result['trees']['count'] == 0
    assert result['deployments'] == []
    for count in result['experiments'].values():
      assert count == 0

  def test_empty_forest_text_mode(self, tmp_path: Path) -> None:
    """Text mode does not crash on empty forest."""
    ws, forest = _make_workspace_with_forest(tmp_path)
    forest.save()

    text = _status_text(ws)
    assert 'Trees:' in text
    assert 'Traceback' not in text
