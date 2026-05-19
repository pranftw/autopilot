"""CLI tests for experiment deploy-log command."""

from autopilot.ai.deployment import DeploymentLog, emit_deployment_event
from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.context import build_context
from autopilot.cli.main import build_parser
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_no_context, run_cli_text
import contextlib
import io
import json
import pytest


def _setup_workspace(tmp_path: Path) -> tuple[Path, FileForest]:
  """Create a workspace with a completed experiment in one tree."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')

  exp_a = Experiment(experiment_id='exp-a', hypothesis='first')
  exp_a.start()
  exp_a.complete(metrics={'accuracy': 0.9})
  tree.add(Node(experiment=exp_a))

  forest.save()
  return ws, forest


def _setup_with_events(tmp_path: Path) -> Path:
  """Create workspace with pre-seeded deployment events."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  (ws / '.autopilot').mkdir(parents=True, exist_ok=True)

  log = DeploymentLog(ws / '.autopilot' / 'deployment_events.jsonl')
  emit_deployment_event(
    log,
    label='prod',
    experiment_id='exp-a',
    action='deploy',
  )
  emit_deployment_event(
    log,
    label='staging',
    experiment_id='exp-b',
    action='deploy',
  )
  emit_deployment_event(
    log,
    label='prod',
    experiment_id='exp-c',
    action='replace',
    previous_experiment_id='exp-a',
  )

  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')
  for eid in ['exp-a', 'exp-b', 'exp-c']:
    exp = Experiment(experiment_id=eid, hypothesis=eid)
    exp.start()
    exp.complete(metrics={'accuracy': 0.9})
    tree.add(Node(experiment=exp))
  forest.save()
  return ws


class TestDeployLogCLI:
  """Tests for the experiment deploy-log CLI command."""

  def test_deploy_log_shows_history(self, tmp_path: Path) -> None:
    """Text output contains expected experiment id / label."""
    ws = _setup_with_events(tmp_path)
    text = run_cli_text(ws, ['experiment', 'deploy-log'])
    assert 'exp-a' in text
    assert 'prod' in text
    assert 'staging' in text

  def test_deploy_log_filter_by_label(self, tmp_path: Path) -> None:
    """--label narrows rows; omitting returns all events."""
    ws = _setup_with_events(tmp_path)
    envelope = run_cli_no_context(ws, ['experiment', 'deploy-log', '--label', 'prod'])
    assert envelope['ok'] is True
    payload = envelope['result']
    events = payload['events']
    assert len(events) == 2
    assert all(e['label'] == 'prod' for e in events)

    envelope_all = run_cli_no_context(ws, ['experiment', 'deploy-log'])
    assert len(envelope_all['result']['events']) == 3

  def test_deploy_log_json(self, tmp_path: Path) -> None:
    """JSON envelope ok and events array shape."""
    ws = _setup_with_events(tmp_path)
    envelope = run_cli_no_context(ws, ['experiment', 'deploy-log'])
    assert envelope['ok'] is True
    payload = envelope['result']
    assert 'events' in payload
    assert isinstance(payload['events'], list)
    assert len(payload['events']) == 3
    first = payload['events'][0]
    assert 'label' in first
    assert 'experiment_id' in first
    assert 'action' in first
    assert 'timestamp' in first

  def test_deploy_log_empty(self, tmp_path: Path) -> None:
    """No file or empty file: success, empty listing."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    (ws / '.autopilot').mkdir(parents=True, exist_ok=True)
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    forest.create_tree('main')
    forest.switch('main')
    forest.save()

    envelope = run_cli_no_context(ws, ['experiment', 'deploy-log'])
    assert envelope['ok'] is True
    assert envelope['result']['events'] == []

  def test_deploy_log_cli_exit_code_success(self, tmp_path: Path) -> None:
    """Exit code 0 with valid workspace."""
    ws = _setup_with_events(tmp_path)
    envelope = run_cli_no_context(ws, ['experiment', 'deploy-log'])
    assert envelope['ok'] is True

  def test_deploy_log_cli_exit_code_no_store(self, tmp_path: Path) -> None:
    """No .autopilot directory -- ctx.fail with non-zero exit and JSON ok: false."""
    ws = tmp_path / 'missing_ws'
    ws.mkdir()
    parser = build_parser()
    full_argv = ['experiment', 'deploy-log', '--workspace', str(ws), '--json']
    parsed = parser.parse_args(full_argv)
    ctx = build_context(parsed)

    buf = io.StringIO()
    with pytest.raises(SystemExit) as exc_info, contextlib.redirect_stdout(buf):
      parsed.handler(ctx, parsed)
    assert exc_info.value.code == 1
    output = buf.getvalue().strip()
    envelope = json.loads(output)
    assert envelope['ok'] is False
    assert 'error' in envelope

  def test_deploy_log_latest_for_nonexistent_label(self, tmp_path: Path) -> None:
    """deploy-log with --label for nonexistent label returns empty events."""
    ws = _setup_with_events(tmp_path)
    envelope = run_cli_no_context(ws, ['experiment', 'deploy-log', '--label', 'nonexistent'])
    assert envelope['ok'] is True
    assert envelope['result']['events'] == []

  def test_deploy_cli_reads_forest_events(self, tmp_path: Path) -> None:
    """Integration: Forest.deploy() writes event, then deploy-log CLI reads it."""
    ws, _forest = _setup_workspace(tmp_path)
    result = run_cli(ws, ['experiment', 'deploy', 'exp-a', '--as', 'production'])
    assert result['ok'] is True

    log_envelope = run_cli_no_context(ws, ['experiment', 'deploy-log'])
    assert log_envelope['ok'] is True
    events = log_envelope['result']['events']
    assert len(events) >= 1
    deploy_events = [e for e in events if e['label'] == 'production']
    assert len(deploy_events) == 1
    assert deploy_events[0]['experiment_id'] == 'exp-a'
    assert deploy_events[0]['action'] == 'deploy'
