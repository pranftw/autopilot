"""Tests for store reflog expire and store recover CLI commands.

Covers:
  10. test_recover_cli_json -- structured JSON with ok and identifiers
  11. test_reflog_expire_cli -- mutates and returns expected text/JSON
  12. test_reflog_expire_dry_run -- count matches would-remove; file unchanged
  13. test_reflog_expire_cli_json -- JSON includes expired_count
  14. test_recover_exit_code -- success 0; invalid index non-zero with JSON error
  15. test_reflog_expire_exit_code -- success 0; propagate non-zero on failure
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from autopilot.tracking.io import append_jsonl
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tests.cli.conftest import run_cli
import pytest


def _setup_workspace(tmp_path: Path) -> Path:
  """Create a workspace with store, forest, experiment, and reflog entries."""
  ws = tmp_path / 'ws'
  ws.mkdir()

  prompts_dir = ws / 'prompts'
  prompts_dir.mkdir()
  (prompts_dir / 'main.txt').write_text('hello')

  config = AutoPilotConfig(workspace=ws)
  param = PathParameter(source=str(prompts_dir), pattern='*.txt')
  store = FileStore(config)
  store.register_parameters({'source': param})
  store.snapshot('exp-001', 0)

  forest = FileForest(store)
  tree = forest.create_tree('main')
  exp = Experiment(experiment_id='exp-001', hypothesis='test')
  exp.start()
  exp.complete(metrics={'accuracy': 0.9})
  tree.add(Node(experiment=exp))
  forest.save()

  return ws


def _inject_old_entry(ws: Path, *, days_old: int = 60) -> None:
  """Inject an old reflog entry into the workspace store."""
  config = AutoPilotConfig(workspace=ws)
  reflog_path = config.store_path / 'reflog.jsonl'
  old_ts = (datetime.now(UTC) - timedelta(days=days_old)).isoformat()
  record = {
    'timestamp': old_ts,
    'operation': 'snapshot',
    'experiment_id': 'exp-001',
    'old_epoch': None,
    'new_epoch': 0,
    'context': None,
  }
  append_jsonl(reflog_path, record)


class TestRecoverCliJson:
  """test_recover_cli_json -- JSON has ok and identifiers."""

  def test_recover_json_output(self, tmp_path: Path) -> None:
    """store recover --reflog-entry returns structured JSON."""
    ws = _setup_workspace(tmp_path)
    result = run_cli(
      ws,
      ['--experiment', 'exp-001', 'store', 'recover', '--reflog-entry', '0'],
    )
    assert result['ok'] is True
    assert 'experiment_id' in result['result']
    assert 'epoch' in result['result']


class TestReflogExpireCli:
  """test_reflog_expire_cli -- mutates and returns expected JSON."""

  def test_expire_removes_old_entries(self, tmp_path: Path) -> None:
    """store reflog expire --older-than 30d removes old entries."""
    ws = _setup_workspace(tmp_path)
    _inject_old_entry(ws, days_old=60)

    result = run_cli(ws, ['store', 'reflog', 'expire', '--older-than', '30d'])
    assert result['ok'] is True
    assert result['result']['expired_count'] >= 1


class TestReflogExpireDryRun:
  """test_reflog_expire_dry_run -- count matches; file unchanged."""

  def test_dry_run_does_not_modify(self, tmp_path: Path) -> None:
    """--dry-run reports count but does not remove entries."""
    ws = _setup_workspace(tmp_path)
    _inject_old_entry(ws, days_old=60)

    config = AutoPilotConfig(workspace=ws)
    reflog_path = config.store_path / 'reflog.jsonl'
    content_before = reflog_path.read_text(encoding='utf-8')

    result = run_cli(
      ws,
      ['store', 'reflog', 'expire', '--older-than', '30d', '--dry-run'],
    )
    assert result['ok'] is True
    assert result['result']['dry_run'] is True
    assert result['result']['expired_count'] >= 1

    content_after = reflog_path.read_text(encoding='utf-8')
    assert content_before == content_after


class TestReflogExpireCliJson:
  """test_reflog_expire_cli_json -- JSON includes expired_count."""

  def test_json_envelope_contains_count(self, tmp_path: Path) -> None:
    """JSON result includes expired_count field."""
    ws = _setup_workspace(tmp_path)
    result = run_cli(ws, ['store', 'reflog', 'expire', '--older-than', '30d'])
    assert result['ok'] is True
    assert 'expired_count' in result['result']
    assert isinstance(result['result']['expired_count'], int)


class TestRecoverExitCode:
  """test_recover_exit_code -- success 0; invalid index non-zero."""

  def test_success_exit_zero(self, tmp_path: Path) -> None:
    """Valid reflog entry index returns success envelope."""
    ws = _setup_workspace(tmp_path)
    result = run_cli(
      ws,
      ['--experiment', 'exp-001', 'store', 'recover', '--reflog-entry', '0'],
    )
    assert result['ok'] is True

  def test_invalid_index_fails(self, tmp_path: Path) -> None:
    """Invalid index produces non-zero exit / error envelope."""
    ws = _setup_workspace(tmp_path)
    with pytest.raises(SystemExit):
      run_cli(
        ws,
        ['--experiment', 'exp-001', 'store', 'recover', '--reflog-entry', '9999'],
      )


class TestReflogExpireExitCode:
  """test_reflog_expire_exit_code -- success 0; failure non-zero."""

  def test_success_exit_zero(self, tmp_path: Path) -> None:
    """Valid expire command returns success envelope."""
    ws = _setup_workspace(tmp_path)
    result = run_cli(ws, ['store', 'reflog', 'expire', '--older-than', '30d'])
    assert result['ok'] is True

  def test_invalid_format_fails(self, tmp_path: Path) -> None:
    """Invalid --older-than format fails."""
    ws = _setup_workspace(tmp_path)
    with pytest.raises(SystemExit):
      run_cli(ws, ['store', 'reflog', 'expire', '--older-than', 'bad'])
