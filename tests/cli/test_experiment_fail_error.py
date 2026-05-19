"""Tests for experiment fail --error flag (clean break from --reason)."""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.primitives import ArgparseCLIError
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli
import pytest


@pytest.fixture
def ws_with_running(tmp_path: Path) -> Path:
  """Workspace with a running experiment for fail tests."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')
  exp = Experiment(experiment_id='exp-run', hypothesis='will fail')
  exp.start()
  tree.add(Node(experiment=exp))
  forest.save()
  return ws


class TestExperimentFailErrorFlag:
  """Tests for the --error flag on experiment fail."""

  def test_experiment_fail_error_flag(self, ws_with_running: Path) -> None:
    """--error sets the persisted experiment error string."""
    result = run_cli(
      ws_with_running,
      ['experiment', 'fail', 'exp-run', '--error', 'OOM on GPU 0'],
    )
    assert result['result']['ok'] is True
    assert result['result']['status'] == 'failed'
    assert result['result']['error'] == 'OOM on GPU 0'

  def test_experiment_fail_reason_flag_rejected(self, ws_with_running: Path) -> None:
    """--reason is no longer accepted (clean break)."""
    with pytest.raises((SystemExit, ArgparseCLIError)):
      run_cli(
        ws_with_running,
        ['experiment', 'fail', 'exp-run', '--reason', 'should fail'],
      )

  def test_experiment_fail_without_error_uses_context(self, ws_with_running: Path) -> None:
    """Omitting --error falls back to --context for the error display field."""
    result = run_cli(
      ws_with_running,
      ['experiment', 'fail', 'exp-run'],
    )
    assert result['result']['status'] == 'failed'
    assert result['result']['error'] == 'test'
