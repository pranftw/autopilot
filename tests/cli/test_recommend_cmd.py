"""CLI tests for the autopilot recommend command."""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli_no_context


def _seed_workspace(tmp_path: Path) -> Path:
  """Create a workspace with a forest and completed experiments."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  tree = forest.create_tree('main')
  forest.switch('main')

  exp1 = Experiment(experiment_id='exp-1')
  exp1.start()
  exp1.complete(metrics={'accuracy': 0.7})
  tree.add(Node(experiment=exp1))

  exp2 = Experiment(experiment_id='exp-2')
  exp2.start()
  exp2.complete(metrics={'accuracy': 0.95})
  tree.add(Node(experiment=exp2))

  forest.save()
  return ws


def _seed_empty_workspace(tmp_path: Path) -> Path:
  """Create a workspace with a forest but no completed experiments."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  forest.create_tree('main')
  forest.switch('main')
  forest.save()
  return ws


class TestRecommendCLI:
  """Tests for the recommend CLI command."""

  def test_recommend_cli_json(self, tmp_path: Path) -> None:
    """recommend --json returns a JSON envelope with ok=true."""
    ws = _seed_workspace(tmp_path)
    result = run_cli_no_context(ws, ['recommend', '--metric', 'accuracy'])
    assert result['ok'] is True
    assert 'result' in result

  def test_recommend_cli_exit_code_success(self, tmp_path: Path) -> None:
    """recommend with experiments exits successfully (no SystemExit)."""
    ws = _seed_workspace(tmp_path)
    result = run_cli_no_context(ws, ['recommend', '--metric', 'accuracy'])
    assert result['ok'] is True

  def test_recommend_cli_exit_code_no_experiments(self, tmp_path: Path) -> None:
    """recommend with no completed experiments returns investigate action."""
    ws = _seed_empty_workspace(tmp_path)
    result = run_cli_no_context(ws, ['recommend', '--metric', 'accuracy'])
    assert result['ok'] is True
    assert result['result']['action'] == 'investigate'
    assert result['result']['confidence'] == 'low'

  def test_recommend_cli_json_schema(self, tmp_path: Path) -> None:
    """recommend --json result payload has all required keys."""
    ws = _seed_workspace(tmp_path)
    result = run_cli_no_context(ws, ['recommend', '--metric', 'accuracy'])
    payload = result['result']
    assert 'action' in payload
    assert 'confidence' in payload
    assert 'experiment_id' in payload
    assert 'reasoning' in payload
    assert 'alternatives' in payload
    assert 'evidence' in payload
