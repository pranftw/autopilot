"""Tests for query --metric-gt and --metric-lt input validation."""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli_no_context
import pytest


@pytest.fixture
def ws_with_tree(tmp_path: Path) -> Path:
  """Workspace with a tree containing one completed experiment."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')
  exp = Experiment(experiment_id='exp-1', hypothesis='test')
  exp.start()
  exp.complete(metrics={'accuracy': 0.9, 'loss': 0.1})
  tree.add(Node(experiment=exp))
  forest.save()
  return ws


class TestMetricGtValidation:
  """Validation tests for --metric-gt flag."""

  def test_metric_gt_rejects_empty_name(self, ws_with_tree: Path) -> None:
    with pytest.raises(SystemExit):
      run_cli_no_context(ws_with_tree, ['query', '--metric-gt', ':0.5'])

  def test_metric_gt_rejects_missing_colon(self, ws_with_tree: Path) -> None:
    with pytest.raises(SystemExit):
      run_cli_no_context(ws_with_tree, ['query', '--metric-gt', 'accuracy'])

  def test_metric_gt_rejects_non_numeric(self, ws_with_tree: Path) -> None:
    with pytest.raises(SystemExit):
      run_cli_no_context(ws_with_tree, ['query', '--metric-gt', 'acc:abc'])

  def test_metric_gt_empty_value_after_colon(self, ws_with_tree: Path) -> None:
    with pytest.raises(SystemExit):
      run_cli_no_context(ws_with_tree, ['query', '--metric-gt', 'accuracy:'])

  def test_metric_gt_whitespace_name(self, ws_with_tree: Path) -> None:
    with pytest.raises(SystemExit):
      run_cli_no_context(ws_with_tree, ['query', '--metric-gt', ' :0.5'])

  def test_metric_gt_multiple_colons(self, ws_with_tree: Path) -> None:
    with pytest.raises(SystemExit):
      run_cli_no_context(ws_with_tree, ['query', '--metric-gt', 'a:b:0.5'])

  def test_metric_gt_negative_value_accepted(self, ws_with_tree: Path) -> None:
    result = run_cli_no_context(ws_with_tree, ['query', '--metric-gt', 'loss:-0.5'])
    assert result['ok'] is True

  def test_metric_gt_scientific_notation_accepted(self, ws_with_tree: Path) -> None:
    result = run_cli_no_context(ws_with_tree, ['query', '--metric-gt', 'loss:1e-3'])
    assert result['ok'] is True

  def test_metric_gt_inf_string_accepted(self, ws_with_tree: Path) -> None:
    result = run_cli_no_context(ws_with_tree, ['query', '--metric-gt', 'loss:inf'])
    assert result['ok'] is True

  def test_metric_gt_nan_string_rejected(self, ws_with_tree: Path) -> None:
    with pytest.raises(SystemExit):
      run_cli_no_context(ws_with_tree, ['query', '--metric-gt', 'loss:nan'])

  def test_metric_gt_valid_passes(self, ws_with_tree: Path) -> None:
    result = run_cli_no_context(ws_with_tree, ['query', '--metric-gt', 'accuracy:0.5'])
    assert result['ok'] is True
    assert result['result']['count'] == 1


class TestMetricLtValidation:
  """Validation tests for --metric-lt flag."""

  def test_metric_lt_rejects_same(self, ws_with_tree: Path) -> None:
    with pytest.raises(SystemExit):
      run_cli_no_context(ws_with_tree, ['query', '--metric-lt', 'accuracy'])

  def test_metric_lt_rejects_non_numeric(self, ws_with_tree: Path) -> None:
    with pytest.raises(SystemExit):
      run_cli_no_context(ws_with_tree, ['query', '--metric-lt', 'acc:abc'])

  def test_metric_lt_valid_passes(self, ws_with_tree: Path) -> None:
    result = run_cli_no_context(ws_with_tree, ['query', '--metric-lt', 'accuracy:0.95'])
    assert result['ok'] is True
    assert result['result']['count'] == 1
