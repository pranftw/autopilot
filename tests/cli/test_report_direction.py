"""Tests for direction-aware report compare (Plan 01, P3#34 and P3#40).

Covers:
  - ``report compare`` classic path respects ``infer_direction`` heuristic
    so latency decrease is treated as improvement.
  - Baseline-centric semantics: first slug is baseline, deltas are relative
    to it.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli_no_context
from unittest.mock import patch
import pytest


@pytest.fixture(autouse=True)
def _patch_store_checkout():
  """Patch FileStore.checkout for tests that don't create snapshots."""
  with patch('autopilot.ai.store.file_store.FileStore.checkout'):
    yield


@pytest.fixture
def ws_report(tmp_path: Path) -> Path:
  """Workspace with experiments having latency and accuracy metrics."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')

  exp_a = Experiment(experiment_id='exp-a', hypothesis='baseline')
  exp_a.start()
  exp_a.complete(metrics={'accuracy': 0.80, 'latency': 200.0, 'loss': 1.0})
  tree.add(Node(experiment=exp_a))

  exp_b = Experiment(experiment_id='exp-b', hypothesis='candidate')
  exp_b.start()
  exp_b.complete(metrics={'accuracy': 0.85, 'latency': 150.0, 'loss': 0.5})
  tree.add(Node(experiment=exp_b))

  exp_c = Experiment(experiment_id='exp-c', hypothesis='third')
  exp_c.start()
  exp_c.complete(metrics={'accuracy': 0.70, 'latency': 250.0, 'loss': 1.5})
  tree.add(Node(experiment=exp_c))

  (config.workspace / '.autopilot').mkdir(parents=True, exist_ok=True)
  forest.save()
  return ws


class TestReportCompareLatencyDirection:
  """P3#34: latency decrease should be treated as improvement."""

  def test_latency_decrease_has_correct_direction(self, ws_report: Path) -> None:
    result = run_cli_no_context(
      ws_report,
      ['report', 'compare', 'exp-a', 'exp-b'],
    )
    comparisons = result['result']['metric_comparisons']
    assert len(comparisons) == 1

    deltas = comparisons[0]
    latency_delta = next((d for d in deltas if d['metric'] == 'latency'), None)
    assert latency_delta is not None
    assert latency_delta['higher_is_better'] is False
    assert latency_delta['delta'] < 0

  def test_loss_decrease_has_correct_direction(self, ws_report: Path) -> None:
    result = run_cli_no_context(
      ws_report,
      ['report', 'compare', 'exp-a', 'exp-b'],
    )
    comparisons = result['result']['metric_comparisons']
    deltas = comparisons[0]
    loss_delta = next((d for d in deltas if d['metric'] == 'loss'), None)
    assert loss_delta is not None
    assert loss_delta['higher_is_better'] is False

  def test_accuracy_has_correct_direction(self, ws_report: Path) -> None:
    result = run_cli_no_context(
      ws_report,
      ['report', 'compare', 'exp-a', 'exp-b'],
    )
    comparisons = result['result']['metric_comparisons']
    deltas = comparisons[0]
    acc_delta = next((d for d in deltas if d['metric'] == 'accuracy'), None)
    assert acc_delta is not None
    assert acc_delta['higher_is_better'] is True

  def test_explicit_lower_metric_flag(self, ws_report: Path) -> None:
    result = run_cli_no_context(
      ws_report,
      ['report', 'compare', 'exp-a', 'exp-b', '--lower-metric', 'latency'],
    )
    comparisons = result['result']['metric_comparisons']
    deltas = comparisons[0]
    latency_delta = next((d for d in deltas if d['metric'] == 'latency'), None)
    assert latency_delta is not None
    assert latency_delta['higher_is_better'] is False


class TestReportCompareBaselineCentric:
  """P3#40: baseline-centric comparison with multiple slugs.

  The first slug is the baseline. Each subsequent slug is compared
  pairwise against the baseline. Deltas are (candidate - baseline).
  """

  def test_multi_slug_baseline_relative(self, ws_report: Path) -> None:
    result = run_cli_no_context(
      ws_report,
      ['report', 'compare', 'exp-a', 'exp-b', 'exp-c'],
    )
    comparisons = result['result']['metric_comparisons']
    assert len(comparisons) == 2

    deltas_ab = {d['metric']: d for d in comparisons[0]}
    assert deltas_ab['accuracy']['baseline'] == 0.80
    assert deltas_ab['accuracy']['candidate'] == 0.85
    assert deltas_ab['accuracy']['delta'] > 0

    deltas_ac = {d['metric']: d for d in comparisons[1]}
    assert deltas_ac['accuracy']['baseline'] == 0.80
    assert deltas_ac['accuracy']['candidate'] == 0.70
    assert deltas_ac['accuracy']['delta'] < 0

  def test_baseline_is_first_slug(self, ws_report: Path) -> None:
    result = run_cli_no_context(
      ws_report,
      ['report', 'compare', 'exp-a', 'exp-b'],
    )
    summaries = result['result']['summaries']
    assert summaries[0]['id'] == 'exp-a'
    assert summaries[1]['id'] == 'exp-b'
