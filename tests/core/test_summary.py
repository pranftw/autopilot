"""Tests for experiment summary functions."""

from autopilot.core.cost_tracker import CostTracker
from autopilot.core.stage_io import write_epoch_artifact
from autopilot.core.stage_models import RegressionAnalysis
from autopilot.core.summary import build_experiment_summary, write_experiment_summary
from autopilot.tracking.io import append_jsonl
from unittest.mock import MagicMock
import json


class TestBuildExperimentSummary:
  def test_zero_epochs(self, tmp_path):
    result = build_experiment_summary(tmp_path, {'epochs': [], 'total_epochs': 0})
    assert result.total_epochs == 0
    assert result.best_epoch == 0

  def test_single_epoch(self, tmp_path):
    loop_result = {
      'epochs': [{'epoch': 1, 'metrics': {'accuracy': 0.8}}],
      'total_epochs': 1,
    }
    result = build_experiment_summary(tmp_path, loop_result)
    assert result.total_epochs == 1
    assert result.best_epoch == 1
    assert result.final_metrics == {'accuracy': 0.8}

  def test_multi_epoch_best(self, tmp_path):
    loop_result = {
      'epochs': [
        {'epoch': 1, 'metrics': {'accuracy': 0.5}},
        {'epoch': 2, 'metrics': {'accuracy': 0.9}},
        {'epoch': 3, 'metrics': {'accuracy': 0.7}},
      ],
      'total_epochs': 3,
    }
    result = build_experiment_summary(tmp_path, loop_result)
    assert result.best_epoch == 2
    assert result.final_metrics == {'accuracy': 0.7}

  def test_with_cost_tracker(self, tmp_path):
    ct = CostTracker(tmp_path)
    trainer = MagicMock()
    ct.on_epoch_start(trainer, 1)
    ct.on_epoch_end(trainer, 1)
    loop_result = {
      'epochs': [{'epoch': 1, 'metrics': {'accuracy': 0.8}}],
      'total_epochs': 1,
    }
    result = build_experiment_summary(tmp_path, loop_result, cost_tracker=ct)
    assert result.cost_total is not None
    assert result.cost_total.wall_clock_s >= 0

  def test_without_cost_tracker(self, tmp_path):
    loop_result = {'epochs': [], 'total_epochs': 0}
    result = build_experiment_summary(tmp_path, loop_result, cost_tracker=None)
    assert result.cost_total is None

  def test_slug_from_dir_name(self, tmp_path):
    exp_dir = tmp_path / 'my-experiment'
    exp_dir.mkdir()
    result = build_experiment_summary(exp_dir, {'epochs': [], 'total_epochs': 0})
    assert result.slug == 'my-experiment'

  def test_reads_per_epoch_regression_artifacts(self, tmp_path):
    ra = RegressionAnalysis(
      epoch=2,
      overall_verdict='net_regression',
      per_category_deltas={'accuracy': -0.1},
      regressions=[{'metric': 'accuracy', 'delta': -0.1}],
    )
    write_epoch_artifact(tmp_path, 2, 'regression_analysis.json', ra.to_dict())
    loop_result = {
      'epochs': [
        {'epoch': 1, 'metrics': {'accuracy': 0.9}},
        {'epoch': 2, 'metrics': {'accuracy': 0.8}},
      ],
      'total_epochs': 2,
    }
    result = build_experiment_summary(tmp_path, loop_result)
    assert len(result.regressions) == 1
    assert result.regressions[0].overall_verdict == 'net_regression'
    assert result.regressions[0].epoch == 2

  def test_reads_knowledge_base_lines(self, tmp_path):
    kb_path = tmp_path / 'knowledge_base.jsonl'
    append_jsonl(kb_path, {'epoch': 1, 'outcome': 'worked'})
    append_jsonl(kb_path, {'epoch': 2, 'outcome': 'failed'})
    loop_result = {'epochs': [], 'total_epochs': 0}
    result = build_experiment_summary(tmp_path, loop_result)
    assert result.memory_entries == 2


class TestWriteExperimentSummary:
  def test_writes_and_reads(self, tmp_path):
    from autopilot.core.stage_models import ExperimentSummaryData

    summary = ExperimentSummaryData(slug='test', total_epochs=3, best_epoch=2)
    path = write_experiment_summary(tmp_path, summary)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data['slug'] == 'test'
    assert data['total_epochs'] == 3
