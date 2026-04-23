"""Tests for experiment summary functions and data models."""

from autopilot.core.artifacts.epoch import MetricComparisonArtifact
from autopilot.core.callbacks.cost import CostEntry, CostTrackerCallback
from autopilot.core.comparison import MetricComparison
from autopilot.core.summary import (
  ExperimentSummaryData,
  build_experiment_summary,
  write_experiment_summary,
)
from unittest.mock import MagicMock
import json

_mc_artifact = MetricComparisonArtifact()


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
    ct = CostTrackerCallback(tmp_path)
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
    mc = MetricComparison(
      epoch=2,
      per_metric_deltas={'accuracy': -0.1},
      regressions=[{'metric': 'accuracy', 'delta': -0.1}],
      improvements=[],
    )
    _mc_artifact.write(mc.to_dict(), tmp_path, epoch=2)
    loop_result = {
      'epochs': [
        {'epoch': 1, 'metrics': {'accuracy': 0.9}},
        {'epoch': 2, 'metrics': {'accuracy': 0.8}},
      ],
      'total_epochs': 2,
    }
    result = build_experiment_summary(tmp_path, loop_result)
    assert len(result.comparisons) == 1
    assert result.comparisons[0].regression_detected is True
    assert result.comparisons[0].epoch == 2

  def test_memory_entries_from_param(self, tmp_path):
    loop_result = {'epochs': [], 'total_epochs': 0}
    result = build_experiment_summary(tmp_path, loop_result, memory_entries=5)
    assert result.memory_entries == 5

  def test_memory_entries_defaults_to_zero(self, tmp_path):
    loop_result = {'epochs': [], 'total_epochs': 0}
    result = build_experiment_summary(tmp_path, loop_result)
    assert result.memory_entries == 0

  def test_monitor_selects_best_epoch(self, tmp_path):
    loop_result = {
      'epochs': [
        {'epoch': 1, 'metrics': {'accuracy': 0.5, 'loss': 0.8}},
        {'epoch': 2, 'metrics': {'accuracy': 0.9, 'loss': 0.2}},
        {'epoch': 3, 'metrics': {'accuracy': 0.7, 'loss': 0.5}},
      ],
      'total_epochs': 3,
    }
    result = build_experiment_summary(
      tmp_path, loop_result, monitor='loss', monitor_higher_is_better=False
    )
    assert result.best_epoch == 2

  def test_monitor_higher_is_better_true(self, tmp_path):
    loop_result = {
      'epochs': [
        {'epoch': 1, 'metrics': {'accuracy': 0.9}},
        {'epoch': 2, 'metrics': {'accuracy': 0.5}},
      ],
      'total_epochs': 2,
    }
    result = build_experiment_summary(
      tmp_path, loop_result, monitor='accuracy', monitor_higher_is_better=True
    )
    assert result.best_epoch == 1

  def test_monitor_higher_is_better_false(self, tmp_path):
    loop_result = {
      'epochs': [
        {'epoch': 1, 'metrics': {'loss': 0.9}},
        {'epoch': 2, 'metrics': {'loss': 0.2}},
      ],
      'total_epochs': 2,
    }
    result = build_experiment_summary(
      tmp_path, loop_result, monitor='loss', monitor_higher_is_better=False
    )
    assert result.best_epoch == 2

  def test_monitor_none_defaults(self, tmp_path):
    loop_result = {
      'epochs': [
        {'epoch': 1, 'metrics': {'accuracy': 0.5}},
        {'epoch': 2, 'metrics': {'accuracy': 0.9}},
      ],
      'total_epochs': 2,
    }
    result = build_experiment_summary(tmp_path, loop_result)
    assert result.best_epoch == 2


class TestWriteExperimentSummary:
  def test_writes_and_reads(self, tmp_path):
    summary = ExperimentSummaryData(slug='test', total_epochs=3, best_epoch=2)
    path = write_experiment_summary(tmp_path, summary)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data['slug'] == 'test'
    assert data['total_epochs'] == 3


class TestExperimentSummaryDataRoundTrip:
  def test_round_trip_without_cost(self):
    s = ExperimentSummaryData(slug='test', total_epochs=5, best_epoch=3)
    d = s.to_dict()
    s2 = ExperimentSummaryData.from_dict(d)
    assert s2.slug == 'test'
    assert s2.cost_total is None

  def test_round_trip_with_cost(self):
    cost = CostEntry(epoch=0, wall_clock_s=30.0)
    mc = MetricComparison(
      epoch=2,
      regressions=[{'metric': 'accuracy', 'delta': -0.1}],
    )
    s = ExperimentSummaryData(slug='test', cost_total=cost, comparisons=[mc])
    d = s.to_dict()
    s2 = ExperimentSummaryData.from_dict(d)
    assert s2.cost_total is not None
    assert s2.cost_total.wall_clock_s == 30.0
    assert len(s2.comparisons) == 1
    assert isinstance(s2.comparisons[0], MetricComparison)

  def test_field_type_preservation(self):
    s = ExperimentSummaryData(final_metrics={'accuracy': 0.95}, memory_entries=10)
    d = s.to_dict()
    s2 = ExperimentSummaryData.from_dict(d)
    assert isinstance(s2.final_metrics['accuracy'], float)
    assert isinstance(s2.memory_entries, int)
