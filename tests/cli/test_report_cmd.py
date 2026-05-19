"""Tests for autopilot.cli.commands.report.command.ReportCommand."""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.commands.report.compare import gather_summary
from autopilot.cli.output import Output
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from unittest.mock import MagicMock, patch
import argparse
import pytest


def _build_forest(tmp_path: Path, slugs_and_epochs: dict[str, int]) -> FileForest:
  """Build a FileForest with experiments in a single tree."""
  config = AutoPilotConfig(workspace=tmp_path)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')
  for slug, epoch in slugs_and_epochs.items():
    exp = Experiment(experiment_id=slug, hypothesis='probe')
    exp.epoch = epoch
    tree.add(Node(experiment=exp))
    exp_dir = config.experiment_path(slug=slug)
    exp_dir.mkdir(parents=True, exist_ok=True)
  forest.save()
  return forest


def test_summary_emits_result_with_experiment_fields(tmp_path: Path) -> None:
  forest = _build_forest(tmp_path, {'exp-a': 2})
  payload = gather_summary(forest, 'exp-a')
  assert payload['id'] == 'exp-a'
  assert payload['epoch'] == 2
  assert payload['hypothesis'] == 'probe'
  assert payload['event_count'] == 0


def test_summary_raises_when_node_missing(tmp_path: Path) -> None:
  """gather_summary raises ValueError for nonexistent experiment."""
  config = AutoPilotConfig(workspace=tmp_path)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  exp_dir = config.experiment_path(slug='missing')
  exp_dir.mkdir(parents=True, exist_ok=True)
  with pytest.raises(ValueError, match='not found in forest'):
    gather_summary(forest, 'missing')


def test_compare_single_slug_calls_fail(tmp_path: Path) -> None:
  """report compare with fewer than 2 slugs calls ctx.fail."""
  from autopilot.cli.commands.report.compare import ReportCompare

  cfg = AutoPilotConfig(workspace=tmp_path)
  cfg.init_workspace()
  ctx = MagicMock()
  ctx.experiment = None
  ctx.config = cfg
  ctx.output = MagicMock(spec=Output)
  ctx.fail = MagicMock(side_effect=SystemExit(1))
  args = argparse.Namespace(
    slugs=['only-one'], lower_metric=None, union_metrics=False, all_trees=False
  )
  with pytest.raises(SystemExit):
    ReportCompare().forward(ctx, args)
  ctx.fail.assert_called_once()
  msg = ctx.fail.call_args[0][0]
  assert 'at least 2' in msg


def test_compare_success(tmp_path: Path) -> None:
  """report compare with two slugs returns summaries and metric_comparisons."""
  from autopilot.cli.commands.report.compare import ReportCompare

  forest = _build_forest(tmp_path, {'base-exp': 3, 'cand-exp': 3})

  ctx = MagicMock()
  ctx.experiment = 'base-exp'
  ctx.config = forest.store.config
  ctx.output = MagicMock(spec=Output)

  args = argparse.Namespace(
    slugs=['base-exp', 'cand-exp'], lower_metric=None, union_metrics=False, all_trees=False
  )
  with patch('autopilot.cli.commands.report.compare.load_forest', return_value=forest):
    ReportCompare().forward(ctx, args)
  ctx.output.result.assert_called_once()
  payload = ctx.output.result.call_args[0][0]
  assert len(payload['summaries']) == 2
  assert payload['summaries'][0]['id'] == 'base-exp'
  assert payload['summaries'][1]['id'] == 'cand-exp'
  assert payload['summaries'][0]['epoch'] == 3


def test_compare_multi_way(tmp_path: Path) -> None:
  """report compare with three slugs returns three summaries and two comparisons."""
  from autopilot.cli.commands.report.compare import ReportCompare

  forest = _build_forest(tmp_path, {'exp-a': 1, 'exp-b': 2, 'exp-c': 3})

  ctx = MagicMock()
  ctx.config = forest.store.config
  ctx.output = MagicMock(spec=Output)

  args = argparse.Namespace(
    slugs=['exp-a', 'exp-b', 'exp-c'], lower_metric=None, union_metrics=False, all_trees=False
  )
  with patch('autopilot.cli.commands.report.compare.load_forest', return_value=forest):
    ReportCompare().forward(ctx, args)
  payload = ctx.output.result.call_args[0][0]
  assert len(payload['summaries']) == 3
  assert len(payload['metric_comparisons']) == 2
