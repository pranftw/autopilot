"""Tests for status CLI command."""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.commands.status import StatusCommand
from autopilot.cli.context import CLIContext
from autopilot.cli.output import Output
from autopilot.core.artifacts.experiment import SummaryArtifact
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from autopilot.tracking.io import atomic_write_json
from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import pytest

_summary = SummaryArtifact()


def _build_forest_ctx(
  tmp_path: Path, slug: str = 'test-exp', epoch: int = 0
) -> tuple[MagicMock, FileForest]:
  """Build a context with a forest containing a single experiment."""
  config = AutoPilotConfig(workspace=tmp_path)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')
  exp = Experiment(experiment_id=slug)
  exp.epoch = epoch
  tree.add(Node(experiment=exp))
  forest.save()

  exp_dir = config.experiment_path(slug=slug)
  exp_dir.mkdir(parents=True, exist_ok=True)

  ctx = MagicMock()
  ctx.experiment = slug
  ctx.config = config
  ctx.output = Output(use_json=True)
  ctx.fail = CLIContext.fail.__get__(ctx, type(ctx))
  return ctx, forest


class TestStatusCommand:
  def test_instantiates(self):
    cmd = StatusCommand()
    assert cmd.name == 'status'

  def test_no_experiment(self):
    cmd = StatusCommand()
    ctx = MagicMock()
    ctx.experiment = ''
    ctx.output = MagicMock()
    ctx.fail = CLIContext.fail.__get__(ctx, type(ctx))
    args = MagicMock()
    with pytest.raises(SystemExit) as exc_info:
      cmd.forward(ctx, args)
    assert exc_info.value.code == 1

  def test_happy_path(self, tmp_path, capsys):
    ctx, forest = _build_forest_ctx(tmp_path, epoch=2)
    exp_dir = ctx.config.experiment_path(slug='test-exp')
    (exp_dir / 'epoch_2').mkdir()

    cmd = StatusCommand()
    args = MagicMock()
    with patch('autopilot.cli.commands.status.load_forest', return_value=forest):
      cmd.forward(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert envelope['result']['id'] == 'test-exp'
    assert envelope['result']['epoch'] == 2
    assert envelope['result']['trained_epochs'] == 1

  def test_includes_stop_reason_from_summary(self, tmp_path, capsys):
    ctx, forest = _build_forest_ctx(tmp_path, epoch=3)
    exp_dir = ctx.config.experiment_path(slug='test-exp')
    _summary.write({'stop_reason': 'plateau', 'last_good_epoch': 2}, exp_dir)

    cmd = StatusCommand()
    with patch('autopilot.cli.commands.status.load_forest', return_value=forest):
      cmd.forward(ctx, MagicMock())
    captured = capsys.readouterr()
    r = json.loads(captured.out)['result']
    assert r['stop_reason'] == 'plateau'
    assert r['last_good_epoch'] == 2

  def test_crash_detection_from_run_state(self, tmp_path, capsys):
    ctx, forest = _build_forest_ctx(tmp_path, epoch=5)
    exp_dir = ctx.config.experiment_path(slug='test-exp')
    atomic_write_json(
      exp_dir / 'run_state.json',
      {
        'epoch': 5,
        'status': 'running',
      },
    )

    cmd = StatusCommand()
    with patch('autopilot.cli.commands.status.load_forest', return_value=forest):
      cmd.forward(ctx, MagicMock())
    captured = capsys.readouterr()
    r = json.loads(captured.out)['result']
    assert r['stop_reason'] == 'crash'

  def test_trained_epochs_count(self, tmp_path, capsys):
    ctx, forest = _build_forest_ctx(tmp_path, epoch=1)
    exp_dir = ctx.config.experiment_path(slug='test-exp')
    for ep in range(1, 4):
      (exp_dir / f'epoch_{ep}').mkdir()

    cmd = StatusCommand()
    with patch('autopilot.cli.commands.status.load_forest', return_value=forest):
      cmd.forward(ctx, MagicMock())
    captured = capsys.readouterr()
    r = json.loads(captured.out)['result']
    assert r['trained_epochs'] == 3

  def test_duplicate_experiment_id_deduplicates(self, tmp_path, capsys):
    """BUG-044: duplicate experiment IDs across trees are deduplicated (first wins)."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)

    tree1 = forest.create_tree('tree1')
    exp1 = Experiment(experiment_id='dup-exp')
    exp1.epoch = 5
    tree1.add(Node(experiment=exp1))

    tree2 = forest.create_tree('tree2')
    exp2 = Experiment(experiment_id='dup-exp')
    exp2.epoch = 10
    tree2.add(Node(experiment=exp2))
    forest.save()

    qb = forest.query()
    nodes = qb.all()
    ids = [n.experiment.id for n in nodes]
    assert ids.count('dup-exp') == 1
    assert nodes[0].experiment.epoch == 5
