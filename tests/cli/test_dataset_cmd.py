"""Tests for autopilot.cli.commands.dataset.DatasetCommand."""

from autopilot.cli.commands.dataset import DatasetCommand
from autopilot.cli.output import Output
from autopilot.core.config import AutoPilotConfig
from pathlib import Path
from unittest.mock import MagicMock
import argparse


def _dataset_ctx(tmp_path: Path) -> MagicMock:
  cfg = AutoPilotConfig(workspace=tmp_path)
  cfg.init_workspace()
  ctx = MagicMock()
  ctx.config = cfg
  ctx.datasets_dir = cfg.datasets_path
  ctx.dataset = None
  ctx.output = MagicMock(spec=Output)
  return ctx


def test_seed_creates_train_val_test(tmp_path: Path) -> None:
  ctx = _dataset_ctx(tmp_path)
  DatasetCommand().seed(ctx, argparse.Namespace())
  base = ctx.datasets_dir
  assert (base / 'train').is_dir()
  assert (base / 'val').is_dir()
  assert (base / 'test').is_dir()
  ctx.output.result.assert_called_once()
  body = ctx.output.result.call_args[0][0]
  assert body['status'] == 'seeded'


def test_list_returns_sorted_directories(tmp_path: Path) -> None:
  ctx = _dataset_ctx(tmp_path)
  base = ctx.datasets_dir
  (base / 'zoo').mkdir(parents=True)
  (base / 'alpha').mkdir(parents=True)
  DatasetCommand().list(ctx, argparse.Namespace())
  payload = ctx.output.result.call_args[0][0]
  assert payload['datasets'] == ['alpha', 'zoo']
  assert payload['count'] == 2


def test_split_no_name_specified(tmp_path: Path) -> None:
  """DatasetSplit.forward with no split_name and ctx.split=None."""
  from autopilot.cli.commands.dataset import DatasetSplit

  ctx = _dataset_ctx(tmp_path)
  ctx.split = None
  DatasetSplit().forward(ctx, argparse.Namespace(split_name=None))
  ctx.output.info.assert_called_once()
  assert 'No split specified' in ctx.output.info.call_args[0][0]
  body = ctx.output.result.call_args[0][0]
  assert body == {'split': None, 'ok': True}


def test_split_uses_ctx_split_when_arg_missing(tmp_path: Path) -> None:
  """DatasetSplit.forward falls back to ctx.split when split_name is None."""
  from autopilot.cli.commands.dataset import DatasetSplit

  ctx = _dataset_ctx(tmp_path)
  ctx.split = 'train'
  DatasetSplit().forward(ctx, argparse.Namespace(split_name=None))
  body = ctx.output.result.call_args[0][0]
  assert body == {'split': 'train', 'ok': True}


def test_list_missing_datasets_root(tmp_path: Path) -> None:
  """dataset list with non-existent datasets_dir returns empty list."""
  ctx = _dataset_ctx(tmp_path)
  ctx.datasets_dir = tmp_path / 'nonexistent'
  DatasetCommand().list(ctx, argparse.Namespace())
  payload = ctx.output.result.call_args[0][0]
  assert payload == {'datasets': [], 'count': 0}


def test_show_default_name(tmp_path: Path) -> None:
  """dataset show with ctx.dataset=None uses 'default' as name."""
  ctx = _dataset_ctx(tmp_path)
  ctx.dataset = None
  DatasetCommand().show(ctx, argparse.Namespace())
  payload = ctx.output.result.call_args[0][0]
  assert payload['dataset'] == 'default'
  assert str(ctx.datasets_dir) in payload['datasets_dir']


def test_show_named_dataset(tmp_path: Path) -> None:
  """dataset show with ctx.dataset set uses that name."""
  ctx = _dataset_ctx(tmp_path)
  ctx.dataset = 'mydata'
  DatasetCommand().show(ctx, argparse.Namespace())
  payload = ctx.output.result.call_args[0][0]
  assert payload['dataset'] == 'mydata'
