"""Tests for diagnose CLI command."""

from autopilot.cli.commands.diagnose_cmd import DiagnoseCommand
from autopilot.cli.output import Output
from autopilot.core.stage_io import append_epoch_artifact, write_epoch_artifact
from pathlib import Path
from unittest.mock import MagicMock
import json


def _make_ctx(tmp_path: Path, experiment: str = 'test-exp') -> MagicMock:
  ctx = MagicMock()
  ctx.experiment = experiment
  ctx.epoch = 1
  ctx.output = Output(use_json=True)
  exp_dir = tmp_path / experiment
  exp_dir.mkdir(parents=True, exist_ok=True)
  ctx.experiment_dir.return_value = exp_dir
  return ctx


class TestDiagnoseCommand:
  def test_instantiates(self):
    cmd = DiagnoseCommand()
    assert cmd.name == 'diagnose'

  def test_run_no_epoch(self, tmp_path):
    ctx = _make_ctx(tmp_path)
    ctx.epoch = 0
    ctx.output = MagicMock()
    cmd = DiagnoseCommand()
    args = MagicMock(epoch=0, category='', node='')
    cmd.run_diagnose(ctx, args)
    ctx.output.error.assert_called_once()
    ctx.output.result.assert_not_called()

  def test_run_empty_traces(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    cmd = DiagnoseCommand()
    args = MagicMock(epoch=1, category='', node='')
    cmd.run_diagnose(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert envelope['result']['diagnoses'] == []

  def test_run_with_traces_and_memory(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    append_epoch_artifact(
      exp_dir,
      1,
      'trace_diagnoses.jsonl',
      {
        'category': 'hallucination',
        'node': 'response_gen',
        'detail': 'made up facts',
      },
    )

    cmd = DiagnoseCommand()
    args = MagicMock(epoch=1, category='', node='')
    cmd.run_diagnose(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['epoch'] == 1
    assert len(envelope['result']['diagnoses']) == 1

  def test_run_filter_by_category(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    append_epoch_artifact(
      exp_dir,
      1,
      'trace_diagnoses.jsonl',
      {
        'category': 'hallucination',
        'node': 'a',
      },
    )
    append_epoch_artifact(
      exp_dir,
      1,
      'trace_diagnoses.jsonl',
      {
        'category': 'other',
        'node': 'b',
      },
    )

    cmd = DiagnoseCommand()
    args = MagicMock(epoch=1, category='hallucination', node='')
    cmd.run_diagnose(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert len(envelope['result']['diagnoses']) == 1

  def test_heatmap_no_epoch(self, tmp_path):
    ctx = _make_ctx(tmp_path)
    ctx.epoch = 0
    ctx.output = MagicMock()
    cmd = DiagnoseCommand()
    args = MagicMock(epoch=0)
    cmd.heatmap(ctx, args)
    ctx.output.error.assert_called_once()
    ctx.output.result.assert_not_called()

  def test_heatmap_happy_path(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    write_epoch_artifact(exp_dir, 1, 'node_heatmap.json', {'node_a': 5, 'node_b': 2})

    cmd = DiagnoseCommand()
    args = MagicMock(epoch=1)
    cmd.heatmap(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['heatmap'] == {'node_a': 5, 'node_b': 2}
