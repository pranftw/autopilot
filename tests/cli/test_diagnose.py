"""Tests for diagnose CLI command."""

from autopilot.cli.commands.diagnose import DiagnoseCommand
from autopilot.core.artifacts.artifact import JSONArtifact, JSONLArtifact
from pathlib import Path
from tests.cli.conftest import make_mock_cli_context
from unittest.mock import MagicMock
import json
import pytest

_diagnoses = JSONLArtifact('trace_diagnoses.jsonl', scope='epoch')
_heatmap = JSONArtifact('node_heatmap.json', scope='epoch')


def _make_ctx(tmp_path: Path, experiment: str = 'test-exp') -> MagicMock:
  return make_mock_cli_context(tmp_path, experiment=experiment, epoch=1)


class TestDiagnoseCommand:
  def test_instantiates(self):
    cmd = DiagnoseCommand()
    assert cmd.name == 'diagnose'

  def test_run_no_epoch(self, tmp_path):
    ctx = _make_ctx(tmp_path)
    ctx.epoch = None
    cmd = DiagnoseCommand()
    args = MagicMock(epoch=None, category='', node='')
    with pytest.raises(SystemExit) as exc_info:
      cmd.run_diagnose(ctx, args)
    assert exc_info.value.code == 1

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
    _diagnoses.append(
      {'category': 'hallucination', 'node': 'response_gen', 'detail': 'made up facts'},
      exp_dir,
      epoch=1,
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
    _diagnoses.append({'category': 'hallucination', 'node': 'a'}, exp_dir, epoch=1)
    _diagnoses.append({'category': 'other', 'node': 'b'}, exp_dir, epoch=1)

    cmd = DiagnoseCommand()
    args = MagicMock(epoch=1, category='hallucination', node='')
    cmd.run_diagnose(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert len(envelope['result']['diagnoses']) == 1

  def test_heatmap_no_epoch(self, tmp_path):
    ctx = _make_ctx(tmp_path)
    ctx.epoch = None
    cmd = DiagnoseCommand()
    args = MagicMock(epoch=None)
    with pytest.raises(SystemExit) as exc_info:
      cmd.heatmap(ctx, args)
    assert exc_info.value.code == 1

  def test_heatmap_happy_path(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    _heatmap.write({'node_a': 5, 'node_b': 2}, exp_dir, epoch=1)

    cmd = DiagnoseCommand()
    args = MagicMock(epoch=1)
    cmd.heatmap(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['heatmap'] == {'node_a': 5, 'node_b': 2}

  def test_heatmap_no_artifact(self, tmp_path):
    ctx = _make_ctx(tmp_path)
    cmd = DiagnoseCommand()
    args = MagicMock(epoch=99)
    with pytest.raises(SystemExit) as exc_info:
      cmd.heatmap(ctx, args)
    assert exc_info.value.code == 1
