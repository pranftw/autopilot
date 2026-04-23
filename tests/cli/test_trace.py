"""Tests for trace CLI command."""

from autopilot.cli.commands.trace import TraceCommand
from autopilot.cli.output import Output
from autopilot.core.artifacts.epoch import DataArtifact
from autopilot.core.memory import FileMemory
from pathlib import Path
from unittest.mock import MagicMock
import json

_data = DataArtifact()


def _make_ctx(tmp_path: Path, experiment: str = 'test-exp') -> MagicMock:
  ctx = MagicMock()
  ctx.experiment = experiment
  ctx.epoch = 1
  ctx.output = Output(use_json=True)
  exp_dir = tmp_path / experiment
  exp_dir.mkdir(parents=True, exist_ok=True)
  ctx.experiment_dir.return_value = exp_dir
  return ctx


class TestTraceCommand:
  def test_instantiates(self):
    cmd = TraceCommand()
    assert cmd.name == 'trace'

  def test_collect_no_epoch(self, tmp_path):
    ctx = _make_ctx(tmp_path)
    ctx.epoch = 0
    ctx.output = MagicMock()
    cmd = TraceCommand()
    args = MagicMock(epoch=0, limit=0)
    cmd.collect(ctx, args)
    ctx.output.error.assert_called_once()
    ctx.output.result.assert_not_called()

  def test_collect_happy_path(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    _data.append({'id': 'a', 'success': True}, exp_dir, epoch=1)
    _data.append({'id': 'b', 'success': False}, exp_dir, epoch=1)

    cmd = TraceCommand()
    args = MagicMock(epoch=1, limit=0)
    cmd.collect(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert envelope['result']['count'] == 2

  def test_collect_with_limit(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    for i in range(5):
      _data.append({'id': str(i)}, exp_dir, epoch=1)

    cmd = TraceCommand()
    args = MagicMock(epoch=1, limit=2)
    cmd.collect(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['count'] == 2

  def test_inspect_no_node(self, tmp_path):
    ctx = _make_ctx(tmp_path)
    ctx.output = MagicMock()
    cmd = TraceCommand()
    args = MagicMock(node='', depth=1, epoch=1)
    cmd.inspect_trace(ctx, args)
    ctx.output.error.assert_called_once()
    ctx.output.result.assert_not_called()

  def test_inspect_filters_by_id(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    _data.append({'id': 'node_a', 'success': True, 'feedback': 'good'}, exp_dir, epoch=1)
    _data.append({'id': 'node_b', 'success': False, 'error_message': 'bad'}, exp_dir, epoch=1)

    cmd = TraceCommand()
    args = MagicMock(node='node_b', depth=1, epoch=1)
    cmd.inspect_trace(ctx, args)
    captured = capsys.readouterr()
    r = json.loads(captured.out)['result']
    assert r['count'] == 1
    assert r['matches'][0]['id'] == 'node_b'
    assert r['matches'][0]['success'] is False
    assert r['matches'][0]['error_message'] == 'bad'

  def test_inspect_depth_2_includes_memory(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    _data.append({'id': 'my_node', 'success': True}, exp_dir, epoch=1)

    memory = FileMemory(exp_dir)
    memory.learn(epoch=1, outcome='worked', node='my_node', metrics={'accuracy': 0.9})

    cmd = TraceCommand()
    args = MagicMock(node='my_node', depth=2, epoch=1)
    cmd.inspect_trace(ctx, args)
    captured = capsys.readouterr()
    r = json.loads(captured.out)['result']
    assert 'memory_records' in r
    assert len(r['memory_records']) >= 1

  def test_inspect_no_matches(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    _data.append({'id': 'other'}, exp_dir, epoch=1)

    cmd = TraceCommand()
    args = MagicMock(node='nonexistent', depth=1, epoch=1)
    cmd.inspect_trace(ctx, args)
    captured = capsys.readouterr()
    r = json.loads(captured.out)['result']
    assert r['count'] == 0
