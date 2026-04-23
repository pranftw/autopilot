"""Tests for memory CLI command."""

from autopilot.cli.commands.memory import MemoryCommand
from autopilot.cli.output import Output
from autopilot.core.memory import FileMemory
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


class TestMemoryCommand:
  def test_instantiates(self):
    cmd = MemoryCommand()
    assert cmd.name == 'memory'

  def test_query_empty(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    cmd = MemoryCommand()
    args = MagicMock(category='', node='', outcome='', strategy='', epoch=0)
    cmd.query(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert envelope['result']['count'] == 0

  def test_record_and_query(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    cmd = MemoryCommand()

    args_record = MagicMock(
      outcome='worked',
      category='rule_change',
      strategy='add_rule',
      node='matcher',
      content='added pattern',
      metrics='{"accuracy": 0.9}',
      epoch=1,
    )
    cmd.record(ctx, args_record)
    capsys.readouterr()

    args_query = MagicMock(category='rule_change', node='', outcome='', strategy='', epoch=0)
    cmd.query(ctx, args_query)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['count'] == 1
    assert envelope['result']['records'][0]['outcome'] == 'worked'
    assert envelope['result']['records'][0]['strategy'] == 'add_rule'

  def test_trends(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    memory = FileMemory(exp_dir)
    memory.learn(epoch=1, outcome='worked', metrics={'accuracy': 0.7})
    memory.learn(epoch=2, outcome='worked', metrics={'accuracy': 0.8})
    memory.learn(epoch=3, outcome='worked', metrics={'accuracy': 0.9})

    cmd = MemoryCommand()
    args = MagicMock(metric='accuracy', window=5)
    cmd.trends(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert 'direction' in envelope['result']

  def test_context(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    memory = FileMemory(exp_dir)
    memory.learn(epoch=1, outcome='worked', metrics={'accuracy': 0.8})

    cmd = MemoryCommand()
    args = MagicMock(epoch=1)
    cmd.context(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert 'total_records' in envelope['result']
