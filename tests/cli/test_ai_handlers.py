"""Tests for AI command handler implementations."""

from autopilot.cli.commands.ai import (
  GenerateCommand,
  GenerateRun,
  JudgeRun,
  _load_generator_config,
  _load_judge_items,
  _require_generator,
  _require_judge,
)
from autopilot.cli.context import CLIContext
from autopilot.cli.output import Output
from pathlib import Path
from unittest.mock import MagicMock
import argparse
import json
import pytest


def _gen_config_json() -> dict:
  return {
    'run': {
      'model': 'openai:gpt-4o',
      'num_parallel': 5,
      'max_rpm': 100,
      'rpm_safety_margin': 0.9,
      'retry': {
        'max_retries': 3,
        'min_timeout_ms': 1000,
        'max_timeout_ms': 30000,
        'backoff_factor': 2,
      },
      'max_tool_steps': 5,
      'max_output_tokens': 4096,
    },
    'dataset_id': 'test-ds',
    'seed': 42,
    'total_count': 10,
    'split_ratios': {'train': 0.8, 'val': 0.1, 'test': 0.1},
    'system_prompt': 'test prompt',
  }


def _write_gen_config(tmp_path: Path) -> Path:
  config_path = tmp_path / 'gen_config.json'
  config_path.write_text(json.dumps(_gen_config_json()), encoding='utf-8')
  return config_path


def _args(**kwargs) -> argparse.Namespace:
  defaults = {
    'total_count': 0,
    'seed': 0,
    'num_parallel': 0,
    'max_rpm': 0,
  }
  defaults.update(kwargs)
  return argparse.Namespace(**defaults)


def _ctx(tmp_path: Path, **kwargs) -> CLIContext:
  return CLIContext(
    workspace=tmp_path,
    output=Output(use_json=True),
    **kwargs,
  )


class TestRequire:
  def test_raises_without_generator(self) -> None:
    ctx = _ctx(Path('/ws'))
    with pytest.raises(ValueError, match='no generator configured'):
      _require_generator(ctx)

  def test_raises_without_judge(self) -> None:
    ctx = _ctx(Path('/ws'))
    with pytest.raises(ValueError, match='no judge configured'):
      _require_judge(ctx)

  def test_passes_when_present(self) -> None:
    ctx = _ctx(Path('/ws'), generator=object())
    _require_generator(ctx)


class TestLoadGeneratorConfig:
  def test_loads_json_file(self, tmp_path: Path) -> None:
    config_path = _write_gen_config(tmp_path)
    args = _args()
    config = _load_generator_config(str(config_path), args)
    assert config.dataset_id == 'test-ds'
    assert config.total_count == 10

  def test_applies_total_count_override(self, tmp_path: Path) -> None:
    config_path = _write_gen_config(tmp_path)
    args = _args(total_count=50)
    config = _load_generator_config(str(config_path), args)
    assert config.total_count == 50

  def test_applies_seed_override(self, tmp_path: Path) -> None:
    config_path = _write_gen_config(tmp_path)
    args = _args(seed=99)
    config = _load_generator_config(str(config_path), args)
    assert config.seed == 99

  def test_zero_means_no_override(self, tmp_path: Path) -> None:
    config_path = _write_gen_config(tmp_path)
    args = _args(total_count=0, seed=0)
    config = _load_generator_config(str(config_path), args)
    assert config.total_count == 10
    assert config.seed == 42


class TestLoadJudgeItems:
  def test_loads_jsonl(self, tmp_path: Path) -> None:
    items_path = tmp_path / 'items.jsonl'
    item = {
      'id': 'item-1',
      'turns': [{'role': 'user', 'content': 'hello'}],
      'custom': {},
    }
    items_path.write_text(json.dumps(item) + '\n', encoding='utf-8')
    items = _load_judge_items(str(items_path))
    assert len(items) == 1
    assert items[0].id == 'item-1'

  def test_skips_blank_lines(self, tmp_path: Path) -> None:
    items_path = tmp_path / 'items.jsonl'
    item = {
      'id': 'item-1',
      'turns': [{'role': 'user', 'content': 'hello'}],
      'custom': {},
    }
    items_path.write_text(
      json.dumps(item) + '\n\n' + json.dumps(item) + '\n',
      encoding='utf-8',
    )
    items = _load_judge_items(str(items_path))
    assert len(items) == 2


class TestHandleGenerateRun:
  def test_calls_generator_run(self, tmp_path: Path) -> None:
    config_path = _write_gen_config(tmp_path)
    gen = MagicMock()
    gen.run = MagicMock(return_value={'total_items': 10})
    ctx = _ctx(tmp_path, generator=gen)
    args = _args(ai_config=str(config_path))
    GenerateRun().forward(ctx, args)
    gen.run.assert_called_once()

  def test_outputs_result(self, tmp_path: Path) -> None:
    config_path = _write_gen_config(tmp_path)
    gen = MagicMock()
    gen.run = MagicMock(return_value={'total_items': 5})
    ctx = _ctx(tmp_path, generator=gen)
    ctx.output = MagicMock()
    args = _args(ai_config=str(config_path))
    GenerateRun().forward(ctx, args)
    ctx.output.result.assert_called()


class TestHandleGenerateDryRun:
  def test_calls_dry_run_sync(self, tmp_path: Path) -> None:
    config_path = _write_gen_config(tmp_path)
    gen = MagicMock()
    gen.dry_run = MagicMock(return_value={'total_slots': 10})
    ctx = _ctx(tmp_path, generator=gen)
    ctx.output = MagicMock()
    args = _args(ai_config=str(config_path))
    GenerateCommand().dry_run(ctx, args)
    gen.dry_run.assert_called_once()


class TestHandleJudgeRun:
  def test_calls_judge_run(self, tmp_path: Path) -> None:
    items_path = tmp_path / 'items.jsonl'
    item = {
      'id': 'item-1',
      'turns': [{'role': 'user', 'content': 'hello'}],
      'custom': {},
    }
    items_path.write_text(json.dumps(item) + '\n', encoding='utf-8')
    judge = MagicMock()
    judge.run = MagicMock(return_value={'summary': {}})
    ctx = _ctx(tmp_path, judge=judge)
    ctx.output = MagicMock()
    args = _args(judge_input=str(items_path), num_parallel=0, max_rpm=0)
    JudgeRun().forward(ctx, args)
    judge.run.assert_called_once()
