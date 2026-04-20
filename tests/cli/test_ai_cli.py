"""Tests for AI CLI command registration and argument parsing."""

from autopilot.cli.commands.ai import GenerateRun, JudgeRun
from autopilot.cli.main import build_parser
import pytest


class TestGenerateParser:
  def test_run_requires_config(self) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
      parser.parse_args(['ai', 'generate', 'run'])

  def test_run_accepts_all_flags(self) -> None:
    parser = build_parser()
    args = parser.parse_args(
      [
        'ai',
        'generate',
        'run',
        '--config',
        'gen.json',
        '--total-count',
        '100',
        '--seed',
        '42',
        '--num-parallel',
        '5',
        '--max-rpm',
        '50',
      ]
    )
    assert args.ai_config == 'gen.json'
    assert args.total_count == 100
    assert args.seed == 42
    assert args.num_parallel == 5
    assert args.max_rpm == 50

  def test_resume_requires_checkpoint(self) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
      parser.parse_args(['ai', 'generate', 'resume'])

  def test_dry_run_requires_config(self) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
      parser.parse_args(['ai', 'generate', 'dry-run'])


class TestJudgeParser:
  def test_run_requires_input(self) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
      parser.parse_args(['ai', 'judge', 'run'])

  def test_run_accepts_parallel_flags(self) -> None:
    parser = build_parser()
    args = parser.parse_args(
      [
        'ai',
        'judge',
        'run',
        '--input',
        'items.jsonl',
        '--num-parallel',
        '3',
        '--max-rpm',
        '30',
      ]
    )
    assert args.judge_input == 'items.jsonl'
    assert args.num_parallel == 3
    assert args.max_rpm == 30

  def test_resume_requires_both(self) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
      parser.parse_args(['ai', 'judge', 'resume', '--checkpoint', 'cp.jsonl'])

  def test_summarize_requires_input(self) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
      parser.parse_args(['ai', 'judge', 'summarize'])


class TestSubcommandRouting:
  def test_generate_run_sets_handler(self) -> None:
    parser = build_parser()
    args = parser.parse_args(
      [
        'ai',
        'generate',
        'run',
        '--config',
        'x.json',
      ]
    )
    assert isinstance(args.handler, GenerateRun)

  def test_judge_run_sets_handler(self) -> None:
    parser = build_parser()
    args = parser.parse_args(
      [
        'ai',
        'judge',
        'run',
        '--input',
        'items.jsonl',
      ]
    )
    assert isinstance(args.handler, JudgeRun)
