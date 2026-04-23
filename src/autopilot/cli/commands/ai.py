"""AI eval generation and judging commands."""

from autopilot.ai.evaluation.schemas import GeneratorConfig, JudgeConfig, JudgeInput
from autopilot.cli.command import Argument, Command, argument, subcommand
from autopilot.cli.context import CLIContext
from autopilot.core.artifacts.epoch import DataArtifact
from autopilot.core.config import load_json
from pathlib import Path
from typing import Any
import argparse
import asyncio


def _require_generator(ctx: CLIContext) -> None:
  if ctx.generator is None:
    raise ValueError('no generator configured -- run via: autopilot -p <project> ai generate run')


def _require_judge(ctx: CLIContext) -> None:
  if ctx.judge is None:
    raise ValueError('no judge configured -- run via: autopilot -p <project> ai judge run')


def _load_generator_config(
  config_path: str,
  args: argparse.Namespace,
) -> GeneratorConfig:
  """Load generator config from JSON, apply CLI overrides."""
  raw = load_json(Path(config_path))
  config = GeneratorConfig.model_validate(raw)
  if args.total_count:
    config.total_count = args.total_count
  if args.seed:
    config.seed = args.seed
  if args.num_parallel:
    config.run.num_parallel = args.num_parallel
  if args.max_rpm:
    config.run.max_rpm = args.max_rpm
  return config


def _load_judge_items(input_path: str) -> list[JudgeInput]:
  """Load JSONL judge input items."""
  items: list[JudgeInput] = []
  with open(input_path) as f:
    for line in f:
      line = line.strip()
      if line:
        items.append(JudgeInput.model_validate_json(line))
  return items


def _build_judge_config(args: argparse.Namespace) -> JudgeConfig:
  """Build JudgeConfig from the judge input file's sibling config, or minimal defaults."""
  config_path = Path(args.judge_input).parent / 'judge_config.json'
  if config_path.exists():
    raw = load_json(config_path)
    config = JudgeConfig.model_validate(raw)
  else:
    config = JudgeConfig(
      run={
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
      system_prompt=None,
    )
  if args.num_parallel:
    config.run.num_parallel = args.num_parallel
  if args.max_rpm:
    config.run.max_rpm = args.max_rpm
  return config


def _run_generate(
  ctx: CLIContext,
  args: argparse.Namespace,
) -> None:
  """Shared logic for generate subcommands."""
  _require_generator(ctx)
  config = _load_generator_config(args.ai_config, args)
  output_dir = ctx.datasets_dir / config.dataset_id
  result = ctx.generator.run(config, output_dir, ctx.output)
  ctx.output.result(result)


class GenerateRun(Command):
  name = 'run'
  help = 'Run eval generation'
  ai_config = Argument('--config', required=True, dest='ai_config', help='generation config path')
  total_count = Argument('--total-count', type=int, default=0, help='override total item count')
  seed = Argument('--seed', type=int, default=0, help='override random seed')
  num_parallel = Argument('--num-parallel', type=int, default=0, help='override parallel workers')
  max_rpm = Argument('--max-rpm', type=int, default=0, help='override max requests per minute')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Handle 'autopilot ai generate run'."""
    _run_generate(ctx, args)


class GenerateResume(Command):
  name = 'resume'
  help = 'Resume eval generation from checkpoint'
  checkpoint = Argument('--checkpoint', required=True, help='checkpoint file path')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Handle 'autopilot ai generate resume'."""
    _require_generator(ctx)
    result = asyncio.run(ctx.generator.resume(Path(args.checkpoint), None, None, ctx.output))
    ctx.output.result(result)


class GenerateCommand(Command):
  name = 'generate'
  help = 'Eval dataset generation'

  def __init__(self) -> None:
    super().__init__()
    self.run = GenerateRun()
    self.resume = GenerateResume()

  @argument('--config', required=True, dest='ai_config', help='generation config path')
  @argument('--total-count', type=int, default=0, help='override total item count')
  @argument('--seed', type=int, default=0, help='override random seed')
  @subcommand(
    'dry-run',
    help='Dry run: plan slots and steps without LLM calls',
  )
  def dry_run(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Handle 'autopilot ai generate dry-run'."""
    _require_generator(ctx)
    config = _load_generator_config(args.ai_config, args)
    result = ctx.generator.dry_run(config, ctx.output)
    ctx.output.result(result)


class JudgeRun(Command):
  name = 'run'
  help = 'Run eval judging'
  judge_input = Argument('--input', required=True, dest='judge_input', help='judge input file path')
  num_parallel = Argument('--num-parallel', type=int, default=0, help='override parallel workers')
  max_rpm = Argument('--max-rpm', type=int, default=0, help='override max requests per minute')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Handle 'autopilot ai judge run'."""
    _require_judge(ctx)
    items = _load_judge_items(args.judge_input)
    output_dir = Path(args.judge_input).parent / 'judge_output'
    config = _build_judge_config(args)
    result = ctx.judge.run(items, config, output_dir, ctx.output)
    ctx.output.result(result)


class JudgeResume(Command):
  name = 'resume'
  help = 'Resume eval judging from checkpoint'
  checkpoint = Argument('--checkpoint', required=True, help='checkpoint file path')
  judge_input = Argument('--input', required=True, dest='judge_input', help='judge input file path')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Handle 'autopilot ai judge resume'."""
    _require_judge(ctx)
    items = _load_judge_items(args.judge_input)
    result = asyncio.run(ctx.judge.resume(Path(args.checkpoint), items, None, None, ctx.output))
    ctx.output.result(result)


class JudgeSummarize(Command):
  name = 'summarize'
  help = 'Summarize judge output'
  judge_input = Argument(
    '--input', required=True, dest='judge_input', help='judge output file path'
  )

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Handle 'autopilot ai judge summarize'."""
    _require_judge(ctx)
    raw = load_json(Path(args.judge_input))
    ctx.output.result(raw['summary'])


class JudgeCommand(Command):
  name = 'judge'
  help = 'Eval judging'

  def __init__(self) -> None:
    super().__init__()
    self.run = JudgeRun()
    self.resume = JudgeResume()
    self.summarize = JudgeSummarize()

  @subcommand('distribution', help='show error category distribution')
  def distribution(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Show failure-type distribution from epoch trace data."""
    epoch = args.epoch or ctx.epoch
    if not epoch:
      ctx.output.error('--epoch is required')
      return

    exp_dir = ctx.experiment_dir()
    data = DataArtifact().read_raw(exp_dir, epoch=epoch)

    categories: dict[str, int] = {}
    for item in data:
      cat = item.get('metadata', {}).get('failure_type', 'unknown')
      if not item.get('success', True):
        categories[cat] = categories.get(cat, 0) + 1

    result: dict[str, Any] = {
      'epoch': epoch,
      'total_items': len(data),
      'failure_distribution': categories,
    }
    ctx.output.result(result)


class AICommand(Command):
  name = 'ai'
  help = 'AI eval generation and judging'

  def __init__(self) -> None:
    super().__init__()
    self.generate = GenerateCommand()
    self.judge = JudgeCommand()
