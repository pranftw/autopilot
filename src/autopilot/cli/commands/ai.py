"""AI eval generation and judging commands."""

from autopilot.ai.evaluation.schemas import (
  GeneratorConfig,
  JudgeConfig,
  JudgeInput,
)
from autopilot.cli.command import Command
from autopilot.cli.context import CLIContext
from autopilot.cli.messages import MSG_EPOCH_REQUIRED
from autopilot.cli.primitives import Argument, argument, subcommand
from autopilot.core.artifacts.epoch import DataArtifact
from autopilot.tracking.io import iter_jsonl_lines, read_json
from pathlib import Path
from typing import Any
import argparse
import asyncio

FALLBACK_JUDGE_CONFIG: dict[str, Any] = {
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
  'system_prompt': None,
}


def require_generator(ctx: CLIContext) -> None:
  """Raise if no generator is configured on the CLI context.

  Args:
    ctx: Current CLI context.

  Raises:
    ValueError: If ``ctx.generator`` is None.
  """
  if ctx.generator is None:
    msg = 'no generator configured -- run via: autopilot -p <project> ai generate run'
    raise ValueError(msg)


def require_judge(ctx: CLIContext) -> None:
  """Raise if no judge is configured on the CLI context.

  Args:
    ctx: Current CLI context.

  Raises:
    ValueError: If ``ctx.judge`` is None.
  """
  if ctx.judge is None:
    msg = 'no judge configured -- run via: autopilot -p <project> ai judge run'
    raise ValueError(msg)


def load_generator_config(
  config_path: str,
  args: argparse.Namespace,
) -> GeneratorConfig:
  """Load generator config from JSON, apply CLI overrides.

  Args:
    config_path: Path to a JSON file with generator settings.
    args: Parsed namespace with optional count, seed, and run overrides.

  Returns:
    Validated ``GeneratorConfig`` with CLI fields applied.

  Raises:
    ValueError: If the config file is missing or empty.
  """
  raw = read_json(Path(config_path))
  if raw is None:
    msg = f'config file not found: {config_path}'
    raise ValueError(msg)
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


def load_judge_items(input_path: str) -> list[JudgeInput]:
  """Load JSONL judge input items.

  Args:
    input_path: Path to a JSONL file; non-empty lines are parsed as ``JudgeInput``.

  Returns:
    List of validated judge inputs in file order.
  """
  return [JudgeInput.model_validate_json(line) for line in iter_jsonl_lines(Path(input_path))]


def build_judge_config(args: argparse.Namespace) -> JudgeConfig:
  """Build JudgeConfig from the judge input file's sibling config, or minimal defaults.

  Args:
    args: Namespace including ``judge_input`` and optional run overrides.

  Returns:
    Loaded or default ``JudgeConfig`` with CLI parallelism overrides applied.
  """
  config_path = Path(args.judge_input).parent / 'judge_config.json'
  if config_path.exists():
    raw = read_json(config_path)
    config = JudgeConfig.model_validate(raw)
  else:
    config = JudgeConfig.model_validate(FALLBACK_JUDGE_CONFIG)
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
  require_generator(ctx)
  generator = ctx.generator
  assert generator is not None
  config = load_generator_config(args.ai_config, args)
  output_dir = ctx.datasets_dir / config.dataset_id
  result = generator.run(config, output_dir, ctx.output)
  ctx.output.result(result)


class GenerateRun(Command):
  """Runs the full eval dataset generation pipeline."""

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
  """Resumes eval generation from an existing checkpoint file."""

  name = 'resume'
  help = 'Resume eval generation from checkpoint'
  checkpoint = Argument('--checkpoint', required=True, help='checkpoint file path')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Handle 'autopilot ai generate resume'."""
    require_generator(ctx)
    generator = ctx.generator
    assert generator is not None
    result = asyncio.run(generator.resume(Path(args.checkpoint), None, ctx.output))
    ctx.output.result(result)


class GenerateCommand(Command):
  """``autopilot ai generate`` group: dataset generation with dry-run and resume."""

  name = 'generate'
  help = 'Eval dataset generation'

  def __init__(self) -> None:
    """Wire generate subcommands (run and resume)."""
    super().__init__()
    self.run = GenerateRun()
    self.resume = GenerateResume()

  @argument('--config', required=True, dest='ai_config', help='generation config path')
  @argument('--total-count', type=int, default=0, help='override total item count')
  @argument('--seed', type=int, default=0, help='override random seed')
  @subcommand(
    'dry-run',
    help_text='Dry run: plan slots and steps without LLM calls',
  )
  def dry_run(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Handle 'autopilot ai generate dry-run'."""
    require_generator(ctx)
    generator = ctx.generator
    assert generator is not None
    config = load_generator_config(args.ai_config, args)
    result = generator.dry_run(config, ctx.output)
    ctx.output.result(result)


class JudgeRun(Command):
  """Runs eval judging on a set of input items."""

  name = 'run'
  help = 'Run eval judging'
  judge_input = Argument('--input', required=True, dest='judge_input', help='judge input file path')
  num_parallel = Argument('--num-parallel', type=int, default=0, help='override parallel workers')
  max_rpm = Argument('--max-rpm', type=int, default=0, help='override max requests per minute')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Handle 'autopilot ai judge run'."""
    require_judge(ctx)
    judge = ctx.judge
    assert judge is not None
    items = load_judge_items(args.judge_input)
    output_dir = Path(args.judge_input).parent / 'judge_output'
    config = build_judge_config(args)
    result = judge.run(items, config, output_dir, ctx.output)
    ctx.output.result(result)


class JudgeResume(Command):
  """Resumes eval judging from an existing checkpoint file."""

  name = 'resume'
  help = 'Resume eval judging from checkpoint'
  checkpoint = Argument('--checkpoint', required=True, help='checkpoint file path')
  judge_input = Argument('--input', required=True, dest='judge_input', help='judge input file path')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Handle 'autopilot ai judge resume'."""
    require_judge(ctx)
    judge = ctx.judge
    assert judge is not None
    items = load_judge_items(args.judge_input)
    result = asyncio.run(judge.resume(Path(args.checkpoint), items, None, ctx.output))
    ctx.output.result(result)


class JudgeSummarize(Command):
  """Displays an aggregated summary from judge output JSON."""

  name = 'summarize'
  help = 'Summarize judge output'
  judge_input = Argument(
    '--input', required=True, dest='judge_input', help='judge output file path'
  )

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Handle 'autopilot ai judge summarize'."""
    require_judge(ctx)
    raw = read_json(Path(args.judge_input))
    if not isinstance(raw, dict) or 'summary' not in raw:
      ctx.fail('invalid judge input: missing summary')
    ctx.output.result(raw['summary'])


class JudgeCommand(Command):
  """``autopilot ai judge`` group: run, resume, summarize, and distribution."""

  name = 'judge'
  help = 'Eval judging'

  def __init__(self) -> None:
    """Wire judge subcommands (run, resume, summarize)."""
    super().__init__()
    self.run = JudgeRun()
    self.resume = JudgeResume()
    self.summarize = JudgeSummarize()

  @subcommand('distribution', help_text='show error category distribution')
  def distribution(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Show failure-type distribution from epoch trace data."""
    epoch = args.epoch if args.epoch is not None else ctx.epoch
    if epoch is None:
      ctx.fail(MSG_EPOCH_REQUIRED)

    exp_dir = ctx.experiment_path()
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
  """Top-level ``autopilot ai`` group: dataset generation and judge workflows."""

  name = 'ai'
  help = 'AI eval generation and judging'

  def __init__(self) -> None:
    """Wire top-level ``generate`` and ``judge`` groups."""
    super().__init__()
    self.generate = GenerateCommand()
    self.judge = JudgeCommand()
