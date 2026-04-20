"""AI eval generation and judging commands."""

from autopilot.ai.models import GeneratorConfig, JudgeConfig, JudgeInput
from autopilot.cli.command import Argument, Command, argument, subcommand
from autopilot.cli.context import CLIContext
from autopilot.core.config import load_json
from pathlib import Path
import argparse
import asyncio


def _require(ctx: CLIContext, component: str) -> None:
  """Raise if the named component is not set on context."""
  if getattr(ctx, component, None) is None:
    raise ValueError(
      f'no {component} configured -- run via: autopilot -p <project> ai {component} run ...'
    )


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
      system_prompt='',
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
  _require(ctx, 'generator')
  config = _load_generator_config(args.ai_config, args)
  output_dir = ctx.datasets_dir / config.dataset_id
  result = ctx.generator.run(config, output_dir, ctx.output)
  ctx.output.result(result)


class GenerateRun(Command):
  name = 'run'
  help = 'Run eval generation'
  include_project_config = False
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
    _require(ctx, 'generator')
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
    include_project_config=False,
  )
  def dry_run(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Handle 'autopilot ai generate dry-run'."""
    _require(ctx, 'generator')
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
    _require(ctx, 'judge')
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
    _require(ctx, 'judge')
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
    _require(ctx, 'judge')
    raw = load_json(Path(args.judge_input))
    ctx.output.result(raw.get('summary', {}))


class JudgeCommand(Command):
  name = 'judge'
  help = 'Eval judging'

  def __init__(self) -> None:
    super().__init__()
    self.run = JudgeRun()
    self.resume = JudgeResume()
    self.summarize = JudgeSummarize()


class AICommand(Command):
  name = 'ai'
  help = 'AI eval generation and judging'

  def __init__(self) -> None:
    super().__init__()
    self.generate = GenerateCommand()
    self.judge = JudgeCommand()
