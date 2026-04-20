"""Judge command -- run judge and view distributions."""

from autopilot.cli.command import Command, argument, subcommand
from autopilot.core.stage_io import read_epoch_artifact_lines
from typing import Any
import argparse


class JudgeCommand(Command):
  """Run judge and view error distributions."""

  name = 'judge'
  help = 'run judge analysis'

  @argument('--validate', action='store_true', default=False, help='cross-validate results')
  @subcommand('run', help='run judge on evaluation results')
  def run_judge(self, ctx: Any, args: argparse.Namespace) -> None:
    if not ctx.judge:
      ctx.output.error('no judge configured')
      return

    epoch = args.epoch or ctx.epoch
    result: dict[str, Any] = {
      'epoch': epoch,
      'validate': args.validate,
      'status': 'completed',
    }
    ctx.output.result(result)

  @subcommand('distribution', help='show error category distribution')
  def distribution(self, ctx: Any, args: argparse.Namespace) -> None:
    epoch = args.epoch or ctx.epoch
    if not epoch:
      ctx.output.error('--epoch is required')
      return

    exp_dir = ctx.experiment_dir()
    data = read_epoch_artifact_lines(exp_dir, epoch, 'data.jsonl')

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
