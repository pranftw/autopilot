"""Diagnose command -- read diagnosis artifacts and cross-reference memory."""

from autopilot.cli.command import Command, argument, subcommand
from autopilot.cli.context import CLIContext
from autopilot.core.artifacts.epoch import DiagnosesArtifact, HeatmapArtifact
from autopilot.core.memory import FileMemory
from typing import Any
import argparse


class DiagnoseCommand(Command):
  """Diagnose failures using trace artifacts."""

  name = 'diagnose'
  help = 'diagnose failures'

  @argument('--category', default=None, help='filter by failure category')
  @argument('--node', default=None, help='filter by node')
  @subcommand('run', help='run diagnosis on epoch artifacts')
  def run_diagnose(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Run diagnosis on epoch artifacts and cross-reference memory."""
    epoch = args.epoch or ctx.epoch
    if not epoch:
      ctx.output.error('--epoch is required')
      return

    exp_dir = ctx.experiment_dir()
    diagnoses = DiagnosesArtifact().read_raw(exp_dir, epoch=epoch)

    if args.category:
      diagnoses = [d for d in diagnoses if d.get('category') == args.category]
    if args.node:
      diagnoses = [d for d in diagnoses if d.get('node') == args.node]

    memory = FileMemory(exp_dir)
    similar_fixes: list[dict[str, Any]] = []
    for d in diagnoses:
      records = memory.recall(
        category=d.get('category'),
        node=d.get('node'),
      )
      if records:
        similar_fixes.append(
          {
            'diagnosis': d,
            'past_fixes': [r.to_dict() for r in records[:3]],
          }
        )

    result: dict[str, Any] = {
      'epoch': epoch,
      'diagnoses': diagnoses,
      'similar_fixes': similar_fixes,
    }
    ctx.output.result(result)

  @subcommand('heatmap', help='show node error heatmap')
  def heatmap(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Display the node error heatmap for an epoch."""
    epoch = args.epoch or ctx.epoch
    if not epoch:
      ctx.output.error('--epoch is required')
      return

    exp_dir = ctx.experiment_dir()
    data = HeatmapArtifact().read_raw(exp_dir, epoch=epoch)

    if data is None:
      ctx.output.error(f'no heatmap artifact found for epoch {epoch}')
      return

    result: dict[str, Any] = {
      'epoch': epoch,
      'heatmap': data,
    }
    ctx.output.result(result)
