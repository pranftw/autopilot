"""Diagnose command -- read diagnosis artifacts."""

from autopilot.cli.command import Command
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import resolve_command_epoch
from autopilot.cli.primitives import argument, subcommand
from autopilot.core.artifacts.epoch import DiagnosesArtifact, HeatmapArtifact
from typing import Any
import argparse


class DiagnoseCommand(Command):
  """Diagnose failures using trace artifacts."""

  name = 'diagnose'
  help = 'diagnose failures'

  @argument('--category', default=None, help='filter by failure category')
  @argument('--node', default=None, help='filter by node')
  @subcommand('run', help_text='run diagnosis on epoch artifacts')
  def run_diagnose(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Run diagnosis on epoch artifacts."""
    epoch = resolve_command_epoch(ctx, args)

    exp_dir = ctx.experiment_path()
    diagnoses = DiagnosesArtifact().read_raw(exp_dir, epoch=epoch)

    if args.category:
      diagnoses = [d for d in diagnoses if d.get('category') == args.category]
    if args.node:
      diagnoses = [d for d in diagnoses if d.get('node') == args.node]

    result: dict[str, Any] = {
      'epoch': epoch,
      'diagnoses': diagnoses,
    }
    ctx.output.result(result)

  @subcommand('heatmap', help_text='show node error heatmap')
  def heatmap(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Display the node error heatmap for an epoch."""
    epoch = resolve_command_epoch(ctx, args)

    exp_dir = ctx.experiment_path()
    data = HeatmapArtifact().read_raw(exp_dir, epoch=epoch)

    if data is None:
      ctx.fail(f'no heatmap artifact found for epoch {epoch}')

    result: dict[str, Any] = {
      'epoch': epoch,
      'heatmap': data,
    }
    ctx.output.result(result)
