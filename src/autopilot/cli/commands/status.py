"""Status command -- comprehensive experiment health overview.

Delegates to core/status.py for the actual status gathering logic.
"""

from autopilot.cli.command import Command
from autopilot.cli.context import CLIContext
from autopilot.core.status import get_experiment_status
import argparse


class StatusCommand(Command):
  """Show experiment status, regression state, and recent metrics."""

  name = 'status'
  help = 'show experiment status'

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Gather and display experiment status."""
    experiment = ctx.experiment
    if not experiment:
      ctx.output.error('no experiment specified (use --experiment)')
      return

    exp_dir = ctx.experiment_dir()
    try:
      result = get_experiment_status(exp_dir)
    except (FileNotFoundError, KeyError, ValueError) as e:
      ctx.output.error(f'cannot load experiment: {e}')
      return

    ctx.output.result(result)
