"""Status command -- comprehensive experiment health overview.

Delegates to core/status.py for the actual status gathering logic,
using Forest for experiment resolution.
"""

from autopilot.cli.command import Command
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import load_forest
from autopilot.core.errors import TrackingError
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
      ctx.fail('no experiment specified (use --experiment)')

    forest = load_forest(ctx)
    try:
      result = get_experiment_status(forest, experiment)
    except TrackingError as e:
      ctx.fail(str(e))

    ctx.output.result(result)
