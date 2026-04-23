"""Debug data collection: inspect module state and forward pass data.

The collect subcommand runs the configured Module's forward pass in debug
mode and reports the result. Requires a configured module (via project CLI).
"""

from autopilot.cli.command import Command, subcommand
from autopilot.cli.context import CLIContext
from autopilot.core.config import resolve_experiment_dir
from typing import Any
import argparse


class DebugCommand(Command):
  """Collect debug data from the Module for inspection."""

  name = 'debug'
  help = 'Debug data collection and analysis'

  @subcommand('collect', help='Collect debug data')
  def collect(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Run the module in debug mode and report the observation."""
    slug = ctx.experiment
    if not slug:
      ctx.output.error('experiment slug required (--experiment)')
      return
    exp_dir = resolve_experiment_dir(ctx.workspace, slug)

    if not ctx.module:
      ctx.output.error('no module configured for debug')
      return

    if ctx.dry_run:
      ctx.output.result({'command': 'debug', 'dry_run': True, 'success': True})
      return

    runtime_ctx: dict[str, Any] = {
      'workspace': str(ctx.workspace),
      'dry_run': ctx.dry_run,
      'experiment_dir': str(exp_dir),
    }
    params: dict[str, Any] = {'command': 'debug'}
    observation = ctx.module(runtime_ctx, params)

    ctx.output.result(
      {
        'command': 'debug',
        'success': observation.success,
        'error': observation.error_message if not observation.success else None,
      },
      ok=observation.success,
    )
