"""Debug workflow: collect data, classify failures, and summarize findings.

Commands delegate to the Trainer for module resolution.
"""

from autopilot.cli.command import Command, subcommand
from autopilot.cli.context import CLIContext
from autopilot.core.config import resolve_experiment_dir
from autopilot.core.models import Datum
from autopilot.tracking.manifest import load_manifest
from typing import Any
import argparse


class DebugCommand(Command):
  name = 'debug'
  help = 'Debug data collection and analysis'

  @subcommand('collect', help='Collect debug data')
  def collect(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    slug = ctx.experiment
    if not slug:
      raise ValueError('experiment slug required (--experiment)')
    exp_dir = resolve_experiment_dir(ctx.workspace, slug)

    if ctx.dry_run:
      observation = Datum(success=True, metadata={'dry_run': True})
    elif ctx.module:
      runtime_ctx: dict[str, Any] = {
        'workspace': str(ctx.workspace),
        'dry_run': ctx.dry_run,
        'experiment_dir': str(exp_dir),
      }
      params: dict[str, Any] = {'command': 'debug'}
      observation = ctx.module(runtime_ctx, params)
    else:
      observation = Datum(
        success=False,
        error_message='no module configured for debug',
      )

    ctx.output.result(
      {
        'command': 'debug',
        'success': observation.success,
        'error': observation.error_message if not observation.success else None,
      },
      ok=observation.success,
    )

  @subcommand('classify', help='Classify failure modes')
  def classify(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    slug = ctx.experiment
    if not slug:
      raise ValueError('experiment slug required (--experiment)')
    ctx.output.info(f'Classifying failures for {slug!r}...')
    ctx.output.result({'experiment': slug})

  @subcommand('summarize', help='Summarize debug session')
  def summarize(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    slug = ctx.experiment
    if not slug:
      raise ValueError('experiment slug required (--experiment)')
    exp_dir = resolve_experiment_dir(ctx.workspace, slug)
    manifest = load_manifest(exp_dir)

    ctx.output.result(
      {
        'experiment': slug,
        'idea': manifest.idea,
        'decision': manifest.decision,
        'decision_reason': manifest.decision_reason,
      }
    )
