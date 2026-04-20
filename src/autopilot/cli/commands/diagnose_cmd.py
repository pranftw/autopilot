"""Diagnose command -- read diagnosis artifacts and cross-reference memory."""

from autopilot.cli.command import Command, argument, subcommand
from autopilot.core.memory import FileMemory
from autopilot.core.stage_io import read_epoch_artifact, read_epoch_artifact_lines
from typing import Any
import argparse


class DiagnoseCommand(Command):
  """Diagnose failures using trace artifacts."""

  name = 'diagnose'
  help = 'diagnose failures'

  @argument('--category', default='', help='filter by failure category')
  @argument('--node', default='', help='filter by node')
  @subcommand('run', help='run diagnosis on epoch artifacts')
  def run_diagnose(self, ctx: Any, args: argparse.Namespace) -> None:
    epoch = args.epoch or ctx.epoch
    if not epoch:
      ctx.output.error('--epoch is required')
      return

    exp_dir = ctx.experiment_dir()
    diagnoses = read_epoch_artifact_lines(exp_dir, epoch, 'trace_diagnoses.jsonl')

    if args.category:
      diagnoses = [d for d in diagnoses if d.get('category') == args.category]
    if args.node:
      diagnoses = [d for d in diagnoses if d.get('node') == args.node]

    memory = FileMemory(exp_dir)
    similar_fixes: list[dict[str, Any]] = []
    for d in diagnoses:
      records = memory.recall(
        category=d.get('category', ''),
        node=d.get('node', ''),
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
  def heatmap(self, ctx: Any, args: argparse.Namespace) -> None:
    epoch = args.epoch or ctx.epoch
    if not epoch:
      ctx.output.error('--epoch is required')
      return

    exp_dir = ctx.experiment_dir()
    data = read_epoch_artifact(exp_dir, epoch, 'node_heatmap.json')

    result: dict[str, Any] = {
      'epoch': epoch,
      'heatmap': data or {},
    }
    ctx.output.result(result)
