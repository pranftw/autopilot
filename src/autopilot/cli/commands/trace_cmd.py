"""Trace command -- collect and inspect execution traces."""

from autopilot.cli.command import Command, argument, subcommand
from autopilot.core.memory import FileMemory
from autopilot.core.stage_io import read_epoch_artifact_lines
from typing import Any
import argparse


class TraceCommand(Command):
  """Collect and inspect execution traces."""

  name = 'trace'
  help = 'trace collection and inspection'

  @argument('--limit', type=int, default=0, help='max items')
  @subcommand('collect', help='collect trace data from epoch')
  def collect(self, ctx: Any, args: argparse.Namespace) -> None:
    epoch = args.epoch or ctx.epoch
    if not epoch:
      ctx.output.error('--epoch is required')
      return

    exp_dir = ctx.experiment_dir()
    data = read_epoch_artifact_lines(exp_dir, epoch, 'data.jsonl')

    limit = args.limit
    if limit > 0:
      data = data[:limit]

    result: dict[str, Any] = {
      'epoch': epoch,
      'items': data,
      'count': len(data),
    }
    ctx.output.result(result)

  @argument('--node', default='', help='node/item_id to inspect')
  @argument('--depth', type=int, default=1, help='inspection depth (2+ pulls memory)')
  @subcommand('inspect', help='inspect trace for a node')
  def inspect_trace(self, ctx: Any, args: argparse.Namespace) -> None:
    node = args.node
    if not node:
      ctx.output.error('--node is required')
      return

    epoch = args.epoch or ctx.epoch
    if not epoch:
      ctx.output.error('--epoch is required')
      return

    exp_dir = ctx.experiment_dir()
    data = read_epoch_artifact_lines(exp_dir, epoch, 'data.jsonl')

    matched = []
    for idx, item in enumerate(data):
      item_id = item.get('item_id', '')
      meta_node = item.get('metadata', {}).get('node', '')
      if node in (item_id, meta_node) or node in str(item_id) or node in str(meta_node):
        matched.append(
          {
            'batch_idx': idx,
            'item_id': item_id,
            'success': item.get('success', True),
            'feedback': item.get('feedback'),
            'error_message': item.get('error_message'),
            'metadata': item.get('metadata', {}),
          }
        )

    result: dict[str, Any] = {
      'node': node,
      'epoch': epoch,
      'matches': matched,
      'count': len(matched),
    }

    if args.depth > 1:
      memory = FileMemory(exp_dir)
      records = memory.recall(node=node)
      result['memory_records'] = [r.to_dict() for r in records[:10]]

    ctx.output.result(result)
