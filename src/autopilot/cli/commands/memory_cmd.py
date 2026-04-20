"""Memory command -- query and manage learning memory."""

from autopilot.cli.command import Command, argument, subcommand
from autopilot.core.memory import FileMemory
from typing import Any
import argparse
import json


class MemoryCommand(Command):
  """Query and manage learning memory."""

  name = 'memory'
  help = 'learning memory operations'

  @argument('--category', default='', help='filter by category')
  @argument('--node', default='', help='filter by node')
  @argument('--outcome', default='', help='filter by outcome')
  @argument('--strategy', default='', help='filter by strategy')
  @subcommand('query', help='query past learnings')
  def query(self, ctx: Any, args: argparse.Namespace) -> None:
    exp_dir = ctx.experiment_dir()
    memory = FileMemory(exp_dir)
    filters: dict[str, Any] = {}
    if args.category:
      filters['category'] = args.category
    if args.node:
      filters['node'] = args.node
    if args.outcome:
      filters['outcome'] = args.outcome
    if args.strategy:
      filters['strategy'] = args.strategy
    if args.epoch:
      filters['epoch'] = args.epoch

    records = memory.recall(**filters)
    ctx.output.result(
      {
        'records': [r.to_dict() for r in records],
        'count': len(records),
      }
    )

  @argument('--outcome', required=True, help='worked|failed|partial')
  @argument('--category', default='', help='entry category')
  @argument('--strategy', default='', help='strategy name')
  @argument('--node', default='', help='node name')
  @argument('--content', default='', help='human-readable note')
  @argument('--metrics', default='', help='JSON metrics dict')
  @subcommand('record', help='record a learning')
  def record(self, ctx: Any, args: argparse.Namespace) -> None:
    exp_dir = ctx.experiment_dir()
    memory = FileMemory(exp_dir)

    metrics: dict[str, float] = {}
    if args.metrics:
      metrics = json.loads(args.metrics)

    memory.learn(
      epoch=args.epoch,
      outcome=args.outcome,
      category=args.category,
      strategy=args.strategy,
      node=args.node,
      content=args.content,
      metrics=metrics,
    )
    ctx.output.result({'status': 'recorded', 'epoch': args.epoch})

  @argument('--metric', default='', help='metric name for trends')
  @argument('--window', type=int, default=5, help='window size')
  @subcommand('trends', help='show metric trends')
  def trends(self, ctx: Any, args: argparse.Namespace) -> None:
    exp_dir = ctx.experiment_dir()
    memory = FileMemory(exp_dir)
    result = memory.trends(metric=args.metric, window=args.window)
    ctx.output.result(result.to_dict())

  @subcommand('context', help='get full decision context')
  def context(self, ctx: Any, args: argparse.Namespace) -> None:
    exp_dir = ctx.experiment_dir()
    memory = FileMemory(exp_dir)
    epoch = args.epoch or ctx.epoch
    result = memory.context(epoch=epoch)
    ctx.output.result(result.to_dict())
