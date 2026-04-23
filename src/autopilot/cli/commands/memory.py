"""Memory command -- query and manage learning memory."""

from autopilot.cli.command import Command, argument, subcommand
from autopilot.cli.context import CLIContext
from autopilot.core.memory import FileMemory
from typing import Any
import argparse
import json


class MemoryCommand(Command):
  """Query and manage learning memory."""

  name = 'memory'
  help = 'learning memory operations'

  @argument('--category', default=None, help='filter by category')
  @argument('--node', default=None, help='filter by node')
  @argument('--outcome', default=None, help='filter by outcome')
  @argument('--strategy', default=None, help='filter by strategy')
  @subcommand('query', help='query past learnings')
  def query(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Query past learnings with optional filters."""
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
  @argument('--category', default=None, help='entry category')
  @argument('--strategy', default=None, help='strategy name')
  @argument('--node', default=None, help='node name')
  @argument('--content', default=None, help='human-readable note')
  @argument('--metrics', default=None, help='JSON metrics dict')
  @subcommand('record', help='record a learning')
  def record(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Record a new learning entry in the memory store."""
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

  @argument('--metric', default=None, help='metric name for trends')
  @argument('--window', type=int, default=5, help='window size')
  @subcommand('trends', help='show metric trends')
  def trends(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Show metric trend analysis over recent epochs."""
    exp_dir = ctx.experiment_dir()
    memory = FileMemory(exp_dir)
    result = memory.trends(metric=args.metric, window=args.window)
    ctx.output.result(result.to_dict())

  @subcommand('context', help='get full decision context')
  def context(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Get the full decision context for the current epoch."""
    exp_dir = ctx.experiment_dir()
    memory = FileMemory(exp_dir)
    epoch = args.epoch or ctx.epoch
    result = memory.context(epoch=epoch)
    ctx.output.result(result.to_dict())
