"""Agent command -- run and manage agent sessions."""

from autopilot.cli.command import Command, argument, subcommand
from typing import Any
import argparse


class AgentCommand(Command):
  """Run and manage agent sessions."""

  name = 'agent'
  help = 'agent operations'

  @argument('--task', default='', help='task description')
  @argument('--session', default='', help='session ID')
  @subcommand('run', help='run an agent task')
  def run_agent(self, ctx: Any, args: argparse.Namespace) -> None:
    task = args.task
    if not task:
      ctx.output.error('--task is required')
      return

    result: dict[str, Any] = {
      'task': task,
      'session': args.session or 'new',
      'status': 'not_implemented',
    }
    ctx.output.result(result)

  @subcommand('list', help='list agent sessions')
  def list_sessions(self, ctx: Any, args: argparse.Namespace) -> None:
    ctx.output.result({'sessions': [], 'count': 0})

  @argument('--session', default='', help='session ID to inspect')
  @subcommand('session', help='inspect agent session')
  def session_info(self, ctx: Any, args: argparse.Namespace) -> None:
    session = args.session
    if not session:
      ctx.output.error('--session is required')
      return

    result: dict[str, Any] = {
      'session': session,
      'status': 'not_found',
    }
    ctx.output.result(result)
