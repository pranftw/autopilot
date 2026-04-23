"""Agent session management: run tasks, list sessions, inspect session state.

Needs a session persistence layer before these subcommands can be fully
implemented. Each subcommand documents its intended behavior.
"""

from autopilot.cli.command import Command, argument, subcommand
from autopilot.cli.context import CLIContext
import argparse


class AgentCommand(Command):
  """Run and manage agent sessions."""

  name = 'agent'
  help = 'agent operations'

  @argument('--task', default=None, help='task description')
  @argument('--session', default=None, help='session ID')
  @subcommand('run', help='run an agent task')
  def run_agent(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Create and run an agent session for a given task.

    Needs: session persistence layer, agent runtime integration,
    task queue management.
    """
    raise NotImplementedError('agent sessions not yet implemented')

  @subcommand('list', help='list agent sessions')
  def list_sessions(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """List all existing agent sessions with their status.

    Needs: session persistence layer with list/filter support.
    """
    raise NotImplementedError('agent sessions not yet implemented')

  @argument('--session', default=None, help='session ID to inspect')
  @subcommand('session', help='inspect agent session')
  def session_info(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Inspect a specific agent session: task, status, outputs.

    Needs: session persistence layer, session artifact reader.
    """
    raise NotImplementedError('agent sessions not yet implemented')
