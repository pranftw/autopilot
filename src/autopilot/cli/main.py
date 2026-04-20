"""AutoPilot CLI entry point.

AutoPilotCLI is the default CLI with all built-in commands.
Project dispatch is internal to CLI.run() via __init_subclass__ registry.
"""

from autopilot.cli.command import CLI
from autopilot.cli.commands.agent_cmd import AgentCommand
from autopilot.cli.commands.ai import AICommand
from autopilot.cli.commands.dataset import DatasetCommand
from autopilot.cli.commands.debug import DebugCommand
from autopilot.cli.commands.diagnose_cmd import DiagnoseCommand
from autopilot.cli.commands.experiment import ExperimentCommand
from autopilot.cli.commands.judge_cmd import JudgeCommand
from autopilot.cli.commands.memory_cmd import MemoryCommand
from autopilot.cli.commands.optimize import OptimizeCommand
from autopilot.cli.commands.policy import PolicyCommand
from autopilot.cli.commands.project_cmd import ProjectCommand
from autopilot.cli.commands.promote import PromoteCommand
from autopilot.cli.commands.propose_cmd import ProposeCommand
from autopilot.cli.commands.report import ReportCommand
from autopilot.cli.commands.status_cmd import StatusCommand
from autopilot.cli.commands.store_cmd import StoreCommand
from autopilot.cli.commands.trace_cmd import TraceCommand
from autopilot.cli.commands.workspace import WorkspaceCommand
from autopilot.cli.context import CLIContext
from pathlib import Path
import argparse
import runpy
import sys
import traceback


class AutoPilotCLI(CLI):
  """The default AutoPilot CLI with all built-in commands."""

  def __init__(self) -> None:
    super().__init__()
    self.ai = AICommand()
    self.workspace = WorkspaceCommand()
    self.project = ProjectCommand()
    self.dataset = DatasetCommand()
    self.experiment = ExperimentCommand()
    self.optimize = OptimizeCommand()
    self.debug = DebugCommand()
    self.policy = PolicyCommand()
    self.report = ReportCommand()
    self.promote = PromoteCommand()
    self.store = StoreCommand()
    self.status = StatusCommand()
    self.judge = JudgeCommand()
    self.diagnose = DiagnoseCommand()
    self.trace = TraceCommand()
    self.propose = ProposeCommand()
    self.memory = MemoryCommand()
    self.agent = AgentCommand()


def _dispatch(ctx: CLIContext, args: argparse.Namespace) -> None:
  """Shared handler dispatch with error handling."""
  handler = getattr(args, 'handler', None)
  if handler is None:
    build_parser().print_help()
    sys.exit(1)
  try:
    handler(ctx, args)
  except Exception as e:
    ctx.output.error(str(e))
    if ctx.verbose:
      traceback.print_exc()
    sys.exit(1)


def _run_project_cli(project_dir: Path, cli_script: Path) -> None:
  """Execute project's cli.py via runpy."""
  sys.path.insert(0, str(project_dir))
  try:
    runpy.run_path(str(cli_script), run_name='__main__')
  finally:
    sys.path.pop(0)


def build_parser() -> argparse.ArgumentParser:
  """Build the complete CLI argument parser."""
  cli = AutoPilotCLI()
  return cli.build_parser()


def main() -> None:
  AutoPilotCLI()()


if __name__ == '__main__':
  main()
