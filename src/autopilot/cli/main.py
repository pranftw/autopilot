"""AutoPilot CLI entry point.

AutoPilotCLI is the default CLI with all built-in commands.
Project dispatch is internal to CLI.run() via __init_subclass__ registry.
"""

from autopilot.cli.command import CLI
from autopilot.cli.commands.ai import AICommand
from autopilot.cli.commands.checkout import CheckoutCommand
from autopilot.cli.commands.dataset import DatasetCommand
from autopilot.cli.commands.debug import DebugCommand
from autopilot.cli.commands.diagnose import DiagnoseCommand
from autopilot.cli.commands.execute import ExecuteCommand
from autopilot.cli.commands.experiment.command import ExperimentCommand
from autopilot.cli.commands.optimize import OptimizeCommand
from autopilot.cli.commands.policy import PolicyCommand
from autopilot.cli.commands.project import ProjectCommand
from autopilot.cli.commands.propose import ProposeCommand
from autopilot.cli.commands.query import QueryCommand
from autopilot.cli.commands.recommend import RecommendCommand
from autopilot.cli.commands.report.command import ReportCommand
from autopilot.cli.commands.stabilize import StabilizeCommand
from autopilot.cli.commands.status import StatusCommand
from autopilot.cli.commands.store.command import StoreCommand
from autopilot.cli.commands.trace import TraceCommand
from autopilot.cli.commands.track import TrackCommand
from autopilot.cli.commands.tree import TreeCommand
from autopilot.cli.commands.undo import UndoGuideCommand
from autopilot.cli.commands.workspace import WorkspaceCommand
from typing import Any
import argparse


class AutoPilotCLI(CLI):
  """The default AutoPilot CLI with all built-in commands."""

  def __init__(self, **kwargs: Any) -> None:
    """Register built-in command groups on this CLI instance.

    Args:
      **kwargs: Forwarded to ``CLI.__init__`` (e.g. ``context_exempt_commands``).
    """
    super().__init__(**kwargs)
    self.ai = AICommand()
    self.workspace = WorkspaceCommand()
    self.project = ProjectCommand()
    self.dataset = DatasetCommand()
    self.experiment = ExperimentCommand()
    self.optimize = OptimizeCommand()
    self.debug = DebugCommand()
    self.policy = PolicyCommand()
    self.report = ReportCommand()
    self.store = StoreCommand()
    self.status = StatusCommand()
    self.diagnose = DiagnoseCommand()
    self.trace = TraceCommand()
    self.propose = ProposeCommand()
    self.tree = TreeCommand()
    self.query = QueryCommand()
    self.recommend = RecommendCommand()
    self.checkout = CheckoutCommand()
    self.stabilize = StabilizeCommand()
    self.execute = ExecuteCommand()
    self.track = TrackCommand()
    self.undo_guide = UndoGuideCommand()


def build_parser() -> argparse.ArgumentParser:
  """Build the complete CLI argument parser.

  Returns:
    Root parser from a default ``AutoPilotCLI`` instance.
  """
  cli = AutoPilotCLI()
  return cli.build_parser()


def main() -> None:
  """Run the default AutoPilot CLI with process sys.argv."""
  AutoPilotCLI()()


if __name__ == '__main__':
  main()
