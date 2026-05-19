"""Tests for agent CLI command removal (BUG-071).

The agent command has been removed entirely. These tests confirm
the module is gone and no NotImplementedError surfaces.
"""

from autopilot.cli.main import AutoPilotCLI
from pathlib import Path


class TestAgentCommandRemoved:
  def test_agent_not_registered(self) -> None:
    """AutoPilotCLI no longer registers an 'agent' command."""
    cli = AutoPilotCLI()
    assert 'agent' not in cli.commands

  def test_agent_module_file_deleted(self) -> None:
    """The agent command module file no longer exists on disk."""
    repo = Path(__file__).resolve().parent.parent.parent
    agent_file = repo / 'src' / 'autopilot' / 'cli' / 'commands' / 'agent.py'
    assert not agent_file.exists()
