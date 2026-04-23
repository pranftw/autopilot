"""Tests for agent CLI command."""

from autopilot.cli.commands.agent import AgentCommand
from unittest.mock import MagicMock
import pytest


class TestAgentCommand:
  def test_instantiates(self):
    cmd = AgentCommand()
    assert cmd.name == 'agent'

  def test_run_raises_not_implemented(self):
    cmd = AgentCommand()
    ctx = MagicMock()
    args = MagicMock(task='optimize rules', session='')
    with pytest.raises(NotImplementedError, match='agent sessions not yet implemented'):
      cmd.run_agent(ctx, args)

  def test_list_raises_not_implemented(self):
    cmd = AgentCommand()
    ctx = MagicMock()
    args = MagicMock()
    with pytest.raises(NotImplementedError, match='agent sessions not yet implemented'):
      cmd.list_sessions(ctx, args)

  def test_session_raises_not_implemented(self):
    cmd = AgentCommand()
    ctx = MagicMock()
    args = MagicMock(session='some-id')
    with pytest.raises(NotImplementedError, match='agent sessions not yet implemented'):
      cmd.session_info(ctx, args)
