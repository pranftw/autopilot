"""Tests for agent CLI command."""

from autopilot.cli.commands.agent_cmd import AgentCommand
from autopilot.cli.output import Output
from unittest.mock import MagicMock
import json


class TestAgentCommand:
  def test_instantiates(self):
    cmd = AgentCommand()
    assert cmd.name == 'agent'

  def test_run_no_task(self):
    cmd = AgentCommand()
    ctx = MagicMock()
    ctx.output = MagicMock()
    args = MagicMock(task='', session='')
    cmd.run_agent(ctx, args)
    ctx.output.error.assert_called_once()
    ctx.output.result.assert_not_called()

  def test_run_with_task(self, capsys):
    cmd = AgentCommand()
    ctx = MagicMock()
    ctx.output = Output(use_json=True)
    args = MagicMock(task='optimize rules', session='')
    cmd.run_agent(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert envelope['result']['task'] == 'optimize rules'
    assert envelope['result']['status'] == 'not_implemented'

  def test_run_with_session(self, capsys):
    cmd = AgentCommand()
    ctx = MagicMock()
    ctx.output = Output(use_json=True)
    args = MagicMock(task='fix errors', session='sess-123')
    cmd.run_agent(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['session'] == 'sess-123'

  def test_list_sessions(self, capsys):
    cmd = AgentCommand()
    ctx = MagicMock()
    ctx.output = Output(use_json=True)
    args = MagicMock()
    cmd.list_sessions(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['sessions'] == []
    assert envelope['result']['count'] == 0

  def test_session_no_id(self):
    cmd = AgentCommand()
    ctx = MagicMock()
    ctx.output = MagicMock()
    args = MagicMock(session='')
    cmd.session_info(ctx, args)
    ctx.output.error.assert_called_once()
    ctx.output.result.assert_not_called()

  def test_session_not_found(self, capsys):
    cmd = AgentCommand()
    ctx = MagicMock()
    ctx.output = Output(use_json=True)
    args = MagicMock(session='nonexistent')
    cmd.session_info(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['status'] == 'not_found'
