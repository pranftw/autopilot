"""Tests for autopilot.ai.coding."""

from autopilot.ai.coding import ClaudeCodeAgent
from autopilot.core.errors import AgentError
from unittest.mock import MagicMock, patch
import json
import pytest


def _ok_stdout(data: dict) -> str:
  return json.dumps(data)


def test_output_format_json():
  agent = ClaudeCodeAgent()
  with patch('autopilot.ai.coding.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=0, stdout=_ok_stdout({'result': ''}))
    agent.forward('p')
  cmd = run_mock.call_args[0][0]
  i = cmd.index('--output-format')
  assert cmd[i + 1] == 'json'


def test_prompt_in_command():
  agent = ClaudeCodeAgent()
  with patch('autopilot.ai.coding.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=0, stdout=_ok_stdout({'result': ''}))
    agent.forward('my prompt')
  cmd = run_mock.call_args[0][0]
  i = cmd.index('-p')
  assert cmd[i + 1] == 'my prompt'


def test_resume_with_session_id():
  agent = ClaudeCodeAgent()
  with patch('autopilot.ai.coding.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=0, stdout=_ok_stdout({'result': ''}))
    agent.forward('p', context={'session_id': 'sess-123'})
  cmd = run_mock.call_args[0][0]
  i = cmd.index('--resume')
  assert cmd[i + 1] == 'sess-123'


def test_resume_absent_without_session():
  agent = ClaudeCodeAgent()
  with patch('autopilot.ai.coding.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=0, stdout=_ok_stdout({'result': ''}))
    agent.forward('p')
  cmd = run_mock.call_args[0][0]
  assert '--resume' not in cmd


def test_allowed_tools_from_constructor():
  agent = ClaudeCodeAgent(allowed_tools=['Bash', 'Read'])
  with patch('autopilot.ai.coding.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=0, stdout=_ok_stdout({'result': ''}))
    agent.forward('p')
  cmd = run_mock.call_args[0][0]
  i = cmd.index('--allowedTools')
  assert cmd[i + 1] == 'Bash,Read'


def test_allowed_tools_from_context_overrides():
  agent = ClaudeCodeAgent(allowed_tools=['Bash', 'Read'])
  with patch('autopilot.ai.coding.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=0, stdout=_ok_stdout({'result': ''}))
    agent.forward('p', context={'allowed_tools': ['Edit']})
  cmd = run_mock.call_args[0][0]
  i = cmd.index('--allowedTools')
  assert cmd[i + 1] == 'Edit'


def test_permission_mode():
  agent = ClaudeCodeAgent(permission_mode='acceptEdits')
  with patch('autopilot.ai.coding.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=0, stdout=_ok_stdout({'result': ''}))
    agent.forward('p')
  cmd = run_mock.call_args[0][0]
  i = cmd.index('--permission-mode')
  assert cmd[i + 1] == 'acceptEdits'


def test_append_system_prompt_from_constructor():
  agent = ClaudeCodeAgent(append_system_prompt='Be concise')
  with patch('autopilot.ai.coding.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=0, stdout=_ok_stdout({'result': ''}))
    agent.forward('p')
  cmd = run_mock.call_args[0][0]
  i = cmd.index('--append-system-prompt')
  assert cmd[i + 1] == 'Be concise'


def test_system_prompt_from_context_overrides():
  agent = ClaudeCodeAgent(append_system_prompt='From ctor')
  with patch('autopilot.ai.coding.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=0, stdout=_ok_stdout({'result': ''}))
    agent.forward('p', context={'system_prompt': 'From ctx'})
  cmd = run_mock.call_args[0][0]
  i = cmd.index('--append-system-prompt')
  assert cmd[i + 1] == 'From ctx'


def test_model_flag():
  agent = ClaudeCodeAgent(model='claude-sonnet-4-20250514')
  with patch('autopilot.ai.coding.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=0, stdout=_ok_stdout({'result': ''}))
    agent.forward('p')
  cmd = run_mock.call_args[0][0]
  i = cmd.index('--model')
  assert cmd[i + 1] == 'claude-sonnet-4-20250514'


def test_cwd_passed_to_subprocess():
  agent = ClaudeCodeAgent(cwd='/tmp/ws')
  with patch('autopilot.ai.coding.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=0, stdout=_ok_stdout({'result': ''}))
    agent.forward('p')
  kwargs = run_mock.call_args[1]
  assert kwargs['cwd'] == '/tmp/ws'


def test_successful_json_parsing():
  agent = ClaudeCodeAgent()
  with patch('autopilot.ai.coding.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(
      returncode=0,
      stdout='{"result":"ok","session_id":"s1"}',
    )
    result = agent.forward('p')
  assert result.output == 'ok'
  assert result.session_id == 's1'


def test_nonzero_exit_raises_agent_error():
  agent = ClaudeCodeAgent()
  with patch('autopilot.ai.coding.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=1, stdout='', stderr='err')
    with pytest.raises(AgentError, match='claude exited with code 1'):
      agent.forward('p')


def test_malformed_json_raises_agent_error():
  agent = ClaudeCodeAgent()
  with patch('autopilot.ai.coding.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=0, stdout='not json')
    with pytest.raises(AgentError, match='failed to parse claude output'):
      agent.forward('p')


def test_binary_not_found_raises_agent_error():
  agent = ClaudeCodeAgent()
  with patch('autopilot.ai.coding.subprocess.run') as run_mock:
    run_mock.side_effect = FileNotFoundError()
    with pytest.raises(AgentError, match='claude binary not found'):
      agent.forward('p')
