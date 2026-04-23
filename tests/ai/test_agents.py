"""Tests for autopilot.ai.agents (Agent, StepAgent, AgentResult, ClaudeCodeAgent)."""

from autopilot.ai.agents.agent import Agent, AgentResult, StepAgent
from autopilot.ai.agents.claude_code import ClaudeCodeAgent
from autopilot.ai.evaluation.steps import Step, python_step
from autopilot.core.errors import AgentError
from unittest.mock import MagicMock, patch
import json
import pytest

# agent base and stepagent tests


def test_agent_run_raises_not_implemented():
  agent = Agent()
  with pytest.raises(NotImplementedError):
    agent.run('x')


@pytest.mark.asyncio
async def test_agent_async_run_raises_not_implemented():
  agent = Agent()
  with pytest.raises(NotImplementedError):
    await agent.async_run()


def test_agent_name_default():
  assert Agent().name() == 'Agent'


def test_agent_setup_teardown_noop():
  a = Agent()
  a.setup(foo=1)
  a.teardown()


def test_agent_state_dict_and_load():
  a = Agent()
  assert a.state_dict() == {}
  a.load_state_dict({})


def test_agent_limiter_get_set():
  a = Agent()
  assert a.limiter is None
  x = object()
  a.limiter = x
  assert a.limiter is x


def test_agent_repr():
  assert repr(Agent()) == 'Agent()'


def test_subclass_of_agent_is_agent():
  class _Sub(Agent):
    def run(self, *args, **kwargs):
      return 1

    async def async_run(self, *args, **kwargs):
      return 2

  s = _Sub()
  assert isinstance(s, Agent)


def test_step_agent_is_subclass_of_agent():
  class _Workflow(StepAgent):
    @python_step('s')
    def s(self, ctx):
      return {}

  w = _Workflow()
  assert isinstance(w, Agent)
  assert isinstance(w, StepAgent)


def test_step_agent_define_steps_default_uses_collect_steps():
  class _W(StepAgent):
    @python_step('a')
    def a(self, ctx):
      return {'k': 1}

  steps = _W().define_steps(None)
  assert len(steps) == 1
  assert steps[0].name == 'a'
  assert isinstance(steps[0], Step)


def test_agent_result_defaults():
  r = AgentResult(output='hello')
  assert r.output == 'hello'
  assert r.session_id is None
  assert r.metadata == {}


def test_agent_result_all_fields():
  r = AgentResult(
    output='out',
    session_id='sid',
    metadata={'k': 1},
  )
  assert r.output == 'out'
  assert r.session_id == 'sid'
  assert r.metadata == {'k': 1}


def test_agent_result_model_dump():
  r = AgentResult(output='a', session_id='b', metadata={'c': 3})
  d = r.model_dump()
  assert d == {'output': 'a', 'session_id': 'b', 'metadata': {'c': 3}}


def test_agent_result_model_validate():
  r = AgentResult.model_validate({'output': 'x', 'session_id': None, 'metadata': {}})
  assert r.output == 'x'


def test_agent_result_dump_validate_round_trip():
  original = AgentResult(output='o', session_id='s', metadata={'m': True})
  restored = AgentResult.model_validate(original.model_dump())
  assert restored == original


# claudecodeagent tests


def _ok_stdout(data: dict) -> str:
  return json.dumps(data)


def test_output_format_json():
  agent = ClaudeCodeAgent()
  with patch('autopilot.ai.agents.claude_code.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=0, stdout=_ok_stdout({'result': ''}))
    agent.run('p')
  cmd = run_mock.call_args[0][0]
  i = cmd.index('--output-format')
  assert cmd[i + 1] == 'json'


def test_prompt_in_command():
  agent = ClaudeCodeAgent()
  with patch('autopilot.ai.agents.claude_code.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=0, stdout=_ok_stdout({'result': ''}))
    agent.run('my prompt')
  cmd = run_mock.call_args[0][0]
  i = cmd.index('-p')
  assert cmd[i + 1] == 'my prompt'


def test_resume_with_session_id():
  agent = ClaudeCodeAgent()
  with patch('autopilot.ai.agents.claude_code.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=0, stdout=_ok_stdout({'result': ''}))
    agent.run('p', context={'session_id': 'sess-123'})
  cmd = run_mock.call_args[0][0]
  i = cmd.index('--resume')
  assert cmd[i + 1] == 'sess-123'


def test_resume_absent_without_session():
  agent = ClaudeCodeAgent()
  with patch('autopilot.ai.agents.claude_code.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=0, stdout=_ok_stdout({'result': ''}))
    agent.run('p')
  cmd = run_mock.call_args[0][0]
  assert '--resume' not in cmd


def test_allowed_tools_from_constructor():
  agent = ClaudeCodeAgent(allowed_tools=['Bash', 'Read'])
  with patch('autopilot.ai.agents.claude_code.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=0, stdout=_ok_stdout({'result': ''}))
    agent.run('p')
  cmd = run_mock.call_args[0][0]
  i = cmd.index('--allowedTools')
  assert cmd[i + 1] == 'Bash,Read'


def test_allowed_tools_from_context_overrides():
  agent = ClaudeCodeAgent(allowed_tools=['Bash', 'Read'])
  with patch('autopilot.ai.agents.claude_code.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=0, stdout=_ok_stdout({'result': ''}))
    agent.run('p', context={'allowed_tools': ['Edit']})
  cmd = run_mock.call_args[0][0]
  i = cmd.index('--allowedTools')
  assert cmd[i + 1] == 'Edit'


def test_permission_mode():
  agent = ClaudeCodeAgent(permission_mode='acceptEdits')
  with patch('autopilot.ai.agents.claude_code.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=0, stdout=_ok_stdout({'result': ''}))
    agent.run('p')
  cmd = run_mock.call_args[0][0]
  i = cmd.index('--permission-mode')
  assert cmd[i + 1] == 'acceptEdits'


def test_append_system_prompt_from_constructor():
  agent = ClaudeCodeAgent(append_system_prompt='Be concise')
  with patch('autopilot.ai.agents.claude_code.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=0, stdout=_ok_stdout({'result': ''}))
    agent.run('p')
  cmd = run_mock.call_args[0][0]
  i = cmd.index('--append-system-prompt')
  assert cmd[i + 1] == 'Be concise'


def test_system_prompt_from_context_overrides():
  agent = ClaudeCodeAgent(append_system_prompt='From ctor')
  with patch('autopilot.ai.agents.claude_code.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=0, stdout=_ok_stdout({'result': ''}))
    agent.run('p', context={'system_prompt': 'From ctx'})
  cmd = run_mock.call_args[0][0]
  i = cmd.index('--append-system-prompt')
  assert cmd[i + 1] == 'From ctx'


def test_model_flag():
  agent = ClaudeCodeAgent(model='claude-sonnet-4-20250514')
  with patch('autopilot.ai.agents.claude_code.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=0, stdout=_ok_stdout({'result': ''}))
    agent.run('p')
  cmd = run_mock.call_args[0][0]
  i = cmd.index('--model')
  assert cmd[i + 1] == 'claude-sonnet-4-20250514'


def test_cwd_passed_to_subprocess():
  agent = ClaudeCodeAgent(cwd='/tmp/ws')
  with patch('autopilot.ai.agents.claude_code.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=0, stdout=_ok_stdout({'result': ''}))
    agent.run('p')
  kwargs = run_mock.call_args[1]
  assert kwargs['cwd'] == '/tmp/ws'


def test_successful_json_parsing():
  agent = ClaudeCodeAgent()
  with patch('autopilot.ai.agents.claude_code.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(
      returncode=0,
      stdout='{"result":"ok","session_id":"s1"}',
    )
    result = agent.run('p')
  assert result.output == 'ok'
  assert result.session_id == 's1'


def test_nonzero_exit_raises_agent_error():
  agent = ClaudeCodeAgent()
  with patch('autopilot.ai.agents.claude_code.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=1, stdout='', stderr='err')
    with pytest.raises(AgentError, match='claude exited with code 1'):
      agent.run('p')


def test_malformed_json_raises_agent_error():
  agent = ClaudeCodeAgent()
  with patch('autopilot.ai.agents.claude_code.subprocess.run') as run_mock:
    run_mock.return_value = MagicMock(returncode=0, stdout='not json')
    with pytest.raises(AgentError, match='failed to parse claude output'):
      agent.run('p')


def test_binary_not_found_raises_agent_error():
  agent = ClaudeCodeAgent()
  with patch('autopilot.ai.agents.claude_code.subprocess.run') as run_mock:
    run_mock.side_effect = FileNotFoundError()
    with pytest.raises(AgentError, match='claude binary not found'):
      agent.run('p')


@pytest.mark.asyncio
async def test_async_run_delegates_to_run() -> None:
  agent = ClaudeCodeAgent()
  with patch.object(ClaudeCodeAgent, 'run', return_value=AgentResult(output='ok')) as run_mock:
    r = await agent.async_run('prompt', context={})
  run_mock.assert_called_once_with('prompt', {})
  assert r.output == 'ok'
