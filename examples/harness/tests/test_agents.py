"""Tests for harness.agents (PydanticAgent AgentCollator bridge)."""

from autopilot.ai.agents.agent import Agent, AgentResult
from harness import DEFAULT_MODEL
from harness.agent import ConversationResult, HarnessAgent
from harness.agents import PydanticAgent
from harness.database import RetailDB
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class TestPydanticAgentInit:
  """Tests for PydanticAgent constructor and attribute storage."""

  def test_default_model(self) -> None:
    agent = PydanticAgent()
    assert agent._model == DEFAULT_MODEL

  def test_custom_model(self) -> None:
    agent = PydanticAgent(model='openrouter:anthropic/claude-3-haiku')
    assert agent._model == 'openrouter:anthropic/claude-3-haiku'

  def test_default_instructions_none(self) -> None:
    agent = PydanticAgent()
    assert agent._instructions is None

  def test_custom_instructions(self) -> None:
    agent = PydanticAgent(instructions='You are a JSON collator.')
    assert agent._instructions == 'You are a JSON collator.'

  def test_is_agent_subclass(self) -> None:
    agent = PydanticAgent()
    assert isinstance(agent, Agent)

  def test_inherits_limiter_default(self) -> None:
    agent = PydanticAgent()
    assert agent.limiter is None

  def test_name_returns_class_name(self) -> None:
    agent = PydanticAgent()
    assert agent.name() == 'PydanticAgent'


class TestPydanticAgentRunMock:
  """Tests for PydanticAgent.run with mocked pydantic_ai."""

  def test_run_returns_agent_result(self) -> None:
    agent = PydanticAgent(model='test-model')
    mock_run_result = SimpleNamespace(output='{"direction": "improve"}')

    with patch('harness.agents.pydantic_ai.Agent') as mock_agent_cls:
      mock_instance = MagicMock()
      mock_instance.run_sync.return_value = mock_run_result
      mock_agent_cls.return_value = mock_instance

      result = agent.run('Analyze the feedback')

    assert isinstance(result, AgentResult)
    assert result.output == '{"direction": "improve"}'

  def test_run_forwards_prompt_to_run_sync(self) -> None:
    agent = PydanticAgent(model='test-model')
    mock_run_result = SimpleNamespace(output='ok')

    with patch('harness.agents.pydantic_ai.Agent') as mock_agent_cls:
      mock_instance = MagicMock()
      mock_instance.run_sync.return_value = mock_run_result
      mock_agent_cls.return_value = mock_instance

      agent.run('Specific prompt text')

    mock_instance.run_sync.assert_called_once_with('Specific prompt text')

  def test_run_coerces_output_to_string(self) -> None:
    agent = PydanticAgent(model='test-model')
    mock_run_result = SimpleNamespace(output=42)

    with patch('harness.agents.pydantic_ai.Agent') as mock_agent_cls:
      mock_instance = MagicMock()
      mock_instance.run_sync.return_value = mock_run_result
      mock_agent_cls.return_value = mock_instance

      result = agent.run('prompt')

    assert result.output == '42'
    assert isinstance(result.output, str)

  def test_run_context_accepted_and_ignored(self) -> None:
    agent = PydanticAgent(model='test-model')
    mock_run_result = SimpleNamespace(output='response')

    with patch('harness.agents.pydantic_ai.Agent') as mock_agent_cls:
      mock_instance = MagicMock()
      mock_instance.run_sync.return_value = mock_run_result
      mock_agent_cls.return_value = mock_instance

      result = agent.run('prompt', context={'key': 'value'})

    assert result.output == 'response'
    mock_instance.run_sync.assert_called_once_with('prompt')


class TestPydanticAgentCustomModel:
  """Tests that model string is correctly forwarded to pydantic_ai.Agent."""

  def test_default_model_passed_to_pydantic_agent(self) -> None:
    agent = PydanticAgent()
    mock_run_result = SimpleNamespace(output='out')

    with patch('harness.agents.pydantic_ai.Agent') as mock_agent_cls:
      mock_instance = MagicMock()
      mock_instance.run_sync.return_value = mock_run_result
      mock_agent_cls.return_value = mock_instance

      agent.run('prompt')

    mock_agent_cls.assert_called_once_with(DEFAULT_MODEL, instructions=None)

  def test_custom_model_passed_to_pydantic_agent(self) -> None:
    agent = PydanticAgent(model='openrouter:meta-llama/llama-3-70b')
    mock_run_result = SimpleNamespace(output='out')

    with patch('harness.agents.pydantic_ai.Agent') as mock_agent_cls:
      mock_instance = MagicMock()
      mock_instance.run_sync.return_value = mock_run_result
      mock_agent_cls.return_value = mock_instance

      agent.run('prompt')

    mock_agent_cls.assert_called_once_with('openrouter:meta-llama/llama-3-70b', instructions=None)


class TestPydanticAgentWithInstructions:
  """Tests that instructions are forwarded to pydantic_ai.Agent."""

  def test_instructions_forwarded_when_provided(self) -> None:
    agent = PydanticAgent(
      model='test-model',
      instructions='Return JSON with direction and parameters.',
    )
    mock_run_result = SimpleNamespace(output='{}')

    with patch('harness.agents.pydantic_ai.Agent') as mock_agent_cls:
      mock_instance = MagicMock()
      mock_instance.run_sync.return_value = mock_run_result
      mock_agent_cls.return_value = mock_instance

      agent.run('prompt')

    mock_agent_cls.assert_called_once_with(
      'test-model',
      instructions='Return JSON with direction and parameters.',
    )

  def test_instructions_none_forwarded(self) -> None:
    agent = PydanticAgent(model='test-model', instructions=None)
    mock_run_result = SimpleNamespace(output='{}')

    with patch('harness.agents.pydantic_ai.Agent') as mock_agent_cls:
      mock_instance = MagicMock()
      mock_instance.run_sync.return_value = mock_run_result
      mock_agent_cls.return_value = mock_instance

      agent.run('prompt')

    mock_agent_cls.assert_called_once_with('test-model', instructions=None)

  def test_empty_string_instructions_forwarded(self) -> None:
    agent = PydanticAgent(model='test-model', instructions='')
    mock_run_result = SimpleNamespace(output='{}')

    with patch('harness.agents.pydantic_ai.Agent') as mock_agent_cls:
      mock_instance = MagicMock()
      mock_instance.run_sync.return_value = mock_run_result
      mock_agent_cls.return_value = mock_instance

      agent.run('prompt')

    mock_agent_cls.assert_called_once_with('test-model', instructions='')


class TestHarnessAgentRunConversationError:
  """Tests for HarnessAgent.run_conversation error handling."""

  def test_run_conversation_error_has_traceback(self) -> None:
    """ConversationResult.error contains 'Traceback' when agent raises."""
    agent = HarnessAgent(model='test-model')
    scenario = {'initial_message': 'hello'}
    db = MagicMock(spec=RetailDB)

    with patch.object(agent, '_build_agent') as mock_build:
      mock_pydantic_agent = MagicMock()
      mock_pydantic_agent.run_sync.side_effect = RuntimeError('connection failed')
      mock_build.return_value = mock_pydantic_agent

      result = agent.run_conversation(
        instructions='test',
        tools={},
        scenario=scenario,
        db=db,
      )

    assert isinstance(result, ConversationResult)
    assert result.error is not None
    assert 'RuntimeError: connection failed' in result.error
    assert 'Traceback' in result.error
