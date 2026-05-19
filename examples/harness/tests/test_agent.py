"""Tests for harness.agent (HarnessAgent and ConversationResult)."""

from dataclasses import fields
from harness.agent import DEFAULT_MAX_CONVERSATION_TURNS, ConversationResult, HarnessAgent
from harness.database import RetailDB
from unittest.mock import MagicMock, patch


class TestConversationResult:
  def test_defaults(self) -> None:
    result = ConversationResult()
    assert result.trajectory == []
    assert result.tool_calls == []
    assert result.turns == 0
    assert result.error is None
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert result.api_calls == 0

  def test_with_values(self) -> None:
    result = ConversationResult(
      trajectory=[{'role': 'assistant', 'content': 'hi', 'turn': 0}],
      tool_calls=[{'tool': 'think'}],
      turns=1,
      error=None,
      input_tokens=100,
      output_tokens=50,
      api_calls=1,
    )
    assert result.turns == 1
    assert len(result.trajectory) == 1
    assert result.input_tokens == 100

  def test_is_dataclass(self) -> None:
    field_names = {f.name for f in fields(ConversationResult)}
    assert 'trajectory' in field_names
    assert 'tool_calls' in field_names
    assert 'turns' in field_names
    assert 'error' in field_names


class TestHarnessAgentInit:
  def test_default_model(self) -> None:
    agent = HarnessAgent()
    assert agent.model == 'openrouter:google/gemma-4-31b-it'

  def test_custom_model(self) -> None:
    agent = HarnessAgent(model='test-model')
    assert agent.model == 'test-model'


class TestBuildAgent:
  def test_build_agent_creates_pydantic_agent(self) -> None:
    harness = HarnessAgent(model='test')

    def calculate(ctx, expression):
      return '1'

    def think(ctx, thought):
      return ''

    tools = {'calculate': calculate, 'think': think}
    agent = harness._build_agent('You are helpful.', tools)
    assert agent is not None

  def test_build_agent_deps_type_and_tool_count(self) -> None:
    from harness.tool_loader import HarnessDeps

    harness = HarnessAgent(model='test')

    def calculate(ctx, expression):
      return '1'

    def think(ctx, thought):
      return ''

    tools = {'calculate': calculate, 'think': think}
    agent = harness._build_agent('Test instructions', tools)
    assert agent._deps_type is HarnessDeps
    registered = agent._function_toolset.tools
    assert len(registered) == 2
    assert 'calculate' in registered
    assert 'think' in registered


class TestRunConversationMock:
  def _make_mock_result(self, output: str, messages: list | None = None) -> MagicMock:
    """Create a mock Pydantic AI RunResult."""
    result = MagicMock()
    result.output = output
    result.all_messages.return_value = messages or []
    usage = MagicMock()
    usage.input_tokens = 10
    usage.output_tokens = 5
    usage.requests = 1
    result.usage.return_value = usage
    return result

  def test_single_turn_conversation(self) -> None:
    harness = HarnessAgent(model='test')
    mock_result = self._make_mock_result('Hello, how can I help?')
    db = RetailDB()
    scenario = {'initial_message': 'Hi there'}

    with patch.object(harness, '_build_agent') as mock_build:
      mock_agent = MagicMock()
      mock_agent.run_sync.return_value = mock_result
      mock_build.return_value = mock_agent

      conv = harness.run_conversation(
        instructions='Be helpful.',
        tools={'think': lambda ctx, t: ''},
        scenario=scenario,
        db=db,
      )

    assert conv.turns == 1
    assert len(conv.trajectory) == 1
    assert conv.trajectory[0]['role'] == 'assistant'
    assert conv.trajectory[0]['content'] == 'Hello, how can I help?'
    assert conv.error is None
    assert conv.input_tokens == 10
    assert conv.output_tokens == 5
    assert conv.api_calls == 1

  def test_multi_turn_with_simulator(self) -> None:
    harness = HarnessAgent(model='test')
    db = RetailDB()
    scenario = {'initial_message': 'Hi'}

    responses = ['Response 1', 'Response 2', 'Response 3']
    call_count = [0]

    def mock_run_sync(msg, deps=None, message_history=None):
      idx = call_count[0]
      call_count[0] += 1
      return self._make_mock_result(responses[idx])

    simulator = MagicMock()
    simulator.next_message.side_effect = ['follow up 1', 'follow up 2', None]

    with patch.object(harness, '_build_agent') as mock_build:
      mock_agent = MagicMock()
      mock_agent.run_sync.side_effect = mock_run_sync
      mock_build.return_value = mock_agent

      conv = harness.run_conversation(
        instructions='Be helpful.',
        tools={},
        scenario=scenario,
        db=db,
        simulator=simulator,
      )

    assert conv.turns == 3
    assert conv.input_tokens == 30
    assert conv.output_tokens == 15
    assert conv.api_calls == 3
    assert conv.error is None
    assert len(conv.trajectory) == 5

  def test_error_during_run_sync(self) -> None:
    harness = HarnessAgent(model='test')
    db = RetailDB()
    scenario = {'initial_message': 'Hi'}

    with patch.object(harness, '_build_agent') as mock_build:
      mock_agent = MagicMock()
      mock_agent.run_sync.side_effect = RuntimeError('API error')
      mock_build.return_value = mock_agent

      conv = harness.run_conversation(
        instructions='Be helpful.',
        tools={},
        scenario=scenario,
        db=db,
      )

    assert conv.turns == 0
    assert conv.error is not None
    assert conv.error.startswith('RuntimeError: API error')
    assert 'Traceback' in conv.error
    assert conv.trajectory == []

  def test_max_turns_reached(self) -> None:
    harness = HarnessAgent(model='test')
    db = RetailDB()
    scenario = {'initial_message': 'Hi'}

    mock_result = self._make_mock_result('response')
    simulator = MagicMock()
    simulator.next_message.return_value = 'keep going'

    with patch.object(harness, '_build_agent') as mock_build:
      mock_agent = MagicMock()
      mock_agent.run_sync.return_value = mock_result
      mock_build.return_value = mock_agent

      conv = harness.run_conversation(
        instructions='Be helpful.',
        tools={},
        scenario=scenario,
        db=db,
        simulator=simulator,
      )

    assert conv.turns == DEFAULT_MAX_CONVERSATION_TURNS
    assert conv.error == 'max_turns'

  def test_tool_calls_from_deps(self) -> None:
    harness = HarnessAgent(model='test')
    db = RetailDB()
    scenario = {'initial_message': 'Hi'}

    def mock_run_sync(msg, deps=None, message_history=None):
      deps.tool_log.append({'tool': 'think', 'args': {'thought': 'hmm'}})
      return self._make_mock_result('done')

    with patch.object(harness, '_build_agent') as mock_build:
      mock_agent = MagicMock()
      mock_agent.run_sync.side_effect = mock_run_sync
      mock_build.return_value = mock_agent

      conv = harness.run_conversation(
        instructions='Be helpful.',
        tools={},
        scenario=scenario,
        db=db,
      )

    assert len(conv.tool_calls) == 1
    assert conv.tool_calls[0]['tool'] == 'think'


class TestDefaultMaxConversationTurns:
  def test_constant_value(self) -> None:
    assert DEFAULT_MAX_CONVERSATION_TURNS == 15


class TestMaxTurnsParameter:
  def _make_mock_result(self, output: str) -> MagicMock:
    """Create a mock Pydantic AI RunResult."""
    result = MagicMock()
    result.output = output
    result.all_messages.return_value = []
    usage = MagicMock()
    usage.input_tokens = 10
    usage.output_tokens = 5
    usage.requests = 1
    result.usage.return_value = usage
    return result

  def test_agent_uses_env_config_max_turns(self) -> None:
    """When max_turns is passed, the loop stops at that limit."""
    harness = HarnessAgent(model='test')
    db = RetailDB()
    scenario = {'initial_message': 'Hi'}
    custom_limit = 3

    mock_result = self._make_mock_result('response')
    simulator = MagicMock()
    simulator.next_message.return_value = 'keep going'

    with patch.object(harness, '_build_agent') as mock_build:
      mock_agent = MagicMock()
      mock_agent.run_sync.return_value = mock_result
      mock_build.return_value = mock_agent

      conv = harness.run_conversation(
        instructions='Be helpful.',
        tools={},
        scenario=scenario,
        db=db,
        simulator=simulator,
        max_turns=custom_limit,
      )

    assert conv.turns == custom_limit
    assert conv.error == 'max_turns'
    assert mock_agent.run_sync.call_count == custom_limit

  def test_agent_default_max_turns(self) -> None:
    """When max_turns is None, effective limit is DEFAULT_MAX_CONVERSATION_TURNS."""
    harness = HarnessAgent(model='test')
    db = RetailDB()
    scenario = {'initial_message': 'Hi'}

    mock_result = self._make_mock_result('response')
    simulator = MagicMock()
    simulator.next_message.return_value = 'keep going'

    with patch.object(harness, '_build_agent') as mock_build:
      mock_agent = MagicMock()
      mock_agent.run_sync.return_value = mock_result
      mock_build.return_value = mock_agent

      conv = harness.run_conversation(
        instructions='Be helpful.',
        tools={},
        scenario=scenario,
        db=db,
        simulator=simulator,
      )

    assert conv.turns == DEFAULT_MAX_CONVERSATION_TURNS
    assert conv.error == 'max_turns'
    assert mock_agent.run_sync.call_count == DEFAULT_MAX_CONVERSATION_TURNS
