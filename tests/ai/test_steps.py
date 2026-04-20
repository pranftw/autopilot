"""Tests for step-based workflow engine."""

from autopilot.ai.models import RetryConfig, RunConfig
from autopilot.ai.steps import BackStep, LLMStep, PythonStep, run_step_workflow
from autopilot.core.errors import AIError
from pydantic import BaseModel
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


class DummyOutput(BaseModel):
  value: str


class DummyOutput2(BaseModel):
  score: int


def _make_run_config() -> RunConfig:
  return RunConfig(
    model='test-model',
    num_parallel=1,
    max_rpm=100,
    rpm_safety_margin=1.0,
    retry=RetryConfig(
      max_retries=1,
      min_timeout_ms=100,
      max_timeout_ms=1000,
      backoff_factor=2,
    ),
    max_tool_steps=5,
    max_output_tokens=1024,
  )


# Helper to create a mock Agent that returns a given output
def _mock_agent_factory(outputs: dict[str, BaseModel]):
  """Create a mock Agent class that returns different outputs based on call count."""
  call_count: dict[str, int] = {}

  def agent_constructor(model, **kwargs):
    mock_agent = MagicMock()
    output_type = kwargs.get('output_type')

    async def mock_run(*args, **run_kwargs):
      # Find the output for this output_type
      for _name, output in outputs.items():
        if isinstance(output, output_type):
          result = MagicMock()
          result.output = output
          return result
      # Default: return first output
      result = MagicMock()
      result.output = list(outputs.values())[call_count.get('idx', 0)]
      call_count['idx'] = call_count.get('idx', 0) + 1
      return result

    mock_agent.run = mock_run
    return mock_agent

  return agent_constructor


class TestLLMStep:
  def test_stores_name(self) -> None:
    s = LLMStep('step-a', DummyOutput)
    assert s.name == 'step-a'

  def test_stores_output_type(self) -> None:
    s = LLMStep('s', DummyOutput2)
    assert s.output_type is DummyOutput2

  def test_stores_instructions(self) -> None:
    s = LLMStep('s', DummyOutput, instructions='do the thing')
    assert s.instructions == 'do the thing'

  def test_stores_instructions_fn(self) -> None:
    def instr_fn(ctx: dict) -> str:
      return 'dynamic'

    s = LLMStep('s', DummyOutput, instructions_fn=instr_fn)
    assert s.instructions_fn is instr_fn

  def test_stores_tools(self) -> None:
    tools = [object()]
    s = LLMStep('s', DummyOutput, tools=tools)
    assert s.tools is tools

  def test_tools_default_none(self) -> None:
    s = LLMStep('s', DummyOutput)
    assert s.tools is None


class TestPythonStep:
  def test_stores_name(self) -> None:
    s = PythonStep('py-step', lambda c: {})
    assert s.name == 'py-step'

  def test_stores_fn(self) -> None:
    def fn(ctx: dict) -> dict:
      return {}

    s = PythonStep('p', fn)
    assert s.fn is fn

  def test_fn_receives_context(self) -> None:
    def fn(ctx: dict) -> dict:
      return {'echo': ctx['key']}

    step = PythonStep('p', fn)
    assert step.fn({'key': 'val'}) == {'echo': 'val'}


class TestBackStep:
  def test_stores_target(self) -> None:
    b = BackStep('b', 'target-a', lambda c: False)
    assert b.target == 'target-a'

  def test_stores_condition(self) -> None:
    def cond(ctx: dict) -> bool:
      return True

    b = BackStep('b', 't', cond)
    assert b.condition is cond

  def test_stores_max_iterations(self) -> None:
    b = BackStep('b', 't', lambda c: False, max_iterations=7)
    assert b.max_iterations == 7

  def test_default_max_iterations(self) -> None:
    b = BackStep('b', 't', lambda c: False)
    assert b.max_iterations == 3

  def test_condition_receives_context(self) -> None:
    seen: list[dict] = []

    def cond(ctx: dict) -> bool:
      seen.append(ctx)
      return False

    b = BackStep('b', 't', cond)
    b.condition({'k': 1})
    assert seen == [{'k': 1}]


class TestRunStepWorkflow:
  @pytest.mark.asyncio
  async def test_python_step_receives_prior_results(self) -> None:
    steps = [
      PythonStep('first', lambda c: {'a': 1}),
      PythonStep('second', lambda c: {'sum': c['first']['a'] + 1}),
    ]
    ctx = await run_step_workflow(steps, {}, 'test-model', _make_run_config())
    assert ctx['first'] == {'a': 1}
    assert ctx['second'] == {'sum': 2}

  @pytest.mark.asyncio
  @patch('autopilot.ai.steps.Agent')
  async def test_llm_step_creates_agent(self, mock_agent_cls) -> None:
    mock_result = MagicMock()
    mock_result.output = DummyOutput(value='test')
    mock_agent_instance = MagicMock()
    mock_agent_instance.run = AsyncMock(return_value=mock_result)
    mock_agent_cls.return_value = mock_agent_instance

    steps = [LLMStep('gen', output_type=DummyOutput, instructions='do it')]
    ctx = await run_step_workflow(steps, {}, 'test-model', _make_run_config())
    assert ctx['gen'].value == 'test'
    mock_agent_cls.assert_called_once()

  @pytest.mark.asyncio
  @patch('autopilot.ai.steps.Agent')
  async def test_llm_step_uses_instructions_fn(self, mock_agent_cls) -> None:
    mock_result = MagicMock()
    mock_result.output = DummyOutput(value='dyn')
    mock_agent_instance = MagicMock()
    mock_agent_instance.run = AsyncMock(return_value=mock_result)
    mock_agent_cls.return_value = mock_agent_instance

    def instr_fn(ctx: dict) -> str:
      return f'prefix:{ctx["setup"]["x"]}'

    steps = [
      PythonStep('setup', lambda c: {'x': 1}),
      LLMStep('gen', output_type=DummyOutput, instructions_fn=instr_fn),
    ]
    ctx = await run_step_workflow(steps, {}, 'm', _make_run_config())
    assert ctx['gen'].value == 'dyn'
    _, kwargs = mock_agent_cls.call_args
    assert kwargs['instructions'] == 'prefix:1'
    run_call = mock_agent_instance.run.call_args
    assert run_call[0][0] == 'prefix:1'

  @pytest.mark.asyncio
  async def test_backstep_loops_back(self) -> None:
    steps = [
      PythonStep(
        'counter',
        lambda c: {'n': c.get('counter', {}).get('n', 0) + 1},
      ),
      BackStep(
        'back',
        'counter',
        lambda c: c.get('counter', {}).get('n', 0) < 2,
      ),
    ]
    ctx = await run_step_workflow(steps, {}, 'm', _make_run_config())
    assert ctx['counter']['n'] == 2
    assert ctx['back_iterations'] == 1

  @pytest.mark.asyncio
  async def test_backstep_max_iterations(self) -> None:
    steps = [
      PythonStep('counter', lambda c: {'n': 1}),
      BackStep(
        'back',
        'counter',
        lambda c: True,
        max_iterations=2,
      ),
    ]
    ctx = await run_step_workflow(steps, {}, 'm', _make_run_config())
    assert ctx['back_iterations'] == 2

  @pytest.mark.asyncio
  async def test_backstep_condition_false_continues(self) -> None:
    steps = [
      PythonStep('a', lambda c: {'v': 1}),
      BackStep('back', 'a', lambda c: False),
      PythonStep('tail', lambda c: {'done': True}),
    ]
    ctx = await run_step_workflow(steps, {}, 'm', _make_run_config())
    assert ctx['tail'] == {'done': True}
    assert ctx['back_iterations'] == 0

  @pytest.mark.asyncio
  async def test_backstep_target_not_found(self) -> None:
    steps = [BackStep('back', 'missing', lambda c: True)]
    with pytest.raises(AIError, match='not found'):
      await run_step_workflow(steps, {}, 'm', _make_run_config())

  @pytest.mark.asyncio
  async def test_backstep_iteration_counter(self) -> None:
    steps = [
      PythonStep(
        'counter',
        lambda c: {'n': c.get('counter', {}).get('n', 0) + 1},
      ),
      BackStep(
        'back',
        'counter',
        lambda c: c.get('counter', {}).get('n', 0) < 2,
      ),
    ]
    ctx = await run_step_workflow(steps, {}, 'm', _make_run_config())
    assert ctx['back_iterations'] == 1

  @pytest.mark.asyncio
  async def test_empty_steps(self) -> None:
    ctx = await run_step_workflow([], {'a': 1}, 'm', _make_run_config())
    assert ctx == {'a': 1}

  @pytest.mark.asyncio
  async def test_single_python_step(self) -> None:
    steps = [PythonStep('only', lambda c: {'x': 1})]
    ctx = await run_step_workflow(steps, {}, 'm', _make_run_config())
    assert ctx['only'] == {'x': 1}

  @pytest.mark.asyncio
  async def test_python_step_exception_propagates(self) -> None:
    def boom(_ctx: dict) -> dict:
      raise ValueError('fail')

    steps = [PythonStep('bad', boom)]
    with pytest.raises(ValueError, match='fail'):
      await run_step_workflow(steps, {}, 'm', _make_run_config())

  @pytest.mark.asyncio
  @patch('autopilot.ai.steps.Agent')
  async def test_llm_step_with_tools(self, mock_agent_cls) -> None:
    mock_result = MagicMock()
    mock_result.output = DummyOutput(value='with-tools')
    mock_agent_instance = MagicMock()
    mock_agent_instance.run = AsyncMock(return_value=mock_result)
    mock_agent_cls.return_value = mock_agent_instance

    tools = [lambda: None]
    steps = [LLMStep('gen', DummyOutput, instructions='x', tools=tools)]
    ctx = await run_step_workflow(steps, {}, 'test-model', _make_run_config())
    assert ctx['gen'].value == 'with-tools'
    _, kwargs = mock_agent_cls.call_args
    assert kwargs['tools'] is tools
