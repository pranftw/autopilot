"""Tests for step-based workflow engine, decorators, and execute() polymorphism."""

from autopilot.ai.evaluation.generator import GeneratorAgent, stratify_by
from autopilot.ai.evaluation.judge import JudgeAgent
from autopilot.ai.evaluation.schemas import RetryConfig, RunConfig
from autopilot.ai.evaluation.steps import (
  BackStep,
  LLMStep,
  PythonStep,
  Step,
  StepLoopback,
  back_step,
  collect_steps,
  llm_step,
  python_step,
  run_step_workflow,
)
from autopilot.core.errors import AIError
from pydantic import BaseModel
from typing import Any
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
  @patch('autopilot.ai.evaluation.steps.Agent')
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
  @patch('autopilot.ai.evaluation.steps.Agent')
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
    assert 'back_iterations' not in ctx

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
  @patch('autopilot.ai.evaluation.steps.Agent')
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


# step decorator tests


class _DecoratorOutput(BaseModel):
  text: str = ''


class TestLLMStepDecorator:
  def test_marks_method_with_meta(self) -> None:
    @llm_step('gen', output_type=_DecoratorOutput)
    def generate(self, ctx):
      return ''

    assert hasattr(generate, '_step_meta')

  def test_meta_has_correct_fields(self) -> None:
    @llm_step('gen', output_type=_DecoratorOutput, instructions='do it')
    def generate(self, ctx):
      return ''

    meta = generate._step_meta
    assert meta.kind == 'llm'
    assert meta.name == 'gen'
    assert meta.output_type is _DecoratorOutput
    assert meta.instructions == 'do it'


class TestPythonStepDecorator:
  def test_marks_method_with_meta(self) -> None:
    @python_step('exec')
    def execute(self, ctx):
      return {}

    assert execute._step_meta.kind == 'python'
    assert execute._step_meta.name == 'exec'


class TestBackStepDecorator:
  def test_marks_method_with_meta(self) -> None:
    @back_step('retry', target='gen', max_iterations=5)
    def should_retry(self, ctx):
      return False

    meta = should_retry._step_meta
    assert meta.kind == 'back'
    assert meta.target == 'gen'
    assert meta.max_iterations == 5

  def test_default_max_iterations(self) -> None:
    @back_step('retry', target='gen')
    def should_retry(self, ctx):
      return False

    assert should_retry._step_meta.max_iterations == 3


class _StubGen:
  @llm_step('generate', output_type=_DecoratorOutput)
  def generate(self, ctx):
    return 'prompt'

  @python_step('execute')
  def execute(self, ctx):
    return {}

  @back_step('retry', target='generate', max_iterations=2)
  def should_retry(self, ctx):
    return True

  def undecorated(self):
    pass


class TestCollectSteps:
  def test_collects_in_definition_order(self) -> None:
    steps = collect_steps(_StubGen())
    assert [s.name for s in steps] == ['generate', 'execute', 'retry']

  def test_builds_llm_step(self) -> None:
    steps = collect_steps(_StubGen())
    assert isinstance(steps[0], LLMStep)
    assert steps[0].output_type is _DecoratorOutput

  def test_builds_python_step(self) -> None:
    steps = collect_steps(_StubGen())
    assert isinstance(steps[1], PythonStep)

  def test_builds_back_step(self) -> None:
    steps = collect_steps(_StubGen())
    s = steps[2]
    assert isinstance(s, BackStep)
    assert s.target == 'generate'
    assert s.max_iterations == 2

  def test_raises_when_no_decorators(self) -> None:
    class Empty:
      pass

    with pytest.raises(NotImplementedError, match='has no @step-decorated methods'):
      collect_steps(Empty())

  def test_ignores_undecorated_methods(self) -> None:
    steps = collect_steps(_StubGen())
    names = [s.name for s in steps]
    assert 'undecorated' not in names


class TestStratifyBy:
  def test_simple_fields(self) -> None:
    @stratify_by('domain')
    class Gen(GeneratorAgent):
      pass

    item = type('Item', (), {'custom': type('C', (), {'domain': 'math'})()})()
    gen = Gen()
    assert gen.stratify_key(item) == 'math'

  def test_dict_field_access(self) -> None:
    @stratify_by('domain', 'difficulty')
    class Gen(GeneratorAgent):
      pass

    item = type('Item', (), {'custom': {'domain': 'math', 'difficulty': 'hard'}})()
    gen = Gen()
    assert gen.stratify_key(item) == 'math:hard'

  def test_dotted_field_paths(self) -> None:
    @stratify_by('metadata.level')
    class Gen(GeneratorAgent):
      pass

    item = type('Item', (), {'custom': {'metadata': {'level': 'expert'}}})()
    gen = Gen()
    assert gen.stratify_key(item) == 'expert'


class TestDefineStepsIntegration:
  def test_generator_collects_decorated_steps(self) -> None:
    class MyGen(GeneratorAgent):
      @llm_step('gen', output_type=_DecoratorOutput)
      def gen(self, ctx):
        return ''

      @python_step('exec')
      def exec_step(self, ctx):
        return {}

    g = MyGen()
    steps = g.define_steps(None)
    assert len(steps) == 2
    assert steps[0].name == 'gen'

  def test_generator_raises_when_no_steps(self) -> None:
    class EmptyGen(GeneratorAgent):
      pass

    with pytest.raises(NotImplementedError):
      EmptyGen().define_steps(None)

  def test_override_define_steps_takes_precedence(self) -> None:
    class CustomGen(GeneratorAgent):
      @llm_step('gen', output_type=_DecoratorOutput)
      def gen(self, ctx):
        return ''

      def define_steps(self, config):
        return [PythonStep('custom', fn=lambda ctx: {})]

    g = CustomGen()
    steps = g.define_steps(None)
    assert len(steps) == 1
    assert steps[0].name == 'custom'

  def test_judge_collects_decorated_steps(self) -> None:
    class MyJudge(JudgeAgent):
      @llm_step('judge', output_type=_DecoratorOutput)
      def judge_step(self, ctx):
        return ''

    j = MyJudge()
    steps = j.define_steps(None)
    assert len(steps) == 1
    assert steps[0].name == 'judge'


# step.execute() polymorphism tests


class TestStepExecuteBase:
  @pytest.mark.asyncio
  async def test_step_execute_base_raises(self) -> None:
    step = Step('base')
    with pytest.raises(NotImplementedError):
      await step.execute({}, 'model', _make_run_config())


class TestPythonStepExecute:
  @pytest.mark.asyncio
  async def test_python_step_execute_runs_fn(self) -> None:
    step = PythonStep('py', lambda c: {'result': c.get('input', 0) + 1})
    result = await step.execute({'input': 5}, 'model', _make_run_config())
    assert result == {'result': 6}

  @pytest.mark.asyncio
  async def test_python_step_execute_returns_result(self) -> None:
    step = PythonStep('py', lambda c: 'string result')
    result = await step.execute({}, 'model', _make_run_config())
    assert result == 'string result'


class TestLLMStepExecute:
  @pytest.mark.asyncio
  @patch('autopilot.ai.evaluation.steps.Agent')
  async def test_llm_step_execute_returns_output(self, mock_agent_cls) -> None:
    mock_result = MagicMock()
    mock_result.output = DummyOutput(value='test')
    mock_agent_instance = MagicMock()
    mock_agent_instance.run = AsyncMock(return_value=mock_result)
    mock_agent_cls.return_value = mock_agent_instance

    step = LLMStep('gen', output_type=DummyOutput, instructions='do it')
    result = await step.execute({}, 'test-model', _make_run_config())
    assert result.value == 'test'
    mock_agent_cls.assert_called_once()


class TestBackStepExecute:
  @pytest.mark.asyncio
  async def test_back_step_execute_returns_loopback_when_condition_true(self) -> None:
    step = BackStep('back', 'target', lambda c: True)
    result = await step.execute({}, 'model', _make_run_config())
    assert isinstance(result, StepLoopback)
    assert result.target_name == 'target'

  @pytest.mark.asyncio
  async def test_back_step_execute_returns_none_when_condition_false(self) -> None:
    step = BackStep('back', 'target', lambda c: False)
    result = await step.execute({}, 'model', _make_run_config())
    assert result is None

  @pytest.mark.asyncio
  async def test_back_step_execute_returns_none_at_max_iterations(self) -> None:
    step = BackStep('back', 'target', lambda c: True, max_iterations=2)
    ctx = {'back_iterations': 2}
    result = await step.execute(ctx, 'model', _make_run_config())
    assert result is None


class TestStepLoopback:
  def test_step_loopback_holds_target_index(self) -> None:
    lb = StepLoopback(target_index=3)
    assert lb.target_index == 3

  def test_step_loopback_holds_target_name(self) -> None:
    lb = StepLoopback(target_index=-1, target_name='counter')
    assert lb.target_name == 'counter'


class TestCustomStepExecute:
  @pytest.mark.asyncio
  async def test_custom_step_in_workflow(self) -> None:
    class CustomStep(Step):
      async def execute(
        self,
        context: dict[str, Any],
        model: str,
        run_config: RunConfig,
      ) -> Any:
        return {'custom': True, 'input_count': len(context)}

    steps = [
      PythonStep('setup', lambda c: {'ready': True}),
      CustomStep('custom'),
    ]
    ctx = await run_step_workflow(steps, {}, 'model', _make_run_config())
    assert ctx['custom'] == {'custom': True, 'input_count': 1}

  @pytest.mark.asyncio
  async def test_backstep_target_not_found_raises(self) -> None:
    steps = [BackStep('back', 'missing', lambda c: True)]
    with pytest.raises(AIError, match='not found'):
      await run_step_workflow(steps, {}, 'm', _make_run_config())

  @pytest.mark.asyncio
  async def test_backstep_loopback_in_workflow(self) -> None:
    steps = [
      PythonStep(
        'counter',
        lambda c: {'n': c.get('counter', {}).get('n', 0) + 1},
      ),
      BackStep(
        'back',
        'counter',
        lambda c: c.get('counter', {}).get('n', 0) < 3,
        max_iterations=5,
      ),
    ]
    ctx = await run_step_workflow(steps, {}, 'm', _make_run_config())
    assert ctx['counter']['n'] == 3
    assert ctx['back_iterations'] == 2

  @pytest.mark.asyncio
  async def test_backstep_max_iterations_stops(self) -> None:
    steps = [
      PythonStep('counter', lambda c: {'n': 1}),
      BackStep('back', 'counter', lambda c: True, max_iterations=2),
    ]
    ctx = await run_step_workflow(steps, {}, 'm', _make_run_config())
    assert ctx['back_iterations'] == 2
