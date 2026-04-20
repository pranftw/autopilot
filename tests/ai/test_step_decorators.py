from autopilot.ai.generator import DataGenerator, stratify_by
from autopilot.ai.judge import Judge
from autopilot.ai.steps import (
  BackStep,
  LLMStep,
  PythonStep,
  back_step,
  collect_steps,
  llm_step,
  python_step,
)
from pydantic import BaseModel
import pytest


class _Output(BaseModel):
  text: str = ''


class TestLLMStepDecorator:
  def test_marks_method_with_meta(self) -> None:
    @llm_step('gen', output_type=_Output)
    def generate(self, ctx):
      return ''

    assert hasattr(generate, '_step_meta')

  def test_meta_has_correct_fields(self) -> None:
    @llm_step('gen', output_type=_Output, instructions='do it')
    def generate(self, ctx):
      return ''

    meta = generate._step_meta
    assert meta.kind == 'llm'
    assert meta.name == 'gen'
    assert meta.output_type is _Output
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
  @llm_step('generate', output_type=_Output)
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
    assert steps[0].output_type is _Output

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
    class Gen(DataGenerator):
      pass

    item = type('Item', (), {'custom': type('C', (), {'domain': 'math'})()})()
    gen = Gen()
    assert gen.stratify_key(item) == 'math'

  def test_dict_field_access(self) -> None:
    @stratify_by('domain', 'difficulty')
    class Gen(DataGenerator):
      pass

    item = type('Item', (), {'custom': {'domain': 'math', 'difficulty': 'hard'}})()
    gen = Gen()
    assert gen.stratify_key(item) == 'math:hard'

  def test_dotted_field_paths(self) -> None:
    @stratify_by('metadata.level')
    class Gen(DataGenerator):
      pass

    item = type('Item', (), {'custom': {'metadata': {'level': 'expert'}}})()
    gen = Gen()
    assert gen.stratify_key(item) == 'expert'


class TestDefineStepsIntegration:
  def test_generator_collects_decorated_steps(self) -> None:
    class MyGen(DataGenerator):
      @llm_step('gen', output_type=_Output)
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
    class EmptyGen(DataGenerator):
      pass

    with pytest.raises(NotImplementedError):
      EmptyGen().define_steps(None)

  def test_override_define_steps_takes_precedence(self) -> None:
    class CustomGen(DataGenerator):
      @llm_step('gen', output_type=_Output)
      def gen(self, ctx):
        return ''

      def define_steps(self, config):
        return [PythonStep('custom', fn=lambda ctx: {})]

    g = CustomGen()
    steps = g.define_steps(None)
    assert len(steps) == 1
    assert steps[0].name == 'custom'

  def test_judge_collects_decorated_steps(self) -> None:
    class MyJudge(Judge):
      @llm_step('judge', output_type=_Output)
      def judge_step(self, ctx):
        return ''

    j = MyJudge()
    steps = j.define_steps(None)
    assert len(steps) == 1
    assert steps[0].name == 'judge'
