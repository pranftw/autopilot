"""Step-based workflow engine for AI structured output pipelines.

Three step types:
  LLMStep: pydantic-ai Agent with structured output_type.
  PythonStep: regular Python function, no LLM.
  BackStep: conditional loopback to a prior step.

Each step type overrides async execute(context, model, run_config) for
polymorphic dispatch. run_step_workflow calls await step.execute() uniformly.
If execute() returns a StepLoopback sentinel, the workflow jumps to the
target step. Otherwise the result is merged into context under the step name.

Step decorators (@llm_step, @python_step, @back_step) attach StepMeta to
methods; collect_steps() gathers them in definition order. The @stratify_by
class decorator (on generator.py) auto-generates stratify_key() from dotted
field paths.

Custom Step subclasses with execute() work in the workflow without any
framework changes.
"""

from autopilot.ai.evaluation.schemas import RunConfig
from autopilot.core.errors import AIError
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import Agent
from typing import Any, Callable


@dataclass
class StepLoopback:
  """Sentinel returned by BackStep.execute() to signal loopback."""

  target_index: int
  target_name: str | None = None


class Step:
  """Base class for workflow steps. Subclass and override execute().

  async execute(context, model, run_config) -> Any
    context: accumulated step results dict
    model: LLM model identifier string
    run_config: RunConfig with max_output_tokens, rate limits, etc.

  Return StepLoopback to signal loopback; any other value is merged
  into context[step.name].
  """

  def __init__(self, name: str) -> None:
    self.name = name

  async def execute(
    self,
    context: dict[str, Any],
    model: str,
    run_config: RunConfig,
  ) -> Any:
    raise NotImplementedError


class LLMStep(Step):
  """Structured output step. Creates a pydantic-ai Agent, returns Pydantic model.

  Tools can optionally be attached if the LLM needs them.
  """

  def __init__(
    self,
    name: str,
    output_type: type[BaseModel],
    instructions: str | None = None,
    instructions_fn: Callable[[dict], str] | None = None,
    tools: list | None = None,
  ) -> None:
    super().__init__(name)
    self.output_type = output_type
    self.instructions = instructions
    self.instructions_fn = instructions_fn
    self.tools = tools

  async def execute(
    self,
    context: dict[str, Any],
    model: str,
    run_config: RunConfig,
  ) -> Any:
    if self.instructions_fn is not None:
      instructions = self.instructions_fn(context)
    else:
      instructions = self.instructions or ''

    agent_kwargs: dict[str, Any] = {
      'output_type': self.output_type,
    }
    if instructions:
      agent_kwargs['instructions'] = instructions
    if self.tools:
      agent_kwargs['tools'] = self.tools

    agent = Agent(model, **agent_kwargs)
    result = await agent.run(
      instructions or f'Execute step: {self.name}',
      model_settings={
        'max_tokens': run_config.max_output_tokens,
      },
    )
    return result.output


class PythonStep(Step):
  """Deterministic execution step. Runs a regular function."""

  def __init__(self, name: str, fn: Callable[[dict], dict]) -> None:
    super().__init__(name)
    self.fn = fn

  async def execute(
    self,
    context: dict[str, Any],
    model: str,
    run_config: RunConfig,
  ) -> Any:
    return self.fn(context)


class BackStep(Step):
  """Conditional loopback to a prior step.

  If condition(context) returns True, jumps back to the target step.
  Tracks iteration count and stops at max_iterations.
  """

  def __init__(
    self,
    name: str,
    target: str,
    condition: Callable[[dict], bool],
    max_iterations: int = 3,
  ) -> None:
    super().__init__(name)
    self.target = target
    self.condition = condition
    self.max_iterations = max_iterations

  async def execute(
    self,
    context: dict[str, Any],
    model: str,
    run_config: RunConfig,
  ) -> StepLoopback | None:
    counter_key = self.name
    count = context.get(f'{counter_key}_iterations', 0)
    if self.condition(context) and count < self.max_iterations:
      return StepLoopback(target_index=-1, target_name=self.target)
    return None


async def run_step_workflow(
  steps: list[Step],
  initial_context: dict[str, Any],
  model: str,
  run_config: RunConfig,
) -> dict[str, Any]:
  """Execute a step workflow. Shared by GeneratorAgent and JudgeAgent.

  Polymorphic dispatch: every step runs via execute(). If execute() returns
  a StepLoopback sentinel, the workflow jumps to the target step. Otherwise
  the result is merged into context under the step name.
  """
  context = dict(initial_context)
  iteration_counts: dict[str, int] = {}

  step_index: dict[str, int] = {}
  for i, step in enumerate(steps):
    step_index[step.name] = i

  idx = 0
  while idx < len(steps):
    step = steps[idx]
    result = await step.execute(context, model, run_config)

    if isinstance(result, StepLoopback):
      target_name = result.target_name
      if target_name is not None and target_name not in step_index:
        raise AIError(f'BackStep target {target_name!r} not found in workflow steps')

      target_idx = result.target_index
      if target_name is not None:
        target_idx = step_index[target_name]

      counter_key = step.name
      count = iteration_counts.get(counter_key, 0)
      iteration_counts[counter_key] = count + 1
      context[f'{step.name}_iterations'] = count + 1
      idx = target_idx
      continue

    if result is not None:
      context[step.name] = result

    idx += 1

  return context


@dataclass
class StepMeta:
  kind: str
  name: str
  output_type: type[BaseModel] | None = None
  instructions: str | None = None
  target: str | None = None
  max_iterations: int = 3


def llm_step(name: str, *, output_type: type[BaseModel], instructions: str | None = None):
  """Mark a method as an LLM step."""

  def decorator(fn):
    fn._step_meta = StepMeta(
      kind='llm',
      name=name,
      output_type=output_type,
      instructions=instructions,
    )
    return fn

  return decorator


def python_step(name: str):
  """Mark a method as a Python step."""

  def decorator(fn):
    fn._step_meta = StepMeta(kind='python', name=name)
    return fn

  return decorator


def back_step(name: str, *, target: str, max_iterations: int = 3):
  """Mark a method as a conditional loopback step."""

  def decorator(fn):
    fn._step_meta = StepMeta(
      kind='back',
      name=name,
      target=target,
      max_iterations=max_iterations,
    )
    return fn

  return decorator


def collect_steps(instance: object) -> list[Step]:
  """Collect @step-decorated methods in definition order. Raises if none found."""
  steps: list[Step] = []
  for attr_name in type(instance).__dict__:
    method = getattr(type(instance), attr_name, None)
    if method is None or not hasattr(method, '_step_meta'):
      continue
    meta: StepMeta = method._step_meta
    bound = getattr(instance, attr_name)
    if meta.kind == 'llm':
      steps.append(
        LLMStep(
          meta.name,
          output_type=meta.output_type,
          instructions=meta.instructions,
          instructions_fn=bound,
        )
      )
    elif meta.kind == 'python':
      steps.append(PythonStep(meta.name, fn=bound))
    elif meta.kind == 'back':
      steps.append(
        BackStep(
          meta.name,
          target=meta.target,
          condition=bound,
          max_iterations=meta.max_iterations,
        )
      )
  if not steps:
    raise NotImplementedError(f'{type(instance).__name__} has no @step-decorated methods')
  return steps
