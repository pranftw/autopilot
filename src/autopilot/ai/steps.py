"""Step-based workflow engine for AI structured output pipelines.

Three step types:
  LLMStep: pydantic-ai Agent with structured output_type.
  PythonStep: regular Python function, no LLM.
  BackStep: conditional loopback to a prior step.
"""

from autopilot.ai.models import RunConfig
from autopilot.core.errors import AIError
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import Agent
from typing import Any, Callable


class Step:
  """Base class for workflow steps."""

  def __init__(self, name: str) -> None:
    self.name = name


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


class PythonStep(Step):
  """Deterministic execution step. Runs a regular function."""

  def __init__(self, name: str, fn: Callable[[dict], dict]) -> None:
    super().__init__(name)
    self.fn = fn


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


async def run_step_workflow(
  steps: list[Step],
  initial_context: dict[str, Any],
  model: str,
  run_config: RunConfig,
) -> dict[str, Any]:
  """Execute a step workflow. Shared by DataGenerator and Judge.

  1. Runs steps in order
  2. LLMStep: creates Agent(model, output_type=...), calls agent.run()
  3. PythonStep: calls fn(context)
  4. BackStep: if condition(context) is True and iterations < max, jump to target
  5. Merges each step result into context for the next step
  6. Returns final accumulated context dict
  """
  context = dict(initial_context)
  iteration_counts: dict[str, int] = {}

  # Build step index for BackStep target resolution
  step_index: dict[str, int] = {}
  for i, step in enumerate(steps):
    step_index[step.name] = i

  idx = 0
  while idx < len(steps):
    step = steps[idx]

    if isinstance(step, LLMStep):
      # Resolve instructions
      if step.instructions_fn is not None:
        instructions = step.instructions_fn(context)
      else:
        instructions = step.instructions or ''

      # Build agent kwargs
      agent_kwargs: dict[str, Any] = {
        'output_type': step.output_type,
      }
      if instructions:
        agent_kwargs['instructions'] = instructions
      if step.tools:
        agent_kwargs['tools'] = step.tools

      agent = Agent(model, **agent_kwargs)
      result = await agent.run(
        instructions or f'Execute step: {step.name}',
        model_settings={
          'max_tokens': run_config.max_output_tokens,
        },
      )
      context[step.name] = result.output

    elif isinstance(step, PythonStep):
      result = step.fn(context)
      if isinstance(result, dict):
        context[step.name] = result
      else:
        context[step.name] = result

    elif isinstance(step, BackStep):
      if step.target not in step_index:
        raise AIError(f'BackStep target {step.target!r} not found in workflow steps')

      counter_key = step.name
      count = iteration_counts.get(counter_key, 0)

      if step.condition(context) and count < step.max_iterations:
        iteration_counts[counter_key] = count + 1
        context[f'{step.name}_iterations'] = count + 1
        idx = step_index[step.target]
        continue
      else:
        context[f'{step.name}_iterations'] = count

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
