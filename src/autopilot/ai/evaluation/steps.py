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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import Agent
from typing import Any, TypeVar
import asyncio


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
    """Create a step with a stable name used as the context key for results.

    Args:
      name: Key under which this step's output is stored in the workflow context.
    """
    self.name = name

  async def execute(
    self,
    context: dict[str, Any],
    model: str,
    run_config: RunConfig,
  ) -> Any:
    """Run this step; subclasses must implement async execution.

    Args:
      context: Accumulated results from prior steps.
      model: LLM model identifier when this step uses an LLM.
      run_config: Token and rate-limit settings for the run.

    Returns:
      Step output to merge into ``context[name]``, or :class:`StepLoopback` to
      jump backward in the workflow.

    Raises:
      NotImplementedError: If the subclass does not implement execution.
    """
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
    instructions_fn: Callable[[dict[str, Any]], str] | None = None,
    tools: list[Any] | None = None,
  ) -> None:
    """Configure structured LLM output and optional tools or dynamic instructions.

    Args:
      name: Step name and context key for results.
      output_type: Pydantic model class for structured agent output.
      instructions: Static system instructions when ``instructions_fn`` is not used.
      instructions_fn: Callable returning instructions from context; if set,
        static ``instructions`` is ignored for the agent.
      tools: Optional pydantic-ai tools to attach to the agent.
    """
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
    """Run a pydantic-ai agent and return structured output.

    Args:
      context: Workflow context; used when ``instructions_fn`` is set.
      model: LLM model identifier passed to the agent.
      run_config: Limits such as ``max_output_tokens`` for the model call.

    Returns:
      Parsed Pydantic instance from the agent run's ``output``.
    """
    if self.instructions_fn is not None:
      instructions = self.instructions_fn(context)
    else:
      instructions = '' if self.instructions is None else self.instructions

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
  """Deterministic execution step. Runs a regular function.

  Callback must be a synchronous function. Async callbacks (coroutines)
  are rejected at construction with TypeError.
  """

  def __init__(self, name: str, fn: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
    """Wrap a synchronous function as an async workflow step.

    Args:
      name: Step name and context key for results.
      fn: Sync callable taking context and returning a dict result.

    Raises:
      TypeError: If ``fn`` is an async coroutine function.
    """
    if asyncio.iscoroutinefunction(fn):
      fn_name = getattr(fn, '__name__', repr(fn))
      msg = f'PythonStep callback must be sync, got async: {fn_name}'
      raise TypeError(msg)
    super().__init__(name)
    self.fn = fn

  async def execute(
    self,
    context: dict[str, Any],
    model: str,
    run_config: RunConfig,
  ) -> Any:
    """Invoke the wrapped synchronous function with the current context.

    Args:
      context: Workflow context passed to ``fn``.
      model: Unused; present for a uniform step signature.
      run_config: Unused; present for a uniform step signature.

    Returns:
      Value returned by ``fn`` (typically a dict merged into context).
    """
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
    condition: Callable[[dict[str, Any]], bool],
    max_iterations: int = 3,
  ) -> None:
    """Configure loopback target, condition, and iteration cap.

    Args:
      name: Step name; iteration counters use this name as a key prefix.
      target: Name of the prior step to jump back to when the condition holds.
      condition: Callable returning True when loopback should occur.
      max_iterations: Maximum loopback iterations for this step.
    """
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
    """Evaluate the condition and possibly return a loopback sentinel.

    Args:
      context: Workflow context; must include ``{name}_iterations`` when counting.
      model: Unused; present for a uniform step signature.
      run_config: Unused; present for a uniform step signature.

    Returns:
      :class:`StepLoopback` when the condition is true and under the cap,
      otherwise ``None``.
    """
    counter_key = self.name
    count = context.get(f'{counter_key}_iterations', 0)
    if self.condition(context) and count < self.max_iterations:
      return StepLoopback(target_index=-1, target_name=self.target)
    return None


async def run_step_workflow(
  steps: Sequence[Step],
  initial_context: dict[str, Any],
  model: str,
  run_config: RunConfig,
) -> dict[str, Any]:
  """Execute a step workflow. Shared by GeneratorAgent and JudgeAgent.

  Polymorphic dispatch: every step runs via execute(). If execute() returns
  a StepLoopback sentinel, the workflow jumps to the target step. Otherwise
  the result is merged into context under the step name.

  Args:
    steps: Ordered workflow steps to run.
    initial_context: Starting context dict (copied before execution).
    model: LLM model identifier forwarded to each step's ``execute``.
    run_config: Run limits forwarded to each step's ``execute``.

  Returns:
    Final workflow context after all steps complete (including merged results).

  Raises:
    AIError: If a loopback target name is not present in ``steps``.
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
        msg = f'BackStep target {target_name!r} not found in workflow steps'
        raise AIError(msg)

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
  """Metadata attached by step decorators to agent methods."""

  kind: str
  name: str
  output_type: type[BaseModel] | None = None
  instructions: str | None = None
  target: str | None = None
  max_iterations: int = 3


_Decorated = TypeVar('_Decorated', bound=Callable[..., Any])


def llm_step(
  name: str,
  *,
  output_type: type[BaseModel],
  instructions: str | None = None,
) -> Callable[[_Decorated], _Decorated]:
  """Mark a method as an LLM step.

  Args:
    name: Step name used as the workflow context key.
    output_type: Pydantic model for structured LLM output.
    instructions: Optional static instructions stored in metadata.

  Returns:
    Decorator that attaches :class:`StepMeta` to the wrapped function.
  """

  def decorator(fn: _Decorated) -> _Decorated:
    fn.step_meta = StepMeta(
      kind='llm',
      name=name,
      output_type=output_type,
      instructions=instructions,
    )
    return fn

  return decorator


def python_step(name: str) -> Callable[[_Decorated], _Decorated]:
  """Mark a method as a Python step.

  Args:
    name: Step name used as the workflow context key.

  Returns:
    Decorator that attaches :class:`StepMeta` for kind ``python``.
  """

  def decorator(fn: _Decorated) -> _Decorated:
    fn.step_meta = StepMeta(kind='python', name=name)
    return fn

  return decorator


def back_step(
  name: str,
  *,
  target: str,
  max_iterations: int = 3,
) -> Callable[[_Decorated], _Decorated]:
  """Mark a method as a conditional loopback step.

  Args:
    name: Step name used as the workflow context key.
    target: Prior step name to jump back to when the method returns True.
    max_iterations: Maximum loopback iterations for this step.

  Returns:
    Decorator that attaches :class:`StepMeta` for kind ``back``.
  """

  def decorator(fn: _Decorated) -> _Decorated:
    fn.step_meta = StepMeta(
      kind='back',
      name=name,
      target=target,
      max_iterations=max_iterations,
    )
    return fn

  return decorator


def collect_steps(instance: object) -> list[Step]:
  """Collect all @step-decorated methods from the instance's class hierarchy.

  Walks the MRO (not just cls.__dict__), so inherited steps are included.
  Deduplicated by step name: first occurrence in MRO wins (child overrides parent).

  Args:
    instance: Object whose class hierarchy is scanned for step metadata.

  Returns:
    Ordered :class:`Step` instances bound to the instance's methods.

  Raises:
    NotImplementedError: If no ``@step``-decorated methods are found.
  """
  seen_names: set[str] = set()
  steps: list[Step] = []
  for klass in type(instance).__mro__:
    for attr_name, method in klass.__dict__.items():
      if not hasattr(method, 'step_meta'):
        continue
      meta: StepMeta = method.step_meta
      if meta.name in seen_names:
        continue
      seen_names.add(meta.name)
      bound = getattr(instance, attr_name)
      if meta.kind == 'llm':
        assert meta.output_type is not None
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
        assert meta.target is not None
        steps.append(
          BackStep(
            meta.name,
            target=meta.target,
            condition=bound,
            max_iterations=meta.max_iterations,
          )
        )
  if not steps:
    msg = f'{type(instance).__name__} has no @step-decorated methods'
    raise NotImplementedError(msg)
  return steps
