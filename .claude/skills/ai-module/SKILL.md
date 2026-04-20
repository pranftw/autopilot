---
name: ai-module
description: AI eval generation and judging module covering generators, judges, step workflows, data abstractions, and checkpointing. Use when creating a generator, judge, or working with eval datasets.
---

## Creating a generator

Two approaches: step decorators (preferred) or override `define_steps()`.

### Decorator-based (preferred)

```python
from autopilot.ai.generator import DataGenerator, stratify_by
from autopilot.ai.steps import back_step, llm_step, python_step

@stratify_by('domain', 'metadata.difficulty_level')
class MyGenerator(DataGenerator[MyConfig, MyCustom]):
  def create_slots(self, config):
    planner = SlotPlanner(vars=config.custom.vars, seed=config.seed)
    return planner.create_slots(config.total_count)

  @llm_step('generate', output_type=MyOutput, instructions='...')
  def generate(self, ctx):
    return build_prompt(ctx['slot'])

  @python_step('validate')
  def validate(self, ctx):
    return validate_item(ctx)

  @back_step('retry', target='generate', max_iterations=3)
  def should_retry(self, ctx):
    return not ctx.get('validate', {}).get('valid', False)

  def assemble_item(self, slot, step_results):
    return DataItem(id=slot['id'], turns=[...], custom=MyCustom(...))
```

### Override-based

```python
class MyGenerator(DataGenerator[MyConfig, MyCustom]):
  def define_steps(self, config):
    return [
      LLMStep('generate', output_type=MyOutput, instructions='...'),
      PythonStep('validate', fn=my_validate_fn),
    ]
  # ... create_slots, assemble_item, stratify_key as before
```

Default `define_steps()` calls `collect_steps(self)` to gather `@step`-decorated methods. Override for full control.

## Step decorators

- `@llm_step(name, *, output_type, instructions=None)` -- marks method as LLM step
- `@python_step(name)` -- marks method as Python step
- `@back_step(name, *, target, max_iterations=3)` -- marks method as conditional loopback
- `@stratify_by(*fields)` -- class decorator, auto-generates `stratify_key()` from dotted field paths on `item.custom`
- `collect_steps(instance)` -- gathers decorated methods in definition order

## Creating a judge

Same two approaches (decorators or override):

```python
class MyJudge(Judge[MyJudgeConfig, MyJudgeCustom, MyResultCustom]):
  @llm_step('classify', output_type=Verdict, instructions='...')
  def classify(self, ctx):
    return build_classify_prompt(ctx['item'])

  def assemble_result(self, item, step_results):
    return JudgeResult(id=item.id, verdict=..., custom=MyResultCustom(...))

  def build_summary(self, results):
    return {'total': len(results), 'correct': sum(...)}
```

## Step workflow

Three step types compose into workflows:

- `LLMStep(name, output_type, instructions)`: Creates a pydantic-ai Agent, returns structured Pydantic model
- `PythonStep(name, fn)`: Runs a regular Python function. `fn(context) -> dict`
- `BackStep(name, target, condition, max_iterations)`: Loops back to target step if condition is True

Steps execute in order. Each step's result is merged into the context dict under its name.

## Data abstractions

- `ListDataset[T]` (`autopilot.data.dataset`): Map-style dataset with `from_jsonl()`, `to_jsonl()`, `subset()`
- `StreamingDataset[T]` (`autopilot.data.dataset`): Lazy JSONL line-by-line reading
- `StratifiedSplitter` (`autopilot.ai.data`): Split dataset with matched distributions across train/val/test
- `SlotPlanner` (`autopilot.ai.data`): Built-in slot generation from `VarDef` weighted distributions
- `[T]` (`autopilot.ai.data`): Lightning-style lifecycle (prepare_data, setup, train/val/test_dataset)

## Checkpointing

`CheckpointManager` provides resumable runs:

```python
from autopilot.ai.checkpoints import CheckpointManager

ckpt = CheckpointManager(path=output_dir / 'checkpoint.jsonl')
ckpt.save_header(config_hash='abc', subsystem='generate')

for item in items:
  if ckpt.is_completed(item.id):
    continue
  result = process(item)
  ckpt.save_event('result', item.id, result.model_dump())
```

## CLI commands

AI commands dispatch through `ctx.generator` / `ctx.judge`. Run via project:

```
autopilot -p <project> ai generate run --config <path> [--total-count N] [--seed N]
autopilot -p <project> ai generate resume --checkpoint <path>
autopilot -p <project> ai generate dry-run --config <path>
autopilot -p <project> ai generate split --config <path> --split <name>
autopilot -p <project> ai judge run --input <path> [--num-parallel N] [--max-rpm N]
autopilot -p <project> ai judge resume --checkpoint <path> --input <path>
autopilot -p <project> ai judge summarize --input <path>
```

Without a project, handlers raise `ValueError('no generator/judge configured')`.

AI handlers call `_require(ctx, 'generator')` / `_require(ctx, 'judge')` before dispatching.
