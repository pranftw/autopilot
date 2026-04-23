"""Protocol and base class for evaluation judges."""

from autopilot.ai.agents.agent import StepAgent
from autopilot.ai.evaluation.checkpoints import CheckpointManager
from autopilot.ai.evaluation.schemas import JC, JI, JR, JudgeConfig, JudgeInput, JudgeResult
from autopilot.ai.evaluation.steps import Step, collect_steps, run_step_workflow
from autopilot.ai.runtime import ParallelRunner, SlidingWindowLimiter
from autopilot.cli.output import Output
from autopilot.tracking.io import atomic_write_json
from pathlib import Path
from typing import Generic, Protocol
import asyncio
import hashlib


class JudgeAgentProtocol(Protocol[JC, JI, JR]):
  """Structural typing contract for judges."""

  def define_steps(self, config: JudgeConfig[JC]) -> list[Step]: ...

  def assemble_result(self, item: JudgeInput[JI], step_results: dict) -> JudgeResult[JR]: ...

  def build_summary(self, results: list[JudgeResult[JR]]) -> dict: ...


class JudgeAgent(StepAgent, Generic[JC, JI, JR]):
  """Agent for evaluation judging using step-based workflows.

  Subclass and override:
    define_steps(config) -> list[Step]                    -- workflow steps
    assemble_result(item, step_results) -> JudgeResult    -- build result from steps
    build_summary(results) -> dict                        -- aggregate results

  Entry points:
    run(items, config, output_dir, output)       -- sync
    async_run(items, config, output_dir, output) -- async

  Same Step abstraction as GeneratorAgent. LLM steps produce structured
  verdicts, Python steps run deterministic checks.
  """

  def define_steps(self, config: JudgeConfig[JC]) -> list[Step]:
    """Return ordered workflow steps from @step-decorated methods. Override for full control."""
    return collect_steps(self)

  def assemble_result(self, item: JudgeInput[JI], step_results: dict) -> JudgeResult[JR]:
    """Assemble final result from all step results."""
    raise NotImplementedError

  def build_summary(self, results: list[JudgeResult[JR]]) -> dict:
    """Aggregate results into summary dict."""
    raise NotImplementedError

  def run(
    self,
    items: list[JudgeInput[JI]],
    config: JudgeConfig[JC],
    output_dir: Path,
    output: Output,
  ) -> dict:
    return asyncio.run(self.async_run(items, config, output_dir, output))

  async def async_run(
    self,
    items: list[JudgeInput[JI]],
    config: JudgeConfig[JC],
    output_dir: Path,
    output: Output,
  ) -> dict:
    """Full judge run: run step workflow per item, write output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = self.define_steps(config)
    step_names = [s.name for s in steps]

    config_hash = hashlib.sha256(config.model_dump_json().encode()).hexdigest()[:16]

    ckpt_path = output_dir / 'checkpoint.jsonl'
    ckpt = CheckpointManager(ckpt_path)
    ckpt.save_header(
      config_hash=config_hash,
      subsystem='judge',
      args={
        'total_items': len(items),
        'model': config.run.model,
        'step_names': step_names,
      },
    )

    output.info(f'Judging {len(items)} items with steps: {step_names}')

    results: list[JudgeResult[JR]] = []

    async def process_item(item: JudgeInput[JI]) -> dict:
      if ckpt.is_completed(item.id):
        return {'id': item.id, 'skipped': True}
      try:
        step_results = await run_step_workflow(
          steps=steps,
          initial_context={'item': item.model_dump()},
          model=config.run.model,
          run_config=config.run,
        )
        result = self.assemble_result(item, step_results)
        ckpt.save_event('result', item.id, {'result': result.model_dump()})
        return {'id': item.id, 'result': result}
      except Exception as exc:
        ckpt.save_event('error', item.id, {'error': str(exc)})
        return {'id': item.id, 'error': str(exc)}

    limiter = SlidingWindowLimiter(config.run.max_rpm, config.run.rpm_safety_margin)
    runner = ParallelRunner(config.run.num_parallel, limiter=limiter)

    def on_complete(result_dict: dict) -> None:
      r = result_dict.get('result')
      if r is not None:
        results.append(r)

    await runner.run(items, process_item, on_complete=on_complete)

    output.info(f'Judged {len(results)} items, writing output...')

    summary = self.build_summary(results)

    output_payload = {
      'summary': summary,
      'results': [r.model_dump() for r in results],
      'config_hash': config_hash,
    }
    atomic_write_json(output_dir / 'output.json', output_payload)

    ckpt_summary = ckpt.summary()
    final = {**ckpt_summary, 'summary': summary}
    output.result(final)
    return final

  async def resume(
    self,
    checkpoint_path: Path,
    items: list[JudgeInput[JI]],
    config: JudgeConfig[JC],
    output_dir: Path,
    output: Output,
  ) -> dict:
    """Resume from checkpoint."""
    steps = self.define_steps(config)

    ckpt = CheckpointManager(checkpoint_path)
    completed = ckpt.completed_ids()
    remaining = [item for item in items if item.id not in completed]

    output.info(f'Resuming: {len(completed)} done, {len(remaining)} remaining')

    results: list[JudgeResult[JR]] = []

    async def process_item(item: JudgeInput[JI]) -> dict:
      try:
        step_results = await run_step_workflow(
          steps=steps,
          initial_context={'item': item.model_dump()},
          model=config.run.model,
          run_config=config.run,
        )
        result = self.assemble_result(item, step_results)
        ckpt.save_event('result', item.id, {'result': result.model_dump()})
        return {'id': item.id, 'result': result}
      except Exception as exc:
        ckpt.save_event('error', item.id, {'error': str(exc)})
        return {'id': item.id, 'error': str(exc)}

    limiter = SlidingWindowLimiter(config.run.max_rpm, config.run.rpm_safety_margin)
    runner = ParallelRunner(config.run.num_parallel, limiter=limiter)

    def on_complete(result_dict: dict) -> None:
      r = result_dict.get('result')
      if r is not None:
        results.append(r)

    await runner.run(remaining, process_item, on_complete=on_complete)

    summary = self.build_summary(results)
    ckpt_summary = ckpt.summary()
    final = {**ckpt_summary, 'resumed_items': len(results), 'summary': summary}
    output.result(final)
    return final
