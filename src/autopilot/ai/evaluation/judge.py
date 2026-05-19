"""Protocol and base class for evaluation judges."""

from autopilot.ai.agents.agent import StepAgent
from autopilot.ai.evaluation.checkpoints import CheckpointManager
from autopilot.ai.evaluation.pipeline import (
  EvalRunContext,
  hash_eval_config,
  log_item_failure,
  resume_from_checkpoint,
  run_parallel_items,
  write_checkpoint_header,
)
from autopilot.ai.evaluation.protocols import EvaluationOutputProtocol
from autopilot.ai.evaluation.schemas import JC, JI, JR, JudgeConfig, JudgeInput, JudgeResult
from autopilot.ai.evaluation.steps import Step, collect_steps, run_step_workflow
from autopilot.ai.runtime import SlidingWindowLimiter
from autopilot.core.errors import AIError
from autopilot.tracking.io import atomic_write_json
from pathlib import Path
from typing import Any, Generic, Protocol
import asyncio
import logging

logger = logging.getLogger(__name__)


class JudgeAgentProtocol(Protocol[JC, JI, JR]):
  """Structural typing contract for judges."""

  def define_steps(self, config: JudgeConfig[JC]) -> list[Step]:
    """Return ordered workflow steps for the judge configuration."""
    ...

  def assemble_result(self, item: JudgeInput[JI], step_results: dict) -> JudgeResult[JR]:
    """Build a structured judge result from step outputs."""
    ...

  def build_summary(self, results: list[JudgeResult[JR]]) -> dict[str, Any]:
    """Aggregate per-item judge results into a summary dict."""
    ...


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

  def build_summary(self, results: list[JudgeResult[JR]]) -> dict[str, Any]:
    """Aggregate results into summary dict."""
    raise NotImplementedError

  async def _process_item_ckpt(
    self,
    ckpt: CheckpointManager,
    steps: list[Step],
    config: JudgeConfig[JC],
    output: EvaluationOutputProtocol,
    unit: JudgeInput[JI],
    *,
    for_resume: bool,
  ) -> dict:
    """Process a single judge item with checkpointing. Shared by async_run and resume.

    Returns:
      Dict with ``id`` and ``skipped``, ``result``, or ``error`` depending on outcome.
    """
    if not for_resume and ckpt.is_completed(unit.item_id):
      return {'id': unit.item_id, 'skipped': True}
    try:
      step_results = await run_step_workflow(
        steps=steps,
        initial_context={'item': unit.model_dump()},
        model=config.run.model,
        run_config=config.run,
      )
      jr = self.assemble_result(unit, step_results)
      ckpt.save_event('result', unit.item_id, {'result': jr.model_dump()})
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
      log_item_failure(unit.item_id, exc, output)
      ckpt.save_event('error', unit.item_id, {'error': str(exc)})
      return {'id': unit.item_id, 'error': str(exc)}
    except AIError as exc:
      log_item_failure(unit.item_id, exc, output, unexpected=True)
      ckpt.save_event('error', unit.item_id, {'error': str(exc)})
      return {'id': unit.item_id, 'error': str(exc)}
    else:
      return {'id': unit.item_id, 'result': jr}

  def run(
    self,
    items: list[JudgeInput[JI]],
    config: JudgeConfig[JC],
    output_dir: Path,
    output: EvaluationOutputProtocol,
  ) -> dict[str, Any]:
    """Runs judging for all ``items`` synchronously.

    Wraps :meth:`async_run` with :func:`asyncio.run`.

    Args:
      items: Inputs to judge (identifiers and payload per schema).
      config: Judge configuration (model, parallelism, prompts, custom payload).
      output_dir: Directory for checkpoint and ``output.json``.
      output: Output sink satisfying :class:`EvaluationOutputProtocol`.

    Returns:
      Final summary dict merging checkpoint statistics and aggregated summary.
    """
    return asyncio.run(self.async_run(items, config, output_dir, output))

  async def async_run(
    self,
    items: list[JudgeInput[JI]],
    config: JudgeConfig[JC],
    output_dir: Path,
    output: EvaluationOutputProtocol,
  ) -> dict[str, Any]:
    """Full judge run: run step workflow per item, write output.

    Returns:
      Final summary dict merging checkpoint statistics and ``build_summary`` output.
    """
    steps = self.define_steps(config)
    step_names = [s.name for s in steps]

    run_ctx = EvalRunContext(
      output_dir,
      hash_eval_config(config.model_dump(mode='json')),
      len(items),
    )

    ckpt = write_checkpoint_header(
      run_ctx.output_dir,
      run_ctx.config_hash,
      'judge',
      {
        'total_items': run_ctx.total_items,
        'model': config.run.model,
        'step_names': step_names,
      },
    )

    output.info(f'Judging {run_ctx.total_items} items with steps: {step_names}')

    results: list[JudgeResult[JR]] = []

    limiter = SlidingWindowLimiter(config.run.max_rpm, config.run.rpm_safety_margin)

    def on_complete(result_dict: dict) -> None:
      judge_result = result_dict.get('result')
      if judge_result is not None:
        results.append(judge_result)

    await run_parallel_items(
      items,
      lambda u: self._process_item_ckpt(ckpt, steps, config, output, u, for_resume=False),
      limiter,
      config.run.num_parallel,
      output,
      on_complete=on_complete,
    )

    output.info(f'Judged {len(results)} items, writing output...')

    summary = self.build_summary(results)

    output_payload = {
      'summary': summary,
      'results': [r.model_dump() for r in results],
      'config_hash': run_ctx.config_hash,
    }
    atomic_write_json(run_ctx.output_dir / 'output.json', output_payload)

    ckpt_summary = ckpt.summary()
    final = {**ckpt_summary, 'summary': summary}
    output.result(final)
    return final

  async def resume(
    self,
    checkpoint_path: Path,
    items: list[JudgeInput[JI]],
    config: JudgeConfig[JC],
    output: EvaluationOutputProtocol,
  ) -> dict[str, Any]:
    """Resumes judging from an existing checkpoint file.

    Loads previously completed results from the checkpoint, identifies
    remaining items, and processes only those that are incomplete.

    Args:
      checkpoint_path: Path to the ``checkpoint.jsonl`` file.
      items: Full item list (same as original run).
      config: Same judge config used for the original run.
      output: Output sink satisfying :class:`EvaluationOutputProtocol`.

    Returns:
      Summary dict with resumed item count, checkpoint stats, and aggregated
      summary from :meth:`build_summary`.
    """
    steps = self.define_steps(config)

    run_ctx = EvalRunContext(checkpoint_path.parent, '', len(items))
    ckpt = CheckpointManager(run_ctx.checkpoint_path())
    completed = ckpt.completed_ids()
    remaining = [item for item in items if item.item_id not in completed]

    prior_results: list[JudgeResult[JR]] = []
    for event in ckpt.load_events():
      if event.get('type') != 'result':
        continue
      payload = event.get('payload', {})
      raw_result = payload.get('result')
      if isinstance(raw_result, dict):
        prior_results.append(JudgeResult.model_validate(raw_result))

    resume_from_checkpoint(ckpt, output, run_ctx.total_items)

    batch_results: list[JudgeResult[JR]] = []

    limiter = SlidingWindowLimiter(config.run.max_rpm, config.run.rpm_safety_margin)

    def on_complete(result_dict: dict) -> None:
      judge_result = result_dict.get('result')
      if judge_result is not None:
        batch_results.append(judge_result)

    await run_parallel_items(
      remaining,
      lambda u: self._process_item_ckpt(ckpt, steps, config, output, u, for_resume=True),
      limiter,
      config.run.num_parallel,
      output,
      on_complete=on_complete,
    )

    all_results = prior_results + batch_results
    summary = self.build_summary(all_results)
    ckpt_summary = ckpt.summary()
    total_success = len(prior_results) + len(batch_results)
    final = {**ckpt_summary, 'resumed_items': total_success, 'summary': summary}
    output.result(final)
    return final
