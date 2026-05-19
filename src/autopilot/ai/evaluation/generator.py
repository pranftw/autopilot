"""Protocol and base class for eval dataset generators."""

from autopilot.ai.agents.agent import StepAgent
from autopilot.ai.data import StratifiedSplitter
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
from autopilot.ai.evaluation.schemas import IT, C, DataItem, GeneratorConfig
from autopilot.ai.evaluation.steps import Step, collect_steps, run_step_workflow
from autopilot.ai.runtime import SlidingWindowLimiter
from autopilot.core.errors import AIError
from autopilot.data.dataset import ListDataset
from autopilot.tracking.io import atomic_write_json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar
import asyncio
import logging

logger = logging.getLogger(__name__)


_ClassT = TypeVar('_ClassT', bound=type[Any])


def stratify_by(*fields: str) -> Callable[[_ClassT], _ClassT]:
  """Class decorator: auto-generate stratify_key from field paths.

  Returns:
    Decorator that adds ``stratify_key`` to the wrapped class.
  """

  def decorator(cls: _ClassT) -> _ClassT:
    def _stratify_key(_self: Any, item: Any) -> str:
      parts: list[str] = []
      for f in fields:
        val = item.custom
        for attr in f.split('.'):
          val = val[attr] if isinstance(val, dict) else getattr(val, attr)
        parts.append(str(val))
      return ':'.join(parts)

    cls.stratify_key = _stratify_key
    return cls

  return decorator


class GeneratorAgentProtocol(Protocol[C, IT]):
  """Structural typing contract for eval generators."""

  def create_slots(self, config: GeneratorConfig[C]) -> list[dict]:
    """Plan generation slots from configuration."""
    ...

  def define_steps(self, config: GeneratorConfig[C]) -> list[Step]:
    """Return ordered workflow steps for the given configuration."""
    ...

  def assemble_item(self, slot: dict, step_results: dict) -> DataItem[IT] | None:
    """Build a dataset item from a slot and step outputs; ``None`` means reject."""
    ...

  def stratify_key(self, item: DataItem[IT]) -> str:
    """Return a key used for stratified splitting of ``item``."""
    ...


class GeneratorAgent(StepAgent, Generic[C, IT]):
  """Agent for eval dataset generation using step-based workflows.

  Subclass and override:
    create_slots(config) -> list[dict]    -- plan generation slots from config
    define_steps(config) -> list[Step]    -- return workflow steps (default: collect_steps)
    assemble_item(slot, step_results)     -- build DataItem from results (None = rejected)
    stratify_key(item) -> str             -- key for stratified splitting

  Entry points:
    run(config, output_dir, output)       -- sync (asyncio.run over async_run)
    async_run(config, output_dir, output) -- async full pipeline

  The @stratify_by class decorator auto-generates stratify_key() from
  dotted field paths on item.custom.
  """

  def create_slots(self, config: GeneratorConfig[C]) -> list[dict]:
    """Plan generation slots from config vars/distributions."""
    raise NotImplementedError

  def define_steps(self, config: GeneratorConfig[C]) -> list[Step]:
    """Return ordered workflow steps from @step-decorated methods. Override for full control."""
    return collect_steps(self)

  def assemble_item(self, slot: dict, step_results: dict) -> DataItem[IT] | None:
    """Assemble final item from all step results. None = rejected."""
    raise NotImplementedError

  def stratify_key(self, item: DataItem[IT]) -> str:
    """Return key for stratified splitting."""
    raise NotImplementedError

  def _slot_result_from_workflow(
    self,
    slot: dict,
    slot_id: str,
    result: dict,
    ckpt: CheckpointManager,
    *,
    for_resume: bool,
  ) -> dict:
    """Assemble item from workflow result and record checkpoint event.

    Returns:
      Dict with ``id`` and either ``item``, ``skipped: True``, or ``error`` string.
    """
    item = self.assemble_item(slot, result)
    if item is not None:
      ckpt.save_event('result', slot_id, {'item': item.model_dump()})
      return {'id': slot_id, 'item': item}
    reason = 'rejected' if for_resume else 'rejected by assemble_item'
    ckpt.save_event('skip', slot_id, {'reason': reason})
    return {'id': slot_id, 'skipped': True}

  async def _process_slot_ckpt(
    self,
    ckpt: CheckpointManager,
    steps: list[Step],
    config: GeneratorConfig[C],
    output: EvaluationOutputProtocol,
    slot: dict,
    *,
    for_resume: bool,
  ) -> dict:
    """Process a single slot with checkpointing. Shared by async_run and resume.

    Returns:
      Dict with ``id`` and ``skipped``, ``item``, or ``error`` depending on outcome.
    """
    slot_id = slot['id']
    if not for_resume and ckpt.is_completed(slot_id):
      return {'id': slot_id, 'skipped': True}
    try:
      result = await run_step_workflow(
        steps=steps,
        initial_context={'slot': slot},
        model=config.run.model,
        run_config=config.run,
      )
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
      log_item_failure(f'slot {slot_id}', exc, output)
      ckpt.save_event('error', slot_id, {'error': str(exc)})
      return {'id': slot_id, 'error': str(exc)}
    except AIError as exc:
      log_item_failure(f'slot {slot_id}', exc, output, unexpected=True)
      ckpt.save_event('error', slot_id, {'error': str(exc)})
      return {'id': slot_id, 'error': str(exc)}
    else:
      return self._slot_result_from_workflow(slot, slot_id, result, ckpt, for_resume=for_resume)

  def run(
    self,
    config: GeneratorConfig[C],
    output_dir: Path,
    output: EvaluationOutputProtocol,
  ) -> dict[str, Any]:
    """Runs the full generation pipeline synchronously.

    Wraps :meth:`async_run` with :func:`asyncio.run`.

    Args:
      config: Generator configuration (model, counts, splits, custom payload).
      output_dir: Directory for checkpoint, split JSONL files, and metadata.
      output: Output sink satisfying :class:`EvaluationOutputProtocol`.

    Returns:
      Summary dict with checkpoint totals, split sizes, and metadata.
    """
    return asyncio.run(self.async_run(config, output_dir, output))

  def _generator_run_setup(
    self,
    config: GeneratorConfig[C],
    output_dir: Path,
  ) -> tuple[list[dict], list[Step], CheckpointManager, str]:
    """Plan slots, collect steps, initialize checkpoint, compute config hash.

    Returns:
      Tuple of ``(slots, steps, checkpoint_manager, config_hash_hex)``.
    """
    slots = self.create_slots(config)
    steps = self.define_steps(config)
    step_names = [s.name for s in steps]

    run_ctx = EvalRunContext(
      output_dir,
      hash_eval_config(config.model_dump(mode='json')),
      len(slots),
    )

    ckpt = write_checkpoint_header(
      run_ctx.output_dir,
      run_ctx.config_hash,
      'generate',
      {
        'dataset_id': config.dataset_id,
        'total_count': config.total_count,
        'model': config.run.model,
        'step_names': step_names,
      },
    )
    return slots, steps, ckpt, run_ctx.config_hash

  async def _dispatch_parallel_slots(
    self,
    slots: list[dict],
    config: GeneratorConfig[C],
    steps: list[Step],
    ckpt: CheckpointManager,
    output: EvaluationOutputProtocol,
  ) -> list[DataItem[IT]]:
    """Run all slots in parallel, collecting successfully assembled items.

    Returns:
      List of successfully assembled :class:`DataItem` instances.
    """
    items: list[DataItem[IT]] = []
    limiter = SlidingWindowLimiter(config.run.max_rpm, config.run.rpm_safety_margin)

    def on_complete(result: dict) -> None:
      item = result.get('item')
      if item is not None:
        items.append(item)

    await run_parallel_items(
      slots,
      lambda s: self._process_slot_ckpt(ckpt, steps, config, output, s, for_resume=False),
      limiter,
      config.run.num_parallel,
      output,
      on_complete=on_complete,
    )
    return items

  def _write_generator_outputs(
    self,
    items: list[DataItem[IT]],
    config: GeneratorConfig[C],
    config_hash: str,
    output_dir: Path,
    ckpt: CheckpointManager,
    output: EvaluationOutputProtocol,
  ) -> dict[str, Any]:
    """Write all.jsonl, splits, metadata.json, and emit summary.

    Returns:
      Summary dict including checkpoint stats, item counts, and split sizes.
    """
    dataset = ListDataset(items)
    splitter = StratifiedSplitter(
      ratios=config.split_ratios,
      key_fn=self.stratify_key,
      seed=config.seed,
    )
    splits = splitter.split(dataset)

    dataset.to_jsonl(output_dir / 'all.jsonl')
    for split_name, split_ds in splits.items():
      split_ds.to_jsonl(output_dir / f'{split_name}.jsonl')

    metadata = {
      'dataset_id': config.dataset_id,
      'total_generated': len(items),
      'total_requested': config.total_count,
      'splits': {name: len(ds) for name, ds in splits.items()},
      'config_hash': config_hash,
    }
    atomic_write_json(output_dir / 'metadata.json', metadata)

    summary: dict[str, Any] = ckpt.summary()
    summary['total_items'] = len(items)
    summary['splits'] = {name: len(ds) for name, ds in splits.items()}
    output.result(summary)
    return summary

  async def async_run(
    self,
    config: GeneratorConfig[C],
    output_dir: Path,
    output: EvaluationOutputProtocol,
  ) -> dict[str, Any]:
    """Full run: plan slots -> run step workflow per slot -> split -> write.

    Returns:
      Summary dict written via ``output`` and returned to the caller.
    """
    slots, steps, ckpt, config_hash = self._generator_run_setup(config, output_dir)
    output.info(f'Generating {len(slots)} items with steps: {[s.name for s in steps]}')
    items = await self._dispatch_parallel_slots(slots, config, steps, ckpt, output)
    output.info(f'Generated {len(items)} items, writing output...')
    return self._write_generator_outputs(items, config, config_hash, output_dir, ckpt, output)

  async def resume(
    self,
    checkpoint_path: Path,
    config: GeneratorConfig[C],
    output: EvaluationOutputProtocol,
  ) -> dict[str, Any]:
    """Resumes generation from an existing checkpoint file.

    Re-creates slots from config, identifies incomplete ones via the
    checkpoint, and re-runs only the remaining slots. Previously
    completed items are loaded from checkpoint events.

    Args:
      checkpoint_path: Path to the ``checkpoint.jsonl`` file.
      config: Same generator config used for the original run.
      output: Output sink satisfying :class:`EvaluationOutputProtocol`.

    Returns:
      Summary dict with resumed item count and checkpoint statistics.
    """
    slots = self.create_slots(config)
    steps = self.define_steps(config)

    run_ctx = EvalRunContext(checkpoint_path.parent, '', len(slots))
    ckpt = CheckpointManager(run_ctx.checkpoint_path())
    completed = ckpt.completed_ids()
    remaining = [s for s in slots if s['id'] not in completed]

    resume_from_checkpoint(ckpt, output, run_ctx.total_items)

    items: list[DataItem[IT]] = []

    for event in ckpt.load_events():
      if event.get('type') != 'result':
        continue
      pl = event.get('payload', {})
      raw_item = pl.get('item')
      if isinstance(raw_item, dict):
        items.append(DataItem.model_validate(raw_item))

    limiter = SlidingWindowLimiter(config.run.max_rpm, config.run.rpm_safety_margin)

    def on_complete(result: dict) -> None:
      item = result.get('item')
      if item is not None:
        items.append(item)

    await run_parallel_items(
      remaining,
      lambda s: self._process_slot_ckpt(ckpt, steps, config, output, s, for_resume=True),
      limiter,
      config.run.num_parallel,
      output,
      on_complete=on_complete,
    )

    summary: dict[str, Any] = ckpt.summary()
    summary['resumed_items'] = len(items)
    output.result(summary)
    return summary

  def dry_run(self, config: GeneratorConfig[C], output: EvaluationOutputProtocol) -> dict[str, Any]:
    """Plan slots + list steps, no LLM calls.

    Returns:
      Dict describing planned slots, step names, split ratios, model, and dataset id.
    """
    slots = self.create_slots(config)
    steps = self.define_steps(config)
    result = {
      'total_slots': len(slots),
      'step_names': [s.name for s in steps],
      'split_ratios': config.split_ratios,
      'model': config.run.model,
      'dataset_id': config.dataset_id,
    }
    output.result(result)
    return result
