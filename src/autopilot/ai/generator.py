"""Protocol and base class for eval dataset generators."""

from autopilot.ai.checkpoints import CheckpointManager
from autopilot.ai.data import StratifiedSplitter
from autopilot.ai.models import IT, C, DataItem, GeneratorConfig
from autopilot.ai.runtime import ParallelRunner, RPMLimiter
from autopilot.ai.steps import Step, collect_steps, run_step_workflow
from autopilot.cli.output import Output
from autopilot.data.dataset import ListDataset
from autopilot.tracking.io import atomic_write_json
from pathlib import Path
from typing import Generic, Protocol
import asyncio
import hashlib


def stratify_by(*fields: str):
  """Class decorator: auto-generate stratify_key from field paths."""

  def decorator(cls):
    def _stratify_key(self, item):
      parts = []
      for f in fields:
        val = item.custom
        for attr in f.split('.'):
          val = val.get(attr, '') if isinstance(val, dict) else getattr(val, attr, '')
        parts.append(str(val))
      return ':'.join(parts)

    cls.stratify_key = _stratify_key
    return cls

  return decorator


class DataGeneratorProtocol(Protocol[C, IT]):
  """Structural typing contract for eval generators."""

  def create_slots(self, config: GeneratorConfig[C]) -> list[dict]: ...

  def define_steps(self, config: GeneratorConfig[C]) -> list[Step]: ...

  def assemble_item(self, slot: dict, step_results: dict) -> DataItem[IT] | None: ...

  def stratify_key(self, item: DataItem[IT]) -> str: ...


class DataGenerator(Generic[C, IT]):
  """Base for eval dataset generation using step-based workflows.

  Subclass and override the abstract methods. Pass the instance directly.
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

  def run(
    self,
    config: GeneratorConfig[C],
    output_dir: Path,
    output: Output,
  ) -> dict:
    return asyncio.run(self.async_run(config, output_dir, output))

  async def async_run(
    self,
    config: GeneratorConfig[C],
    output_dir: Path,
    output: Output,
  ) -> dict:
    """Full run: plan slots -> run step workflow per slot -> split -> write."""
    output_dir.mkdir(parents=True, exist_ok=True)
    slots = self.create_slots(config)
    steps = self.define_steps(config)
    step_names = [s.name for s in steps]

    # Config hash for checkpoint
    config_hash = hashlib.sha256(config.model_dump_json().encode()).hexdigest()[:16]

    # Checkpoint
    ckpt_path = output_dir / 'checkpoint.jsonl'
    ckpt = CheckpointManager(ckpt_path)
    ckpt.save_header(
      config_hash=config_hash,
      subsystem='generate',
      args={
        'dataset_id': config.dataset_id,
        'total_count': config.total_count,
        'model': config.run.model,
        'step_names': step_names,
      },
    )

    output.info(f'Generating {len(slots)} items with steps: {step_names}')

    # Process slots
    items: list[DataItem[IT]] = []

    async def process_slot(slot: dict) -> dict:
      slot_id = slot.get('id', '')
      if ckpt.is_completed(slot_id):
        return {'id': slot_id, 'skipped': True}
      try:
        result = await run_step_workflow(
          steps=steps,
          initial_context={'slot': slot},
          model=config.run.model,
          run_config=config.run,
        )
        item = self.assemble_item(slot, result)
        if item is not None:
          ckpt.save_event('result', slot_id, {'item': item.model_dump()})
          return {'id': slot_id, 'item': item}
        else:
          ckpt.save_event('skip', slot_id, {'reason': 'rejected by assemble_item'})
          return {'id': slot_id, 'skipped': True}
      except Exception as exc:
        ckpt.save_event('error', slot_id, {'error': str(exc)})
        return {'id': slot_id, 'error': str(exc)}

    limiter = RPMLimiter(config.run.max_rpm, config.run.rpm_safety_margin)
    runner = ParallelRunner(config.run.num_parallel, limiter)

    def on_complete(result: dict) -> None:
      item = result.get('item')
      if item is not None:
        items.append(item)

    await runner.run(slots, process_slot, on_complete=on_complete)

    output.info(f'Generated {len(items)} items, writing output...')

    # Split
    dataset = ListDataset(items)
    splitter = StratifiedSplitter(
      ratios=config.split_ratios,
      key_fn=self.stratify_key,
      seed=config.seed,
    )
    splits = splitter.split(dataset)

    # Write all.jsonl
    dataset.to_jsonl(output_dir / 'all.jsonl')

    # Write per-split files
    for split_name, split_ds in splits.items():
      split_ds.to_jsonl(output_dir / f'{split_name}.jsonl')

    # Write metadata
    metadata = {
      'dataset_id': config.dataset_id,
      'total_generated': len(items),
      'total_requested': config.total_count,
      'splits': {name: len(ds) for name, ds in splits.items()},
      'config_hash': config_hash,
    }
    atomic_write_json(output_dir / 'metadata.json', metadata)

    summary = ckpt.summary()
    summary['total_items'] = len(items)
    summary['splits'] = {name: len(ds) for name, ds in splits.items()}
    output.result(summary)
    return summary

  async def resume(
    self,
    checkpoint_path: Path,
    config: GeneratorConfig[C],
    output_dir: Path,
    output: Output,
  ) -> dict:
    """Resume from checkpoint, re-run only incomplete slots."""
    slots = self.create_slots(config)
    steps = self.define_steps(config)

    ckpt = CheckpointManager(checkpoint_path)
    completed = ckpt.completed_ids()
    remaining = [s for s in slots if s.get('id', '') not in completed]

    output.info(f'Resuming: {len(completed)} done, {len(remaining)} remaining')

    items: list[DataItem[IT]] = []

    # Load already-completed items from checkpoint
    for event in ckpt.load_events():
      if event.get('type') != 'result':
        continue
      pl = event.get('payload') or {}
      raw_item = pl.get('item')
      if isinstance(raw_item, dict):
        items.append(DataItem.model_validate(raw_item))

    async def process_slot(slot: dict) -> dict:
      slot_id = slot.get('id', '')
      try:
        result = await run_step_workflow(
          steps=steps,
          initial_context={'slot': slot},
          model=config.run.model,
          run_config=config.run,
        )
        item = self.assemble_item(slot, result)
        if item is not None:
          ckpt.save_event('result', slot_id, {'item': item.model_dump()})
          return {'id': slot_id, 'item': item}
        else:
          ckpt.save_event('skip', slot_id, {'reason': 'rejected'})
          return {'id': slot_id, 'skipped': True}
      except Exception as exc:
        ckpt.save_event('error', slot_id, {'error': str(exc)})
        return {'id': slot_id, 'error': str(exc)}

    limiter = RPMLimiter(config.run.max_rpm, config.run.rpm_safety_margin)
    runner = ParallelRunner(config.run.num_parallel, limiter)

    def on_complete(result: dict) -> None:
      item = result.get('item')
      if item is not None:
        items.append(item)

    await runner.run(remaining, process_slot, on_complete=on_complete)

    summary = ckpt.summary()
    summary['resumed_items'] = len(items)
    output.result(summary)
    return summary

  def dry_run(self, config: GeneratorConfig[C], output: Output) -> dict:
    """Plan slots + list steps, no LLM calls."""
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
