"""Tests for DataGenerator base class."""

from autopilot.ai.checkpoints import CheckpointManager
from autopilot.ai.generator import DataGenerator
from autopilot.ai.models import (
  ConversationTurn,
  DataItem,
  GeneratorConfig,
  RetryConfig,
  RunConfig,
)
from autopilot.ai.steps import PythonStep
from autopilot.cli.output import Output
from autopilot.data.dataset import ListDataset
from pathlib import Path
from pydantic import BaseModel
from unittest.mock import AsyncMock, patch
import pytest


class StubCustom(BaseModel):
  value: str


class StubGenConfig(BaseModel):
  prefix: str = 'STUB'


def _make_run_config() -> RunConfig:
  return RunConfig(
    model='test-model',
    num_parallel=1,
    max_rpm=100,
    rpm_safety_margin=1.0,
    retry=RetryConfig(max_retries=1, min_timeout_ms=100, max_timeout_ms=1000, backoff_factor=2),
    max_tool_steps=5,
    max_output_tokens=1024,
  )


def _make_config(total: int = 5) -> GeneratorConfig[StubGenConfig]:
  return GeneratorConfig(
    run=_make_run_config(),
    dataset_id='test_ds',
    seed=42,
    total_count=total,
    split_ratios={'train': 0.8, 'val': 0.1, 'test': 0.1},
    system_prompt='test',
    custom=StubGenConfig(),
  )


class StubGenerator(DataGenerator[StubGenConfig, StubCustom]):
  def create_slots(self, config):
    return [{'id': f'S{i:04d}'} for i in range(config.total_count)]

  def define_steps(self, config):
    return [PythonStep('gen', fn=lambda ctx: {'value': 'generated'})]

  def assemble_item(self, slot, step_results):
    return DataItem(
      id=slot['id'],
      turns=[ConversationTurn(role='user', content='test')],
      custom=StubCustom(value=step_results.get('gen', {}).get('value', 'default')),
    )

  def stratify_key(self, item):
    return 'default'


class TestDataGeneratorAbstract:
  def test_cannot_instantiate(self) -> None:
    gen = DataGenerator()
    with pytest.raises(NotImplementedError):
      gen.create_slots(_make_config())

  def test_stub_subclass_instantiates(self) -> None:
    StubGenerator()


class TestDryRun:
  def test_returns_slot_count(self) -> None:
    out = Output()
    r = StubGenerator().dry_run(_make_config(5), out)
    assert r['total_slots'] == 5

  def test_returns_step_names(self) -> None:
    out = Output()
    r = StubGenerator().dry_run(_make_config(), out)
    assert r['step_names'] == ['gen']

  def test_no_llm_calls(self) -> None:
    out = Output()
    r = StubGenerator().dry_run(_make_config(), out)
    assert r['dataset_id'] == 'test_ds'
    assert r['model'] == 'test-model'


class TestRun:
  @pytest.mark.asyncio
  @patch('autopilot.ai.generator.run_step_workflow', new_callable=AsyncMock)
  async def test_processes_all_slots(self, mock_workflow: AsyncMock, tmp_path: Path) -> None:
    mock_workflow.return_value = {'gen': {'value': 'mocked'}}
    out = Output()
    summary = await StubGenerator().async_run(_make_config(5), tmp_path, out)
    assert summary['total_items'] == 5
    lines = (tmp_path / 'all.jsonl').read_text(encoding='utf-8').strip().splitlines()
    assert len(lines) == 5

  @pytest.mark.asyncio
  @patch('autopilot.ai.generator.run_step_workflow', new_callable=AsyncMock)
  async def test_checkpoint_written(self, mock_workflow: AsyncMock, tmp_path: Path) -> None:
    mock_workflow.return_value = {'gen': {'value': 'mocked'}}
    await StubGenerator().async_run(_make_config(3), tmp_path, Output())
    assert (tmp_path / 'checkpoint.jsonl').is_file()

  @pytest.mark.asyncio
  @patch('autopilot.ai.generator.run_step_workflow', new_callable=AsyncMock)
  async def test_splits_assigned(self, mock_workflow: AsyncMock, tmp_path: Path) -> None:
    mock_workflow.return_value = {'gen': {'value': 'mocked'}}
    await StubGenerator().async_run(_make_config(5), tmp_path, Output())
    assert (tmp_path / 'train.jsonl').is_file()
    assert (tmp_path / 'val.jsonl').is_file()
    assert (tmp_path / 'test.jsonl').is_file()

  @pytest.mark.asyncio
  @patch('autopilot.ai.generator.run_step_workflow', new_callable=AsyncMock)
  async def test_output_files_written(self, mock_workflow: AsyncMock, tmp_path: Path) -> None:
    mock_workflow.return_value = {'gen': {'value': 'mocked'}}
    await StubGenerator().async_run(_make_config(2), tmp_path, Output())
    assert (tmp_path / 'all.jsonl').is_file()
    assert (tmp_path / 'metadata.json').is_file()

  @pytest.mark.asyncio
  @patch('autopilot.ai.generator.run_step_workflow', new_callable=AsyncMock)
  async def test_rejected_items_excluded(self, mock_workflow: AsyncMock, tmp_path: Path) -> None:
    mock_workflow.return_value = {'gen': {'value': 'mocked'}}

    class RejectingGenerator(StubGenerator):
      def assemble_item(self, slot, step_results):
        if slot['id'] in ('S0000', 'S0001'):
          return None
        return super().assemble_item(slot, step_results)

    await RejectingGenerator().async_run(_make_config(5), tmp_path, Output())
    ds = ListDataset.from_jsonl(tmp_path / 'all.jsonl', DataItem[StubCustom])
    assert len(ds) == 3


class TestResume:
  @pytest.mark.asyncio
  @patch('autopilot.ai.generator.run_step_workflow', new_callable=AsyncMock)
  async def test_skips_completed_slots(self, mock_workflow: AsyncMock, tmp_path: Path) -> None:
    ckpt_path = tmp_path / 'checkpoint.jsonl'
    ckpt = CheckpointManager(ckpt_path)
    ckpt.save_header(config_hash='abc', subsystem='generate', args={})
    for sid in ('S0000', 'S0001'):
      item = DataItem(
        id=sid,
        turns=[ConversationTurn(role='user', content='hi')],
        custom=StubCustom(value='done'),
      )
      ckpt.save_event('result', sid, {'item': item.model_dump()})

    mock_workflow.return_value = {'gen': {'value': 'mocked'}}
    out_dir = tmp_path / 'resume_out'
    await StubGenerator().resume(ckpt_path, _make_config(5), out_dir, Output())
    assert mock_workflow.call_count == 3
