"""Tests for GeneratorAgent base class."""

from autopilot.ai.evaluation.checkpoints import CheckpointManager
from autopilot.ai.evaluation.generator import GeneratorAgent
from autopilot.ai.evaluation.schemas import (
  ConversationTurn,
  DataItem,
  GeneratorConfig,
)
from autopilot.ai.evaluation.steps import PythonStep
from autopilot.cli.output import Output
from autopilot.data.dataset import ListDataset
from pathlib import Path
from pydantic import BaseModel
from tests.doubles import make_run_config
from unittest.mock import AsyncMock, Mock, patch
import json
import pytest


class StubCustom(BaseModel):
  value: str


class StubGenConfig(BaseModel):
  prefix: str = 'STUB'


def _make_config(total: int = 5) -> GeneratorConfig[StubGenConfig]:
  return GeneratorConfig(
    run=make_run_config(),
    dataset_id='test_ds',
    seed=42,
    total_count=total,
    split_ratios={'train': 0.8, 'val': 0.1, 'test': 0.1},
    system_prompt='test',
    custom=StubGenConfig(),
  )


class StubGenerator(GeneratorAgent[StubGenConfig, StubCustom]):
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


class TestGeneratorAgentAbstract:
  def test_cannot_instantiate(self) -> None:
    gen = GeneratorAgent()
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
  @patch('autopilot.ai.evaluation.generator.run_step_workflow', new_callable=AsyncMock)
  async def test_processes_all_slots(self, mock_workflow: AsyncMock, tmp_path: Path) -> None:
    mock_workflow.return_value = {'gen': {'value': 'mocked'}}
    out = Output()
    summary = await StubGenerator().async_run(_make_config(5), tmp_path, out)
    assert summary['total_items'] == 5
    lines = (tmp_path / 'all.jsonl').read_text(encoding='utf-8').strip().splitlines()
    assert len(lines) == 5

  @pytest.mark.asyncio
  @patch('autopilot.ai.evaluation.generator.run_step_workflow', new_callable=AsyncMock)
  async def test_checkpoint_written(self, mock_workflow: AsyncMock, tmp_path: Path) -> None:
    mock_workflow.return_value = {'gen': {'value': 'mocked'}}
    await StubGenerator().async_run(_make_config(3), tmp_path, Output())
    assert (tmp_path / 'checkpoint.jsonl').is_file()

  @pytest.mark.asyncio
  @patch('autopilot.ai.evaluation.generator.run_step_workflow', new_callable=AsyncMock)
  async def test_splits_assigned(self, mock_workflow: AsyncMock, tmp_path: Path) -> None:
    mock_workflow.return_value = {'gen': {'value': 'mocked'}}
    await StubGenerator().async_run(_make_config(5), tmp_path, Output())
    assert (tmp_path / 'train.jsonl').is_file()
    assert (tmp_path / 'val.jsonl').is_file()
    assert (tmp_path / 'test.jsonl').is_file()

  @pytest.mark.asyncio
  @patch('autopilot.ai.evaluation.generator.run_step_workflow', new_callable=AsyncMock)
  async def test_output_files_written(self, mock_workflow: AsyncMock, tmp_path: Path) -> None:
    mock_workflow.return_value = {'gen': {'value': 'mocked'}}
    await StubGenerator().async_run(_make_config(2), tmp_path, Output())
    assert (tmp_path / 'all.jsonl').is_file()
    assert (tmp_path / 'metadata.json').is_file()

  @pytest.mark.asyncio
  @patch('autopilot.ai.evaluation.generator.run_step_workflow', new_callable=AsyncMock)
  async def test_rejected_items_excluded(self, mock_workflow: AsyncMock, tmp_path: Path) -> None:
    mock_workflow.return_value = {'gen': {'value': 'mocked'}}

    class RejectingGenerator(StubGenerator):
      def assemble_item(self, slot, step_results):
        if slot['id'] in {'S0000', 'S0001'}:
          return None
        return super().assemble_item(slot, step_results)

    await RejectingGenerator().async_run(_make_config(5), tmp_path, Output())
    ds = ListDataset.from_jsonl(tmp_path / 'all.jsonl', DataItem[StubCustom])
    assert len(ds) == 3


class TestResume:
  @pytest.mark.asyncio
  @patch('autopilot.ai.evaluation.generator.run_step_workflow', new_callable=AsyncMock)
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
    await StubGenerator().resume(ckpt_path, _make_config(5), Output())
    assert mock_workflow.call_count == 3

  @pytest.mark.asyncio
  @patch('autopilot.ai.evaluation.generator.run_step_workflow', new_callable=AsyncMock)
  async def test_resume_appends_new_events_and_grows_file(
    self, mock_workflow: AsyncMock, tmp_path: Path
  ) -> None:
    ckpt_path = tmp_path / 'checkpoint.jsonl'
    ckpt = CheckpointManager(ckpt_path)
    ckpt.save_header(config_hash='abc', subsystem='generate', args={})
    for sid in ('S0000',):
      item = DataItem(
        id=sid,
        turns=[ConversationTurn(role='user', content='hi')],
        custom=StubCustom(value='prior'),
      )
      ckpt.save_event('result', sid, {'item': item.model_dump()})
    size_before = ckpt_path.stat().st_size
    line_count_before = len(ckpt_path.read_text(encoding='utf-8').strip().splitlines())

    mock_workflow.return_value = {'gen': {'value': 'mocked'}}
    out = Output()
    summary = await StubGenerator().resume(ckpt_path, _make_config(3), out)
    size_after = ckpt_path.stat().st_size
    lines_after = ckpt_path.read_text(encoding='utf-8').strip().splitlines()
    assert size_after > size_before
    assert len(lines_after) >= line_count_before + 2
    assert summary['resumed_items'] == 3
    result_lines = sum(1 for line in lines_after if '"type": "result"' in line)
    assert result_lines >= 3


class TestWriteGeneratorOutputs:
  def test_writes_all_jsonl(self, tmp_path: Path) -> None:
    gen = StubGenerator()
    items = [
      DataItem(
        id=f'I{i}',
        turns=[ConversationTurn(role='user', content='hi')],
        custom=StubCustom(value='v'),
      )
      for i in range(4)
    ]
    ckpt_path = tmp_path / 'checkpoint.jsonl'
    ckpt = CheckpointManager(ckpt_path)
    ckpt.save_header(config_hash='abc', subsystem='generate', args={})
    for item in items:
      ckpt.save_event('result', item.item_id, {'item': item.model_dump()})

    out = Output()
    gen._write_generator_outputs(items, _make_config(4), 'abc123', tmp_path, ckpt, out)
    assert (tmp_path / 'all.jsonl').is_file()
    lines = (tmp_path / 'all.jsonl').read_text(encoding='utf-8').strip().splitlines()
    assert len(lines) == 4

  def test_writes_splits(self, tmp_path: Path) -> None:
    gen = StubGenerator()
    items = [
      DataItem(
        id=f'I{i}',
        turns=[ConversationTurn(role='user', content='hi')],
        custom=StubCustom(value='v'),
      )
      for i in range(10)
    ]
    ckpt_path = tmp_path / 'checkpoint.jsonl'
    ckpt = CheckpointManager(ckpt_path)
    ckpt.save_header(config_hash='abc', subsystem='generate', args={})
    for item in items:
      ckpt.save_event('result', item.item_id, {'item': item.model_dump()})

    out = Output()
    gen._write_generator_outputs(items, _make_config(10), 'hash16', tmp_path, ckpt, out)
    assert (tmp_path / 'train.jsonl').is_file()
    assert (tmp_path / 'val.jsonl').is_file()
    assert (tmp_path / 'test.jsonl').is_file()

  def test_writes_metadata_json(self, tmp_path: Path) -> None:
    gen = StubGenerator()
    items = [
      DataItem(
        id='I0',
        turns=[ConversationTurn(role='user', content='hi')],
        custom=StubCustom(value='v'),
      )
    ]
    ckpt_path = tmp_path / 'checkpoint.jsonl'
    ckpt = CheckpointManager(ckpt_path)
    ckpt.save_header(config_hash='abc', subsystem='generate', args={})
    ckpt.save_event('result', 'I0', {'item': items[0].model_dump()})

    out = Output()
    gen._write_generator_outputs(items, _make_config(1), 'hash16', tmp_path, ckpt, out)
    meta = json.loads((tmp_path / 'metadata.json').read_text(encoding='utf-8'))
    assert meta['total_generated'] == 1
    assert meta['config_hash'] == 'hash16'

  def test_summary_has_total_items(self, tmp_path: Path) -> None:
    gen = StubGenerator()
    items = [
      DataItem(
        id=f'I{i}',
        turns=[ConversationTurn(role='user', content='hi')],
        custom=StubCustom(value='v'),
      )
      for i in range(3)
    ]
    ckpt_path = tmp_path / 'checkpoint.jsonl'
    ckpt = CheckpointManager(ckpt_path)
    ckpt.save_header(config_hash='abc', subsystem='generate', args={})
    for item in items:
      ckpt.save_event('result', item.item_id, {'item': item.model_dump()})

    out = Output()
    summary = gen._write_generator_outputs(items, _make_config(3), 'abc', tmp_path, ckpt, out)
    assert summary['total_items'] == 3
    assert 'splits' in summary


class TestSlotResultFromWorkflow:
  def test_successful_assembly(self) -> None:
    gen = StubGenerator()
    ckpt = Mock()
    result = gen._slot_result_from_workflow(
      {'id': 'S0001'}, 'S0001', {'gen': {'value': 'ok'}}, ckpt, for_resume=False
    )
    assert result['id'] == 'S0001'
    assert result['item'] is not None
    ckpt.save_event.assert_called_once()
    assert ckpt.save_event.call_args[0][0] == 'result'

  def test_rejected_assembly(self) -> None:
    class RejectingGen(StubGenerator):
      def assemble_item(self, slot, step_results):
        return None

    gen = RejectingGen()
    ckpt = Mock()
    result = gen._slot_result_from_workflow({'id': 'S0001'}, 'S0001', {}, ckpt, for_resume=False)
    assert result['skipped'] is True
    ckpt.save_event.assert_called_once()
    assert ckpt.save_event.call_args[0][0] == 'skip'
    payload = ckpt.save_event.call_args[0][2]
    assert 'rejected by assemble_item' in payload['reason']

  def test_rejected_for_resume(self) -> None:
    class RejectingGen(StubGenerator):
      def assemble_item(self, slot, step_results):
        return None

    gen = RejectingGen()
    ckpt = Mock()
    result = gen._slot_result_from_workflow({'id': 'S0001'}, 'S0001', {}, ckpt, for_resume=True)
    assert result['skipped'] is True
    payload = ckpt.save_event.call_args[0][2]
    assert payload['reason'] == 'rejected'
