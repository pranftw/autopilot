"""Tests for JudgeAgent base class."""

from autopilot.ai.evaluation.checkpoints import CheckpointManager
from autopilot.ai.evaluation.judge import JudgeAgent
from autopilot.ai.evaluation.schemas import (
  ConversationTurn,
  JudgeConfig,
  JudgeInput,
  JudgeResult,
  JudgeVerdict,
  RetryConfig,
  RunConfig,
)
from autopilot.ai.evaluation.steps import PythonStep
from autopilot.cli.output import Output
from pydantic import BaseModel
from unittest.mock import AsyncMock, patch
import json
import pytest


class StubJudgeConfig(BaseModel):
  threshold: float = 0.5


class StubJudgeCustom(BaseModel):
  query: str


class StubResultCustom(BaseModel):
  score: float


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


def _make_config() -> JudgeConfig[StubJudgeConfig]:
  return JudgeConfig(
    run=_make_run_config(),
    system_prompt='judge test',
    custom=StubJudgeConfig(),
  )


def _make_items(count: int = 3) -> list[JudgeInput[StubJudgeCustom]]:
  return [
    JudgeInput(
      id=f'J{i:04d}',
      turns=[ConversationTurn(role='user', content=f'q{i}')],
      response=f'response {i}',
      custom=StubJudgeCustom(query=f'query {i}'),
    )
    for i in range(count)
  ]


class StubJudge(JudgeAgent[StubJudgeConfig, StubJudgeCustom, StubResultCustom]):
  def define_steps(self, config):
    return [PythonStep('analyze', fn=lambda ctx: {'score': 0.9})]

  def assemble_result(self, item, step_results):
    score = step_results.get('analyze', {}).get('score', 0.0)
    return JudgeResult(
      id=item.id,
      verdict=JudgeVerdict(
        category='correct',
        rationale='looks good',
        confidence=score,
      ),
      custom=StubResultCustom(score=score),
    )

  def build_summary(self, results):
    return {
      'total': len(results),
      'correct': sum(1 for r in results if r.verdict and r.verdict.category == 'correct'),
    }


class TestJudgeAgentAbstract:
  def test_cannot_instantiate(self) -> None:
    j = JudgeAgent()
    with pytest.raises(NotImplementedError):
      j.define_steps(_make_config())

  def test_stub_subclass_instantiates(self) -> None:
    j = StubJudge()
    assert j.define_steps(_make_config())


class TestRun:
  @pytest.mark.asyncio
  @patch('autopilot.ai.evaluation.judge.run_step_workflow', new_callable=AsyncMock)
  async def test_processes_all_items(self, mock_workflow: AsyncMock, tmp_path) -> None:
    mock_workflow.return_value = {
      'analyze': {'score': 0.9},
      'item': {},
    }
    judge = StubJudge()
    items = _make_items(3)
    output = Output()
    result = await judge.async_run(items, _make_config(), tmp_path, output)
    assert mock_workflow.await_count == 3
    assert result['summary']['total'] == 3
    assert result['summary']['correct'] == 3

  @pytest.mark.asyncio
  @patch('autopilot.ai.evaluation.judge.run_step_workflow', new_callable=AsyncMock)
  async def test_checkpoint_per_item(self, mock_workflow: AsyncMock, tmp_path) -> None:
    mock_workflow.return_value = {'analyze': {'score': 0.9}, 'item': {}}
    judge = StubJudge()
    items = _make_items(3)
    await judge.async_run(items, _make_config(), tmp_path, Output())
    ckpt_path = tmp_path / 'checkpoint.jsonl'
    assert ckpt_path.is_file()
    lines = ckpt_path.read_text(encoding='utf-8').strip().split('\n')
    assert len(lines) >= 4
    result_events = [ev for line in lines if (ev := json.loads(line)).get('type') == 'result']
    assert len(result_events) == 3

  @pytest.mark.asyncio
  @patch('autopilot.ai.evaluation.judge.run_step_workflow', new_callable=AsyncMock)
  async def test_build_summary_called(self, mock_workflow: AsyncMock, tmp_path) -> None:
    mock_workflow.return_value = {'analyze': {'score': 0.9}, 'item': {}}
    judge = StubJudge()
    result = await judge.async_run(_make_items(2), _make_config(), tmp_path, Output())
    assert 'summary' in result
    assert result['summary'] == {'total': 2, 'correct': 2}

  @pytest.mark.asyncio
  @patch('autopilot.ai.evaluation.judge.run_step_workflow', new_callable=AsyncMock)
  async def test_output_written(self, mock_workflow: AsyncMock, tmp_path) -> None:
    mock_workflow.return_value = {'analyze': {'score': 0.9}, 'item': {}}
    judge = StubJudge()
    await judge.async_run(_make_items(1), _make_config(), tmp_path, Output())
    out_path = tmp_path / 'output.json'
    assert out_path.is_file()
    payload = json.loads(out_path.read_text(encoding='utf-8'))
    assert 'summary' in payload
    assert 'results' in payload
    assert 'config_hash' in payload


class TestResume:
  @pytest.mark.asyncio
  @patch('autopilot.ai.evaluation.judge.run_step_workflow', new_callable=AsyncMock)
  async def test_skips_completed_items(self, mock_workflow: AsyncMock, tmp_path) -> None:
    mock_workflow.return_value = {'analyze': {'score': 0.9}, 'item': {}}
    ckpt_path = tmp_path / 'checkpoint.jsonl'
    cm = CheckpointManager(ckpt_path)
    cm.save_header(config_hash='abc', subsystem='judge', args={})
    cm.save_event('result', 'J0000', {'result': {'id': 'J0000'}})
    cm.save_event('result', 'J0001', {'result': {'id': 'J0001'}})

    judge = StubJudge()
    items = _make_items(3)
    await judge.resume(ckpt_path, items, _make_config(), tmp_path, Output())

    assert mock_workflow.await_count == 1
