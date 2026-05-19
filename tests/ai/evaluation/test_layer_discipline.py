"""Tests for Plan 05: Layer discipline (evaluation vs CLI, Experiment flags)."""

from autopilot.ai.evaluation.generator import GeneratorAgent
from autopilot.ai.evaluation.judge import JudgeAgent
from autopilot.ai.evaluation.protocols import EvaluationOutputProtocol
from autopilot.ai.evaluation.schemas import (
  ConversationTurn,
  DataItem,
  GeneratorConfig,
  JudgeConfig,
  JudgeResult,
  JudgeVerdict,
)
from autopilot.ai.evaluation.steps import PythonStep
from autopilot.core.experiment import Experiment
from pathlib import Path
from pydantic import BaseModel
from tests.doubles import MockEvaluationOutput, make_run_config
from typing import Any, cast
from unittest.mock import AsyncMock, patch
import pytest


class StubGenCustom(BaseModel):
  value: str


class StubGenConfig(BaseModel):
  prefix: str = 'STUB'


def _make_gen_config(total: int = 5) -> GeneratorConfig[StubGenConfig]:
  return GeneratorConfig(
    run=make_run_config(),
    dataset_id='test_ds',
    seed=42,
    total_count=total,
    split_ratios={'train': 0.8, 'val': 0.1, 'test': 0.1},
    system_prompt='test',
    custom=StubGenConfig(),
  )


class StubGenerator(GeneratorAgent[StubGenConfig, StubGenCustom]):
  def create_slots(self, config):
    return [{'id': f'S{i:04d}'} for i in range(config.total_count)]

  def define_steps(self, config):
    return [PythonStep('gen', fn=lambda ctx: {'value': 'generated'})]

  def assemble_item(self, slot, step_results):
    return DataItem(
      id=slot['id'],
      turns=[ConversationTurn(role='user', content='test')],
      custom=StubGenCustom(value=step_results.get('gen', {}).get('value', 'default')),
    )

  def stratify_key(self, item):
    return 'default'


class StubJudgeCustom(BaseModel):
  query: str


class StubResultCustom(BaseModel):
  score: float


class StubJudgeConfig(BaseModel):
  threshold: float = 0.5


class StubJudge(JudgeAgent[StubJudgeConfig, StubJudgeCustom, StubResultCustom]):
  def define_steps(self, config):
    return [PythonStep('analyze', fn=lambda ctx: {'score': 0.9})]

  def assemble_result(self, item, step_results):
    score = step_results.get('analyze', {}).get('score', 0.0)
    return JudgeResult(
      id=item.item_id,
      verdict=JudgeVerdict(
        category='correct',
        rationale='looks good',
        confidence=score,
      ),
      custom=StubResultCustom(score=score),
    )

  def build_summary(self, results):
    return {'count': len(results)}


def test_protocols_module_import() -> None:
  """EvaluationOutputProtocol imports without error and is usable as annotation."""
  assert hasattr(EvaluationOutputProtocol, 'info')
  assert hasattr(EvaluationOutputProtocol, 'result')

  output: EvaluationOutputProtocol = MockEvaluationOutput()
  output.info('test')
  output.result({'key': 'val'})


def test_evaluation_output_protocol_mock_generator() -> None:
  """Generator dry_run works with MockEvaluationOutput (no CLI Output)."""
  mock = MockEvaluationOutput()
  StubGenerator().dry_run(_make_gen_config(3), mock)
  assert len(mock.results) == 1
  assert mock.results[0][0]['total_slots'] == 3
  assert mock.results[0][1] is True
  assert mock.infos == []


@pytest.mark.asyncio
async def test_evaluation_output_protocol_mock_judge(tmp_path: Path) -> None:
  """Judge async_run works with MockEvaluationOutput (no CLI Output)."""
  mock = MockEvaluationOutput()
  judge_config = JudgeConfig(
    run=make_run_config(),
    system_prompt='judge test',
    custom=StubJudgeConfig(),
  )
  await StubJudge().async_run([], judge_config, tmp_path, mock)

  assert len(mock.infos) == 2
  assert 'Judging' in mock.infos[0]
  assert '0 items' in mock.infos[0]
  payload = mock.results[-1][0]
  assert 'summary' in payload
  assert payload['summary'] == {'count': 0}
  assert mock.results[-1][1] is True


def test_experiment_should_rollback_default() -> None:
  """Experiment instances default should_rollback to False."""
  exp = Experiment(experiment_id='e1')
  assert exp.should_rollback is False


def test_experiment_should_rollback_settable() -> None:
  """Assigning should_rollback = True persists on the instance."""
  exp = Experiment(experiment_id='e2')
  exp.should_rollback = True
  assert exp.should_rollback is True


def test_no_cli_import_in_ai_evaluation() -> None:
  """No file under src/autopilot/ai/evaluation/ imports from autopilot.cli."""
  eval_dir = Path('src/autopilot/ai/evaluation')
  for py_file in eval_dir.glob('*.py'):
    content = py_file.read_text(encoding='utf-8')
    for line in content.splitlines():
      stripped = line.strip()
      if stripped.startswith('#'):
        continue
      assert not stripped.startswith('from autopilot.cli'), (
        f'{py_file}: imports from autopilot.cli: {stripped}'
      )
      assert not stripped.startswith('import autopilot.cli'), (
        f'{py_file}: imports autopilot.cli: {stripped}'
      )


def test_mock_evaluation_output_records_callbacks() -> None:
  """infos and results lists match scripted calls."""
  mock_out = MockEvaluationOutput()
  mock_out.info('ping')
  mock_out.result({'k': 1}, ok=True)
  assert mock_out.infos == ['ping']
  assert mock_out.results == [({'k': 1}, True)]


@pytest.mark.asyncio
@patch('autopilot.ai.evaluation.generator.run_step_workflow', new_callable=AsyncMock)
async def test_stub_generator_async_run_accepts_mock_protocol(
  mock_wf: AsyncMock, tmp_path: Path
) -> None:
  """Generator async_run works with MockEvaluationOutput through full pipeline."""
  mock_wf.return_value = {'gen': {'value': 'mocked'}}
  mock_out = MockEvaluationOutput()
  summary = await StubGenerator().async_run(_make_gen_config(2), tmp_path, mock_out)
  assert summary['total_items'] == 2
  assert mock_out.results[-1][0]['total_items'] == 2


@pytest.mark.asyncio
@patch('autopilot.ai.evaluation.judge.run_step_workflow', new_callable=AsyncMock)
async def test_stub_judge_async_run_accepts_mock_protocol(
  mock_wf: AsyncMock, tmp_path: Path
) -> None:
  """Judge async_run works with MockEvaluationOutput; summary total and emitted payload."""
  mock_wf.return_value = {'analyze': {'score': 0.9}, 'item': {}}
  mock_out = MockEvaluationOutput()
  judge_config = JudgeConfig(
    run=make_run_config(),
    system_prompt='judge test',
    custom=StubJudgeConfig(),
  )
  items = [
    DataItem(
      id=f'J{i:04d}',
      turns=[ConversationTurn(role='user', content=f'q{i}')],
      custom=StubJudgeCustom(query=f'q{i}'),
    )
    for i in range(2)
  ]
  final = await StubJudge().async_run(cast(Any, items), judge_config, tmp_path, mock_out)
  assert final['summary']['count'] == 2
  assert any(
    'summary' in p and 'count' in p['summary'] for p, _ in mock_out.results if isinstance(p, dict)
  )
