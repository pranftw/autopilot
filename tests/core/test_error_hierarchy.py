"""Tests for error hierarchy and enriched error messages (sub-plan 06)."""

from autopilot.ai.agents.claude_code import ClaudeCodeAgent
from autopilot.ai.data import split_names_and_normalized_ratios
from autopilot.ai.evaluation.judge import JudgeAgent
from autopilot.ai.evaluation.schemas import (
  ConversationTurn,
  JudgeConfig,
  JudgeInput,
  JudgeResult,
  JudgeVerdict,
)
from autopilot.ai.evaluation.steps import PythonStep
from autopilot.ai.gradient import AgentCollator
from autopilot.cli.output import Output
from autopilot.core.artifacts.experiment import EventsArtifact
from autopilot.core.comparison import MetricsComparator
from autopilot.core.config import AutoPilotConfig
from autopilot.core.enums import Status
from autopilot.core.errors import AgentError, AIError, ConfigError, ExperimentError
from autopilot.core.experiment import Experiment
from autopilot.core.metric import Metric
from autopilot.core.types import Datum
from autopilot.data.dataset import IterableDataset
from pydantic import BaseModel
from tests.doubles import make_run_config
from unittest.mock import AsyncMock, MagicMock, patch
import json
import logging
import pytest


class TestConfigErrorOnNoProject:
  """2.1: ConfigError raised when project is unset."""

  def test_config_error_on_no_project(self, tmp_path):
    config = AutoPilotConfig(workspace=tmp_path, project=None)
    config.autopilot_path.mkdir(parents=True, exist_ok=True)
    config.projects_path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ConfigError) as exc_info:
      config.init_project()
    msg = str(exc_info.value)
    assert 'workspace=' in msg
    assert str(tmp_path) in msg

  def test_config_error_is_autopilot_error(self, tmp_path):
    config = AutoPilotConfig(workspace=tmp_path, project=None)
    config.autopilot_path.mkdir(parents=True, exist_ok=True)
    config.projects_path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ConfigError):
      config.init_project()


class TestExperimentErrorOnDoubleStart:
  """2.2: ExperimentError on invalid lifecycle transitions."""

  def test_experiment_error_on_double_start(self):
    e = Experiment(experiment_id='x')
    e.start()
    with pytest.raises(ExperimentError) as exc_info:
      e.start()
    msg = str(exc_info.value)
    assert 'id=' in msg
    assert 'status=' in msg
    assert 'x' in msg
    assert 'running' in msg

  def test_experiment_error_on_complete_from_terminal(self):
    e = Experiment(experiment_id='y')
    e.start()
    e.complete()
    with pytest.raises(ExperimentError) as exc_info:
      e.complete()
    msg = str(exc_info.value)
    assert 'id=' in msg
    assert 'status=' in msg
    assert 'y' in msg
    assert 'completed' in msg

  def test_experiment_fail_from_pending_succeeds(self):
    """BUG-DFV1-002: fail() accepts pending status for CLI-only workflows."""
    e = Experiment(experiment_id='z')
    e.fail('reason')
    assert e.status == Status.failed
    assert e.error == 'reason'
    assert e.failed_at is not None

  def test_experiment_error_on_cancel_from_terminal(self):
    e = Experiment(experiment_id='w')
    e.start()
    e.complete()
    with pytest.raises(ExperimentError) as exc_info:
      e.cancel()
    msg = str(exc_info.value)
    assert 'id=' in msg
    assert 'status=' in msg
    assert 'w' in msg
    assert 'completed' in msg

  def test_experiment_error_on_advance_epoch_from_pending(self):
    e = Experiment(experiment_id='v')
    with pytest.raises(ExperimentError) as exc_info:
      e.advance_epoch()
    msg = str(exc_info.value)
    assert 'id=' in msg
    assert 'status=' in msg
    assert 'v' in msg
    assert 'pending' in msg


@pytest.mark.parametrize(
  ('case_id', 'trigger', 'assertion'),
  [
    pytest.param(
      'ratios',
      lambda: split_names_and_normalized_ratios({'a': 0.0}),
      lambda msg: 'sum=0.0' in msg,
      id='ratios',
    ),
    pytest.param(
      'events_type',
      lambda: EventsArtifact().validate([]),
      lambda msg: 'type=' in msg and 'list' in msg,
      id='events_type',
    ),
    pytest.param(
      'events_keys',
      lambda: EventsArtifact().validate({'foo': 1}),
      lambda msg: 'keys=' in msg,
      id='events_keys',
    ),
    pytest.param(
      'best_index_empty',
      None,
      lambda msg: 'metric=' in msg,
      id='best_index_empty',
    ),
    pytest.param(
      'best_index_missing',
      None,
      lambda msg: 'metric' in msg and 'keys_present' in msg,
      id='best_index_missing',
    ),
    pytest.param(
      'datum_backward',
      lambda: Datum().backward(None),
      lambda msg: 'id=' in msg,
      id='datum_backward',
    ),
    pytest.param(
      'iterable_getitem',
      lambda: IterableDataset()[0],
      lambda msg: 'IterableDataset' in msg,
      id='iterable_getitem',
    ),
  ],
)
class TestErrorMessageIncludesContext:
  """2.3-2.7: Error messages include contextual data."""

  def test_error_message_includes_context(self, case_id, trigger, assertion):
    if case_id == 'best_index_empty':
      m = _HigherMetric()
      comp = MetricsComparator([m])
      with pytest.raises(ValueError, match='results list is empty') as exc_info:
        comp.best_index([], m.name())
      assert assertion(str(exc_info.value))
      return

    if case_id == 'best_index_missing':
      m = _HigherMetric()
      comp = MetricsComparator([m])
      with pytest.raises(ValueError, match='keys_present_across_results') as exc_info:
        comp.best_index([{}], m.name())
      assert assertion(str(exc_info.value))
      return

    if case_id == 'datum_backward':
      with pytest.raises(RuntimeError, match='cannot backward') as exc_info:
        trigger()
      assert assertion(str(exc_info.value))
      return

    if case_id == 'iterable_getitem':
      with pytest.raises(TypeError, match='IterableDataset') as exc_info:
        trigger()
      assert assertion(str(exc_info.value))
      return

    if case_id == 'events_type':
      with pytest.raises(TypeError, match='got type=') as exc_info:
        trigger()
      assert assertion(str(exc_info.value))
      return

    with pytest.raises(
      ValueError,
      match=(r'(ratios must sum to a positive value|event requires timestamp and event_type)'),
    ) as exc_info:
      trigger()
    assert assertion(str(exc_info.value))


class _HigherMetric(Metric):
  higher_is_better = True

  def update(self, datum):
    pass

  def compute(self):
    return {}


class _NoneMetric(Metric):
  higher_is_better = None

  def update(self, datum):
    pass

  def compute(self):
    return {}


class TestBestIndexHigherIsBetterNone:
  """2.5: higher_is_better=None raises ValueError with mode context."""

  def test_higher_is_better_none(self):
    m = _NoneMetric()
    comp = MetricsComparator([m])
    with pytest.raises(ValueError, match='higher_is_better not set') as exc_info:
      comp.best_index([{m.name(): 1.0}], m.name())
    msg = str(exc_info.value)
    assert 'higher_is_better not set' in msg
    assert 'mode must be set' in msg


class TestExceptionChaining:
  """2.9: Exception chaining with from exc."""

  def test_agent_collator_json_parse_chains_cause(self):
    agent = MagicMock()
    agent.run.return_value = MagicMock(output='not json{{{')
    collator = AgentCollator(agent)
    with pytest.raises(RuntimeError) as exc_info:
      collator.parse_result('not json{{{', [])
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)

  def test_claude_code_file_not_found_chains_cause(self):
    agent = ClaudeCodeAgent()
    with patch('subprocess.run', side_effect=FileNotFoundError('no claude')):
      with pytest.raises(AgentError) as exc_info:
        agent.run('hello')
      assert exc_info.value.__cause__ is not None
      assert isinstance(exc_info.value.__cause__, FileNotFoundError)

  def test_claude_code_json_parse_chains_cause(self):
    agent = ClaudeCodeAgent()
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = 'not json!!!'
    with patch('subprocess.run', return_value=mock_proc):
      with pytest.raises(AgentError) as exc_info:
        agent.run('hello')
      assert exc_info.value.__cause__ is not None
      assert isinstance(exc_info.value.__cause__, (json.JSONDecodeError, TypeError))


class _StubJudgeConfig(BaseModel):
  threshold: float = 0.5


class _StubJudgeCustom(BaseModel):
  query: str


class _StubResultCustom(BaseModel):
  score: float


class _TestJudge(JudgeAgent['_StubJudgeConfig', '_StubJudgeCustom', '_StubResultCustom']):
  def define_steps(self, config):
    return [PythonStep('analyze', fn=lambda ctx: {'score': 0.9})]

  def assemble_result(self, item, step_results):
    score = step_results.get('analyze', {}).get('score', 0.0)
    return JudgeResult(
      id=item.item_id,
      verdict=JudgeVerdict(
        category='correct',
        rationale='ok',
        confidence=score,
      ),
      custom=_StubResultCustom(score=score),
    )

  def build_summary(self, results):
    return {'total': len(results)}


def _make_judge_config() -> JudgeConfig:
  return JudgeConfig(
    run=make_run_config(),
    system_prompt='test',
    custom=_StubJudgeConfig(),
  )


def _make_items(count: int = 1) -> list:
  return [
    JudgeInput(
      id=f'J{i:04d}',
      turns=[ConversationTurn(role='user', content=f'q{i}')],
      response=f'resp {i}',
      custom=_StubJudgeCustom(query=f'query {i}'),
    )
    for i in range(count)
  ]


class TestNarrowExceptionCatch:
  """2.8: Judge/generator exception narrowing and unexpected fallback."""

  @pytest.mark.asyncio
  @patch('autopilot.ai.evaluation.judge.run_step_workflow', new_callable=AsyncMock)
  async def test_known_exception_checkpoints_error(
    self, mock_workflow: AsyncMock, tmp_path, caplog
  ):
    mock_workflow.side_effect = ValueError('bad input')
    judge = _TestJudge()
    items = _make_items(1)
    with caplog.at_level(logging.WARNING):
      await judge.async_run(items, _make_judge_config(), tmp_path, Output())
    ckpt_path = tmp_path / 'checkpoint.jsonl'
    lines = ckpt_path.read_text(encoding='utf-8').strip().split('\n')
    error_events = [json.loads(line) for line in lines if 'error' in line and '"type"' in line]
    assert len(error_events) >= 1
    assert 'failed' in caplog.text

  @pytest.mark.asyncio
  @patch('autopilot.ai.evaluation.judge.run_step_workflow', new_callable=AsyncMock)
  async def test_unexpected_exception_logs_unexpected(
    self, mock_workflow: AsyncMock, tmp_path, caplog
  ):
    mock_workflow.side_effect = AIError('surprise')
    judge = _TestJudge()
    items = _make_items(1)
    with caplog.at_level(logging.WARNING):
      await judge.async_run(items, _make_judge_config(), tmp_path, Output())
    assert 'unexpected' in caplog.text
    ckpt_path = tmp_path / 'checkpoint.jsonl'
    lines = ckpt_path.read_text(encoding='utf-8').strip().split('\n')
    error_events = [json.loads(line) for line in lines if '"error"' in line and '"type"' in line]
    assert len(error_events) >= 1

  @pytest.mark.asyncio
  @patch('autopilot.ai.evaluation.judge.run_step_workflow', new_callable=AsyncMock)
  async def test_type_error_caught_by_known_tuple(self, mock_workflow: AsyncMock, tmp_path, caplog):
    mock_workflow.side_effect = TypeError('type issue')
    judge = _TestJudge()
    items = _make_items(1)
    with caplog.at_level(logging.WARNING):
      await judge.async_run(items, _make_judge_config(), tmp_path, Output())
    assert 'unexpected' not in caplog.text
    assert 'failed' in caplog.text


class TestIterableDatasetSubclass:
  """2.7: Subclass name appears in error."""

  def test_subclass_name_in_error(self):
    class MyStreamDataset(IterableDataset):
      def __iter__(self):
        return iter([])

    with pytest.raises(TypeError) as exc_info:
      MyStreamDataset()[0]
    assert 'MyStreamDataset' in str(exc_info.value)
