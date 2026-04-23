"""Tests for autopilot.ai.evaluation.schemas Pydantic schemas."""

from autopilot.ai.evaluation.schemas import (
  CheckpointEvent,
  CheckpointHeader,
  ConversationTurn,
  DataItem,
  GeneratorConfig,
  JudgeConfig,
  JudgeInput,
  JudgeResult,
  JudgeVerdict,
  RetryConfig,
  RunConfig,
  VarDef,
)
from pydantic import BaseModel, ValidationError
import pytest


class SimpleCustom(BaseModel):
  domain: str
  difficulty: str


class AltCustom(BaseModel):
  topic: str
  level: int


def _retry() -> RetryConfig:
  return RetryConfig(
    max_retries=3,
    min_timeout_ms=1000,
    max_timeout_ms=5000,
    backoff_factor=2,
  )


def _run() -> RunConfig:
  return RunConfig(
    model='gpt-4',
    num_parallel=4,
    max_rpm=60,
    rpm_safety_margin=0.9,
    retry=_retry(),
    max_tool_steps=10,
    max_output_tokens=4096,
  )


class TestConversationTurn:
  def test_round_trip_all_fields(self) -> None:
    tool_calls = [{'id': 'call1', 'type': 'function', 'function': {'name': 'fn'}}]
    turn = ConversationTurn(
      role='assistant',
      content='hello',
      name='bot',
      tool_call_id='tc1',
      tool_calls=tool_calls,
    )
    data = turn.model_dump()
    back = ConversationTurn.model_validate(data)
    assert back.role == 'assistant'
    assert back.content == 'hello'
    assert back.name == 'bot'
    assert back.tool_call_id == 'tc1'
    assert back.tool_calls == tool_calls

  def test_round_trip_defaults(self) -> None:
    turn = ConversationTurn(role='user', content='hi')
    assert turn.name is None
    assert turn.tool_call_id is None
    assert turn.tool_calls is None
    back = ConversationTurn.model_validate(turn.model_dump())
    assert back.name is None
    assert back.tool_call_id is None
    assert back.tool_calls is None

  def test_jsonl_serialization(self) -> None:
    turn = ConversationTurn(role='user', content='q')
    js = turn.model_dump_json()
    back = ConversationTurn.model_validate_json(js)
    assert back == turn


class TestDataItemGeneric:
  def test_concrete_type_round_trip(self) -> None:
    item = DataItem[SimpleCustom](
      id='e1',
      turns=[ConversationTurn(role='user', content='q')],
      custom=SimpleCustom(domain='math', difficulty='hard'),
    )
    back = DataItem[SimpleCustom].model_validate(item.model_dump())
    assert back.id == 'e1'
    assert back.custom.domain == 'math'
    assert back.custom.difficulty == 'hard'

  def test_split_none_default(self) -> None:
    item = DataItem[SimpleCustom](
      id='e1',
      turns=[ConversationTurn(role='user', content='q')],
      custom=SimpleCustom(domain='x', difficulty='y'),
    )
    assert item.split is None

  def test_json_schema_includes_custom(self) -> None:
    schema = DataItem[SimpleCustom].model_json_schema()
    assert 'custom' in schema['properties']

  def test_jsonl_round_trip(self) -> None:
    item = DataItem[SimpleCustom](
      id='e1',
      turns=[ConversationTurn(role='user', content='q')],
      split='train',
      custom=SimpleCustom(domain='d', difficulty='easy'),
    )
    js = item.model_dump_json()
    back = DataItem[SimpleCustom].model_validate_json(js)
    assert back == item

  def test_different_custom_types(self) -> None:
    a = DataItem[SimpleCustom](
      id='1',
      turns=[ConversationTurn(role='user', content='a')],
      custom=SimpleCustom(domain='d', difficulty='e'),
    )
    b = DataItem[AltCustom](
      id='2',
      turns=[ConversationTurn(role='user', content='b')],
      custom=AltCustom(topic='t', level=3),
    )
    assert isinstance(a.custom, SimpleCustom)
    assert isinstance(b.custom, AltCustom)


class TestJudgeInputGeneric:
  def test_round_trip_all_fields(self) -> None:
    ji = JudgeInput[SimpleCustom](
      id='j1',
      turns=[ConversationTurn(role='user', content='q')],
      response='ans',
      is_error=True,
      error_message='e',
      trace_present=True,
      trace_summary='s',
      custom=SimpleCustom(domain='d', difficulty='x'),
    )
    back = JudgeInput[SimpleCustom].model_validate(ji.model_dump())
    assert back.response == 'ans'
    assert back.is_error is True
    assert back.error_message == 'e'
    assert back.trace_present is True
    assert back.trace_summary == 's'

  def test_defaults(self) -> None:
    ji = JudgeInput[SimpleCustom](
      id='j1',
      turns=[ConversationTurn(role='user', content='q')],
      custom=SimpleCustom(domain='d', difficulty='x'),
    )
    assert ji.response is None
    assert ji.is_error is False
    assert ji.trace_present is False

  def test_with_custom_type(self) -> None:
    ji = JudgeInput[SimpleCustom](
      id='j1',
      turns=[],
      custom=SimpleCustom(domain='a', difficulty='b'),
    )
    assert ji.custom.domain == 'a'


class TestJudgeVerdict:
  def test_confidence_lower_bound(self) -> None:
    v = JudgeVerdict(category='c', rationale='r', confidence=0.0)
    assert v.confidence == 0.0

  def test_confidence_upper_bound(self) -> None:
    v = JudgeVerdict(category='c', rationale='r', confidence=1.0)
    assert v.confidence == 1.0

  def test_confidence_above_range(self) -> None:
    with pytest.raises(ValidationError):
      JudgeVerdict(category='c', rationale='r', confidence=1.5)

  def test_confidence_below_range(self) -> None:
    with pytest.raises(ValidationError):
      JudgeVerdict(category='c', rationale='r', confidence=-0.1)

  def test_subcategory_optional(self) -> None:
    v = JudgeVerdict(category='c', rationale='r', confidence=0.5)
    assert v.subcategory is None


class TestJudgeResultGeneric:
  def test_verdict_none(self) -> None:
    jr = JudgeResult[SimpleCustom](
      id='r1',
      verdict=None,
      custom=SimpleCustom(domain='d', difficulty='e'),
    )
    assert jr.verdict is None

  def test_with_verdict(self) -> None:
    verdict = JudgeVerdict(category='ok', rationale='fine', confidence=0.9)
    jr = JudgeResult[SimpleCustom](
      id='r1',
      verdict=verdict,
      custom=SimpleCustom(domain='d', difficulty='e'),
    )
    back = JudgeResult[SimpleCustom].model_validate(jr.model_dump())
    assert back.verdict is not None
    assert back.verdict.category == 'ok'
    assert back.verdict.confidence == 0.9

  def test_with_custom(self) -> None:
    jr = JudgeResult[SimpleCustom](
      id='r1',
      custom=SimpleCustom(domain='math', difficulty='hard'),
    )
    assert jr.custom.domain == 'math'
    assert isinstance(jr.custom, SimpleCustom)


class TestRetryConfig:
  def test_all_fields_required(self) -> None:
    with pytest.raises(ValidationError):
      RetryConfig.model_validate({})


class TestRunConfig:
  def test_all_fields_required(self) -> None:
    with pytest.raises(ValidationError):
      RunConfig.model_validate({})

  def test_nested_retry(self) -> None:
    rc = _run()
    assert rc.retry.max_retries == 3
    assert isinstance(rc.retry, RetryConfig)


class TestGeneratorConfig:
  def test_custom_none(self) -> None:
    gc = GeneratorConfig[SimpleCustom](
      run=_run(),
      dataset_id='ds1',
      seed=42,
      total_count=100,
      split_ratios={'train': 0.8, 'val': 0.2},
      system_prompt='p',
      custom=None,
    )
    assert gc.custom is None

  def test_with_custom(self) -> None:
    gc = GeneratorConfig[SimpleCustom](
      run=_run(),
      dataset_id='ds1',
      seed=1,
      total_count=10,
      split_ratios={'train': 1.0},
      system_prompt='sys',
      custom=SimpleCustom(domain='d', difficulty='e'),
    )
    assert gc.custom is not None
    assert gc.custom.domain == 'd'

  def test_split_ratios(self) -> None:
    ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}
    gc = GeneratorConfig[SimpleCustom](
      run=_run(),
      dataset_id='ds',
      seed=0,
      total_count=1,
      split_ratios=ratios,
      system_prompt='s',
    )
    assert gc.split_ratios == ratios


class TestJudgeConfig:
  def test_custom_none(self) -> None:
    jc = JudgeConfig[SimpleCustom](run=_run(), system_prompt='p', custom=None)
    assert jc.custom is None

  def test_round_trip(self) -> None:
    jc = JudgeConfig[SimpleCustom](
      run=_run(),
      system_prompt='evaluate',
      custom=SimpleCustom(domain='d', difficulty='e'),
    )
    back = JudgeConfig[SimpleCustom].model_validate(jc.model_dump())
    assert back.system_prompt == 'evaluate'
    assert back.run.model == 'gpt-4'


class TestVarDef:
  def test_round_trip(self) -> None:
    vd = VarDef(
      choices=['a', 'b'],
      distribution=[0.5, 0.5],
      metadata=[{'k': 1}],
    )
    back = VarDef.model_validate(vd.model_dump())
    assert back.choices == ['a', 'b']
    assert back.distribution == [0.5, 0.5]
    assert back.metadata == [{'k': 1}]

  def test_metadata_optional(self) -> None:
    vd = VarDef(choices=['x'], distribution=[1.0])
    assert vd.metadata is None


class TestCheckpointHeader:
  def test_round_trip(self) -> None:
    h = CheckpointHeader(
      subsystem='gen',
      config_hash='abc',
      created_at='2026-01-01T00:00:00Z',
    )
    back = CheckpointHeader.model_validate(h.model_dump())
    assert back.subsystem == 'gen'
    assert back.config_hash == 'abc'
    assert back.created_at == '2026-01-01T00:00:00Z'

  def test_default_type(self) -> None:
    h = CheckpointHeader(
      subsystem='s',
      config_hash='h',
      created_at='t',
    )
    assert h.type == 'header'

  def test_args_default(self) -> None:
    h = CheckpointHeader(
      subsystem='s',
      config_hash='h',
      created_at='t',
    )
    assert h.args == {}


class TestCheckpointEvent:
  def test_round_trip(self) -> None:
    ev = CheckpointEvent(
      type='done',
      id='i1',
      timestamp='2026-01-01T00:00:00Z',
      payload={'x': 1},
    )
    back = CheckpointEvent.model_validate(ev.model_dump())
    assert back.type == 'done'
    assert back.id == 'i1'
    assert back.payload == {'x': 1}

  def test_payload_default(self) -> None:
    ev = CheckpointEvent(type='x', id='i', timestamp='t')
    assert ev.payload == {}
