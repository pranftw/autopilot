"""Tests for shared evaluation pipeline helpers."""

from autopilot.ai.evaluation.checkpoints import CheckpointManager
from autopilot.ai.evaluation.generator import GeneratorAgent
from autopilot.ai.evaluation.judge import JudgeAgent
from autopilot.ai.evaluation.pipeline import (
  EvalRunContext,
  hash_eval_config,
  log_item_failure,
  resume_from_checkpoint,
  run_parallel_items,
  write_checkpoint_header,
)
from autopilot.ai.evaluation.schemas import (
  ConversationTurn,
  DataItem,
  GeneratorConfig,
  JudgeConfig,
  JudgeInput,
  JudgeResult,
  JudgeVerdict,
)
from autopilot.ai.evaluation.steps import PythonStep
from autopilot.cli.output import Output
from pathlib import Path
from pydantic import BaseModel
from tests.doubles import make_run_config
from unittest.mock import AsyncMock, patch
import hashlib
import inspect
import pytest


class StubCustom(BaseModel):
  value: str


class StubGenConfig(BaseModel):
  prefix: str = 'STUB'


class StubJudgeConfig(BaseModel):
  threshold: float = 0.5


class StubJudgeCustom(BaseModel):
  query: str


class StubResultCustom(BaseModel):
  score: float


def _make_gen_config(total: int = 5) -> GeneratorConfig[StubGenConfig]:
  return GeneratorConfig(
    run=make_run_config(num_parallel=2),
    dataset_id='test_ds',
    seed=42,
    total_count=total,
    split_ratios={'train': 0.8, 'val': 0.1, 'test': 0.1},
    system_prompt='test',
    custom=StubGenConfig(),
  )


def _make_judge_config() -> JudgeConfig[StubJudgeConfig]:
  return JudgeConfig(
    run=make_run_config(num_parallel=2),
    system_prompt='judge test',
    custom=StubJudgeConfig(),
  )


def _make_judge_items(count: int = 3) -> list[JudgeInput[StubJudgeCustom]]:
  return [
    JudgeInput(
      id=f'J{i:04d}',
      turns=[ConversationTurn(role='user', content=f'q{i}')],
      response=f'response {i}',
      custom=StubJudgeCustom(query=f'query {i}'),
    )
    for i in range(count)
  ]


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
    return {
      'total': len(results),
      'correct': sum(1 for r in results if r.verdict and r.verdict.category == 'correct'),
    }


class TestRunParallelItemsSuccess:
  @pytest.mark.asyncio
  async def test_processes_all_items(self) -> None:
    items = [1, 2, 3, 4, 5]

    async def identity(x: int) -> int:
      return x * 2

    results = await run_parallel_items(items, identity, None, 3, Output())
    assert len(results) == len(items)
    assert set(results) == {2, 4, 6, 8, 10}

  @pytest.mark.asyncio
  async def test_on_complete_called_for_each(self) -> None:
    items = ['a', 'b', 'c']
    completed: list[str] = []

    async def upper(x: str) -> str:
      return x.upper()

    def on_complete(r: str) -> None:
      completed.append(r)

    results = await run_parallel_items(items, upper, None, 2, Output(), on_complete=on_complete)
    assert len(results) == 3
    assert set(completed) == {'A', 'B', 'C'}


class TestRunParallelItemsWithFailures:
  @pytest.mark.asyncio
  async def test_error_results_still_returned(self) -> None:
    """Simulates the pipeline pattern: process_fn catches exceptions and returns error dicts."""
    items = ['ok1', 'fail', 'ok2']

    async def process(x: str) -> dict:
      if x == 'fail':
        return {'id': x, 'error': 'simulated failure'}
      return {'id': x, 'result': x.upper()}

    completed: list[dict] = []

    def on_complete(r: dict) -> None:
      completed.append(r)

    results = await run_parallel_items(items, process, None, 3, Output(), on_complete=on_complete)
    assert len(results) == 3
    assert len(completed) == 3
    error_results = [r for r in results if 'error' in r]
    success_results = [r for r in results if 'result' in r]
    assert len(error_results) == 1
    assert len(success_results) == 2


class TestRunParallelItemsEmpty:
  @pytest.mark.asyncio
  async def test_empty_items_returns_empty(self) -> None:
    call_count = 0

    async def should_not_be_called(x: int) -> int:
      nonlocal call_count
      call_count += 1
      return x

    results = await run_parallel_items([], should_not_be_called, None, 5, Output())
    assert results == []
    assert call_count == 0


class TestRunParallelItemsLimiterNone:
  @pytest.mark.asyncio
  async def test_works_without_limiter(self) -> None:
    items = [10, 20, 30]

    async def double(x: int) -> int:
      return x * 2

    results = await run_parallel_items(items, double, None, 3, Output())
    assert len(results) == 3
    assert set(results) == {20, 40, 60}

  @pytest.mark.asyncio
  async def test_limiter_none_no_acquire_called(self) -> None:
    """When limiter=None, no rate-limit acquire path is exercised."""
    from unittest.mock import MagicMock

    fake_limiter = MagicMock()
    fake_limiter.async_acquire = AsyncMock(side_effect=RuntimeError('should not be called'))

    items = [1, 2]

    async def identity(x: int) -> int:
      return x

    results = await run_parallel_items(items, identity, None, 2, Output())
    assert len(results) == 2
    fake_limiter.async_acquire.assert_not_called()


class TestResumeFromCheckpoint:
  def test_appends_banner(self, tmp_path: Path, capsys) -> None:
    ckpt_path = tmp_path / 'checkpoint.jsonl'
    ckpt = CheckpointManager(ckpt_path)
    ckpt.save_header(config_hash='abc', subsystem='test', args={})
    ckpt.save_event('result', 'item1', {'data': 'x'})
    ckpt.save_event('result', 'item2', {'data': 'y'})

    n_done, n_remaining = resume_from_checkpoint(ckpt, Output(), total_planned=5)
    assert n_done == 2
    assert n_remaining == 3
    captured = capsys.readouterr()
    assert 'Resuming: 2 done, 3 remaining' in captured.out

  def test_zero_completed(self, tmp_path: Path, capsys) -> None:
    ckpt_path = tmp_path / 'checkpoint.jsonl'
    ckpt = CheckpointManager(ckpt_path)
    ckpt.save_header(config_hash='abc', subsystem='test', args={})

    n_done, n_remaining = resume_from_checkpoint(ckpt, Output(), total_planned=10)
    assert n_done == 0
    assert n_remaining == 10
    captured = capsys.readouterr()
    assert 'Resuming: 0 done, 10 remaining' in captured.out


class TestWriteCheckpointHeader:
  def test_creates_dir_and_file(self, tmp_path: Path) -> None:
    out_dir = tmp_path / 'nested' / 'output'
    ckpt = write_checkpoint_header(out_dir, 'hash123', 'generate', {'k': 'v'})
    assert out_dir.exists()
    assert (out_dir / 'checkpoint.jsonl').is_file()
    assert ckpt.header is not None
    assert ckpt.header['config_hash'] == 'hash123'
    assert ckpt.header['subsystem'] == 'generate'

  def test_returns_checkpoint_manager(self, tmp_path: Path) -> None:
    ckpt = write_checkpoint_header(tmp_path, 'h', 'judge', {'items': 5})
    assert isinstance(ckpt, CheckpointManager)


class TestLogItemFailure:
  def test_outputs_warning(self, capsys) -> None:
    exc = ValueError('bad input')
    log_item_failure('item-42', exc, Output())
    captured = capsys.readouterr()
    assert 'item-42 failed (ValueError): bad input' in captured.err


class TestResumedItemsCountsTotal:
  @pytest.mark.asyncio
  @patch('autopilot.ai.evaluation.judge.run_step_workflow', new_callable=AsyncMock)
  async def test_resumed_items_is_total_success(
    self, mock_workflow: AsyncMock, tmp_path: Path
  ) -> None:
    """Checkpoint holds 2 results, batch completes 1 more: resumed_items == 3."""
    mock_workflow.return_value = {'analyze': {'score': 0.9}, 'item': {}}
    ckpt_path = tmp_path / 'checkpoint.jsonl'
    cm = CheckpointManager(ckpt_path)
    cm.save_header(config_hash='abc', subsystem='judge', args={})
    cm.save_event(
      'result',
      'J0000',
      {
        'result': {
          'id': 'J0000',
          'verdict': {'category': 'correct', 'rationale': 'ok', 'confidence': 0.9},
          'custom': {'score': 0.9},
        },
      },
    )
    cm.save_event(
      'result',
      'J0001',
      {
        'result': {
          'id': 'J0001',
          'verdict': {'category': 'correct', 'rationale': 'ok', 'confidence': 0.9},
          'custom': {'score': 0.9},
        },
      },
    )

    judge = StubJudge()
    items = _make_judge_items(3)
    result = await judge.resume(ckpt_path, items, _make_judge_config(), Output())

    assert result['resumed_items'] == 3
    assert result['summary']['total'] == 3


class TestOutputDirRemovedFromResume:
  def test_generator_resume_no_output_dir(self) -> None:
    sig = inspect.signature(GeneratorAgent.resume)
    assert 'output_dir' not in sig.parameters

  def test_judge_resume_no_output_dir(self) -> None:
    sig = inspect.signature(JudgeAgent.resume)
    assert 'output_dir' not in sig.parameters


class TestForResumeSkipsCompleted:
  @pytest.mark.asyncio
  @patch('autopilot.ai.evaluation.generator.run_step_workflow', new_callable=AsyncMock)
  async def test_for_resume_true_still_processes(
    self, mock_workflow: AsyncMock, tmp_path: Path
  ) -> None:
    """With for_resume=True, even completed slots are re-processed if in remaining list."""
    mock_workflow.return_value = {'gen': {'value': 'mocked'}}
    ckpt_path = tmp_path / 'checkpoint.jsonl'
    ckpt = CheckpointManager(ckpt_path)
    ckpt.save_header(config_hash='abc', subsystem='generate', args={})
    ckpt.save_event(
      'result',
      'S0000',
      {
        'item': {
          'id': 'S0000',
          'turns': [{'role': 'user', 'content': 'hi'}],
          'custom': {'value': 'done'},
        },
      },
    )

    gen = StubGenerator()
    slot = {'id': 'S0000'}
    result = await gen._process_slot_ckpt(
      ckpt,
      gen.define_steps(_make_gen_config()),
      _make_gen_config(),
      Output(),
      slot,
      for_resume=True,
    )
    assert mock_workflow.await_count == 1
    assert 'item' in result

  @pytest.mark.asyncio
  @patch('autopilot.ai.evaluation.generator.run_step_workflow', new_callable=AsyncMock)
  async def test_for_resume_false_skips_completed(
    self, mock_workflow: AsyncMock, tmp_path: Path
  ) -> None:
    """With for_resume=False, completed slots are skipped."""
    mock_workflow.return_value = {'gen': {'value': 'mocked'}}
    ckpt_path = tmp_path / 'checkpoint.jsonl'
    ckpt = CheckpointManager(ckpt_path)
    ckpt.save_header(config_hash='abc', subsystem='generate', args={})
    ckpt.save_event(
      'result',
      'S0000',
      {
        'item': {
          'id': 'S0000',
          'turns': [{'role': 'user', 'content': 'hi'}],
          'custom': {'value': 'done'},
        },
      },
    )

    gen = StubGenerator()
    slot = {'id': 'S0000'}
    result = await gen._process_slot_ckpt(
      ckpt,
      gen.define_steps(_make_gen_config()),
      _make_gen_config(),
      Output(),
      slot,
      for_resume=False,
    )
    assert mock_workflow.await_count == 0
    assert result == {'id': 'S0000', 'skipped': True}


class TestHashEvalConfigMatchesLegacy:
  """Golden vector tests: hash_eval_config MUST match legacy sha256(model_dump_json())."""

  def test_generator_config_matches_legacy(self) -> None:
    config = _make_gen_config()
    legacy = hashlib.sha256(config.model_dump_json().encode()).hexdigest()[:16]
    new = hash_eval_config(config.model_dump(mode='json'))
    assert new == legacy

  def test_judge_config_matches_legacy(self) -> None:
    config = _make_judge_config()
    legacy = hashlib.sha256(config.model_dump_json().encode()).hexdigest()[:16]
    new = hash_eval_config(config.model_dump(mode='json'))
    assert new == legacy

  def test_generator_config_varied_params_matches_legacy(self) -> None:
    config = _make_gen_config(total=100)
    legacy = hashlib.sha256(config.model_dump_json().encode()).hexdigest()[:16]
    new = hash_eval_config(config.model_dump(mode='json'))
    assert new == legacy


class TestHashEvalConfigDeterministic:
  def test_same_config_same_hash(self) -> None:
    config = _make_gen_config()
    data = config.model_dump(mode='json')
    assert hash_eval_config(data) == hash_eval_config(data)

  def test_repeated_calls_identical(self) -> None:
    data = {'model': 'gpt-4', 'temperature': 0.7}
    results = [hash_eval_config(data) for _ in range(10)]
    assert len(set(results)) == 1


class TestHashEvalConfigDifferentConfigs:
  def test_different_values_different_hash(self) -> None:
    config_a = _make_gen_config(total=5)
    config_b = _make_gen_config(total=10)
    hash_a = hash_eval_config(config_a.model_dump(mode='json'))
    hash_b = hash_eval_config(config_b.model_dump(mode='json'))
    assert hash_a != hash_b

  def test_generator_vs_judge_different_hash(self) -> None:
    gen = hash_eval_config(_make_gen_config().model_dump(mode='json'))
    judge = hash_eval_config(_make_judge_config().model_dump(mode='json'))
    assert gen != judge


class TestHashEvalConfigNonSerializableValue:
  def test_non_serializable_raises_type_error(self) -> None:
    data = {'key': object()}
    with pytest.raises(TypeError):
      hash_eval_config(data)


class TestEvalRunContextCheckpointPath:
  def test_returns_checkpoint_jsonl(self, tmp_path: Path) -> None:
    ctx = EvalRunContext(tmp_path, 'abc123', 10)
    assert ctx.checkpoint_path() == tmp_path / 'checkpoint.jsonl'

  def test_properties_expose_constructor_args(self, tmp_path: Path) -> None:
    ctx = EvalRunContext(tmp_path, 'hash16', 42)
    assert ctx.output_dir == tmp_path
    assert ctx.config_hash == 'hash16'
    assert ctx.total_items == 42


class TestEvalRunContextLoadCompletedCount:
  def test_with_results(self, tmp_path: Path) -> None:
    ckpt_path = tmp_path / 'checkpoint.jsonl'
    mgr = CheckpointManager(ckpt_path)
    mgr.save_header(config_hash='abc', subsystem='test', args={})
    mgr.save_event('result', 'item1', {'data': 'x'})
    mgr.save_event('result', 'item2', {'data': 'y'})
    mgr.save_event('error', 'item3', {'error': 'fail'})

    ctx = EvalRunContext(tmp_path, 'abc', 5)
    assert ctx.load_completed_count() == 2

  def test_agrees_with_checkpoint_manager(self, tmp_path: Path) -> None:
    ckpt_path = tmp_path / 'checkpoint.jsonl'
    mgr = CheckpointManager(ckpt_path)
    mgr.save_header(config_hash='abc', subsystem='test', args={})
    mgr.save_event('result', 'a', {})
    mgr.save_event('result', 'b', {})
    mgr.save_event('skip', 'c', {'reason': 'rejected'})

    ctx = EvalRunContext(tmp_path, 'abc', 10)
    assert ctx.load_completed_count() == len(mgr.completed_ids())


class TestEvalRunContextMissingCheckpointFile:
  def test_returns_zero(self, tmp_path: Path) -> None:
    ctx = EvalRunContext(tmp_path, 'abc', 5)
    assert ctx.load_completed_count() == 0


class TestEvalRunContextEmptyCheckpointFile:
  def test_returns_zero(self, tmp_path: Path) -> None:
    ckpt_path = tmp_path / 'checkpoint.jsonl'
    ckpt_path.write_text('', encoding='utf-8')
    ctx = EvalRunContext(tmp_path, 'abc', 5)
    assert ctx.load_completed_count() == 0


class TestEvalRunContextShouldSkip:
  def test_completed_item_skipped(self, tmp_path: Path) -> None:
    ckpt_path = tmp_path / 'checkpoint.jsonl'
    mgr = CheckpointManager(ckpt_path)
    mgr.save_header(config_hash='abc', subsystem='test', args={})
    mgr.save_event('result', 'done_item', {'data': 'ok'})

    ctx = EvalRunContext(tmp_path, 'abc', 5)
    assert ctx.should_skip('done_item') is True

  def test_incomplete_item_not_skipped(self, tmp_path: Path) -> None:
    ckpt_path = tmp_path / 'checkpoint.jsonl'
    mgr = CheckpointManager(ckpt_path)
    mgr.save_header(config_hash='abc', subsystem='test', args={})
    mgr.save_event('result', 'done_item', {'data': 'ok'})

    ctx = EvalRunContext(tmp_path, 'abc', 5)
    assert ctx.should_skip('pending_item') is False

  def test_agrees_with_checkpoint_manager(self, tmp_path: Path) -> None:
    ckpt_path = tmp_path / 'checkpoint.jsonl'
    mgr = CheckpointManager(ckpt_path)
    mgr.save_header(config_hash='abc', subsystem='test', args={})
    mgr.save_event('result', 'x1', {})
    mgr.save_event('error', 'x2', {'error': 'fail'})
    mgr.save_event('result', 'x3', {})

    ctx = EvalRunContext(tmp_path, 'abc', 10)
    completed = mgr.completed_ids()
    for item_id in ['x1', 'x2', 'x3', 'x4']:
      assert ctx.should_skip(item_id) == (item_id in completed)

  def test_missing_checkpoint_never_skips(self, tmp_path: Path) -> None:
    ctx = EvalRunContext(tmp_path, 'abc', 5)
    assert ctx.should_skip('any_item') is False
