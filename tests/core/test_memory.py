"""Tests for Memory base class, FileMemory, MemoryCallback, and memory data models."""

from autopilot.core.callbacks.memory import MemoryCallback
from autopilot.core.memory import (
  BlockedStrategy,
  FileMemory,
  Memory,
  MemoryContext,
  MemoryRecord,
  TrendResult,
)
from autopilot.core.models import Result
from autopilot.core.optimizer import Optimizer
from autopilot.core.trainer import Trainer
from unittest.mock import MagicMock
import json


class TestMemoryBase:
  def test_learn_noop(self):
    m = Memory()
    m.learn(epoch=1, outcome='worked')

  def test_recall_empty(self):
    m = Memory()
    assert m.recall() == []

  def test_trends_empty(self):
    m = Memory()
    result = m.trends(metric='accuracy')
    assert isinstance(result, TrendResult)

  def test_context_empty(self):
    m = Memory()
    result = m.context(epoch=1)
    assert isinstance(result, MemoryContext)
    assert result.epoch == 1

  def test_block_strategy_noop(self):
    m = Memory()
    m.block_strategy('x')
    assert not m.is_strategy_blocked('x')

  def test_state_dict_empty(self):
    m = Memory()
    assert m.state_dict() == {}


class TestFileMemory:
  def test_learn_and_recall_all(self, tmp_path):
    m = FileMemory(tmp_path)
    m.learn(epoch=1, outcome='worked', metrics={'accuracy': 0.7})
    m.learn(epoch=2, outcome='failed', metrics={'accuracy': 0.5})
    m.learn(epoch=3, outcome='worked', metrics={'accuracy': 0.9})
    records = m.recall()
    assert len(records) == 3

  def test_recall_returns_typed_records(self, tmp_path):
    m = FileMemory(tmp_path)
    m.learn(epoch=1, outcome='worked', metrics={'accuracy': 0.8})
    records = m.recall()
    assert isinstance(records[0], MemoryRecord)
    assert records[0].epoch == 1
    assert records[0].outcome == 'worked'
    assert records[0].metrics == {'accuracy': 0.8}

  def test_recall_by_category(self, tmp_path):
    m = FileMemory(tmp_path)
    m.learn(epoch=1, outcome='worked', category='eval')
    m.learn(epoch=2, outcome='failed', category='train')
    m.learn(epoch=3, outcome='worked', category='eval')
    records = m.recall(category='eval')
    assert len(records) == 2

  def test_recall_by_node(self, tmp_path):
    m = FileMemory(tmp_path)
    m.learn(epoch=1, outcome='worked', node='prompts')
    m.learn(epoch=2, outcome='worked', node='config')
    records = m.recall(node='prompts')
    assert len(records) == 1

  def test_recall_by_category_and_node(self, tmp_path):
    m = FileMemory(tmp_path)
    m.learn(epoch=1, outcome='worked', category='eval', node='prompts')
    m.learn(epoch=2, outcome='worked', category='train', node='prompts')
    m.learn(epoch=3, outcome='worked', category='eval', node='config')
    records = m.recall(category='eval', node='prompts')
    assert len(records) == 1

  def test_recall_by_epoch_range(self, tmp_path):
    m = FileMemory(tmp_path)
    for i in range(1, 6):
      m.learn(epoch=i, outcome='worked')
    records = m.recall(epoch_min=2, epoch_max=4)
    assert len(records) == 3
    assert all(2 <= r.epoch <= 4 for r in records)

  def test_recall_empty(self, tmp_path):
    m = FileMemory(tmp_path)
    assert m.recall() == []

  def test_recall_by_outcome(self, tmp_path):
    m = FileMemory(tmp_path)
    m.learn(epoch=1, outcome='worked')
    m.learn(epoch=2, outcome='failed')
    m.learn(epoch=3, outcome='partial')
    assert len(m.recall(outcome='failed')) == 1

  def test_recall_metrics_queryable(self, tmp_path):
    m = FileMemory(tmp_path)
    m.learn(epoch=1, outcome='worked', metrics={'accuracy': 0.85})
    records = m.recall()
    assert records[0].metrics['accuracy'] == 0.85

  def test_learn_structured_metrics(self, tmp_path):
    m = FileMemory(tmp_path)
    m.learn(epoch=1, outcome='worked', metrics={'accuracy': 0.85, 'latency_ms': 120.0})
    records = m.recall()
    assert records[0].metrics == {'accuracy': 0.85, 'latency_ms': 120.0}

  def test_trends_returns_typed_result(self, tmp_path):
    m = FileMemory(tmp_path)
    m.learn(epoch=1, outcome='worked', metrics={'accuracy': 0.5})
    m.learn(epoch=2, outcome='worked', metrics={'accuracy': 0.6})
    result = m.trends(metric='accuracy')
    assert isinstance(result, TrendResult)
    assert result.metric == 'accuracy'

  def test_trends_improving(self, tmp_path):
    m = FileMemory(tmp_path)
    for i in range(1, 6):
      m.learn(epoch=i, outcome='worked', metrics={'accuracy': 0.5 + i * 0.1})
    result = m.trends(metric='accuracy')
    assert result.direction == 'improving'

  def test_trends_plateau(self, tmp_path):
    m = FileMemory(tmp_path)
    for i in range(1, 6):
      m.learn(epoch=i, outcome='worked', metrics={'accuracy': 0.8})
    result = m.trends(metric='accuracy')
    assert result.direction == 'plateau'

  def test_trends_oscillating(self, tmp_path):
    m = FileMemory(tmp_path)
    values = [0.5, 0.9, 0.5, 0.9, 0.5]
    for i, v in enumerate(values, 1):
      m.learn(epoch=i, outcome='worked', metrics={'accuracy': v})
    result = m.trends(metric='accuracy')
    assert result.direction == 'oscillating'

  def test_trends_empty(self, tmp_path):
    m = FileMemory(tmp_path)
    result = m.trends(metric='accuracy')
    assert result.values == []
    assert result.epochs == []

  def test_trends_window(self, tmp_path):
    m = FileMemory(tmp_path)
    for i in range(1, 6):
      m.learn(epoch=i, outcome='worked', metrics={'accuracy': i * 0.1})
    result = m.trends(metric='accuracy', window=3)
    assert len(result.values) == 3
    assert len(result.epochs) == 3

  def test_context_returns_typed_result(self, tmp_path):
    m = FileMemory(tmp_path)
    m.learn(epoch=1, outcome='worked', metrics={'accuracy': 0.7})
    ctx = m.context(epoch=2)
    assert isinstance(ctx, MemoryContext)
    assert ctx.epoch == 2

  def test_context_assembly(self, tmp_path):
    m = FileMemory(tmp_path)
    m.learn(epoch=1, outcome='failed', strategy='add_rule', metrics={'accuracy': 0.5})
    m.learn(epoch=2, outcome='worked', strategy='fix_regex', metrics={'accuracy': 0.7})
    m.block_strategy('bad_approach')
    ctx = m.context(epoch=3)
    assert 'add_rule' in ctx.strategies_tried
    assert 'fix_regex' in ctx.strategies_tried
    assert 'bad_approach' in ctx.blocked
    assert ctx.total_records == 2

  def test_block_strategy(self, tmp_path):
    m = FileMemory(tmp_path)
    m.block_strategy('dangerous', reason='causes regression', epoch=2)
    assert m.is_strategy_blocked('dangerous')

  def test_block_strategy_records_metadata(self, tmp_path):
    m = FileMemory(tmp_path)
    m.block_strategy('dangerous', reason='causes regression', epoch=2)
    blocklist = m._load_blocklist()
    assert blocklist[0].reason == 'causes regression'
    assert blocklist[0].epoch_blocked == 2
    assert blocklist[0].timestamp != ''

  def test_block_duplicate(self, tmp_path):
    m = FileMemory(tmp_path)
    m.block_strategy('x')
    m.block_strategy('x')
    assert len(m.blocked_strategies()) == 1

  def test_is_blocked_false(self, tmp_path):
    m = FileMemory(tmp_path)
    assert not m.is_strategy_blocked('anything')

  def test_blocked_strategies_list(self, tmp_path):
    m = FileMemory(tmp_path)
    m.block_strategy('a')
    m.block_strategy('b')
    m.block_strategy('c')
    assert set(m.blocked_strategies()) == {'a', 'b', 'c'}

  def test_state_dict_round_trip(self, tmp_path):
    m = FileMemory(tmp_path)
    m.learn(epoch=1, outcome='worked', metrics={'accuracy': 0.8})
    m.block_strategy('bad')
    state = m.state_dict()

    m2 = FileMemory(tmp_path / 'new')
    m2.load_state_dict(state)
    records = m2.recall()
    assert len(records) == 1
    assert records[0].metrics == {'accuracy': 0.8}
    assert m2.is_strategy_blocked('bad')

  def test_file_memory_persistence(self, tmp_path):
    m1 = FileMemory(tmp_path)
    m1.learn(epoch=1, outcome='worked', metrics={'accuracy': 0.7})

    m2 = FileMemory(tmp_path)
    records = m2.recall()
    assert len(records) == 1
    assert isinstance(records[0], MemoryRecord)

  def test_file_memory_jsonl_structure(self, tmp_path):
    m = FileMemory(tmp_path)
    m.learn(epoch=1, outcome='worked', metrics={'accuracy': 0.8}, category='eval')
    lines = (tmp_path / 'knowledge_base.jsonl').read_text().strip().splitlines()
    data = json.loads(lines[0])
    assert data['epoch'] == 1
    assert data['outcome'] == 'worked'
    assert data['metrics'] == {'accuracy': 0.8}
    assert data['category'] == 'eval'

  def test_file_memory_concurrent_learns(self, tmp_path):
    m1 = FileMemory(tmp_path)
    m1.learn(epoch=1, outcome='worked')
    m2 = FileMemory(tmp_path)
    m2.learn(epoch=2, outcome='failed')
    m3 = FileMemory(tmp_path)
    assert len(m3.recall()) == 2

  def test_file_memory_empty_dir(self, tmp_path):
    m = FileMemory(tmp_path / 'empty')
    assert m.recall() == []
    assert m.blocked_strategies() == []

  def test_file_memory_corrupted_jsonl(self, tmp_path):
    kb_path = tmp_path / 'knowledge_base.jsonl'
    kb_path.write_text(
      '{"epoch": 1, "outcome": "worked"}\nnot json\n{"epoch": 2, "outcome": "failed"}\n',
    )
    m = FileMemory(tmp_path)
    records = m.recall()
    assert len(records) == 2

  def test_file_memory_corrupted_blocklist(self, tmp_path):
    bl_path = tmp_path / 'strategy_blocklist.json'
    bl_path.write_text('not valid json')
    m = FileMemory(tmp_path)
    assert m.blocked_strategies() == []


class TestMemoryRecordRoundTrip:
  def test_round_trip(self):
    r = MemoryRecord(epoch=1, outcome='worked', metrics={'accuracy': 0.85}, category='eval')
    d = r.to_dict()
    r2 = MemoryRecord.from_dict(d)
    assert r2.epoch == 1
    assert r2.outcome == 'worked'
    assert r2.metrics == {'accuracy': 0.85}
    assert r2.category == 'eval'

  def test_metrics_preserved_as_floats(self):
    r = MemoryRecord(metrics={'a': 0.5, 'b': 1.0})
    d = r.to_dict()
    r2 = MemoryRecord.from_dict(d)
    assert isinstance(r2.metrics['a'], float)
    assert isinstance(r2.metrics['b'], float)

  def test_defaults(self):
    r = MemoryRecord.from_dict({})
    assert r.epoch == 0
    assert r.metrics == {}
    assert r.content is None

  def test_unknown_keys_ignored(self):
    r = MemoryRecord.from_dict({'epoch': 1, 'unknown_key': 'ignored'})
    assert r.epoch == 1

  def test_timestamp_auto_populated(self):
    r = MemoryRecord(epoch=1, outcome='worked')
    assert r.timestamp != ''


class TestTrendResultRoundTrip:
  def test_round_trip(self):
    t = TrendResult(
      metric='accuracy',
      direction='improving',
      rate=0.05,
      values=[0.7, 0.75, 0.8],
      epochs=[1, 2, 3],
    )
    d = t.to_dict()
    t2 = TrendResult.from_dict(d)
    assert t2.metric == 'accuracy'
    assert t2.direction == 'improving'
    assert t2.values == [0.7, 0.75, 0.8]
    assert t2.epochs == [1, 2, 3]


class TestMemoryContextRoundTrip:
  def test_round_trip(self):
    rec = MemoryRecord(epoch=1, outcome='failed')
    trend = TrendResult(metric='accuracy', direction='improving')
    ctx = MemoryContext(
      epoch=3,
      top_failures=[rec],
      strategies_tried=['a', 'b'],
      blocked=['c'],
      trends={'accuracy': trend},
      total_records=5,
    )
    d = ctx.to_dict()
    ctx2 = MemoryContext.from_dict(d)
    assert ctx2.epoch == 3
    assert len(ctx2.top_failures) == 1
    assert isinstance(ctx2.top_failures[0], MemoryRecord)
    assert ctx2.strategies_tried == ['a', 'b']
    assert isinstance(ctx2.trends['accuracy'], TrendResult)


class TestBlockedStrategyRoundTrip:
  def test_round_trip(self):
    b = BlockedStrategy(strategy='x', reason='bad', epoch_blocked=2)
    d = b.to_dict()
    b2 = BlockedStrategy.from_dict(d)
    assert b2.strategy == 'x'
    assert b2.reason == 'bad'
    assert b2.epoch_blocked == 2

  def test_timestamp_auto(self):
    b = BlockedStrategy(strategy='x')
    assert b.timestamp != ''


# memorycallback tests


def _memory_trainer(
  *,
  fit_context: dict | None = None,
  experiment: MagicMock | None = None,
) -> MagicMock:
  trainer = MagicMock()
  trainer.fit_context = fit_context if fit_context is not None else {}
  trainer.experiment = experiment
  return trainer


class TestMemoryCallback:
  def test_records_structured_learnings(self, tmp_path):
    memory = FileMemory(tmp_path)
    cb = MemoryCallback(memory)
    trainer = _memory_trainer()
    result = Result(metrics={'accuracy': 0.8}, passed=True)
    cb.on_epoch_end(trainer=trainer, epoch=1, result=result)
    records = memory.recall()
    assert len(records) == 1
    assert records[0].epoch == 1
    assert records[0].outcome == 'worked'
    assert records[0].metrics == {'accuracy': 0.8}
    assert records[0].category == 'epoch_result'

  def test_default_category(self, tmp_path):
    memory = FileMemory(tmp_path)
    cb = MemoryCallback(memory)
    trainer = _memory_trainer()
    cb.on_epoch_end(trainer=trainer, epoch=1, result=Result(passed=True))
    records = memory.recall()
    assert records[0].category == 'epoch_result'

  def test_custom_category(self, tmp_path):
    memory = FileMemory(tmp_path)
    cb = MemoryCallback(memory, default_category='custom')
    trainer = _memory_trainer()
    cb.on_epoch_end(trainer=trainer, epoch=1, result=Result(passed=True))
    records = memory.recall()
    assert records[0].category == 'custom'

  def test_recorded_entry_has_metrics(self, tmp_path):
    memory = FileMemory(tmp_path)
    cb = MemoryCallback(memory)
    trainer = _memory_trainer()
    result = Result(metrics={'accuracy': 0.9, 'f1': 0.85}, passed=True)
    cb.on_epoch_end(trainer=trainer, epoch=1, result=result)
    records = memory.recall()
    assert records[0].metrics['accuracy'] == 0.9
    assert records[0].metrics['f1'] == 0.85

  def test_result_metrics_includes_merged_val_metrics(self, tmp_path):
    memory = FileMemory(tmp_path)
    cb = MemoryCallback(memory)
    trainer = _memory_trainer()
    result = Result(
      metrics={'loss': 0.3, 'val_loss': 0.4, 'val_accuracy': 0.88},
      passed=True,
    )
    cb.on_epoch_end(trainer=trainer, epoch=1, result=result)
    records = memory.recall()
    assert records[0].metrics == result.metrics
    assert records[0].metrics['val_accuracy'] == 0.88

  def test_rollback_outcome_when_experiment_should_rollback(self, tmp_path):
    memory = FileMemory(tmp_path)
    cb = MemoryCallback(memory)
    exp = MagicMock()
    exp.should_rollback = True
    trainer = _memory_trainer(experiment=exp)
    result = Result(metrics={'val_accuracy': 0.9}, passed=True)
    cb.on_epoch_end(trainer=trainer, epoch=1, result=result)
    records = memory.recall()
    assert records[0].outcome == 'rollback'

  def test_strategy_from_fit_context(self, tmp_path):
    memory = FileMemory(tmp_path)
    cb = MemoryCallback(memory)
    trainer = _memory_trainer(fit_context={'strategy': 'try_smaller_batch'})
    result = Result(metrics={'accuracy': 0.8}, passed=True)
    cb.on_epoch_end(trainer=trainer, epoch=1, result=result)
    records = memory.recall()
    assert records[0].strategy == 'try_smaller_batch'

  def test_populates_blocked_via_method(self, tmp_path):
    memory = FileMemory(tmp_path)
    memory.block_strategy('bad_approach')
    cb = MemoryCallback(memory)
    opt = Optimizer(parameters=[], lr=1.0)
    trainer = MagicMock()
    trainer.optimizer = opt
    cb.on_before_optimizer_step(trainer=trainer)
    assert opt.is_strategy_blocked('bad_approach')

  def test_no_blocklist(self, tmp_path):
    memory = FileMemory(tmp_path)
    cb = MemoryCallback(memory)
    opt = Optimizer(parameters=[], lr=1.0)
    trainer = MagicMock()
    trainer.optimizer = opt
    cb.on_before_optimizer_step(trainer=trainer)
    assert opt.blocked_strategies == frozenset()

  def test_accesses_optimizer_via_trainer_property(self, tmp_path):
    memory = FileMemory(tmp_path)
    memory.block_strategy('x')
    cb = MemoryCallback(memory)
    opt = Optimizer(parameters=[], lr=1.0)
    trainer = Trainer()
    trainer._optimizer = opt
    cb.on_before_optimizer_step(trainer=trainer)
    assert opt.is_strategy_blocked('x')

  def test_state_dict_delegates_to_memory(self, tmp_path):
    memory = FileMemory(tmp_path)
    memory.learn(epoch=1, outcome='worked')
    cb = MemoryCallback(memory)
    state = cb.state_dict()
    assert 'records' in state
    assert len(state['records']) == 1

  def test_memory_file_written_structured(self, tmp_path):
    memory = FileMemory(tmp_path)
    cb = MemoryCallback(memory)
    trainer = _memory_trainer()
    result = Result(metrics={'accuracy': 0.7}, passed=False)
    cb.on_epoch_end(trainer=trainer, epoch=1, result=result)

    lines = (tmp_path / 'knowledge_base.jsonl').read_text().strip().splitlines()
    data = json.loads(lines[0])
    assert data['epoch'] == 1
    assert data['outcome'] == 'failed'
    assert data['metrics'] == {'accuracy': 0.7}
    assert data['category'] == 'epoch_result'

  def test_load_state_dict_round_trip(self, tmp_path):
    memory = FileMemory(tmp_path)
    memory.learn(epoch=1, outcome='worked', metrics={'accuracy': 0.8})
    memory.learn(epoch=2, outcome='failed', metrics={'accuracy': 0.6})
    cb = MemoryCallback(memory)
    state = cb.state_dict()

    memory2 = FileMemory(tmp_path / 'other')
    cb2 = MemoryCallback(memory2)
    cb2.load_state_dict(state)
    state2 = cb2.state_dict()
    assert len(state2['records']) == 2
