"""Tests for Memory base class and FileMemory implementation."""

from autopilot.core.memory import FileMemory, Memory
from autopilot.core.stage_models import MemoryContext, MemoryRecord, TrendResult
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
