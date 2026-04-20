"""Tests for stage data models to_dict/from_dict round-trips."""

from autopilot.core.stage_models import (
  BlockedStrategy,
  ChangeProposal,
  CostEntry,
  EpochMetrics,
  ExperimentSummaryData,
  JudgeValidation,
  MemoryContext,
  MemoryRecord,
  ProposalVerdict,
  RegressionAnalysis,
  TrendResult,
)


class TestMemoryRecord:
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
    assert r.content == ''

  def test_unknown_keys_ignored(self):
    r = MemoryRecord.from_dict({'epoch': 1, 'unknown_key': 'ignored'})
    assert r.epoch == 1

  def test_timestamp_auto_populated(self):
    r = MemoryRecord(epoch=1, outcome='worked')
    assert r.timestamp != ''


class TestTrendResult:
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


class TestMemoryContext:
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


class TestBlockedStrategy:
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


class TestEpochMetrics:
  def test_round_trip(self):
    em = EpochMetrics(epoch=1, split='val', total=10, passed=8, accuracy=0.8)
    d = em.to_dict()
    em2 = EpochMetrics.from_dict(d)
    assert em2.epoch == 1
    assert em2.accuracy == 0.8


class TestProposalVerdict:
  def test_round_trip(self):
    v = ProposalVerdict(proposal_id='p1', items_tested=10, items_fixed=3, verdict='fix_confirmed')
    d = v.to_dict()
    v2 = ProposalVerdict.from_dict(d)
    assert v2.proposal_id == 'p1'
    assert v2.verdict == 'fix_confirmed'


class TestChangeProposal:
  def test_round_trip_without_verification(self):
    p = ChangeProposal(proposal_id='p1', hypothesis='test', epoch=1)
    d = p.to_dict()
    p2 = ChangeProposal.from_dict(d)
    assert p2.proposal_id == 'p1'
    assert p2.verification is None

  def test_round_trip_with_verification(self):
    v = ProposalVerdict(proposal_id='p1', verdict='fix_confirmed')
    p = ChangeProposal(proposal_id='p1', verification=v)
    d = p.to_dict()
    p2 = ChangeProposal.from_dict(d)
    assert p2.verification is not None
    assert p2.verification.verdict == 'fix_confirmed'

  def test_timestamp_auto(self):
    p = ChangeProposal(proposal_id='p1')
    assert p.timestamp != ''


class TestJudgeValidation:
  def test_round_trip(self):
    j = JudgeValidation(judge_id='j1', agreement_rate=0.95, confidence='high')
    d = j.to_dict()
    j2 = JudgeValidation.from_dict(d)
    assert j2.judge_id == 'j1'
    assert j2.confidence == 'high'


class TestRegressionAnalysis:
  def test_round_trip(self):
    r = RegressionAnalysis(
      epoch=3,
      overall_verdict='net_regression',
      per_category_deltas={'billing': -0.1},
      regressions=[{'metric': 'accuracy', 'delta': -0.1}],
    )
    d = r.to_dict()
    r2 = RegressionAnalysis.from_dict(d)
    assert r2.overall_verdict == 'net_regression'
    assert r2.per_category_deltas == {'billing': -0.1}


class TestCostEntry:
  def test_round_trip(self):
    c = CostEntry(epoch=1, wall_clock_s=5.0, api_calls=3, tokens_used=1000)
    d = c.to_dict()
    c2 = CostEntry.from_dict(d)
    assert c2.wall_clock_s == 5.0
    assert c2.tokens_used == 1000


class TestExperimentSummaryData:
  def test_round_trip_without_cost(self):
    s = ExperimentSummaryData(slug='test', total_epochs=5, best_epoch=3)
    d = s.to_dict()
    s2 = ExperimentSummaryData.from_dict(d)
    assert s2.slug == 'test'
    assert s2.cost_total is None

  def test_round_trip_with_cost(self):
    cost = CostEntry(epoch=0, wall_clock_s=30.0)
    reg = RegressionAnalysis(epoch=2, overall_verdict='net_regression')
    s = ExperimentSummaryData(slug='test', cost_total=cost, regressions=[reg])
    d = s.to_dict()
    s2 = ExperimentSummaryData.from_dict(d)
    assert s2.cost_total is not None
    assert s2.cost_total.wall_clock_s == 30.0
    assert len(s2.regressions) == 1
    assert isinstance(s2.regressions[0], RegressionAnalysis)

  def test_field_type_preservation(self):
    s = ExperimentSummaryData(final_metrics={'accuracy': 0.95}, memory_entries=10)
    d = s.to_dict()
    s2 = ExperimentSummaryData.from_dict(d)
    assert isinstance(s2.final_metrics['accuracy'], float)
    assert isinstance(s2.memory_entries, int)
